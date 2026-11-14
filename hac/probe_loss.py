"""Loss-resilience probe for a trained+decoded HAC model (no change to HAC itself).

Loads the decoded model that HAC saved after its own encode->decode round trip
(point_cloud.ply + checkpoint.pth at iteration 30000), then renders the test views with
HAC's renderer while some anchors are withheld (passed as invisible in visible_mask, so
none of their neural Gaussians is generated). Metrics use HAC's own psnr/ssim/lpips(vgg)
functions on float renders, exactly as in HAC's training_report test evaluation.

Modes
  intact        : no loss (reference for this script)
  spatial_block : anchors binned on a G^3 grid over the robust (1-99 pct) anchor box;
                  round(B/6) of the B non-empty blocks are chosen at random and dropped
  hac_chunk     : HAC codes anchors in coding chunks of 3000 (feat_s/scaling_s/offsets_s.b);
                  round(S/6) of the S chunks are dropped (their files lost, everything else kept)
Byte-level check (feat_<k>.b): reconstruct x_bound from _quantized_v.npy and the decoded
anchors, decode the intact file (must match the decoded model bit-exactly), then decode
after (a) flipping one byte and (b) zero-filling a 1024-byte run in the middle of the
arithmetic-coded payload, and count corrupted symbols / anchors.
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import prefilter_voxel, render
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.encodings import Q_anchor
from utils.encodings_cuda import decoder_gaussian_chunk

ap = argparse.ArgumentParser()
lp = ModelParams(ap)
pp = PipelineParams(ap)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--grid", type=int, nargs="+", default=[8, 4])
ap.add_argument("--frac", type=float, default=1 / 6)
ap.add_argument("--out", type=str, required=True)
ap.add_argument("--byte_chunk", type=int, default=-1)
args = ap.parse_args()
dataset, pipe = lp.extract(args), pp.extract(args)
torch.manual_seed(0)

with torch.no_grad():
    g = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                      dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                      n_features_per_level=4, log2_hashmap_size=13, log2_hashmap_size_2D=15,
                      decoded_version=True)
    scene = Scene(dataset, g, load_iteration=-1, shuffle=False)
    g.eval()
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    cams = scene.getTestCameras()
    valid = g.get_mask_anchor.clone()                  # anchors with >=1 active Gaussian
    N = int(valid.sum().item()); N_full = valid.shape[0]
    assert bool(valid[:N].all()) and not bool(valid[N:].any()), "decoded anchors must be the first N rows"
    n_gauss = g._mask[:, :, 0].sum(dim=1)              # active neural Gaussians per anchor

    def evaluate(keep):
        P = S = L = 0.0
        for cam in cams:
            vis = prefilter_voxel(cam, g, pipe, bg)
            if keep is not None:
                vis = vis & keep
            img = torch.clamp(render(cam, g, pipe, bg, visible_mask=vis)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            P += psnr(img, gt).mean().double().item()
            S += ssim(img, gt).mean().double().item()
            L += lpips(img, gt, net_type="vgg").mean().double().item()
        n = len(cams)
        return {"PSNR": P / n, "SSIM": S / n, "LPIPS": L / n}

    res = {"model_path": dataset.model_path, "n_test_views": len(cams), "N_anchors_coded": N,
           "N_gaussians_active": int(n_gauss.sum().item()), "frac": args.frac}
    t0 = time.time(); res["intact"] = evaluate(None); res["t_eval_s"] = time.time() - t0
    print("intact", res["intact"], flush=True)

    A = g._anchor[:N]
    lo = torch.quantile(A, 0.01, dim=0); hi = torch.quantile(A, 0.99, dim=0)
    for G in args.grid:
        cell = torch.clamp(((A - lo) / (hi - lo) * G).floor().long(), 0, G - 1)
        bid = cell[:, 0] * G * G + cell[:, 1] * G + cell[:, 2]
        blocks = torch.unique(bid); B = blocks.numel(); k = int(round(B * args.frac))
        runs = []
        for s in range(args.seeds):
            gen = torch.Generator().manual_seed(1000 + s)
            drop_b = blocks[torch.randperm(B, generator=gen)[:k].to(blocks.device)]
            drop = torch.zeros(N_full, dtype=torch.bool, device="cuda"); drop[:N] = torch.isin(bid, drop_b)
            r = evaluate(~drop)
            r.update({"seed": s, "anchors_dropped_frac": drop.sum().item() / N,
                      "gaussians_dropped_frac": (n_gauss[drop].sum() / n_gauss.sum()).item()})
            runs.append(r); print("block", G, r, flush=True)
        res[f"spatial_block_G{G}"] = {"n_nonempty_blocks": B, "n_dropped_blocks": k, "runs": runs}

    MB = 3000
    S_ = (N + MB - 1) // MB; k = int(round(S_ * args.frac)); runs = []
    chunk_id = torch.arange(N, device="cuda") // MB
    for s in range(args.seeds):
        gen = torch.Generator().manual_seed(2000 + s)
        drop_c = torch.randperm(S_, generator=gen)[:k].cuda()
        drop = torch.zeros(N_full, dtype=torch.bool, device="cuda"); drop[:N] = torch.isin(chunk_id, drop_c)
        r = evaluate(~drop)
        r.update({"seed": s, "anchors_dropped_frac": drop.sum().item() / N,
                  "gaussians_dropped_frac": (n_gauss[drop].sum() / n_gauss.sum()).item()})
        runs.append(r); print("chunk", r, flush=True)
    res["hac_chunk"] = {"chunk_anchors": MB, "n_chunks": S_, "n_dropped_chunks": k, "runs": runs}

    # ---------- byte-level corruption of one feat chunk file ----------
    bs = os.path.join(dataset.model_path, "bitstreams")
    q = np.load(os.path.join(bs, "_quantized_v.npy")).astype(np.int64)
    assert q.shape[0] == N
    Ad = A.double().cpu().numpy()
    xmin = np.zeros(3); itv = np.zeros(3)
    for c in range(3):
        X = np.stack([q[:, c].astype(np.float64), np.ones(N)], 1)
        itv[c], xmin[c] = np.linalg.lstsq(X, Ad[:, c], rcond=None)[0]
    g.x_bound_min = torch.tensor(xmin, dtype=torch.float32, device="cuda")[None]
    g.x_bound_max = torch.tensor((itv - 1e-6) / Q_anchor + xmin, dtype=torch.float32, device="cuda")[None]
    interval = (g.x_bound_max - g.x_bound_min) * Q_anchor + 1e-6
    A_re = torch.from_numpy(q).cuda().to(torch.int32) * interval + g.x_bound_min
    res["xbound_reconstruction_max_abs_err"] = (A_re - A).abs().max().item()

    kc = args.byte_chunk if args.byte_chunk >= 0 else S_ // 2
    s0, s1 = kc * MB, min((kc + 1) * MB, N)
    ctx = g.get_grid_mlp(g.calc_interp_feat(A[s0:s1]))
    mean, scale, _, _, _, _, Qf_adj, _, _ = torch.split(
        ctx, [g.feat_dim, g.feat_dim, 6, 6, 3 * g.n_offsets, 3 * g.n_offsets, 1, 1, 1], dim=-1)
    Qf = (1 * (1 + torch.tanh(Qf_adj.contiguous()))).repeat(1, g.feat_dim).view(-1)
    mean = mean.contiguous().view(-1); scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
    fname = os.path.join(bs, f"feat_{kc}.b")
    ref = g._anchor_feat[s0:s1].reshape(-1)

    def dec(path):
        return decoder_gaussian_chunk(mean, scale, Qf, file_name=path).view(-1)

    raw = open(fname.replace(".b", "_0.b"), "rb").read()
    lc = int(np.frombuffer(raw[8:12], dtype=np.int32)[0]); hdr = 12 + lc
    pay = len(raw) - hdr
    out = {"chunk": kc, "file_bytes": len(raw), "payload_bytes": pay,
           "cuda_substreams": lc // 4, "symbols": int(ref.numel())}
    d0 = dec(fname)
    out["intact_symbol_mismatch"] = int((d0 - ref).abs().gt(1e-4).sum().item())
    tmpdir = os.path.join(bs, "..", "probe_tmp"); os.makedirs(tmpdir, exist_ok=True)
    for name, mutate in [("flip_1_byte", lambda b: b[:hdr + pay // 2] + bytes([b[hdr + pay // 2] ^ 0xFF]) + b[hdr + pay // 2 + 1:]),
                         ("zero_1024_bytes", lambda b: b[:hdr + pay // 2] + bytes(1024) + b[hdr + pay // 2 + 1024:])]:
        p = os.path.join(tmpdir, "feat_x.b")
        open(p.replace(".b", "_0.b"), "wb").write(mutate(raw))
        d = dec(p)
        bad = (d - d0).abs().gt(1e-4).view(-1, g.feat_dim)
        good_feat = g._anchor_feat.clone()
        g._anchor_feat[s0:s1] = d.view(-1, g.feat_dim)
        r = evaluate(None)
        g._anchor_feat.copy_(good_feat)
        out[name] = {"symbols_wrong": int(bad.sum().item()), "symbols_wrong_frac": bad.float().mean().item(),
                     "anchors_wrong": int(bad.any(1).sum().item()), "anchors_in_chunk": s1 - s0,
                     "first_wrong_anchor": int(bad.any(1).nonzero()[0].item()) if bad.any() else None,
                     "last_wrong_anchor": int(bad.any(1).nonzero()[-1].item()) if bad.any() else None,
                     "render": r}
        print(name, out[name], flush=True)
    import shutil; shutil.rmtree(tmpdir)
    res["byte_level"] = out

json.dump(res, open(args.out, "w"), indent=1)
print("saved", args.out)
