"""Per-frame end-to-end latency of the edge pipeline on real frame paths and cold starts (one process, one CUDA stream).

Every frame is replayed through the complete edge pipeline with its real data:
  receive complete (bitstream in pinned host memory)
  -> [dec]  host-to-device copy and GPU rANS decoding (range_code2.py kernel, groups of 6 channels, tokens of 64 Morton-
            consecutive anchors) of the received base-layer and enhancement-layer tokens, dequantization and update of
            the edge attribute table and cache
  -> [enc]  deployment variant only: encodings of the newly received anchors (cached keys/values)
  -> [gath] visible set of the view, lost (not cached) anchors and the gathering of their neighbourhoods
  -> [conc] concealment network, FP16 eager, batches of 16,384 lost anchors
  -> [ren]  decoding of the visible anchors into neural Gaussians and rasterization (model.render, FP32)
Stage times come from CUDA events on the one stream; the per-frame total T_edge is wall clock (time.perf_counter) from
receive complete to the synchronized rendered image, so it includes host overhead and the implicit synchronizations of
the implementation. The bitstreams are encoded per frame on the CPU before the timed region (cloud side), and the
decoded table is checked bit-exact against the quantized attributes once per scene. E2E = t_rx + T_edge, where t_rx is
the receive-complete time within the frame period from the transmission schedule (1-ms slots):
  session: the new anchors of the frame (edge cache, eval_frames.py, interleaved mapping, base layer on n_A modes) are
           sent in slot 1, t_rx = 1 ms; ARQ+conceal (retransmissions in slots 3 and 5 on the streams that did not fail in
           the previous transmission slot, eps = 1/6): t_rx = end of the last retransmission slot used. Loss patterns of
           the correlated Model-B traces, chains of eval_frames.py (same seeds, so the same losses as Table tab:frames).
  cold:    the whole view (eval_cold3.py schedule: anchor payload in S_A slots, enhancement tokens interleaved over the
           M - n_A data streams in the same slots, network in slot S_A + 1); lost tokens from the stratum-E windows of
           eval_cold3.py (same seeds), plus one loss-free window per view; t_rx = S_A + 1 ms with losses (the network
           must have arrived), S_A ms without. The whole view is entropy-decoded after reception (no overlap with
           reception), every lost anchor is concealed by the network (no anytime truncation), neighbours among the
           anchors of the view (precomputed).
Networks: deploy = jointly trained MLP (tag _v2m_r1) on garden, room and playroom, EdgeConcealNet (tag _v2a_r1)
elsewhere; attn = the cross-attention concealment network of the paper (tag _v2a_r1).
Pacing (--barrier, used for concurrent sessions): frame k is released at start + k * 11.11 ms + t_rx; a frame that is
released while the previous one is still processed waits, and the wait counts in its E2E.
Output: --out-json (summary) and <out-json without .json>_rows.json.gz (one row per timed frame).
usage:
  python bench_e2e_frames.py --mode session --scenes garden --nets deploy,attn --reps 3 --out-json results/rev6_raw/session_garden.json
  python bench_e2e_frames.py --mode cold --scenes garden --nets deploy,attn --reps 3 --out-json results/rev6_raw/cold_garden.json
  python bench_e2e_frames.py --mode session --scenes room --nets deploy --runs T8R8:10:conceal --chains 3 \
      --duration 60 --barrier /tmp/rev6_bar2 --n-procs 2 --out-json results/rev6_raw/conc2_room.json
  python bench_e2e_frames.py --aggregate 'results/rev6_raw/session_*.json' --out-json results/rev6_latency_session.json
"""
import argparse
import glob
import gzip
import itertools
import json
import math
import os
import subprocess
import time

import numpy as np
import torch

from conceal_net import knn
from eval_frames import pattern_stream
from eval_layered import load
from range_code2 import CPP, CU, CU_DECL, tables
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams

T_FRAME, SLOT, R0, MLP_MBIT = 11.1, 1.0, 9.35e6, 0.14e6
GROUP, TOK, CHUNK = 6, 64, 16384
STAGES = ("dec", "enc", "gath", "conc", "ren")
EXT = {}


def exts():
    if not EXT:
        from torch.utils.cpp_extension import load_inline
        EXT["cpu"] = load_inline("rans2_ext", cpp_sources=CPP, functions=["encode", "decode"], extra_cflags=["-O3", "-fopenmp"],
                                 extra_ldflags=["-fopenmp"], verbose=False)
        EXT["gpu"] = load_inline("rans2_cuda", cpp_sources=CU_DECL, cuda_sources=CU, functions=["decode_cuda"],
                                 extra_cuda_cflags=["-O3"], verbose=False)
    return EXT["cpu"], EXT["gpu"]


def gpu_apps():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except OSError:
        return []


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def wait_idle(allowed, log):
    """Wait until no process other than `allowed` uses the GPU (checked with nvidia-smi)."""
    t0 = time.time()
    while True:
        others = [x for x in gpu_apps() if int(x.split(",")[0]) not in allowed]
        if not others:
            return time.time() - t0
        print(f"GPU busy ({others}), waiting", flush=True); log.append({"t": time.time(), "busy": others})
        time.sleep(30)


def probe(seconds=1.0):
    """Synchronized 1024^3 FP32 matrix products for `seconds`: (median ms, max ms, load average)."""
    x = torch.randn(1024, 1024, device="cuda"); ts = []; t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t = time.perf_counter(); x @ x; torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t))
    ts = np.array(ts)
    return float(np.median(ts)), float(ts.max()), open("/proc/loadavg").read().split()[:3]


def quiet(log, a, tag="", wait=True):
    """Host/GPU quietness probe over 1 s; the machine counts as quiet if no operation takes more than 3x the median +
    5 ms (heavy CPU load of other jobs on this virtual machine stalls the launching thread for 0.1 to 0.4 s, while
    ordinary background activity causes spikes of a few ms). With wait, waits in 30-s steps until quiet; returns the
    probe statistics and "ok"."""
    if a.no_probe:
        return {"tag": tag, "ok": True, "skipped": True}
    t_wait = time.time()
    while True:
        med, mx, load = probe()
        ok = mx <= 3 * med + 5.0
        if ok or not wait:
            return {"tag": tag, "ok": ok, "median_ms": med, "max_ms": mx, "load": load, "waited_s": time.time() - t_wait}
        log.append({"t": time.time(), "tag": tag, "probe_median": med, "probe_max": mx, "load": load})
        print(f"not quiet ({tag}): probe median {med:.3f} max {mx:.1f} ms, load {load}; waiting", flush=True)
        time.sleep(30)


class Coder:
    """rANS tables of both layers of a scene (factorized Laplace model of the checkpoint); encode on the CPU (cloud),
    decode on the GPU (edge). Symbols are computed exactly as the quantizer of the model, so symbol * step equals the
    quantized attribute."""

    def __init__(self, model, L, ck, threads):
        self.threads = threads
        cpu, _ = exts()
        fb, A = ck["base_dim"], model.xyz.shape[0]
        rm = ck["rate_model"]
        parts = {"base": [("feat", slice(0, fb)), ("scaling", slice(None))], "enh": [("feat", slice(fb, None)), ("offsets", slice(None))]}
        self.q, self.tab, self.gtab, self.gstep, self.C, self.S, self.bits = {}, {}, {}, {}, {}, {}, {}
        with torch.no_grad():
            sym = {n: torch.round(getattr(model, n).reshape(A, -1) / model.qstep(n)).long().cpu().numpy() for n in model.ATTRS}
            step = {n: float(model.qstep(n)) for n in model.ATTRS}
            for lay, pp in parts.items():
                q = np.concatenate([sym[n][:, sl] for n, sl in pp], 1).astype(np.int32)
                mu = np.concatenate([rm[f"loc.{n}"].cpu().numpy().astype(np.float64)[sl] for n, sl in pp])
                bb = np.concatenate([np.exp(rm[f"log_b.{n}"].cpu().numpy().astype(np.float64))[sl] for n, sl in pp])
                stp = np.concatenate([np.full(sym[n][:, sl].shape[1], step[n]) for n, sl in pp])
                Cn = q.shape[1]; fr, cu, lo, sl_, S = [], [], [], [], 0
                for c in range(Cn):
                    f_, c_, l_, s_, S_, _, _ = tables(q[:, c:c + 1].astype(np.int64), mu[c:c + 1], bb[c:c + 1], stp[c])
                    fr.append(f_[0]); cu.append(c_[0]); lo.append(int(l_[0])); sl_.append(s_[0]); S = max(S, S_)
                freq = np.zeros((Cn, S), np.int32); cum = np.zeros((Cn, S + 1), np.int32)
                for c in range(Cn):
                    freq[c, :fr[c].size] = fr[c]; cum[c, :cu[c].size] = cu[c]; cum[c, cu[c].size:] = cu[c][-1]
                t = (torch.from_numpy(freq.reshape(-1)), torch.from_numpy(cum.reshape(-1)), torch.tensor(lo, dtype=torch.int32),
                     torch.from_numpy(np.stack(sl_).astype(np.int16).reshape(-1)))
                self.q[lay], self.tab[lay], self.gtab[lay] = q, t, tuple(x.cuda() for x in t)
                self.C[lay], self.S[lay] = Cn, S
                self.gstep[lay] = torch.cat([model.qstep(n).detach().expand(sym[n][:, sl].shape[1]) for n, sl in pp]).float().cuda()

    def encode(self, lay, ids):
        """ids: Morton-sorted anchor ids (numpy int64). Returns the packet (pinned host tensors)."""
        cpu, _ = exts()
        q = self.q[lay][ids]; n, C = q.shape; ng = C // GROUP; T = (n + TOK - 1) // TOK; full = n // TOK; r = n - full * TOK
        parts = []
        if full:
            parts.append(q[:full * TOK].reshape(full, TOK, ng, GROUP).transpose(0, 2, 1, 3).reshape(-1))
        if r:
            parts.append(q[full * TOK:].reshape(r, ng, GROUP).transpose(1, 0, 2).reshape(-1))
        flat = np.ascontiguousarray(np.concatenate(parts)).astype(np.int32)
        lens = np.full(T * ng, TOK * GROUP, np.int64)
        if r:
            lens[-ng:] = r * GROUP
        tok_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        sc0 = np.tile(np.arange(0, C, GROUP, dtype=np.int32), T); snc = np.full(T * ng, GROUP, np.int32)
        F_, C_, L_, _ = self.tab[lay]
        data, doff = cpu.encode(torch.from_numpy(flat), torch.from_numpy(tok_off), torch.from_numpy(sc0), torch.from_numpy(snc), F_, C_, L_,
                                self.S[lay], self.threads)
        pin = lambda x: x.contiguous().pin_memory()
        return {"lay": lay, "n": n, "bits": 16 * int(data.numel()), "data": pin(data), "doff": pin(doff), "tok_off": pin(torch.from_numpy(tok_off)),
                "sc0": pin(torch.from_numpy(sc0)), "snc": pin(torch.from_numpy(snc)), "ids": pin(torch.from_numpy(ids.astype(np.int64)))}

    def decode(self, pk, table):
        """GPU: copy, decode, reorder to (anchor, channel), dequantize and write into the edge table; returns the ids."""
        _, gpu = exts()
        lay = pk["lay"]
        g = lambda x: x.cuda(non_blocking=True)
        d, do, to, c0, nc, ids = (g(pk[k]) for k in ("data", "doff", "tok_off", "sc0", "snc", "ids"))
        F_, C_, L_, Z_ = self.gtab[lay]
        out = gpu.decode_cuda(d, do, to, c0, nc, F_, C_, L_, Z_, self.S[lay])
        n, C = pk["n"], self.C[lay]; ng = C // GROUP; full = n // TOK; r = n - full * TOK
        parts = []
        if full:
            parts.append(out[:full * TOK * C].view(full, ng, TOK, GROUP).permute(0, 2, 1, 3).reshape(full * TOK, C))
        if r:
            parts.append(out[full * TOK * C:].view(ng, r, GROUP).permute(1, 0, 2).reshape(r, C))
        q = torch.cat(parts) if len(parts) > 1 else parts[0]
        table[ids] = q.float() * self.gstep[lay]
        return ids


class Edge:
    """Edge state of one scene (attribute tables, cache, neighbour table, concealment network in FP16)."""

    def __init__(self, scene_name, net_kind, a):
        self.scene = scene_name
        has_mlp_joint = os.path.exists(os.path.join(a.out, scene_name, "layered_v2m_r1.pt"))
        if net_kind == "deploy" and has_mlp_joint:
            self.tag, self.netname, self.kind = "_v2m_r1", "gen_fp", "mlp"
        elif net_kind == "deploy":
            self.tag, self.netname, self.kind = "_v2a_r1", "edge_fp", "edge"
        else:
            self.tag, self.netname, self.kind = "_v2a_r1", "gen_fp", "attn"
        model, L, nets, c, ck = load(a.out, scene_name, self.tag)
        self.model, self.L, self.c, self.ck = model, L, c, ck
        self.net = nets[self.netname].half().eval()
        self.ref_net = None
        for k in list(nets):
            if k != self.netname:
                del nets[k]
        with torch.no_grad():
            at = model.attributes(quantized=True); self.base, self.enh = L.split(at)
            self.base, self.enh = self.base.detach().contiguous(), self.enh.detach().contiguous()
            self.stats = (self.base.mean(0), self.base.std(0).clamp_min(0.02), self.enh.mean(0), self.enh.std(0).clamp_min(0.02))
        self.nbr_global = knn(model.xyz, c["kn"]); self.nbr = self.nbr_global
        self.radius = ck["radius"]
        self.pk = Packetizer(model.xyz, model.voxel, ck["importance"].cuda(), G=c["G"])
        self.coder = Coder(model, L, ck, a.threads)
        self.A = model.xyz.shape[0]
        self.check = self.self_check()
        self.reset()

    @torch.no_grad()
    def self_check(self):
        """Bit-exact round trip of all anchors through the encoder and the GPU decoder."""
        ids = self.pk.token_ids(torch.arange(self.A, device="cuda"))[0].cpu().numpy()
        res = {}
        for lay, ref in (("base", self.base), ("enh", self.enh)):
            pk = self.coder.encode(lay, ids); tab = torch.zeros_like(ref)
            self.coder.decode(pk, tab); torch.cuda.synchronize()
            res[lay] = {"bit_exact": bool(torch.equal(tab, ref)), "bits_per_anchor": pk["bits"] / self.A}
        assert res["base"]["bit_exact"] and res["enh"]["bit_exact"], res
        return res

    @torch.no_grad()
    def reset(self):
        self.cached = torch.zeros(self.A, dtype=torch.bool, device="cuda")
        self.base_tab = self.base.clone()          # base layer of every anchor available for concealment, as in eval_frames
        self.enh_tab = torch.zeros_like(self.enh)
        if self.kind == "edge":
            bm, bs, em, es = self.stats
            self.kv = [t.contiguous() for t in self.net.encode(((self.base_tab - bm) / bs).half(), torch.zeros_like(self.enh_tab).half(),
                                                               torch.zeros(self.A, dtype=torch.bool, device="cuda"))]

    def on_arrival(self, ids):
        bm, bs, em, es = self.stats
        kv = self.net.encode(((self.base_tab[ids] - bm) / bs).half(), ((self.enh_tab[ids] - em) / es).half(), self.cached[ids])
        for i, t in enumerate(kv):
            self.kv[i][ids] = t

    def gather(self, miss):
        bm, bs, em, es = self.stats
        nb = self.nbr[miss]
        xyz = self.model.xyz
        rel = ((xyz[nb] - xyz[miss][:, None]) / self.radius).half()
        own = ((self.base_tab[miss] - bm) / bs).half()
        if self.kind == "edge":
            return (nb, rel, own)
        if self.kind == "mlp":                  # FP32 inputs: the weighted means are formed in FP32 (see mlp_forward)
            return (rel.float(), (self.base_tab[nb] - bm) / bs, (self.enh_tab[nb] - em) / es, self.cached[nb], own.float())
        return (rel, ((self.base_tab[nb] - bm) / bs).half(), ((self.enh_tab[nb] - em) / es).half(), self.cached[nb], own)

    def mlp_forward(self, rel, base, enh, recv, own_base):
        """MLPConcealNet.forward with the inverse-distance means in FP32 (the clamp 1e-12 of the reference underflows
        in FP16 when no neighbour is received) and the MLP in FP16."""
        w = 1.0 / rel.norm(dim=-1).clamp_min(1e-6)
        wr = w * recv.to(w.dtype)
        has = (wr.sum(1, keepdim=True) > 0).to(w.dtype)
        enh_m = (wr[..., None] * enh).sum(1) / wr.sum(1, keepdim=True).clamp_min(1e-12)
        base_m = (w[..., None] * base).sum(1) / w.sum(1, keepdim=True)
        frac = recv.to(w.dtype).mean(1, keepdim=True)
        return self.net.mlp(torch.cat([own_base, base_m, enh_m * has, frac, has], -1).half())

    def conceal(self, inp):
        bm, bs, em, es = self.stats
        n = inp[0].shape[0]; outs = []
        for i in range(0, n, CHUNK):
            x = [t[i:i + CHUNK] for t in inp]
            outs.append(self.net.forward_table(self.kv, *x) if self.kind == "edge" else self.mlp_forward(*x) if self.kind == "mlp" else self.net(*x))
        return torch.cat(outs).float() * es + em

    @torch.no_grad()
    def run_frame(self, fr, K, V, W, H):
        """One frame through the pipeline; returns stage times (ms, CUDA events), wall time (ms) and counts."""
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(STAGES) + 1)]
        torch.cuda.synchronize()
        t0 = time.perf_counter(); ev[0].record()
        bids = None
        if fr.get("base") is not None:
            bids = self.coder.decode(fr["base"], self.base_tab)
        if fr.get("enh") is not None:
            eids = self.coder.decode(fr["enh"], self.enh_tab); self.cached[eids] = True
        ev[1].record()
        if self.kind == "edge" and bids is not None:
            self.on_arrival(bids)
        ev[2].record()
        idx = self.model.visible(K, V, W, H)
        mv = ~self.cached[idx]; miss = idx[mv]; n_miss = int(miss.numel())
        inp = self.gather(miss) if n_miss else None
        ev[3].record()
        pred = self.conceal(inp) if n_miss else None
        ev[4].record()
        b = self.base_tab[idx]; e = self.enh_tab[idx]
        if n_miss:
            e[mv] = pred
        img = self.model.render(self.L.merge(b, e), idx, K, V, W, H)[0]
        ev[5].record(); torch.cuda.synchronize()
        wall = 1e3 * (time.perf_counter() - t0)
        st = [ev[i].elapsed_time(ev[i + 1]) for i in range(len(STAGES))]
        return st, wall, n_miss, int(idx.numel()), img

    @torch.no_grad()
    def equivalence(self, miss):
        """Max abs deviation of the FP16 deployed prediction from the FP32 reference forward (same inputs)."""
        from train_layered import LayeredConcealer
        if miss.numel() == 0:
            return None
        import copy
        n32 = copy.deepcopy(self.net).float()
        conc = LayeredConcealer(self.model, self.L, n32, self.stats, self.nbr, self.radius)
        ref = conc.predict(miss, self.cached, self.base_tab, self.enh_tab)
        dep = self.conceal(self.gather(miss))
        del n32
        return {"n": int(miss.numel()), "max_abs": float((ref - dep).abs().max()), "rel_rms": float((ref - dep).pow(2).mean().sqrt() / ref.std())}


def setup_phy(a, cfg, tau, n_anchor):
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    cf = ms["configs"][cfg]; tx_ms = cf["tx"]; M = len(tx_ms)
    rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx_ms)
    return ms, cf, tx_ms, M, rel


def morton(E, ids):
    return E.pk.token_ids(ids)[0].cpu().numpy() if ids.numel() else None


class Views:
    def __init__(self, E, scene):
        self.E, self.scene, self.vis, self.cam = E, scene, {}, {}

    def get(self, vi):
        if vi not in self.vis:
            K = torch.from_numpy(self.scene.K[vi]).cuda(); V = torch.from_numpy(self.scene.viewmats[vi]).cuda()
            self.cam[vi] = (K, V); self.vis[vi] = self.E.model.visible(K, V, self.scene.W, self.scene.H)
        return self.vis[vi], self.cam[vi]


@torch.no_grad()
def session_chains(E, scene, views, a, cfg, tau, n_anchor, schemes, chains, max_frames=0):
    """Frame paths of eval_frames.py (same seeds): yields (chain, scheme, frames) with the real packets of every frame."""
    _, _, tx_ms, M, rel = setup_phy(a, cfg, tau, n_anchor)
    stream, tx = pattern_stream(a.phy, cfg, tau)
    assert [int(x) for x in tx] == [int(x) for x in tx_ms]
    best = min(itertools.combinations(range(M), n_anchor),
               key=lambda sub: int(np.count_nonzero((stream & sum(1 << i for i in sub)) == sum(1 << i for i in sub))))
    anchor_modes = [rel.index(int(tx[i])) for i in best]
    rng = np.random.default_rng(a.seed)
    order = np.argsort(scene.names); T = len(order); F = 11
    test = set(int(i) for i in scene.test_idx)
    failed = lambda p: set(patterns_to_streams(p, tx, rel))
    for ch in range(chains):
        off = rng.integers(0, stream.size - T * F)
        pats = stream[off:off + T * F].reshape(T, F)
        for sch in schemes:
            cached = torch.zeros(E.A, dtype=torch.bool, device="cuda"); prng = np.random.default_rng(a.seed * 1000 + ch)
            frames = []
            for f, vi in enumerate(order):
                if max_frames and f >= max_frames:
                    break
                vi = int(vi); idx, _ = views.get(vi)
                new = idx[~cached[idx]]
                fr = {"f": f, "view": vi, "test": vi in test, "n_new": int(new.numel()), "t_rx": SLOT, "n_lost_tx": 0, "base": None, "enh": None,
                      "burst": int(pats[f, 0] != 0)}
                if new.numel():
                    S0 = failed(pats[f, 0])
                    if all(s in S0 for s in anchor_modes):
                        lost = new; base_ok = False
                    else:
                        base_ok = True
                        lm = E.pk.lost_mask(new, M, "interleaved", sorted(S0), prng) if S0 else torch.zeros(new.numel(), dtype=torch.bool, device="cuda")
                        lost = new[lm]
                        if sch.startswith("arq") and lost.numel():
                            budget = int(a.eps * new.numel()); prev = S0
                            for s in range(2, min(F, a.arq_last_slot), 2):
                                if lost.numel() == 0 or budget <= 0:
                                    break
                                send = lost[:budget]; budget -= send.numel()
                                Ss = failed(pats[f, s])
                                ok_ = [m_ for m_ in range(M) if m_ not in prev] or list(range(M))
                                fl_ = sorted(ok_.index(m_) for m_ in Ss if m_ in ok_)
                                lm2 = E.pk.lost_mask(send, len(ok_), "interleaved", fl_, prng) if fl_ else torch.zeros(send.numel(), dtype=torch.bool, device="cuda")
                                prev = Ss
                                lost = torch.cat([send[lm2], lost[send.numel():]])
                                fr["t_rx"] = float(s + 1) * SLOT
                    delivered = new[~torch.isin(new, lost)] if lost.numel() else new
                    cached[delivered] = True
                    fr["n_lost_tx"] = int(lost.numel())
                    if base_ok:
                        fr["base"] = E.coder.encode("base", morton(E, new))
                    if delivered.numel():
                        fr["enh"] = E.coder.encode("enh", morton(E, delivered))
                frames.append(fr)
            yield ch, sch, frames


@torch.no_grad()
def cold_windows(E, scene, views, a, cfg, tau, n_anchor):
    """Cold-start windows of eval_cold3.py (same seeds; stratum E, conceal scheme) plus one loss-free window per view."""
    _, cf, tx, M, rel = setup_phy(a, cfg, tau, n_anchor)
    xs = []
    for f in sorted(glob.glob(os.path.join(a.phy, "patterns", f"traces_corr_2.5_{tau:g}ms_pat*.npz"))):
        z = np.load(f); xs.append(z[cfg])
    pats = np.concatenate(xs)
    bits = np.unpackbits(pats[..., None], axis=-1, bitorder="little")[..., :M].astype(bool)
    flat = bits.reshape(-1, M)
    anchor = min(itertools.combinations(range(M), n_anchor), key=lambda s: int(flat[:, list(s)].all(1).sum()))
    anchor_streams = sorted(rel.index(int(tx[i])) for i in anchor)
    data_tx = [i for i in range(M) if i not in anchor]; Md = len(data_tx)
    data_streams = torch.tensor(sorted(rel.index(int(tx[i])) for i in data_tx), device="cuda")
    anyf = bits.any(-1); Kd = bits[..., data_tx].sum(-1); Ka = bits.sum(-1); alost = bits[..., list(anchor)].all(-1)
    strata = {}

    def strata_for(SA, SR):
        if (SA, SR) not in strata:
            WE = max(SA + 4, SR); nW = anyf.shape[1] - WE - 1
            Em = np.zeros((anyf.shape[0], nW), bool); D = np.zeros_like(Em)
            for t in range(WE):
                Em |= anyf[:, t:nW + t]
            for t in range(SR):
                D |= (Kd[:, t:nW + t] > Md // 2) if t < SA else (Ka[:, t:nW + t] > M // 2)
            for t in range(SA):
                D |= alost[:, t:nW + t]
            strata[(SA, SR)] = dict(E=np.flatnonzero(Em.ravel()), D=np.flatnonzero(D.ravel()), pE=float(Em.mean()), nW=nW, WE=WE)
        return strata[(SA, SR)]

    tag = E.tag
    mt = json.load(open(os.path.join(a.out, E.scene, f"metrics_layered{tag}.json")))
    rc_path = os.path.join(a.out, E.scene, f"rangecode{tag}.json")
    base_bits = (json.load(open(rc_path))["layers"]["base"]["real_bits_per_anchor"] if os.path.exists(rc_path) else 1.012 * mt["rate_bits_per_anchor"]["base_layer"])
    b_A = mt["rate_bits_per_anchor"]["position_fixed"] + base_bits
    b_E = (json.load(open(rc_path))["layers"]["enhancement"]["real_bits_per_anchor"] if os.path.exists(rc_path) else 1.013 * mt["rate_bits_per_anchor"]["enh_layer"])
    rng = np.random.default_rng(a.seed)
    for vi in scene.test_idx:
        vi = int(vi); idx, _ = views.get(vi); n_vis = int(idx.numel())
        SA = max(1, math.ceil((n_vis * b_A + MLP_MBIT) / R0)); L2 = 2 * n_vis * b_E
        SR = SA + max(0, math.ceil((L2 - Md * R0 * SA) / (M * R0)))
        st = strata_for(SA, SR)
        evE = [int(i) for i in rng.choice(st["E"], min(a.n_ev, st["E"].size), replace=False)]
        rng.choice(st["D"], min(2, st["D"].size), replace=False)          # keeps the random stream of eval_cold3.py
        tv = E.nbr_global.clone(); tv[idx] = idx[knn(E.model.xyz[idx], E.c["kn"])]
        s_, tid = E.pk.token_ids(idx); T = int(tid[-1]) + 1
        slot_of = ((torch.arange(T, device="cuda") * SA) // T)[tid]
        s_np = s_.cpu().numpy()
        base_pk = E.coder.encode("base", s_np)
        wins = [{"kind": "free", "view": vi, "S_A": SA, "n_visible": n_vis, "t_rx": SA * SLOT, "n_lost_tx": 0, "anchor_lost": False, "net_lost": False,
                 "base": base_pk, "enh": E.coder.encode("enh", s_np), "nbr": tv}]
        for w in evE:
            ch, s = divmod(w, st["nW"])
            prng = np.random.default_rng(vi * 7919 + ch * 251 + s)
            St = [set(patterns_to_streams(int(pats[ch, s + t]), tx, rel)) for t in range(st["WE"] + 1)]
            fs = [torch.tensor([m in St[t] for m in range(M)], device="cuda") for t in range(st["WE"] + 1)]
            aset = [all(x in St[t] for x in anchor_streams) for t in range(st["WE"] + 1)]
            tok_stream = data_streams[E.pk.assign(s_, tid, Md, "interleaved", prng)][tid]
            fail = torch.zeros_like(tok_stream, dtype=torch.bool)
            for t in range(SA):
                fail |= (slot_of == t) & fs[t][tok_stream]
            n_lost = int(fail.sum())
            win = {"kind": "E", "view": vi, "S_A": SA, "n_visible": n_vis, "t_rx": (SA + (1 if n_lost else 0)) * SLOT, "n_lost_tx": n_lost,
                   "anchor_lost": bool(any(aset[:SA])), "net_lost": bool(aset[SA]), "base": base_pk, "nbr": tv,
                   "enh": E.coder.encode("enh", s_np[~fail.cpu().numpy()]) if n_lost < n_vis else None}
            wins.append(win)
        yield vi, wins


def cp_interval(k, n, conf=0.95):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf((1 - conf) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))
    hi1 = 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    return lo, hi, hi1


def dist(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return {}
    return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)),
            "p999": float(np.percentile(x, 99.9)), "max": float(x.max())}


def summarize(rows, key_fields):
    """rows: list of dicts. Groups by key_fields; distribution of T_edge and E2E, deadline misses with Clopper-Pearson
    bounds, stage statistics and stage correlations."""
    groups = {}
    for r in rows:
        groups.setdefault(" | ".join(str(r[k]) for k in key_fields), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        e2e = np.array([r["e2e"] for r in rs]); wall = np.array([r["wall"] for r in rs])
        st = np.array([[r[s] for s in STAGES] for r in rs])
        k = int((e2e > T_FRAME).sum()); lo, hi, hi1 = cp_interval(k, len(rs))
        k2 = int((wall > T_FRAME).sum()); lo2, hi2, hi21 = cp_interval(k2, len(rs))
        evs = st.sum(1)
        stc = {}
        sd = st.std(0)
        active = [i for i in range(len(STAGES)) if sd[i] > 0]
        if len(active) > 1:
            cm = np.corrcoef(st[:, active].T)
            stc = {f"{STAGES[active[i]]}~{STAGES[active[j]]}": float(cm[i, j]) for i in range(len(active)) for j in range(i + 1, len(active))}
        out[g] = {"frames": len(rs), "T_edge_wall": dist(wall), "E2E": dist(e2e), "t_rx": dist([r["t_rx"] for r in rs]),
                  "frac_E2E_gt_11.1": k / len(rs), "n_E2E_gt_11.1": k, "cp95_E2E": [lo, hi], "cp95_one_sided_upper_E2E": hi1,
                  "frac_Tedge_gt_11.1": k2 / len(rs), "n_Tedge_gt_11.1": k2, "cp95_Tedge": [lo2, hi2], "cp95_one_sided_upper_Tedge": hi21,
                  "stages": {s: dist(st[:, i]) for i, s in enumerate(STAGES)},
                  "sum_stage_means": float(st.mean(0).sum()), "sum_stage_p99": float(np.percentile(st, 99, axis=0).sum()),
                  "p99_of_event_total": float(np.percentile(evs, 99)), "host_overhead_mean": float((wall - evs).mean()),
                  "stage_corr": stc,
                  "stage_corr_with_total": {s: float(np.corrcoef(st[:, i], wall)[0, 1]) for i, s in enumerate(STAGES) if sd[i] > 0},
                  "n_miss": dist([r["n_miss"] for r in rs]), "frames_with_miss": int(sum(r["n_miss"] > 0 for r in rs)),
                  "worst": sorted(rs, key=lambda r: -r["e2e"])[:5]}
    return out


def row_of(meta, st, wall, n_miss, n_vis, extra=None):
    r = dict(meta); r.update({s: round(float(v), 4) for s, v in zip(STAGES, st)})
    r.update(wall=round(wall, 4), n_miss=n_miss, n_vis=n_vis)
    r["e2e"] = round(r["t_rx"] + r.get("wait", 0.0) + wall, 4)
    if extra:
        r.update(extra)
    return r


def barrier(a):
    os.makedirs(a.barrier, exist_ok=True)
    open(os.path.join(a.barrier, f"ready_{os.getpid()}"), "w").write(str(time.time()))
    while len(glob.glob(os.path.join(a.barrier, "ready_*"))) < a.n_procs:
        time.sleep(0.2)
    t = max(os.path.getmtime(f) for f in glob.glob(os.path.join(a.barrier, "ready_*")))
    start = t + 2.0
    while time.time() < start:
        time.sleep(0.01)
    return start


def run_session(a, E, scene, views, rows, info):
    W, H = scene.W, scene.H
    runs = [r.split(":") for r in a.runs.split(",")]
    if E.kind == "attn" and not a.attn_all_runs:
        runs = [r for r in runs if r[0] == "T8R8" and r[2] == "conceal"]
    warmed = False
    for cfg, tau, sch in runs:
        tau = float(tau); na = 4 if cfg == "T8R8" else 2
        n_ch = a.chains_attn if (E.kind == "attn" and a.chains_attn) else a.chains_t6 if (cfg.startswith("T6") and a.chains_t6) else a.chains
        for ch, sch_, frames in session_chains(E, scene, views, a, cfg, tau, na, [sch], n_ch, a.max_frames):
            if not warmed:                          # warm-up: one untimed pass over the first chain
                E.reset()
                for fr in frames:
                    vi = fr["view"]; _, (K, V) = views.get(vi); E.run_frame(fr, K, V, W, H)
                eq = None
                for fr in frames:                   # FP16 deployment against the FP32 reference, one frame with losses
                    if eq is None and fr["n_lost_tx"] > 0:
                        E.reset()
                        for f2 in frames[:fr["f"] + 1]:
                            _, (K, V) = views.get(f2["view"]); E.run_frame(f2, K, V, W, H)
                        idx, _ = views.get(fr["view"]); eq = E.equivalence(idx[~E.cached[idx]])
                info.setdefault("fp16_check", {})[E.kind] = eq
                warmed = True
            for attempt in range(a.max_retry):     # probe before and after; a chain disturbed by host load is timed again
                tag = f"{E.scene}/{E.kind}/{cfg}/{tau:g}/{sch}/{ch}/{attempt}"
                info.setdefault("probes", []).append(quiet(info.setdefault("busy_log", []), a, tag))
                new_rows = run_chain(a, E, views, frames, cfg, tau, sch, ch, W, H)
                post = quiet(info["busy_log"], a, tag + "/post", wait=False); info["probes"].append(post)
                if post["ok"] or attempt == a.max_retry - 1:
                    rows += new_rows
                    break
            print(f"{E.scene} {E.kind} {cfg} tau{tau:g} {sch} chain {ch}: {len(rows)} rows", flush=True)


def run_chain(a, E, views, frames, cfg, tau, sch, ch, W, H):
    rows = []
    for rep in range(a.reps):
        E.reset()
        for fr in frames[:5]:               # short re-warm after the untimed simulation of the chain
            _, (K, V) = views.get(fr["view"]); E.run_frame(fr, K, V, W, H)
        E.reset()
        for fr in frames:
            _, (K, V) = views.get(fr["view"])
            st, wall, n_miss, n_vis, _ = E.run_frame(fr, K, V, W, H)
            rows.append(row_of({"mode": "session", "scene": E.scene, "net": E.kind, "cfg": cfg, "tau": tau, "scheme": sch, "chain": ch, "rep": rep,
                                "frame": fr["f"], "view": fr["view"], "test": fr["test"], "n_new": fr["n_new"], "n_lost_tx": fr["n_lost_tx"],
                                "t_rx": fr["t_rx"], "first": fr["f"] == 0, "enh_bits": fr["enh"]["bits"] if fr["enh"] else 0,
                                "base_bits": fr["base"]["bits"] if fr["base"] else 0}, st, wall, n_miss, n_vis))
    return rows


def run_paced(a, E, scene, views, rows, info):
    """Concurrent sessions: frames released at 90 frames/s, for --duration seconds after a common start."""
    W, H = scene.W, scene.H
    cfg, tau, sch = a.runs.split(",")[0].split(":"); tau = float(tau); na = 4 if cfg == "T8R8" else 2
    paths = [(ch, frames) for ch, _, frames in session_chains(E, scene, views, a, cfg, tau, na, [sch], a.chains)]
    E.reset()
    for fr in paths[0][1][:30]:
        _, (K, V) = views.get(fr["view"]); E.run_frame(fr, K, V, W, H)
    start = barrier(a); info["start"] = start
    k = 0; busy_until = start; p = 0
    while True:
        ch, frames = paths[p % len(paths)]; E.reset()
        for fr in frames:
            release = start + (k * T_FRAME + fr["t_rx"]) / 1e3
            if release > start + a.duration:
                return
            now = time.time()
            if now < release:
                time.sleep(max(0.0, release - now - 0.0005))
                while time.time() < release:
                    pass
            wait = max(0.0, (time.time() - release) * 1e3)
            _, (K, V) = views.get(fr["view"])
            st, wall, n_miss, n_vis, _ = E.run_frame(fr, K, V, W, H)
            rows.append(row_of({"mode": "paced", "n_procs": a.n_procs, "scene": E.scene, "net": E.kind, "cfg": cfg, "tau": tau, "scheme": sch, "chain": ch,
                                "rep": p // len(paths), "frame": fr["f"], "view": fr["view"], "test": fr["test"], "n_new": fr["n_new"],
                                "n_lost_tx": fr["n_lost_tx"], "t_rx": fr["t_rx"], "wait": round(wait, 4), "first": fr["f"] == 0}, st, wall, n_miss, n_vis))
            k += 1
        p += 1


def run_cold(a, E, scene, views, rows, info):
    W, H = scene.W, scene.H
    for spec in a.cold_runs.split(","):
        cfg, tau = spec.split(":"); tau = float(tau); na = 4 if cfg == "T8R8" else 2
        first = True; batch = []

        def flush(batch, first):
            """Time a batch of views between a probe before and a probe after; a disturbed batch is timed again."""
            for attempt in range(a.max_retry):
                tag = f"{E.scene}/{E.kind}/{cfg}/{tau:g}/{batch[0][0]}-{batch[-1][0]}/{attempt}"
                info.setdefault("probes", []).append(quiet(info.setdefault("busy_log", []), a, tag))
                new_rows = []
                for j, (vi_, wins_) in enumerate(batch):
                    _, (K, V) = views.get(vi_)
                    new_rows += cold_view(a, E, wins_, K, V, W, H, cfg, tau, vi_, first and j == 0)
                post = quiet(info["busy_log"], a, tag + "/post", wait=False); info["probes"].append(post)
                if post["ok"] or attempt == a.max_retry - 1:
                    return new_rows
        for vi, wins in cold_windows(E, scene, views, a, cfg, tau, na):
            batch.append((vi, wins))
            if len(batch) == a.probe_every:
                rows += flush(batch, first); batch = []; first = False
        if batch:
            rows += flush(batch, first)
        print(f"{E.scene} {E.kind} cold {cfg} tau{tau:g}: {len(rows)} rows", flush=True)


def cold_view(a, E, wins, K, V, W, H, cfg, tau, vi, first):
    rows = []
    for w in wins:
        E.nbr = w["nbr"]
        if w["anchor_lost"]:                 # anchor payload lost: the view cannot be rendered at all
            rows.append({"mode": "cold", "scene": E.scene, "net": E.kind, "cfg": cfg, "tau": tau, "view": vi, "kind": w["kind"], "skipped": "anchor_lost"})
            continue
        for rep in range(a.reps + (2 if first else 1)):   # untimed warm-up repetitions first
            E.reset()
            st, wall, n_miss, n_vis, _ = E.run_frame(w, K, V, W, H)
            if rep >= (2 if first else 1):
                rows.append(row_of({"mode": "cold", "scene": E.scene, "net": E.kind, "cfg": cfg, "tau": tau, "view": vi, "kind": w["kind"], "rep": rep,
                                    "S_A": w["S_A"], "t_rx": w["t_rx"], "n_lost_tx": w["n_lost_tx"], "net_lost": w["net_lost"],
                                    "base_bits": w["base"]["bits"], "enh_bits": w["enh"]["bits"] if w["enh"] else 0}, st, wall, n_miss, n_vis))
    E.nbr = E.nbr_global
    return rows


def rep_median(rows, frame_keys):
    """One row per frame (same work in every repetition) with the median E2E and T_edge over its repetitions: the tail
    of this distribution comes from the frame content, the tail of the raw rows also contains the run-to-run jitter."""
    g = {}
    for r in rows:
        g.setdefault(tuple(r.get(k) for k in frame_keys), []).append(r)
    out = []
    for rs in g.values():
        m = dict(rs[0]); m["e2e"] = float(np.median([x["e2e"] for x in rs])); m["wall"] = float(np.median([x["wall"] for x in rs]))
        for s_ in STAGES:
            m[s_] = float(np.median([x[s_] for x in rs]))
        m["n_reps"] = len(rs); out.append(m)
    return out


def tail_summary(rows, key_fields):
    out = {}
    for g, v in summarize(rows, key_fields).items():
        out[g] = {k: v[k] for k in ("frames", "E2E", "T_edge_wall", "n_E2E_gt_11.1", "frac_E2E_gt_11.1", "cp95_E2E", "cp95_one_sided_upper_E2E")}
    return out


def aggregate(a):
    rows, infos = [], []
    for f in sorted(glob.glob(a.aggregate)):
        rf = f[:-5] + "_rows.json.gz"
        if not os.path.exists(rf):
            continue
        rows += [r for r in json.load(gzip.open(rf, "rt")) if "skipped" not in r]
        infos.append(json.load(open(f)).get("info", {}))
    mode = rows[0]["mode"]
    for r in rows:                      # deploy = jointly trained MLP (3 scenes) or EdgeConcealNet (8 scenes)
        r["variant"] = "deploy" if r["net"] in ("mlp", "edge") else "attn"
    res = {"files": sorted(glob.glob(a.aggregate)), "T_frame_ms": T_FRAME, "infos": infos}
    if mode == "session":
        res["all"] = summarize(rows, ("net", "cfg", "tau", "scheme"))
        res["steady_state"] = summarize([r for r in rows if not r["first"]], ("net", "cfg", "tau", "scheme"))
        res["first_frame_full_view"] = summarize([r for r in rows if r["first"]], ("net", "cfg", "tau", "scheme"))
        res["frames_with_loss"] = summarize([r for r in rows if r["n_miss"] > 0 and not r["first"]], ("net", "cfg", "tau", "scheme"))
        res["test_frames"] = summarize([r for r in rows if r["test"]], ("net", "cfg", "tau", "scheme"))
        res["per_scene"] = summarize(rows, ("net", "scene", "cfg", "tau", "scheme"))
        res["deploy"] = {"all": summarize(rows, ("variant", "cfg", "tau", "scheme")),
                         "steady_state": summarize([r for r in rows if not r["first"]], ("variant", "cfg", "tau", "scheme")),
                         "first_frame_full_view": summarize([r for r in rows if r["first"]], ("variant", "cfg", "tau", "scheme"))}
        med = rep_median(rows, ("net", "scene", "cfg", "tau", "scheme", "chain", "frame"))
        res["deploy"]["median_over_reps"] = tail_summary(med, ("variant", "cfg", "tau", "scheme"))
        res["deploy"]["median_over_reps_steady_state"] = tail_summary([r for r in med if not r["first"]], ("variant", "cfg", "tau", "scheme"))
        res["median_over_reps"] = tail_summary(med, ("net", "cfg", "tau", "scheme"))
        res["median_over_reps_steady_state"] = tail_summary([r for r in med if not r["first"]], ("net", "cfg", "tau", "scheme"))
    elif mode == "cold":
        res["all"] = summarize(rows, ("net", "cfg", "tau", "kind"))
        res["per_scene"] = summarize(rows, ("net", "scene", "cfg", "tau", "kind"))
        res["median_over_reps"] = tail_summary(rep_median(rows, ("net", "scene", "cfg", "tau", "view", "kind", "n_lost_tx", "t_rx")), ("net", "cfg", "tau", "kind"))
        # entropy decoding overlapped with reception (the tokens of a slot decoded while the next slot arrives, as in the
        # cold-start schedule of the paper): only max(dec / S_A, dec - (S_A - 1) ms) of the measured decoding remains
        ov = []
        for r in rows:
            o = dict(r); rem = max(r["dec"] / r["S_A"], r["dec"] - (r["S_A"] - 1) * SLOT)
            o["e2e"] = r["e2e"] - r["dec"] + rem; o["wall"] = r["wall"] - r["dec"] + rem; ov.append(o)
        res["decode_overlapped"] = tail_summary(ov, ("net", "cfg", "tau", "kind"))
        res["deploy"] = {"all": summarize(rows, ("variant", "cfg", "tau", "kind")),
                         "median_over_reps": tail_summary(rep_median(rows, ("net", "scene", "cfg", "tau", "view", "kind", "n_lost_tx", "t_rx")), ("variant", "cfg", "tau", "kind")),
                         "decode_overlapped": tail_summary(ov, ("variant", "cfg", "tau", "kind"))}
        res["frac_E2E_gt_22.2"] = {g: float(np.mean([r["e2e"] > 2 * T_FRAME for r in rows if " | ".join(str(r[k]) for k in ("net", "cfg", "tau", "kind")) == g]))
                                   for g in res["all"]}
        res["miss_causes"] = {}
        for g in res["all"]:
            rs = [r for r in rows if " | ".join(str(r[k]) for k in ("net", "cfg", "tau", "kind")) == g and r["e2e"] > T_FRAME]
            res["miss_causes"][g] = {"n": len(rs), "t_rx_alone_ge_frame": sum(r["t_rx"] >= T_FRAME for r in rs),
                                     "by_S_A": {str(s): sum(r["S_A"] == s for r in rs) for s in sorted({r["S_A"] for r in rs})},
                                     "by_scene": {s: sum(r["scene"] == s for r in rs) for s in sorted({r["scene"] for r in rs})},
                                     "T_edge_mean_of_misses": float(np.mean([r["wall"] for r in rs])) if rs else None}
    else:
        res["per_n_procs"] = summarize(rows, ("n_procs",))
        res["per_session"] = summarize(rows, ("n_procs", "scene"))
    with open(a.out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    for g, v in res.get("all", res.get("per_n_procs", {})).items():
        print(g, json.dumps({k: v[k] for k in ("frames", "frac_E2E_gt_11.1", "cp95_E2E")}), json.dumps(v["E2E"]), json.dumps(v["T_edge_wall"]), flush=True)


def main(a):
    if a.aggregate:
        return aggregate(a)
    torch.backends.cuda.matmul.allow_tf32 = False
    log = []
    info = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "args": vars(a), "gpu_apps_before": gpu_apps(), "gpu_before": gpu_state()}
    if a.barrier is None:
        info["waited_s"] = wait_idle({os.getpid()}, log)
    rows = []
    for sc in a.scenes.split(","):
        scene = Scene(*scene_location(sc, a.data))
        for nk in a.nets.split(","):
            E = Edge(sc, nk, a); views = Views(E, scene)
            info.setdefault("checks", {})[f"{sc}/{E.kind}"] = {"tag": E.tag, "net": E.netname, "rans": E.check, "A": E.A}
            if a.barrier is not None:
                run_paced(a, E, scene, views, rows, info)
            elif a.mode == "session":
                run_session(a, E, scene, views, rows, info)
            else:
                run_cold(a, E, scene, views, rows, info)
            del E, views
            torch.cuda.empty_cache()
    info["gpu_apps_after"] = gpu_apps(); info["gpu_after"] = gpu_state(); info["wait_log"] = log
    keys = ("net", "cfg", "tau", "scheme") if a.mode == "session" else ("net", "cfg", "tau", "kind")
    if a.barrier is not None:
        keys = ("n_procs", "scene")
    good = [r for r in rows if "skipped" not in r]
    res = {"info": info, "summary": summarize(good, keys) if good else {}}
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    with gzip.open(a.out_json[:-5] + "_rows.json.gz", "wt") as fh:
        json.dump(rows, fh)
    for g, v in res["summary"].items():
        print(g, json.dumps(v["E2E"]), v["frac_E2E_gt_11.1"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="session", choices=("session", "cold"))
    ap.add_argument("--scenes", default="garden")
    ap.add_argument("--nets", default="deploy,attn", help="deploy (MLP joint / EdgeConcealNet) and/or attn (cross-attention reference)")
    ap.add_argument("--runs", default="T8R8:10:conceal,T8R8:10:arq+conceal,T8R8:2.5:conceal,T8R8:2.5:arq+conceal,T6R11:10:conceal,T6R11:2.5:conceal",
                    help="session runs cfg:tau:scheme (attn: only the T8R8 conceal runs)")
    ap.add_argument("--cold-runs", default="T8R8:2.5,T8R8:10,T6R11:2.5")
    ap.add_argument("--chains", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chains-t6", type=int, default=0, help="chains of the six-mode runs (0: --chains)")
    ap.add_argument("--chains-attn", type=int, default=0, help="chains of the runs of the reference network (0: --chains)")
    ap.add_argument("--probe-every", type=int, default=8, help="cold start: views timed between two probes")
    ap.add_argument("--n-ev", type=int, default=3)
    ap.add_argument("--eps", type=float, default=1 / 6)
    ap.add_argument("--arq-last-slot", type=int, default=5)
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads of the (untimed) cloud-side encoder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--out-json", default="results/rev6_raw/e2e.json")
    ap.add_argument("--barrier", default=None, help="directory of the start barrier of concurrent sessions (implies --pace)")
    ap.add_argument("--n-procs", type=int, default=1)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--max-retry", type=int, default=5, help="attempts of a chain (cold: view) disturbed by host load")
    ap.add_argument("--max-frames", type=int, default=0, help="time only the first frames of every path (1: the full-view first frame)")
    ap.add_argument("--attn-all-runs", action="store_true", help="time the reference network on every run (default: T8R8 conceal only)")
    ap.add_argument("--no-probe", action="store_true", help="skip the quietness probe (concurrent sessions)")
    ap.add_argument("--aggregate", default="", help="glob of per-scene summary files to pool")
    main(ap.parse_args())
