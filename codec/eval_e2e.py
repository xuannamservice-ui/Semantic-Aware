"""Codeword-level evaluation of the service failed-slot rate: the failed 10-us segments of a 1-ms slot are passed through
the bitstream of the slot, token by token, instead of failing whole streams.

PHY input (load_phy): per variant (configuration, link variant, FEC threshold) the total number N of simulated slots and,
for every slot with a failed segment, a boolean array [n_rows, 100] of failed 10-us segments, row r = transmitted OAM mode
modes[r]. Per slot, the view (scene, test view) is drawn uniformly from the pooled test views of the 11 scenes by a hash
of the slot index, so a slot gets the same view in every variant (paired); with more than --n-sample records, the
records with the smallest hash value of the slot index are taken (a uniform subsample, nested across variants).

Bitstream of a slot = the per-view evaluation of eval_layered.py (full-view exposure, policy interleaved, network gen_fp
= the 'Conceal' column): the visible anchors form tokens of 64 Morton-consecutive anchors; enhancement token t goes to
stream t mod M (streams in reliability order, stream m on mode rel[m]); the anchor token (fixed-length positions and base
layer) is replicated on every anchor mode (Table S1, sigma_R^2 = 2.5: T6R11 (-1,+2), T8R8 (-2,-1,+1,+3)). Token sizes are
the real byte lengths of the grouped-substream rANS coder of range_code2.py (6 channels per independently terminated
substream, 16-bit words; tables of the scene), encoded per view, plus --hdr-bytes per token (default 0: the paper
accounts no token header). Every stream sends its tokens back to back in token order; an anchor mode sends the anchor
payload first, then its enhancement tokens. Timing of the slot:
  fill : the payload of every stream spans the slot (time proportional to the cumulative bytes of the stream)
  rate : all streams at one rate, the longest stream spans the slot (the others end earlier)
Loss rules (combos):
  stream     : a stream with a failed segment loses all its tokens; the anchor payload is lost if every anchor mode has a
               failed segment (the rule behind Table abar and eval_layered.py)
  token_fill : a token is lost if a failed segment of its stream overlaps its time interval (fill timing); an anchor token
               is lost if all its copies are hit
  token_rate : the same with rate timing
  token_fill_stag : fill timing, copy j of the anchor payload starts at token j T / n_A (cyclic), so that the copies of
               one token are sent at different times (not in the paper; optional)
Anchors whose anchor token is lost are neither rendered nor used as received neighbours (their base layer still enters
the network input, so Psi of such slots is indicative only); a slot with a lost anchor token fails regardless of Psi.
Psi = PSNR of the rendered view against the loss-free rendering, capped at PSI_MAX (no loss).
Output: <out>/<scene>/e2e<tag><suffix>.json (rows), <out>/e2e_summary<suffix>.json (--summary)
usage: python eval_e2e.py --phy-e2e ../phy_sim/results/e2e --scenes garden,room --summary
       python eval_e2e.py --make-synth results/e2e_synth --scenes garden      (synthetic variants for the tests)
       python eval_e2e.py --validate --scenes garden --cfgs T8R8               (rANS sizes, zero masks, whole-stream masks)
"""
import argparse
import fnmatch
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.stats import beta as beta_dist
from torch.utils.cpp_extension import load_inline

import metrics
import range_code2
from conceal_net import knn
from eval_layered import load
from scene_data import Scene, scene_location
from streaming import Packetizer, mode_reliability_order, patterns_to_streams
from train_layered import LayeredConcealer

SCENES = ("bicycle", "bonsai", "counter", "drjohnson", "garden", "kitchen", "playroom", "room", "stump", "train", "truck")
ANCHOR_MODES = {"T6R11": (-1, 2), "T6R8": (-1, 2), "T8R11": (-3, 1), "T8R8": (-2, -1, 1, 3)}   # Table S1, sigma_R^2 = 2.5
NSEG, PSI_MAX, GROUP = 100, 99.0, 6
COMBOS = ("stream", "token_fill", "token_rate", "token_fill_stag")


# ------------------------------------------------------------------------------------------------ PHY failure records
def splitmix(x):
    """Vectorized splitmix64 hash of uint64 values -> uint64."""
    x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def slot_hash(slots, seed, salt):
    with np.errstate(over="ignore"):
        return splitmix(splitmix(np.asarray(slots, dtype=np.int64).astype(np.uint64) ^ np.uint64(seed * 1000003 + salt)))


class Variant:
    """name, cfg, N (simulated slots), modes (OAM mode of each mask row), slots (int64 [n]) and a loader of the masks
    (bool [k, n_rows, NSEG]) of selected records; anchor_modes may be given by the PHY output."""

    def __init__(self, name, cfg, N, modes, slots, get_masks, anchor_modes=None, meta=None):
        self.name, self.cfg, self.N, self.modes, self.slots = name, cfg, int(N), [int(x) for x in modes], np.asarray(slots, np.int64)
        self.get_masks, self.anchor_modes, self.meta = get_masks, anchor_modes, meta or {}


def load_phy(path):
    """Variants of the PHY output (phy_sim/run_e2e.py): <path>/rec_<cfg>_<variant>_<threshold>.npz with slot (int64 [n],
    slots with a failed segment), fail (uint8 [n, M, NSEG], row r = stream on tx_modes[r]), tx_modes, rx_modes, N.
    Otherwise the synthetic format of make_synth: <path>/*.npz with name, cfg, N, modes, slot, mask_bits (np.packbits
    of the mask along the last axis, little bit order) and optionally anchor_modes."""
    out = []
    recs = sorted(glob.glob(os.path.join(path, "rec_*.npz")))
    for f in recs:
        name = os.path.basename(f)[4:-4]
        parts = name.split("_")
        z = np.load(f, allow_pickle=False)
        out.append(Variant(name, parts[0], int(z["N"]), z["tx_modes"], z["slot"], lambda sel, z=z: z["fail"][sel].astype(bool),
                           meta={"phy_variant": "_".join(parts[1:-1]), "threshold": parts[-1],
                                 "rx_modes": [int(x) for x in z["rx_modes"]]}))
    for f in ([] if recs else sorted(glob.glob(os.path.join(path, "*.npz")))):
        z = np.load(f, allow_pickle=False)
        out.append(Variant(str(z["name"]), str(z["cfg"]), int(z["N"]), z["modes"], z["slot"],
                           lambda sel, z=z: np.unpackbits(z["mask_bits"][sel], axis=-1, count=NSEG, bitorder="little").astype(bool),
                           tuple(int(x) for x in z["anchor_modes"]) if "anchor_modes" in z else None))
    return out


def make_synth(a):
    """Synthetic variants for the tests (N = 10^6 slots each, the same slot indices in all of them):
    zero (no failed segment), burst30 (one random stream, contiguous burst of 30 segments), fade3 (all streams, 3
    segments), whole (failure patterns of the eval_layered traces, every segment of the failed modes)."""
    os.makedirs(a.make_synth, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
    n, N = a.n_synth, 10 ** 6
    slots = np.sort(rng.choice(N, n, replace=False))
    for cfg in a.cfgs.split(","):
        tx = ms["configs"][cfg]["tx"]; R = len(tx)
        tr = np.load(os.path.join(a.phy, f"traces_{a.s2}.npz"))[cfg]
        pats, cnt = np.unique(tr[tr > 0], return_counts=True)
        var = {k: np.zeros((n, R, NSEG), bool) for k in ("zero", "burst30", "fade3", "whole")}
        for i in range(n):
            r, s0 = rng.integers(R), rng.integers(NSEG - 30 + 1)
            var["burst30"][i, r, s0:s0 + 30] = True
            s0 = rng.integers(NSEG - 3 + 1)
            var["fade3"][i, :, s0:s0 + 3] = True
            p = int(pats[rng.choice(len(pats), p=cnt / cnt.sum())])
            var["whole"][i, [j for j in range(R) if (p >> j) & 1]] = True
        for k, m in var.items():
            np.savez_compressed(os.path.join(a.make_synth, f"{cfg}_{k}.npz"), name=f"{cfg}_{k}", cfg=cfg, N=N,
                                modes=np.asarray(tx), slot=slots, mask_bits=np.packbits(m, axis=-1, bitorder="little"))
        print(f"synthetic {cfg}: {n} records per variant in {a.make_synth}", flush=True)


def view_pool(a):
    """Pooled (scene, view) list: the test views of eval_layered (all scenes have results for every view)."""
    pool = []
    for sc in SCENES:
        e = json.load(open(os.path.join(a.out, sc, f"eval_layered{a.tag}_T8R8_{a.s2}.json")))
        pool += [(sc, v) for v in sorted({r["view"] for r in e["rows"]})]
    return pool


def selection(v, a):
    """Indices of the rendered records of a variant (all, or the n_sample with the smallest slot hash)."""
    if len(v.slots) <= a.n_sample:
        return np.arange(len(v.slots))
    return np.sort(np.argsort(slot_hash(v.slots, a.seed, 1), kind="stable")[:a.n_sample])


# ------------------------------------------------------------------------------------------------ rANS token sizes
class TokenSizes:
    """Per-view byte length of every token: anchor token (fixed-length positions + base layer) and enhancement token,
    encoded with the per-channel Laplace tables and the substream layout of range_code2.py."""

    def __init__(self, ck, fb, pos_bits, threads):
        self.ext = load_inline("rans2_ext", cpp_sources=range_code2.CPP, functions=["encode", "decode"],
                               extra_cflags=["-O3", "-fopenmp"], extra_ldflags=["-fopenmp"], verbose=False)
        st = {k: v.detach().cpu() for k, v in ck["state"].items()}
        A = st["feat"].shape[0]
        step = {n: float(np.exp(float(st[f"log_qstep.{n}"]))) for n in ("feat", "offsets", "scaling")}
        rm = ck["rate_model"]
        loc = {n: rm[f"loc.{n}"].cpu().numpy().astype(np.float64) for n in step}
        scl = {n: np.exp(rm[f"log_b.{n}"].cpu().numpy().astype(np.float64)) for n in step}
        sym = {"feat": torch.round(st["feat"] / step["feat"]).long().numpy(),
               "offsets": torch.round(st["offsets"].reshape(A, -1) / step["offsets"]).long().numpy(),
               "scaling": torch.round(st["scaling"] / step["scaling"]).long().numpy()}
        layers = {"base": [("feat", slice(0, fb)), ("scaling", slice(None))],
                  "enhancement": [("feat", slice(fb, None)), ("offsets", slice(None))]}
        self.tab, self.pos_bits, self.threads = {}, pos_bits, threads
        for lname, parts in layers.items():
            q = np.concatenate([sym[n][:, sl] for n, sl in parts], 1)
            mu = np.concatenate([loc[n][sl] for n, sl in parts]); bb = np.concatenate([scl[n][sl] for n, sl in parts])
            stp = np.concatenate([np.full(sym[n][:, sl].shape[1], step[n]) for n, sl in parts])
            Cn = q.shape[1]; S = 0; fr, cu, lo_ = [], [], []
            for c in range(Cn):
                f_, c_, l_, _, S_, _, _ = range_code2.tables(q[:, c:c + 1], mu[c:c + 1], bb[c:c + 1], stp[c])
                fr.append(f_[0]); cu.append(c_[0]); lo_.append(int(l_[0])); S = max(S, S_)
            freq = np.zeros((Cn, S), np.int32); cum = np.zeros((Cn, S + 1), np.int32)
            for c in range(Cn):
                freq[c, :fr[c].size] = fr[c]; cum[c, :cu[c].size] = cu[c]; cum[c, cu[c].size:] = cu[c][-1]
            assert Cn % GROUP == 0
            self.tab[lname] = (q.astype(np.int32), torch.from_numpy(freq.reshape(-1)), torch.from_numpy(cum.reshape(-1)),
                               torch.tensor(lo_, dtype=torch.int32), S)

    def layer_bytes(self, lname, anchors):
        """anchors: (n,) anchor indices in transmission (Morton) order -> bytes of each token of 64."""
        q, F_, C_, L_, S = self.tab[lname]
        qs = q[anchors]; n, Cn = qs.shape; G = Cn // GROUP; T = (n + 63) // 64
        nt = np.minimum(64, n - 64 * np.arange(T))                       # anchors per token
        # (token, group) substreams in the order of range_code2: token-major, groups of GROUP channels, anchor-major inside
        pad = np.zeros((T * 64, Cn), np.int32); pad[:n] = qs
        blk = pad.reshape(T, 64, G, GROUP).transpose(0, 2, 1, 3)        # (T, G, 64, GROUP)
        keep = (np.arange(64)[None, :] < nt[:, None])                    # (T, 64)
        flat = blk[np.broadcast_to(keep[:, None, :], (T, G, 64))].reshape(-1)
        sizes = np.repeat(nt * GROUP, G)
        tok_off = torch.from_numpy(np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64))
        sc0 = torch.from_numpy(np.tile(np.arange(G) * GROUP, T).astype(np.int32))
        snc = torch.full((T * G,), GROUP, dtype=torch.int32)
        _, doff = self.ext.encode(torch.from_numpy(np.ascontiguousarray(flat)), tok_off, sc0, snc, F_, C_, L_, S, self.threads)
        words = (doff[1:] - doff[:-1]).numpy().reshape(T, G).sum(1)
        return 2.0 * words, nt

    def view(self, anchors):
        eb, nt = self.layer_bytes("enhancement", anchors)
        bb, _ = self.layer_bytes("base", anchors)
        return bb + nt * self.pos_bits / 8.0, eb


# ------------------------------------------------------------------------------------------------ slot layout and losses
def intervals(sizes_by_stream, denom):
    """Segment range [lo, hi] (inclusive) of consecutive items of the given byte sizes over a slot of denom bytes."""
    c = np.concatenate([[0.0], np.cumsum(sizes_by_stream)]) / denom * NSEG
    lo = np.floor(c[:-1] + 1e-9).astype(np.int64)
    hi = np.ceil(c[1:] - 1e-9).astype(np.int64) - 1
    return np.clip(lo, 0, NSEG - 1), np.clip(np.maximum(hi, lo), 0, NSEG - 1)


class ViewLayout:
    """Tokens of a view, their streams and their segment ranges under each timing."""

    def __init__(self, pk, idx, M, anchor_streams, sizes, hdr):
        self.order = torch.argsort(pk.morton[idx])
        s = idx[self.order]
        self.tid = (torch.arange(s.numel(), device=idx.device) // pk.G)
        T = int(self.tid[-1]) + 1 if s.numel() else 0
        self.T, self.M, self.anchor_streams = T, M, list(anchor_streams)
        self.st = (torch.arange(T) % M).numpy()                         # interleaved policy (Packetizer.assign)
        ab, eb = sizes(s.cpu().numpy())
        ab, eb = ab + hdr, eb + hdr
        self.bytes_anchor, self.bytes_enh = float(ab.sum()), float(eb.sum())
        tot = np.array([eb[self.st == m].sum() + (ab.sum() if m in self.anchor_streams else 0.0) for m in range(M)])
        self.lay = {}
        for timing in ("fill", "rate", "fill_stag"):
            enh_lo, enh_hi = np.zeros(T, np.int64), np.zeros(T, np.int64)
            anc_lo, anc_hi = np.zeros((len(self.anchor_streams), T), np.int64), np.zeros((len(self.anchor_streams), T), np.int64)
            for m in range(M):
                den = tot[m] if timing != "rate" else tot.max()
                sel = np.nonzero(self.st == m)[0]
                if m in self.anchor_streams:
                    j = self.anchor_streams.index(m)
                    rot = (j * T) // len(self.anchor_streams) if timing == "fill_stag" else 0
                    aord = np.roll(np.arange(T), -rot)                    # transmission order of the anchor tokens
                    lo, hi = intervals(np.concatenate([ab[aord], eb[sel]]), den)
                    anc_lo[j, aord], anc_hi[j, aord] = lo[:T], hi[:T]
                    enh_lo[sel], enh_hi[sel] = lo[T:], hi[T:]
                else:
                    lo, hi = intervals(eb[sel], den)
                    enh_lo[sel], enh_hi[sel] = lo, hi
            self.lay[timing] = (enh_lo, enh_hi, anc_lo, anc_hi)
        self.share_anchor_mode = float(ab.sum() / tot[self.anchor_streams].mean()) if self.anchor_streams else 0.0

    def losses(self, fail, combo):
        """fail: bool [M, NSEG] per stream -> (enhancement token lost [T], anchor token lost [T])."""
        if combo == "stream":
            f = fail.any(1)
            return f[self.st], np.full(self.T, bool(f[self.anchor_streams].all()))
        enh_lo, enh_hi, anc_lo, anc_hi = self.lay[combo[len("token_"):]]
        cs = np.concatenate([np.zeros((self.M, 1), np.int64), np.cumsum(fail, 1)], 1)
        el = cs[self.st, enh_hi + 1] - cs[self.st, enh_lo] > 0
        al = np.ones(self.T, bool)
        for j, m in enumerate(self.anchor_streams):
            al &= cs[m, anc_hi[j] + 1] - cs[m, anc_lo[j]] > 0
        return el, al

    def anchor_masks(self, tok_lost):
        """Token mask -> anchor mask in the order of idx (as Packetizer.lost_mask)."""
        t = torch.as_tensor(tok_lost, device=self.tid.device)[self.tid]
        out = torch.empty_like(t); out[self.order] = t
        return out


# ------------------------------------------------------------------------------------------------ rendering (eval_layered)
class SceneEval:
    def __init__(self, a, scene_name):
        self.scene = Scene(*scene_location(scene_name, a.data))
        self.model, self.L, nets, c, ck = load(a.out, scene_name, a.tag)
        with torch.no_grad():
            self.at = self.model.attributes(quantized=True)
            self.base, self.enh = self.L.split(self.at)
            stats = (self.base.mean(0), self.base.std(0).clamp_min(0.02), self.enh.mean(0), self.enh.std(0).clamp_min(0.02))
        nbr = knn(self.model.xyz, c["kn"])
        self.conc = LayeredConcealer(self.model, self.L, nets[a.method], stats, nbr, ck["radius"])
        self.pk = Packetizer(self.model.xyz, self.model.voxel, ck["importance"].cuda(), G=c["G"])
        mt = json.load(open(os.path.join(a.out, scene_name, f"metrics_layered{a.tag}.json")))
        self.ts = TokenSizes(ck, ck["base_dim"], mt["rate_bits_per_anchor"]["position_fixed"], a.threads)
        self.ms = json.load(open(os.path.join(a.phy, f"modeset_{a.s2}.json")))
        self.hdr, self.half, self.views = a.hdr_bytes, a.half, {}

    def cfg(self, cfg, anchor_modes):
        cf = self.ms["configs"][cfg]; tx = cf["tx"]
        rel = mode_reliability_order(cf["per_mode_count"][cf["snr_labels"].index("op")], tx)
        am = anchor_modes or ANCHOR_MODES[cfg]
        return tx, rel, sorted(rel.index(int(x)) for x in am)

    def view(self, vi):
        if vi not in self.views:
            self.views = {}                                             # one view at a time (records sorted by view)
            gt, K, V = self.scene.view(vi)
            idx = self.model.visible(K, V, self.scene.W, self.scene.H)
            with torch.no_grad():
                full, _, _ = self.model.render(self.at, idx, K, V, self.scene.W, self.scene.H)
            self.views[vi] = {"gt": gt, "K": K, "V": V, "idx": idx, "full": full, "lay": {}}
        return self.views[vi]

    def layout(self, vw, M, anchor_streams):
        key = (M, tuple(anchor_streams))
        if key not in vw["lay"]:
            vw["lay"][key] = ViewLayout(self.pk, vw["idx"], M, anchor_streams, self.ts.view, self.hdr)
        return vw["lay"][key]

    @torch.no_grad()
    def render(self, vw, lost_m, gone_m):
        """lost_m: enhancement lost (anchor received), gone_m: anchor token lost; masks over idx. Returns Psi, PSNR."""
        idx, model = vw["idx"], self.model
        lost_m = lost_m & ~gone_m
        lost = idx[lost_m]
        avail = torch.zeros(model.xyz.shape[0], dtype=torch.bool, device="cuda")
        avail[idx[~lost_m & ~gone_m]] = True
        e = self.enh.clone(); ridx = idx[~gone_m]
        if lost.numel():
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                e[lost] = self.conc.predict(lost, avail, self.base, self.enh).float()
        if ridx.numel():
            img, _, _ = model.render(self.L.merge(self.base, e), ridx, vw["K"], vw["V"], self.scene.W, self.scene.H)
        else:
            img = torch.zeros_like(vw["full"])
        psi = metrics.psnr(img.clamp(0, 1), vw["full"].clamp(0, 1))
        return min(psi, PSI_MAX), metrics.psnr(img.clamp(0, 1), vw["gt"])

    def evaluate(self, cfg, anchor_modes, modes, vi, mask, combos):
        """mask: bool [n_rows, NSEG], row r = mode modes[r]. One row per combo; equal loss masks are rendered once."""
        tx, rel, anchor_streams = self.cfg(cfg, anchor_modes)
        M = len(tx)
        fail = np.zeros((M, NSEG), bool)
        for r, md in enumerate(modes):
            fail[rel.index(int(md))] |= mask[r]
        vw = self.view(vi)
        lay = self.layout(vw, M, anchor_streams)
        out, cache = {}, {}
        for combo in combos:
            el, al = lay.losses(fail, combo)
            key = (el.tobytes(), al.tobytes())
            if key not in cache:
                psi, ps = self.render(vw, lay.anchor_masks(el), lay.anchor_masks(al))
                cache[key] = {"psi": psi, "psnr": ps, "anchor_lost": bool(al.any()), "n_anchor_tok_lost": int(al.sum()),
                              "enh_tok_lost_frac": float(el.mean()) if lay.T else 0.0}
            out[combo] = cache[key]
        return out, {"T": lay.T, "fail_frac": float(fail.mean()), "K_any": int(fail.any(1).sum()),
                     "anchor_share": lay.share_anchor_mode}


# ------------------------------------------------------------------------------------------------ main runs
def run(a):
    variants = [v for v in load_phy(a.phy_e2e) if v.cfg in a.cfgs.split(",")
                and (not a.variants or any(fnmatch.fnmatch(v.name, p) for p in a.variants.split(",")))]
    pool = view_pool(a)
    plan = {}                                                           # scene -> list of (view, variant index, record)
    meta = []
    for k, v in enumerate(variants):
        sel = selection(v, a)
        vid = slot_hash(v.slots[sel], a.seed, 2) % np.uint64(len(pool))
        for r, j in zip(sel, vid):
            sc, vi = pool[int(j)]
            plan.setdefault(sc, []).append((vi, k, int(r)))
        v.sel_masks = dict(zip((int(r) for r in sel), v.get_masks(sel)))   # selected records only
        meta.append({"name": v.name, "cfg": v.cfg, "N": v.N, "n_records": int(len(v.slots)), "n_sample": int(len(sel)),
                     "modes": v.modes, "anchor_modes": list(v.anchor_modes or ANCHOR_MODES[v.cfg]), **v.meta})
        print(f"variant {v.name} ({v.cfg}): N={v.N} records={len(v.slots)} sampled={len(sel)}", flush=True)
    combos = a.combos.split(",")
    for sc in a.scenes.split(","):
        out_f = os.path.join(a.out, sc, f"e2e{a.tag}{a.suffix}.json")
        jobs = sorted(plan.get(sc, []))
        if a.skip_done and os.path.exists(out_f):
            print(f"{sc}: done", flush=True); continue
        se = SceneEval(a, sc); rows = []; t0 = time.time()
        for n_, (vi, k, r) in enumerate(jobs):
            v = variants[k]
            res, info = se.evaluate(v.cfg, v.anchor_modes, v.modes, vi, v.sel_masks[r], combos)
            for combo, x in res.items():
                rows.append({"variant": v.name, "cfg": v.cfg, "slot": int(v.slots[r]), "view": int(vi), "combo": combo, **x, **info})
            if n_ % 200 == 0:
                print(f"{sc}: {n_ + 1}/{len(jobs)} records, {time.time() - t0:.0f}s", flush=True)
        json.dump({"scene": sc, "tag": a.tag, "s2": a.s2, "phy_e2e": a.phy_e2e, "seed": a.seed, "hdr_bytes": a.hdr_bytes,
                   "method": a.method, "policy": "interleaved", "combos": combos, "variants": meta, "rows": rows},
                  open(out_f, "w"))
        print(f"{sc}: {len(jobs)} records, {len(rows)} rows, {time.time() - t0:.0f}s -> {out_f}", flush=True)
        del se; torch.cuda.empty_cache()


def clopper_pearson(k, n, conf=0.95):
    lo = 0.0 if k == 0 else float(beta_dist.ppf((1 - conf) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - (1 - conf) / 2, k + 1, n - k))
    return lo, hi


def summary(a):
    rows, meta, sufs = [], {}, a.summary_suffixes.split(",") if a.summary_suffixes else [a.suffix]
    for suf in sufs:                                                    # runs with different variants, one suffix each
        for sc in SCENES:
            f = os.path.join(a.out, sc, f"e2e{a.tag}{suf}.json")
            if os.path.exists(f):
                e = json.load(open(f)); rows += e["rows"]; meta.update({v["name"]: v for v in e["variants"]})
    out = {"tag": a.tag, "suffixes": sufs, "seed": a.seed, "psi_max": PSI_MAX, "variants": {}}
    for vm in meta.values():
        res = {}
        for combo in sorted({r["combo"] for r in rows}):
            r = [x for x in rows if x["variant"] == vm["name"] and x["combo"] == combo]
            if not r:
                continue
            n = len(r); ns, nrec, N = vm["n_sample"], vm["n_records"], vm["N"]
            al = np.array([x["anchor_lost"] for x in r]); psi = np.array([x["psi"] for x in r])
            d = {"n_rendered": n, "n_anchor_lost": int(al.sum())}
            for th in (30, 35):
                k = int((al | (psi < th)).sum())
                d[f"n_psi{th}"] = int(((psi < th) & ~al).sum())
                if ns == nrec:                                          # every failed slot rendered: k failures in N slots
                    lo, hi = clopper_pearson(k, N); ab = k / N
                else:                                                   # uniform subsample of the records
                    lo, hi = (x * nrec / N for x in clopper_pearson(k, n)); ab = k / n * nrec / N
                d[f"abar_{th}"], d[f"abar_{th}_ci"] = ab, [lo, hi]
            d["mean_psi"] = float(psi.mean()); d["mean_psi_anchor_ok"] = float(psi[~al].mean()) if (~al).any() else None
            d["median_psi"] = float(np.median(psi)); d["n_no_loss"] = int((psi >= PSI_MAX).sum())
            d["mean_enh_tok_lost_frac"] = float(np.mean([x["enh_tok_lost_frac"] for x in r]))
            res[combo] = d
            if n != ns:
                print(f"warning: {vm['name']} {combo}: {n} rendered of {ns} sampled (scenes missing)")
        out["variants"][vm["name"]] = {**vm, "results": res}
    f = os.path.join(a.out, f"e2e_summary{a.suffix}.json")
    json.dump(out, open(f, "w"), indent=1)
    for name, v in out["variants"].items():
        for combo, d in v["results"].items():
            print(f"{name:28s} {combo:16s} n={d['n_rendered']:5d} anchor={d['n_anchor_lost']:4d} psi<30={d['n_psi30']:4d} "
                  f"psi<35={d['n_psi35']:4d} meanPsi={d['mean_psi']:.2f} abar30={d['abar_30']:.3g} "
                  f"[{d['abar_30_ci'][0]:.3g},{d['abar_30_ci'][1]:.3g}] abar35={d['abar_35']:.3g}", flush=True)
    print("->", f)


def validate(a):
    """(1) per-view rANS coder over all anchors in Morton order = range_code2 size; (2) zero masks -> loss-free;
    (3) whole-stream masks of the eval_layered patterns -> the stored psnr_vs_full (interleaved, gen_fp)."""
    rep = {}
    for sc in a.scenes.split(","):
        se = SceneEval(a, sc); r = {}
        rc = json.load(open(os.path.join(a.out, sc, f"rangecode{a.tag}_g{GROUP}.json")))["layers"]
        allm = torch.argsort(se.pk.morton).cpu().numpy(); A = allm.size
        for ln in ("base", "enhancement"):
            b, _ = se.ts.layer_bytes(ln, allm)
            r[f"rans_{ln}_bits_per_anchor"] = [8 * float(b.sum()) / A, rc[ln]["real_bits_per_anchor"]]
        for cfg in a.cfgs.split(","):
            e = json.load(open(os.path.join(a.out, sc, f"eval_layered{a.tag}_{cfg}_{a.s2}.json")))
            tx, rel, anchor_streams = se.cfg(cfg, None)
            ev = [x for x in e["rows"] if x["policy"] == "interleaved" and x["method"] == a.method]
            views = sorted({x["view"] for x in ev})[:a.n_val_views]
            zero, diffs, ok = [], [], 0
            for vi in views:
                z, _ = se.evaluate(cfg, None, tx, vi, np.zeros((len(tx), NSEG), bool), ("stream", "token_fill", "token_rate"))
                zero += [x["psi"] for x in z.values()]
                for x in [x for x in ev if x["view"] == vi]:
                    pat = x["pattern"]
                    m = np.zeros((len(tx), NSEG), bool)
                    for j in range(len(tx)):
                        if (pat >> j) & 1:
                            m[j] = True
                    assert sorted(patterns_to_streams(pat, tx, rel)) == sorted(rel.index(int(tx[j])) for j in range(len(tx)) if m[j].any())
                    res, _ = se.evaluate(cfg, None, tx, vi, m, ("stream", "token_fill", "token_rate"))
                    if res["stream"]["anchor_lost"]:
                        continue                                        # eval_layered keeps the anchor channel
                    ok += 1
                    diffs.append([x["psnr_vs_full"], res["stream"]["psi"], res["token_fill"]["psi"], res["token_rate"]["psi"],
                                  x["lost_frac"], res["stream"]["enh_tok_lost_frac"]])
            d = np.array(diffs)
            r[cfg] = {"views": views, "zero_mask_psi_min": float(min(zero)), "n_whole": ok,
                      "max_abs_diff_stream": float(np.abs(d[:, 0] - d[:, 1]).max()),
                      "max_abs_diff_token_fill": float(np.abs(d[:, 0] - d[:, 2]).max()),
                      "max_abs_diff_token_rate": float(np.abs(d[:, 0] - d[:, 3]).max()),
                      "mean_eval_layered": float(d[:, 0].mean()), "mean_e2e_stream": float(d[:, 1].mean()),
                      "max_abs_diff_lost_frac_vs_token_frac": float(np.abs(d[:, 4] - d[:, 5]).max())}
        rep[sc] = r
        print(sc, json.dumps(r), flush=True)
        del se; torch.cuda.empty_cache()
    json.dump(rep, open(os.path.join(a.out, f"e2e_validate{a.suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--cfgs", default="T6R11,T8R8")
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--s2", default="2.5")
    ap.add_argument("--data", default="data/360_v2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--phy", default="phy")
    ap.add_argument("--phy-e2e", default="../phy_sim/results/e2e", help="folder of the PHY failure records (load_phy)")
    ap.add_argument("--method", default="gen_fp")
    ap.add_argument("--combos", default="stream,token_fill,token_rate")
    ap.add_argument("--n-sample", type=int, default=2000, help="records rendered per variant (all if fewer)")
    ap.add_argument("--hdr-bytes", type=float, default=0.0, help="header bytes per token (0: none, as in the paper)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--variants", default="", help="comma list of fnmatch patterns of the variant names (default: all)")
    ap.add_argument("--summary-suffixes", default="", help="comma list of run suffixes pooled into one summary")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--summary", action="store_true", help="write the summary after the run (or alone with --no-run)")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n-val-views", type=int, default=6)
    ap.add_argument("--make-synth", default="", help="write synthetic variants to this folder and exit")
    ap.add_argument("--n-synth", type=int, default=400)
    a = ap.parse_args()
    if a.make_synth:
        make_synth(a)
    elif a.validate:
        validate(a)
    else:
        if not a.no_run:
            run(a)
        if a.summary:
            summary(a)
