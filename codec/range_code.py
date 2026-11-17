"""Implemented entropy coder for the layered codec: per-token rANS with the factorized Laplace model of the paper.

For one scene: the quantized symbols of the base layer (12 feature channels and 6 scaling values per anchor) and of the
enhancement layer (36 feature channels and 30 offsets), per-channel frequency tables built from the transmitted Laplace
parameters (location, scale, quantization step) on the per-channel symbol range (both sent once per scene as side
information), and one independently terminated rANS stream per token of 64 Morton-ordered anchors and per layer, so that
a lost token does not affect the others. The script reports the real size against the empirical entropy and the model
code length used in the paper, checks the bit-exact round trip, and times decoding with 1 and N CPU threads for 5,500,
35,000 and 112,000 anchors (a session frame at the 95th percentile, the largest new-anchor frame, an average garden view).
rANS: 32-bit state, 16-bit renormalization, 16-bit probability precision; every symbol in the range gets a frequency of
at least one. Runs on the CPU only.
Output: <out>/<scene>/rangecode<tag>.json
usage: python range_code.py --scene garden [--threads 32]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

from streaming import Packetizer

CPP = r"""
#include <torch/extension.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cstdint>
static const uint32_t PREC = 16, RL = 1u << 16;

std::vector<torch::Tensor> encode(torch::Tensor sym, torch::Tensor tok_off, int64_t nch, torch::Tensor freq, torch::Tensor cum,
                                  torch::Tensor lo, int64_t S, int64_t nthreads) {
  const int32_t* s = sym.data_ptr<int32_t>(); const int64_t* off = tok_off.data_ptr<int64_t>(); int64_t T = tok_off.numel() - 1;
  const int32_t* F = freq.data_ptr<int32_t>(); const int32_t* C = cum.data_ptr<int32_t>(); const int32_t* LO = lo.data_ptr<int32_t>();
  std::vector<std::vector<uint16_t>> outs(T);
  omp_set_num_threads((int)nthreads);
  #pragma omp parallel for schedule(dynamic, 8)
  for (int64_t t = 0; t < T; ++t) {
    std::vector<uint16_t> buf; buf.reserve((size_t)(off[t + 1] - off[t]) + 4);
    uint32_t x = RL;
    for (int64_t j = off[t + 1] - 1; j >= off[t]; --j) {
      int64_t c = (j - off[t]) % nch; int32_t i = s[j] - LO[c];
      uint32_t f = (uint32_t)F[c * S + i], cm = (uint32_t)C[c * (S + 1) + i];
      uint64_t xmax = (uint64_t)f << 16;
      while ((uint64_t)x >= xmax) { buf.push_back((uint16_t)(x & 0xffff)); x >>= 16; }
      x = ((x / f) << PREC) + (x % f) + cm;
    }
    buf.push_back((uint16_t)(x & 0xffff)); buf.push_back((uint16_t)(x >> 16));
    std::reverse(buf.begin(), buf.end());
    outs[t] = std::move(buf);
  }
  std::vector<int64_t> doff(T + 1, 0);
  for (int64_t t = 0; t < T; ++t) doff[t + 1] = doff[t] + (int64_t)outs[t].size();
  auto data = torch::empty({doff[T]}, torch::dtype(torch::kInt32));
  auto d = data.data_ptr<int32_t>();
  for (int64_t t = 0; t < T; ++t) for (size_t k = 0; k < outs[t].size(); ++k) d[doff[t] + (int64_t)k] = outs[t][k];
  auto dofft = torch::from_blob(doff.data(), {T + 1}, torch::dtype(torch::kInt64)).clone();
  return {data, dofft};
}

torch::Tensor decode(torch::Tensor data, torch::Tensor data_off, torch::Tensor tok_off, int64_t nch, torch::Tensor freq, torch::Tensor cum,
                     torch::Tensor lo, torch::Tensor slot2sym, int64_t S, int64_t nthreads) {
  const int32_t* d = data.data_ptr<int32_t>(); const int64_t* doff = data_off.data_ptr<int64_t>(); const int64_t* off = tok_off.data_ptr<int64_t>();
  int64_t T = tok_off.numel() - 1;
  const int32_t* F = freq.data_ptr<int32_t>(); const int32_t* C = cum.data_ptr<int32_t>(); const int32_t* LO = lo.data_ptr<int32_t>();
  const int16_t* Z = slot2sym.data_ptr<int16_t>();
  auto out = torch::empty({off[T]}, torch::dtype(torch::kInt32)); int32_t* o = out.data_ptr<int32_t>();
  omp_set_num_threads((int)nthreads);
  #pragma omp parallel for schedule(dynamic, 8)
  for (int64_t t = 0; t < T; ++t) {
    int64_t p = doff[t];
    uint32_t x = ((uint32_t)(uint16_t)d[p] << 16) | (uint32_t)(uint16_t)d[p + 1]; p += 2;
    for (int64_t j = off[t]; j < off[t + 1]; ++j) {
      int64_t c = (j - off[t]) % nch;
      uint32_t slot = x & 0xffff; int32_t i = Z[c * 65536 + slot];
      uint32_t f = (uint32_t)F[c * S + i], cm = (uint32_t)C[c * (S + 1) + i];
      x = f * (x >> PREC) + slot - cm;
      while (x < RL) { x = (x << 16) | (uint32_t)(uint16_t)d[p++]; }
      o[j] = i + LO[c];
    }
  }
  return out;
}
"""


CU = r"""
#include <torch/extension.h>
#include <cstdint>
__global__ void dec_kernel(const int32_t* d, const int64_t* doff, const int64_t* off, int64_t T, int64_t nch, const int32_t* F,
                           const int32_t* C, const int32_t* LO, const int16_t* Z, int64_t S, int32_t* o) {
  int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int64_t p = doff[t];
  uint32_t x = ((uint32_t)(uint16_t)d[p] << 16) | (uint32_t)(uint16_t)d[p + 1]; p += 2;
  for (int64_t j = off[t]; j < off[t + 1]; ++j) {
    int64_t c = (j - off[t]) % nch;
    uint32_t slot = x & 0xffff; int32_t i = Z[c * 65536 + slot];
    uint32_t f = (uint32_t)F[c * S + i], cm = (uint32_t)C[c * (S + 1) + i];
    x = f * (x >> 16) + slot - cm;
    while (x < (1u << 16)) { x = (x << 16) | (uint32_t)(uint16_t)d[p++]; }
    o[j] = i + LO[c];
  }
}
torch::Tensor decode_cuda(torch::Tensor data, torch::Tensor data_off, torch::Tensor tok_off, int64_t nch, torch::Tensor freq, torch::Tensor cum,
                          torch::Tensor lo, torch::Tensor slot2sym, int64_t S) {
  int64_t T = tok_off.numel() - 1;
  auto out = torch::empty({tok_off[T].item<int64_t>()}, torch::dtype(torch::kInt32).device(data.device()));
  int threads = 64; int blocks = (int)((T + threads - 1) / threads);
  dec_kernel<<<blocks, threads>>>(data.data_ptr<int32_t>(), data_off.data_ptr<int64_t>(), tok_off.data_ptr<int64_t>(), T, nch,
                                  freq.data_ptr<int32_t>(), cum.data_ptr<int32_t>(), lo.data_ptr<int32_t>(), slot2sym.data_ptr<int16_t>(), S,
                                  out.data_ptr<int32_t>());
  return out;
}
"""
CU_DECL = "torch::Tensor decode_cuda(torch::Tensor data, torch::Tensor data_off, torch::Tensor tok_off, int64_t nch, torch::Tensor freq, torch::Tensor cum, torch::Tensor lo, torch::Tensor slot2sym, int64_t S);"


def laplace_cdf(x):
    return 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-np.abs(x)))


def tables(q, mu, b, step):
    """q: (A, C) integer symbols; mu, b: (C,) Laplace location and scale in value units. Returns freq, cum, lo, slot2sym, S
    and the model code length in bits (with the quantized frequencies) and the empirical entropy in bits."""
    A, Cn = q.shape
    lo = q.min(0); hi = q.max(0); n = hi - lo + 1; S = int(n.max())
    freq = np.zeros((Cn, S), np.int64); cum = np.zeros((Cn, S + 1), np.int64); slot = np.zeros((Cn, 65536), np.int16)
    model_bits = 0.0; emp_bits = 0.0
    for c in range(Cn):
        i = np.arange(lo[c], hi[c] + 1)
        bb = max(b[c], 1e-3 * step) / step; m = mu[c] / step
        up = laplace_cdf((i + 0.5 - m) / bb); dn = laplace_cdf((i - 0.5 - m) / bb)
        up[-1] = 1.0; dn[0] = 0.0                                      # tails folded into the extreme symbols
        p = np.clip(up - dn, 0, None); p = p / p.sum()
        f = np.floor(p * (65536 - n[c])).astype(np.int64) + 1
        f[np.argmax(f)] += 65536 - f.sum()
        assert f.min() >= 1 and f.sum() == 65536
        freq[c, :n[c]] = f; cum[c, 1:n[c] + 1] = np.cumsum(f)
        slot[c] = np.repeat(np.arange(n[c]), f).astype(np.int16)
        cnt = np.bincount(q[:, c] - lo[c], minlength=n[c])
        model_bits += float(-(cnt * np.log2(f / 65536)).sum())
        pe = cnt[cnt > 0] / A; emp_bits += float(-(cnt[cnt > 0] * np.log2(pe)).sum())
    return freq, cum, lo, slot, S, model_bits, emp_bits


def main(a):
    ext = load_inline("rans_ext", cpp_sources=CPP, functions=["encode", "decode"], extra_cflags=["-O3", "-fopenmp"],
                      extra_ldflags=["-fopenmp"], verbose=False)
    gext = load_inline("rans_cuda", cpp_sources=CU_DECL, cuda_sources=CU, functions=["decode_cuda"], extra_cuda_cflags=["-O3"], verbose=False) if a.gpu else None
    ck = torch.load(os.path.join(a.out, a.scene, f"layered{a.tag}.pt"), map_location="cpu", weights_only=False)
    st = ck["state"]; fb = ck["base_dim"]; A = st["feat"].shape[0]
    step = {n: float(math.exp(float(st[f"log_qstep.{n}"]))) for n in ("feat", "offsets", "scaling")}
    rm = ck["rate_model"]
    loc = {n: rm[f"loc.{n}"].numpy().astype(np.float64) for n in step}
    scl = {n: np.exp(rm[f"log_b.{n}"].numpy().astype(np.float64)) for n in step}
    sym = {"feat": torch.round(st["feat"] / step["feat"]).long().numpy(),
           "offsets": torch.round(st["offsets"].reshape(A, -1) / step["offsets"]).long().numpy(),
           "scaling": torch.round(st["scaling"] / step["scaling"]).long().numpy()}
    layers = {"base": [("feat", slice(0, fb)), ("scaling", slice(None))], "enhancement": [("feat", slice(fb, None)), ("offsets", slice(None))]}
    pk = Packetizer(st["xyz"].float(), ck["voxel"], ck["importance"].float(), G=64)
    order = torch.argsort(pk.morton).numpy()                           # Morton order of all anchors; tokens of 64
    res = {"scene": a.scene, "anchors": int(A), "threads": a.threads, "cpu": os.cpu_count(), "load_avg": os.getloadavg(), "layers": {}}
    for lname, parts in layers.items():
        q = np.concatenate([sym[n][:, sl] for n, sl in parts], 1)
        mu = np.concatenate([loc[n][sl] for n, sl in parts]); bb = np.concatenate([scl[n][sl] for n, sl in parts])
        stp = np.concatenate([np.full(sym[n][:, sl].shape[1], step[n]) for n, sl in parts])
        # per-channel tables (each channel has its own step)
        Cn = q.shape[1]; S = 0; fr, cu, lo_, sl_, mbits, ebits = [], [], [], [], 0.0, 0.0
        for c in range(Cn):
            f_, c_, l_, s_, S_, m_, e_ = tables(q[:, c:c + 1], mu[c:c + 1], bb[c:c + 1], stp[c])
            fr.append(f_[0]); cu.append(c_[0]); lo_.append(int(l_[0])); sl_.append(s_[0]); S = max(S, S_); mbits += m_; ebits += e_
        freq = np.zeros((Cn, S), np.int32); cum = np.zeros((Cn, S + 1), np.int32)
        for c in range(Cn):
            freq[c, :fr[c].size] = fr[c]; cum[c, :cu[c].size] = cu[c]; cum[c, cu[c].size:] = cu[c][-1]
        slot = np.stack(sl_).astype(np.int16)
        qs = q[order]                                                   # Morton order
        T = (A + 63) // 64
        tok_off = torch.tensor([min(64 * t, A) * Cn for t in range(T + 1)], dtype=torch.int64)
        flat = torch.from_numpy(np.ascontiguousarray(qs.reshape(-1).astype(np.int32)))
        F_, C_, L_ = torch.from_numpy(freq.reshape(-1)), torch.from_numpy(cum.reshape(-1)), torch.tensor(lo_, dtype=torch.int32)
        t0 = time.time(); data, doff = ext.encode(flat, tok_off, Cn, F_, C_, L_, S, a.threads); t_enc = time.time() - t0
        dec = ext.decode(data, doff, tok_off, Cn, F_, C_, L_, torch.from_numpy(slot.reshape(-1)), S, a.threads)
        exact = bool(torch.equal(dec, flat))
        real_bits = 16.0 * data.numel()
        timing = {}
        for n_anc in (5500, 35000, 112000):
            if n_anc > A:
                continue
            Tn = (n_anc + 63) // 64; to_n = tok_off[:Tn + 1].clone(); to_n[-1] = min(Tn * 64, n_anc) * Cn
            dn = data[:int(doff[Tn])]; don = doff[:Tn + 1]
            slot_t = torch.from_numpy(slot.reshape(-1))
            for th in sorted({1, 4, 8, 16, a.threads}):
                ts = []
                for _ in range(10):
                    t0 = time.perf_counter(); ext.decode(dn, don, to_n, Cn, F_, C_, L_, slot_t, S, th); ts.append(time.perf_counter() - t0)
                timing[f"{n_anc}_t{th}_ms_median"] = 1e3 * float(np.median(ts)); timing[f"{n_anc}_t{th}_ms_min"] = 1e3 * float(np.min(ts))
        if a.gpu:                                                       # one CUDA thread per token
            g = lambda x_: x_.cuda()
            gd, gdo, gF, gC, gL, gZ = g(data), g(doff), g(F_), g(C_), g(L_), g(torch.from_numpy(slot.reshape(-1)))
            assert torch.equal(gext.decode_cuda(gd, gdo, g(tok_off), Cn, gF, gC, gL, gZ, S).cpu(), flat)
            for n_anc in (5500, 35000, 112000):
                if n_anc > A:
                    continue
                Tn = (n_anc + 63) // 64; to_n = tok_off[:Tn + 1].clone(); to_n[-1] = min(Tn * 64, n_anc) * Cn; gto = g(to_n)
                ts = []
                for _ in range(20):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); gext.decode_cuda(gd, gdo[:Tn + 1], gto, Cn, gF, gC, gL, gZ, S); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
                timing[f"{n_anc}_gpu_ms_median"] = float(np.median(ts[5:])); timing[f"{n_anc}_gpu_ms_min"] = float(np.min(ts[5:]))
        side_bits = Cn * (2 * 32 + 2 * 16)                              # location and scale (float32), symbol range (int16)
        res["layers"][lname] = {"channels": Cn, "tokens": T, "real_bits_per_anchor": real_bits / A, "side_bits": side_bits,
                                "model_bits_per_anchor": mbits / A, "empirical_bits_per_anchor": ebits / A,
                                "overhead_vs_empirical": real_bits / ebits - 1, "overhead_vs_model": real_bits / mbits - 1,
                                "bit_exact": exact, "encode_s_all": t_enc, "decode_ms": timing}
        print(lname, json.dumps(res["layers"][lname]), flush=True)
    with open(os.path.join(a.out, a.scene, f"rangecode{a.tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tag", default="_v2a_r1")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--gpu", action="store_true", help="also decode with one CUDA thread per token and time it")
    main(ap.parse_args())
