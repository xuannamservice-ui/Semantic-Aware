"""Exact k-nearest-neighbour search on the GPU for the edge server (cold-start view tables and whole-scene tables).

Semantics (identical to a float64 brute force over the FP32 positions, knn_brute64 below):
  d2(q, r) = (dx*dx + dy*dy) + dz*dz in float64, dx = float64(q.x) - float64(r.x), every operation rounded to nearest
  (no fused multiply-add); the k neighbours of a query are the k references with the smallest (d2, reference index) in
  lexicographic order, i.e. equal distances go to the lower reference index; a query never returns itself (self_idx);
  rows are sorted by (d2, index); with fewer than k other references the row is padded with index -1, distance inf.
  The returned distances are float32(sqrt(d2)).
Algorithm (custom CUDA, no host synchronization; all list lengths stay on the device):
  grid    cells of size h, Morton keys of the integer cell coordinates (x: bit 0, y: bit 1, z: bit 2), one radix sort
          of the reference keys (torch.sort), references copied into key order as float4, a directory of 2^18 entries
          over the sorted keys (O(1) cell lookup); a cell of size 2^L h is a contiguous key range, so the sorted array
          is an implicit octree;
  search  one warp per query (k_search_warp). Level L = 0, 1, ...: the 3 x 3 x 3 cells of size 2^L h around the
          query (cells already scanned at level L - 1 skipped), looked up in parallel by lanes 0..26; cells farther
          than the current 32nd candidate are pruned, cells with more than 128 points are split into their children
          (looked up in parallel, depth first); the candidate ranges are concatenated and scanned 32 at a time with an
          FP32 distance; the 32 smallest keys (FP32 d2 bits << 32 | position) are held one per lane (insertion = one
          ballot and one shuffle). The level loop stops when the k-th FP32 distance (x (1 + 1e-5)) is below the
          distance from the query to the boundary of the scanned block minus a rounding margin (4e-6 of the coordinate
          magnitude): every reference outside the block is then farther. The 32 survivors are ranked by the exact
          (float64 d2, index) key; the result is certified when the 32nd FP32 distance x (1 - 4e-6) exceeds the k-th
          float64 d2 (no excluded point can tie or precede it);
  fallback uncertified queries (near-ties beyond 32 candidates, stack overflow) are searched again by one thread per
          query with a float64 list (k_search_exact, same traversal); queries that still fail (outside the grid,
          fewer than k references) by brute force (k_brute, one block per query). On the 11 scenes neither fallback
          was needed (see knn_gpu_eval.py).
Cell size: h given by the caller (recommended: the scene's median k-NN distance, ck["radius"], a scalar the edge
receives with the decoding networks), or None: h_factor x the median (k/m)-th neighbour distance of 256 strided points
among every m-th reference (m = k for large sets; brute force on the GPU, about 0.15 ms).
API:
  knn_view(query_pos, ref_pos, k=16, self_idx=None, h=None) -> (idx (Q, k) int64 into ref_pos, dist (Q, k) float32)
  knn_view_global(xyz, view_idx, lost_idx, k=16, h=None) -> (Q, k) int64 global anchor ids: the neighbours of the lost
      anchors among the anchors of the view (self excluded), i.e. the rows lost_idx of the view table of eval_cold3.py
      --view-nbr (tv_[idx] = idx[knn(xyz[idx], k)]), up to the tie convention
  knn_table(pos, k=16, h=None) -> (idx (N, k) int64, dist (N, k) float32): whole-scene table, self excluded
      (drop-in for conceal_net.knn(xyz, k), up to the tie convention)
  knn_brute64(query_pos, ref_pos, k, self_idx=None) -> (idx, d2 float64): reference (float64 brute force, torch only)
k <= 16 (the kernels are instantiated for k = 4, 8, 16; smaller k use the next instance and are sliced).
"""
import os

import torch
from torch.utils.cpp_extension import load_inline

CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cmath>

#define AXMAX ((1 << 21) - 1)
#define BT_BRUTE 128

__device__ __forceinline__ unsigned long long spread3(unsigned long long v) {
    v &= 0x1FFFFFULL;
    v = (v | v << 32) & 0x1F00000000FFFFULL;
    v = (v | v << 16) & 0x1F0000FF0000FFULL;
    v = (v | v << 8) & 0x100F00F00F00F00FULL;
    v = (v | v << 4) & 0x10C30C30C30C30C3ULL;
    v = (v | v << 2) & 0x1249249249249249ULL;
    return v;
}
// Morton key of a cell (x: bit 0, y: bit 1, z: bit 2); the cells of size 2^L h are contiguous key ranges
__device__ __forceinline__ long long mkey(int x, int y, int z) {
    return (long long)(spread3((unsigned)x) | (spread3((unsigned)y) << 1) | (spread3((unsigned)z) << 2));
}
__device__ __forceinline__ int cellc(float p, float lo, float h) {
    double f = floor(__ddiv_rn(__dsub_rn((double)p, (double)lo), (double)h));
    f = fmin(fmax(f, -1.0e9), 1.0e9);
    return (int)f;
}
__device__ __forceinline__ double d2_64(float ax, float ay, float az, float bx, float by, float bz) {
    double dx = __dsub_rn((double)ax, (double)bx), dy = __dsub_rn((double)ay, (double)by), dz = __dsub_rn((double)az, (double)bz);
    return __dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz));
}
__device__ __forceinline__ float d2_32(float ax, float ay, float az, float bx, float by, float bz) {
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}
// distances are kept as the bit patterns of the (non-negative) float64 d2: same order, integer comparisons only
typedef unsigned long long u64;
#define DINF 0x7FF0000000000000ULL
__device__ __forceinline__ bool lessp(u64 d, int i, u64 D, int I) { return d < D || (d == D && i < I); }

template <int K>
__device__ __forceinline__ void ins(u64 (&D)[K], int (&I)[K], u64 d, int i) {
    if (!lessp(d, i, D[K - 1], I[K - 1])) return;
    D[K - 1] = d; I[K - 1] = i;
#pragma unroll
    for (int j = K - 1; j > 0; --j) {
        bool sw = lessp(D[j], I[j], D[j - 1], I[j - 1]);
        u64 td = sw ? D[j - 1] : D[j]; u64 td2 = sw ? D[j] : D[j - 1];
        int ti = sw ? I[j - 1] : I[j]; int ti2 = sw ? I[j] : I[j - 1];
        D[j] = td; D[j - 1] = td2; I[j] = ti; I[j - 1] = ti2;
    }
}
__device__ __forceinline__ u64 dbits(double d) { return (u64)__double_as_longlong(d); }
__device__ __forceinline__ double bitsd(u64 b) { return __longlong_as_double((long long)b); }
__device__ __forceinline__ float thr32(u64 D) {             // FP32 pre-filter bound (inf while the list is not full)
    return D >= DINF ? INFINITY : (float)bitsd(D) * 1.00001f + 1e-30f;
}

// keys of the reference points; g = [lo.x, lo.y, lo.z, h]
__global__ void k_keys(const float* __restrict__ p, int n, const float* __restrict__ g, long long* __restrict__ keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float h = g[3];
    int x = min(max(cellc(p[3 * i], g[0], h), 0), AXMAX), y = min(max(cellc(p[3 * i + 1], g[1], h), 0), AXMAX),
        z = min(max(cellc(p[3 * i + 2], g[2], h), 0), AXMAX);
    keys[i] = mkey(x, y, z);
}
__global__ void k_gather(const float* __restrict__ p, const long long* __restrict__ perm, int n, float4* __restrict__ sp,
                         int* __restrict__ sidx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int i = (int)perm[j];
    sp[j] = make_float4(p[3 * i], p[3 * i + 1], p[3 * i + 2], 0.f);
    sidx[j] = i;
}
__device__ __forceinline__ int lower_bound(const long long* __restrict__ a, int n, long long v) {
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) >> 1; if (__ldg(a + m) < v) lo = m + 1; else hi = m; }
    return lo;
}

#define DB 18
// directory over the sorted keys: dir[p] = lower_bound(keys, p << dsh), p < 2^DB, dir[2^DB] = N
__global__ void k_dir(const long long* __restrict__ skeys, int N, int* __restrict__ dir, int* __restrict__ dinfo) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p > (1 << DB)) return;
    long long mx = N ? skeys[N - 1] : 0;
    int bits = mx ? 64 - __clzll(mx) : 0;
    int sh = max(0, bits - DB);
    if (p == 0) dinfo[0] = sh;
    dir[p] = p == (1 << DB) ? N : lower_bound(skeys, N, (long long)p << sh);
}
__device__ __forceinline__ int dlb(const long long* __restrict__ a, int n, const int* __restrict__ dir, int dsh, long long v) {
    long long p = v >> dsh;
    if (p >= (1 << DB)) return n;
    int lo = __ldg(dir + p), hi = __ldg(dir + p + 1);
    if (((long long)p << dsh) == v) return lo;
    while (lo < hi) { int m = (lo + hi) >> 1; if (__ldg(a + m) < v) lo = m + 1; else hi = m; }
    return lo;
}

// first position after the keys <= k1 (no signed overflow when k1 is the largest key value)
__device__ __forceinline__ int dlb_end(const long long* __restrict__ a, int n, const int* __restrict__ dir, int dsh, long long k1) {
    return k1 >= LLONG_MAX ? n : dlb(a, n, dir, dsh, k1 + 1);
}

#define LEAF 32
#define STK 168
struct QCtx {
    float qx, qy, qz;                          // FP32 query (distances)
    double dx, dy, dz, lx, ly, lz, h, delta;   // float64 geometry (cells, boxes, block test)
};
// true when every point of the box of fine cells [X, X + 2^l) h + lo is farther than sqrt(thr2) (FP32 d2 threshold)
__device__ __forceinline__ bool box_far(const QCtx& c, int l, int X, int Y, int Z, float thr2) {
    if (isinf(thr2)) return false;
    double w = (double)(1 << l) * c.h;
    double x0 = c.lx + (double)X * c.h, y0 = c.ly + (double)Y * c.h, z0 = c.lz + (double)Z * c.h;
    double tx = fmax(fmax(x0 - c.dx, c.dx - (x0 + w)), 0.0);
    double ty = fmax(fmax(y0 - c.dy, c.dy - (y0 + w)), 0.0);
    double tz = fmax(fmax(z0 - c.dz, c.dz - (z0 + w)), 0.0);
    double r = (double)sqrtf(thr2 * 1.00002f) + c.delta;
    return tx * tx + ty * ty + tz * tz > r * r;
}
// true when the k-th FP32 d2 (kth2) is below the squared distance from the query to the boundary of the block of
// fine cells [(c - 1) 2^L, (c + 2) 2^L) (minus the margin): every reference outside the block is then farther
__device__ __forceinline__ bool block_ok(const QCtx& c, int L, int cx, int cy, int cz, float kth2) {
    if (isinf(kth2)) return false;
    double m = fmin(fmin(c.dx - (c.lx + (double)((cx - 1) << L) * c.h), (c.lx + (double)((cx + 2) << L) * c.h) - c.dx),
                    fmin(c.dy - (c.ly + (double)((cy - 1) << L) * c.h), (c.ly + (double)((cy + 2) << L) * c.h) - c.dy));
    m = fmin(m, fmin(c.dz - (c.lz + (double)((cz - 1) << L) * c.h), (c.lz + (double)((cz + 2) << L) * c.h) - c.dz));
    m -= c.delta;
    return m > 0.0 && (double)kth2 * 1.00002 < m * m;
}
__device__ __forceinline__ void set_delta(QCtx& c, int fx, int fy, int fz, int L) {
    double sc = fmax(fmax(fabs(c.dx), fmax(fabs(c.dy), fabs(c.dz))), fmax(fabs(c.lx), fmax(fabs(c.ly), fabs(c.lz))));
    c.delta = 1e-9 * (sc + (double)(max(fx, max(fy, fz)) + (2 << L)) * c.h) + 1e-300;
}

// Exact list: the K smallest (float64 d2, original index)
template <int K>
struct ExactList {
    u64 D[K]; int I[K];
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int j = 0; j < K; ++j) { D[j] = DINF; I[j] = INT_MAX; }
    }
    __device__ __forceinline__ float filt() const { return thr32(D[K - 1]); }
    __device__ __forceinline__ float prune2() const { return filt(); }
    __device__ __forceinline__ float kth2() const { return filt(); }
    __device__ __forceinline__ void add(float d32, int j, const float4& r, const int* __restrict__ sidx, int me, const QCtx& c) {
        int id = __ldg(sidx + j);
        if (id == me) return;
        ins<K>(D, I, dbits(d2_64(c.qx, c.qy, c.qz, r.x, r.y, r.z)), id);
    }
};

#define UNR 4
// scan the sorted positions [j, e): loads are issued UNR at a time (independent of the loop control)
template <class LST>
__device__ __forceinline__ bool scan_keys(const float4* __restrict__ sp, const int* __restrict__ sidx, const long long* __restrict__ skeys,
        int j, int e, unsigned mask, int sh, const QCtx& c, int me, int mej, LST& lst, float& t32, int& scanned, int cap) {
    for (; j < e; j += UNR) {
        float4 r[UNR]; unsigned sub[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            int jj = min(j + u, e - 1);
            r[u] = __ldg(sp + jj);
            sub[u] = mask ? (unsigned)((__ldg(skeys + jj) >> sh) & 7) : 0u;
        }
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            int jj = j + u;
            if (jj >= e) break;
            if ((mask >> sub[u]) & 1u) continue;
            if (++scanned > cap) return false;
            float d = d2_32(c.qx, c.qy, c.qz, r[u].x, r[u].y, r[u].z);
            if (d > t32 || jj == mej) continue;
            lst.add(d, jj, r[u], sidx, me, c);
            t32 = lst.filt();
        }
    }
    return true;
}

__constant__ signed char ORD27[27] = {13, 4, 10, 12, 14, 16, 22, 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25, 0, 2, 6, 8, 18, 20, 24, 26};

// traversal: level L = the 3 x 3 x 3 cells of size 2^L h around the query (cells of level L - 1 already scanned are
// skipped); a cell with more than LEAF points is split into its children, depth first, nearest child first; cells
// farther than the current threshold are pruned. Returns true when the block test proves the list complete.
template <class LST>
__device__ __forceinline__ bool traverse(QCtx& c, int fx, int fy, int fz, int me, int mej, LST& lst,
        const float4* __restrict__ sp, const int* __restrict__ sidx, const long long* __restrict__ skeys, int N, int cap, int lmax,
        const int* __restrict__ dir, int dsh, bool& abort) {
    float t32 = INFINITY;
    int scanned = 0;
    u64 stk[STK]; unsigned char stl[STK];
    for (int L = 0; L <= lmax; ++L) {
        int cx = fx >> L, cy = fy >> L, cz = fz >> L, cm = AXMAX >> L;
        int px = L ? fx >> (L - 1) : 0, py = L ? fy >> (L - 1) : 0, pz = L ? fz >> (L - 1) : 0;
        set_delta(c, fx, fy, fz, L);
        for (int ci = 0; ci < 27; ++ci) {
            int cc = ORD27[ci];
            int X = cx + cc % 3 - 1, Y = cy + (cc / 3) % 3 - 1, Z = cz + cc / 9 - 1;
            if (X < 0 || X > cm || Y < 0 || Y > cm || Z < 0 || Z > cm) continue;
            unsigned mask = 0;
            if (L) {
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    int sx = 2 * X + (s & 1), sy = 2 * Y + ((s >> 1) & 1), sz = 2 * Z + (s >> 2);
                    if (abs(sx - px) <= 1 && abs(sy - py) <= 1 && abs(sz - pz) <= 1) mask |= 1u << s;
                }
                if (mask == 0xFFu) continue;
            }
            int FX = X << L, FY = Y << L, FZ = Z << L;
            if (box_far(c, L, FX, FY, FZ, lst.prune2())) continue;
            long long k0 = mkey(FX, FY, FZ), k1 = k0 + ((1LL << (3 * L)) - 1);
            int j = dlb(skeys, N, dir, dsh, k0);
            if (L == 0 || j + LEAF >= N || __ldg(skeys + j + LEAF) > k1) {
                int e = dlb_end(skeys, N, dir, dsh, k1);
                if (!scan_keys(sp, sidx, skeys, j, e, mask, L ? 3 * (L - 1) : 0, c, me, mej, lst, t32, scanned, cap)) { abort = true; return false; }
                continue;
            }
            int n = 0;
            {
                int hl = 1 << (L - 1);
                int qo = (fx >= FX + hl ? 1 : 0) | (fy >= FY + hl ? 2 : 0) | (fz >= FZ + hl ? 4 : 0);
                for (int r = 3; r >= 0; --r)
                    for (int s = 0; s < 8; ++s)
                        if (__popc(s ^ qo) == r && !((mask >> s) & 1u)) {
                            stk[n] = (u64)(FX + (s & 1) * hl) | ((u64)(FY + ((s >> 1) & 1) * hl) << 21) | ((u64)(FZ + (s >> 2) * hl) << 42);
                            stl[n++] = (unsigned char)(L - 1);
                        }
            }
            while (n > 0) {
                --n;
                int l = stl[n]; u64 e = stk[n];
                int X2 = (int)(e & AXMAX), Y2 = (int)((e >> 21) & AXMAX), Z2 = (int)((e >> 42) & AXMAX);
                if (box_far(c, l, X2, Y2, Z2, lst.prune2())) continue;
                long long a0 = mkey(X2, Y2, Z2), a1 = a0 + ((1LL << (3 * l)) - 1);
                int j2 = dlb(skeys, N, dir, dsh, a0);
                if (j2 >= N || __ldg(skeys + j2) > a1) continue;
                if (l == 0 || j2 + LEAF >= N || __ldg(skeys + j2 + LEAF) > a1 || n + 8 > STK) {
                    int e2 = dlb_end(skeys, N, dir, dsh, a1);
                    if (!scan_keys(sp, sidx, skeys, j2, e2, 0u, 0, c, me, mej, lst, t32, scanned, cap)) { abort = true; return false; }
                    continue;
                }
                int hl = 1 << (l - 1);
                int qo = (fx >= X2 + hl ? 1 : 0) | (fy >= Y2 + hl ? 2 : 0) | (fz >= Z2 + hl ? 4 : 0);
                for (int r = 3; r >= 0; --r)
                    for (int s = 0; s < 8; ++s)
                        if (__popc(s ^ qo) == r) {
                            stk[n] = (u64)(X2 + (s & 1) * hl) | ((u64)(Y2 + ((s >> 1) & 1) * hl) << 21) | ((u64)(Z2 + (s >> 2) * hl) << 42);
                            stl[n++] = (unsigned char)(l - 1);
                        }
            }
        }
        // distance from the query to the boundary of the scanned block, fine cells [(c - 1) 2^L, (c + 2) 2^L)
        if (block_ok(c, L, cx, cy, cz, lst.kth2())) return true;
    }
    return false;
}

__device__ __forceinline__ bool query_setup(const float* __restrict__ q, int qs, int t, const float* __restrict__ g, QCtx& c,
                                            int& fx, int& fy, int& fz) {
    c.qx = q[(long long)qs * t]; c.qy = q[(long long)qs * t + 1]; c.qz = q[(long long)qs * t + 2];
    c.dx = c.qx; c.dy = c.qy; c.dz = c.qz; c.lx = g[0]; c.ly = g[1]; c.lz = g[2]; c.h = g[3];
    fx = cellc(c.qx, g[0], g[3]); fy = cellc(c.qy, g[1], g[3]); fz = cellc(c.qz, g[2], g[3]);
    return fx >= 0 && fx <= AXMAX && fy >= 0 && fy <= AXMAX && fz >= 0 && fz <= AXMAX;
}

// exact pass over the listed queries (float64 list during the traversal); failures go to the brute force (blist)
template <int K>
__global__ void k_search_exact(const float* __restrict__ q, int qs, const int* __restrict__ list, const int* __restrict__ cnt,
        const int* __restrict__ self, const int* __restrict__ selfj, const int* __restrict__ orow, const float4* __restrict__ sp,
        const int* __restrict__ sidx, const long long* __restrict__ skeys, int N, const float* __restrict__ g,
        const int* __restrict__ dir, const int* __restrict__ dinfo, int cap, int lmax, long long* __restrict__ oidx, float* __restrict__ od, int* __restrict__ blist, int* __restrict__ bcnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *cnt) return;
    int t = list[i];
    QCtx c; int fx, fy, fz;
    int me = self ? self[t] : -1, mej = selfj ? selfj[t] : -1;
    bool ok = false, abort = !query_setup(q, qs, t, g, c, fx, fy, fz);
    ExactList<K> lst; lst.init();
    if (!abort) ok = traverse(c, fx, fy, fz, me, mej, lst, sp, sidx, skeys, N, cap, lmax, dir, dinfo[0], abort);
    long long row = orow ? (long long)orow[t] : (long long)t;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        oidx[row * K + j] = (lst.I[j] == INT_MAX) ? -1LL : (long long)lst.I[j];
        od[row * K + j] = (float)sqrt(bitsd(lst.D[j]));
    }
    if (!ok) blist[atomicAdd(bcnt, 1)] = t;
}

// ---------------------------------------------------------------- warp-per-query search (main path)
// One warp per query. The list of the 32 smallest candidates is distributed over the lanes (lane i holds the i-th
// smallest key = FP32 d2 bits << 32 | sorted position), so an insertion is one ballot and one shuffle. Per level, lanes
// 0..26 look up the 27 cells in parallel; the ranges of the cells to scan are concatenated (prefix sum over the lanes)
// and scanned 32 candidates at a time; a cell (or node) with more than LEAFW points is split into its children, which
// are again looked up in parallel (lanes 0..7); nodes farther than the 32-th candidate are pruned. When the block test
// holds, the 32 survivors are ranked by (float64 d2, original index) and the result is certified if the 32-th FP32
// distance exceeds the k-th float64 distance by the rounding margin.
#define LEAFW 128
#define WSTK 192
#define WPB 8
#define FULL 0xFFFFFFFFu

__device__ __forceinline__ void warp_insert(u64& E_, u64& thr, u64 key, bool pass, int lane) {
    unsigned bal = __ballot_sync(FULL, pass && key < thr);
    while (bal) {
        int b = __ffs(bal) - 1; bal &= bal - 1;
        u64 kb = __shfl_sync(FULL, key, b);
        if (kb >= thr) continue;
        int pos = __popc(__ballot_sync(FULL, E_ < kb));
        u64 up = __shfl_up_sync(FULL, E_, 1);
        if (lane > pos) E_ = up; else if (lane == pos) E_ = kb;
        thr = __shfl_sync(FULL, E_, 31);
    }
}
__device__ __forceinline__ float keyf(u64 k) { return k == ~0ULL ? INFINITY : __uint_as_float((unsigned)(k >> 32)); }

// scan the concatenation of the lanes' ranges [j, j + len) (mask: sub-cells to skip, sh: their key shift)
__device__ __forceinline__ void warp_scan(int j, int len, unsigned mask, int sh, const float4* __restrict__ sp,
        const long long* __restrict__ skeys, const QCtx& c, int mej, u64& E_, u64& thr, int lane) {
    int incl = len;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(FULL, incl, o); if (lane >= o) incl += v; }
    int total = __shfl_sync(FULL, incl, 31);
    for (int base = 0; base < total; base += 32) {
        int x = base + lane;
        int lo = 0;
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) { int v = __shfl_sync(FULL, incl, lo + s - 1); if (v <= x) lo += s; }
        int cl = min(lo, 31);
        int cj = __shfl_sync(FULL, j, cl), cinc = __shfl_sync(FULL, incl, cl), clen = __shfl_sync(FULL, len, cl);
        unsigned cm = __shfl_sync(FULL, mask, cl);
        bool act = x < total;
        int jj = cj + (x - (cinc - clen));
        bool pass = false; u64 key = ~0ULL;
        if (act) {
            float4 r = __ldg(sp + jj);
            bool skip = cm && ((cm >> (unsigned)((__ldg(skeys + jj) >> sh) & 7)) & 1u);
            if (!skip && jj != mej) {
                float d = d2_32(c.qx, c.qy, c.qz, r.x, r.y, r.z);
                key = ((u64)__float_as_uint(d) << 32) | (unsigned)jj;
                pass = true;
            }
        }
        warp_insert(E_, thr, key, pass, lane);
    }
}

template <int K>
__global__ void __launch_bounds__(32 * WPB) k_search_warp(const float* __restrict__ q, int qs, int Q, const int* __restrict__ list,
        const int* __restrict__ cnt, const int* __restrict__ self, const int* __restrict__ selfj, const int* __restrict__ orow,
        const float4* __restrict__ sp, const int* __restrict__ sidx, const long long* __restrict__ skeys, int N,
        const float* __restrict__ g, const int* __restrict__ dir, const int* __restrict__ dinfo, int lmax,
        long long* __restrict__ oidx, float* __restrict__ od, int* __restrict__ flist, int* __restrict__ fcnt) {
    __shared__ u64 wst[WPB][WSTK];
    __shared__ unsigned char wsl[WPB][WSTK];
    int lane = threadIdx.x & 31, wb = threadIdx.x >> 5;
    int nq = cnt ? *cnt : Q;
    int dsh = dinfo[0];
    for (int w = blockIdx.x * WPB + wb; w < nq; w += gridDim.x * WPB) {
        int t = list ? list[w] : w;
        QCtx c; int fx, fy, fz;
        bool inside = query_setup(q, qs, t, g, c, fx, fy, fz);
        int me = self ? self[t] : -1, mej = selfj ? selfj[t] : -1;
        u64 E_ = ~0ULL, thr = ~0ULL;
        bool ok = false, ovf = false;
        for (int L = 0; L <= lmax && inside && !ok; ++L) {
            int cx = fx >> L, cy = fy >> L, cz = fz >> L, cm = AXMAX >> L;
            set_delta(c, fx, fy, fz, L);
            // lanes 0..26: the cells of this level
            int j = 0, len = 0; unsigned mask = 0; bool split = false; int FX = 0, FY = 0, FZ = 0;
            if (lane < 27) {
                int X = cx + lane % 3 - 1, Y = cy + (lane / 3) % 3 - 1, Z = cz + lane / 9 - 1;
                bool valid = X >= 0 && X <= cm && Y >= 0 && Y <= cm && Z >= 0 && Z <= cm;
                if (valid && L) {
                    int px = fx >> (L - 1), py = fy >> (L - 1), pz = fz >> (L - 1);
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        int sx = 2 * X + (s & 1), sy = 2 * Y + ((s >> 1) & 1), sz = 2 * Z + (s >> 2);
                        if (abs(sx - px) <= 1 && abs(sy - py) <= 1 && abs(sz - pz) <= 1) mask |= 1u << s;
                    }
                    valid = mask != 0xFFu;
                }
                FX = X << L; FY = Y << L; FZ = Z << L;
                if (valid) valid = !box_far(c, L, FX, FY, FZ, keyf(thr));
                if (valid) {
                    long long k0 = mkey(FX, FY, FZ), k1 = k0 + ((1LL << (3 * L)) - 1);
                    j = dlb(skeys, N, dir, dsh, k0);
                    int e = dlb_end(skeys, N, dir, dsh, k1);
                    len = e - j;
                    split = L > 0 && len > LEAFW;
                    if (split) len = 0;
                }
            }
            warp_scan(j, len, mask, L ? 3 * (L - 1) : 0, sp, skeys, c, mej, E_, thr, lane);
            // split cells: their children (outside the previous block) go on the warp stack, then depth first
            unsigned sb = __ballot_sync(FULL, split);
            int n = 0;
            while (sb) {
                int b = __ffs(sb) - 1; sb &= sb - 1;
                int bx = __shfl_sync(FULL, FX, b), by = __shfl_sync(FULL, FY, b), bz = __shfl_sync(FULL, FZ, b);
                unsigned bm = __shfl_sync(FULL, mask, b);
                unsigned keep = ~bm & 0xFFu;
                if (n + __popc(keep) > WSTK) { ovf = true; break; }
                if (lane < 8 && ((keep >> lane) & 1u)) {
                    int hl = 1 << (L - 1);
                    int pos = n + __popc(keep & ((1u << lane) - 1));
                    wst[wb][pos] = (u64)(bx + (lane & 1) * hl) | ((u64)(by + ((lane >> 1) & 1) * hl) << 21) | ((u64)(bz + (lane >> 2) * hl) << 42);
                    wsl[wb][pos] = (unsigned char)(L - 1);
                }
                n += __popc(keep);
                __syncwarp();
            }
            while (n > 0) {
                --n;
                u64 e0 = wst[wb][n]; int l = wsl[wb][n];
                __syncwarp();
                int X2 = (int)(e0 & AXMAX), Y2 = (int)((e0 >> 21) & AXMAX), Z2 = (int)((e0 >> 42) & AXMAX);
                int jc = 0, lc = 0; bool sc = false; int CX = 0, CY = 0, CZ = 0;
                if (lane < 8) {
                    if (l == 0) { if (lane == 0) { CX = X2; CY = Y2; CZ = Z2; } }
                    else { int hl = 1 << (l - 1); CX = X2 + (lane & 1) * hl; CY = Y2 + ((lane >> 1) & 1) * hl; CZ = Z2 + (lane >> 2) * hl; }
                    int lv = l ? l - 1 : 0;
                    bool v = (l > 0 || lane == 0);
                    if (v) v = !box_far(c, lv, CX, CY, CZ, keyf(thr));
                    if (v) {
                        long long a0 = mkey(CX, CY, CZ), a1 = a0 + ((1LL << (3 * lv)) - 1);
                        jc = dlb(skeys, N, dir, dsh, a0);
                        lc = dlb_end(skeys, N, dir, dsh, a1) - jc;
                        sc = lv > 0 && lc > LEAFW;
                        if (sc) lc = 0;
                    }
                }
                warp_scan(jc, lc, 0u, 0, sp, skeys, c, mej, E_, thr, lane);
                unsigned pb = __ballot_sync(FULL, sc);
                if (n + __popc(pb) > WSTK) { ovf = true; break; }
                if (sc) {
                    int pos = n + __popc(pb & ((1u << lane) - 1));
                    wst[wb][pos] = (u64)CX | ((u64)CY << 21) | ((u64)CZ << 42);
                    wsl[wb][pos] = (unsigned char)(l - 1);
                }
                n += __popc(pb);
                __syncwarp();
            }
            if (ovf) break;
            // block test: every reference outside the scanned block is farther than the k-th candidate
            ok = block_ok(c, L, cx, cy, cz, keyf(__shfl_sync(FULL, E_, K - 1)));
        }
        // exact ranking of the 32 survivors by (float64 d2, original index)
        u64 d64 = DINF; int id = INT_MAX;
        if (E_ != ~0ULL) {
            int jj = (int)(E_ & 0xFFFFFFFFULL);
            float4 r = __ldg(sp + jj);
            id = __ldg(sidx + jj);
            if (id != me) d64 = dbits(d2_64(c.qx, c.qy, c.qz, r.x, r.y, r.z)); else id = INT_MAX;
        }
        int rank = 0;
#pragma unroll 4
        for (int b = 0; b < 32; ++b) {
            u64 od_ = __shfl_sync(FULL, d64, b); int oi_ = __shfl_sync(FULL, id, b);
            rank += lessp(od_, oi_, d64, id) ? 1 : 0;
        }
        unsigned kb = __ballot_sync(FULL, rank == K - 1);
        u64 dk = __shfl_sync(FULL, d64, kb ? __ffs(kb) - 1 : 0);
        float last = keyf(__shfl_sync(FULL, E_, 31));
        ok = ok && !ovf && kb && ((double)last * (1.0 - 4e-6) > bitsd(dk));
        long long row = orow ? (long long)orow[t] : (long long)t;
        if (ok) {
            if (rank < K) {
                oidx[row * K + rank] = (id == INT_MAX) ? -1LL : (long long)id;
                od[row * K + rank] = (float)sqrt(bitsd(d64));
            }
        } else if (lane == 0) flist[atomicAdd(fcnt, 1)] = t;
        __syncwarp();
    }
}

// brute force for listed queries: one block per query (grid-stride over the list, whose length is read on the device)
template <int K>
__global__ void k_brute(const float* __restrict__ q, int qs, const int* __restrict__ self, const int* __restrict__ orow,
        const float* __restrict__ rp, int rs, const int* __restrict__ ridx, int N,
        const int* __restrict__ list, const int* __restrict__ cnt, int bound, long long* __restrict__ oidx, float* __restrict__ od) {
    __shared__ u64 sD[BT_BRUTE * K];
    __shared__ int sI[BT_BRUTE * K];
    int n = *cnt;
    for (int f = blockIdx.x; f < n; f += gridDim.x) {
        int t = list[f];
        float qx = q[(long long)qs * t], qy = q[(long long)qs * t + 1], qz = q[(long long)qs * t + 2];
        int me = self ? self[t] : -1;
        u64 D[K]; int I[K];
#pragma unroll
        for (int j = 0; j < K; ++j) { D[j] = DINF; I[j] = INT_MAX; }
        float t32 = INFINITY;
        if (bound) {               // the k-th distance of the incomplete search bounds the true one
            long long row = orow ? (long long)orow[t] : (long long)t;
            float b = od[row * K + K - 1];
            if (isfinite(b)) t32 = b * b * 1.00002f + 1e-30f;
        }
        __syncthreads();
        constexpr int U = 8;
        for (int base = threadIdx.x; base < N; base += BT_BRUTE * U) {
            float rx[U], ry[U], rz[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                int j = base + u * BT_BRUTE;
                if (j < N) { rx[u] = rp[(long long)rs * j]; ry[u] = rp[(long long)rs * j + 1]; rz[u] = rp[(long long)rs * j + 2]; }
                else { rx[u] = INFINITY; ry[u] = 0.f; rz[u] = 0.f; }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                int j = base + u * BT_BRUTE;
                if (j >= N || !(d2_32(qx, qy, qz, rx[u], ry[u], rz[u]) <= t32)) continue;
                int id = ridx ? ridx[j] : j;
                if (id == me) continue;
                ins<K>(D, I, dbits(d2_64(qx, qy, qz, rx[u], ry[u], rz[u])), id);
                t32 = thr32(D[K - 1]);
            }
        }
#pragma unroll
        for (int j = 0; j < K; ++j) { sD[threadIdx.x * K + j] = D[j]; sI[threadIdx.x * K + j] = I[j]; }
        __syncthreads();
        for (int s = BT_BRUTE / 2; s > 0; s >>= 1) {
            u64 mD[K]; int mI[K];
            if (threadIdx.x < s) {
                int a = threadIdx.x * K, b = (threadIdx.x + s) * K, ia = 0, ib = 0;
                for (int j = 0; j < K; ++j) {
                    bool ta = lessp(sD[a + ia], sI[a + ia], sD[b + ib], sI[b + ib]);
                    if (ta) { mD[j] = sD[a + ia]; mI[j] = sI[a + ia]; ++ia; } else { mD[j] = sD[b + ib]; mI[j] = sI[b + ib]; ++ib; }
                }
            }
            __syncthreads();
            if (threadIdx.x < s)
                for (int j = 0; j < K; ++j) { sD[threadIdx.x * K + j] = mD[j]; sI[threadIdx.x * K + j] = mI[j]; }
            __syncthreads();
        }
        if (threadIdx.x < K) {
            long long row = orow ? (long long)orow[t] : (long long)t;
            int j = threadIdx.x;
            oidx[row * K + j] = (sI[j] == INT_MAX) ? -1LL : (long long)sI[j];
            od[row * K + j] = (float)sqrt(bitsd(sD[j]));
        }
        __syncthreads();
    }
}

static const float* fptr(const torch::Tensor& t) { return t.data_ptr<float>(); }
static const int* iptr_opt(const torch::Tensor& t) { return t.numel() ? t.data_ptr<int>() : nullptr; }

torch::Tensor keys(torch::Tensor p, torch::Tensor g) {
    int n = p.size(0);
    auto k = torch::empty({n}, p.options().dtype(torch::kLong));
    if (n) k_keys<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(fptr(p), n, fptr(g), (long long*)k.data_ptr<int64_t>());
    return k;
}
std::vector<torch::Tensor> gather(torch::Tensor p, torch::Tensor perm) {
    int n = p.size(0);
    auto sp = torch::empty({n, 4}, p.options());
    auto si = torch::empty({n}, p.options().dtype(torch::kInt));
    if (n) k_gather<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(fptr(p), (long long*)perm.data_ptr<int64_t>(), n,
                                                                                   (float4*)sp.data_ptr<float>(), si.data_ptr<int>());
    return {sp, si};
}

#define DISPATCH_K(K, ...) \
    switch (K) { case 4: { constexpr int KK = 4; __VA_ARGS__; } break; case 8: { constexpr int KK = 8; __VA_ARGS__; } break; \
                 case 16: { constexpr int KK = 16; __VA_ARGS__; } break; \
                 default: TORCH_CHECK(false, "K must be 4, 8 or 16"); }

std::vector<torch::Tensor> directory(torch::Tensor skeys) {
    int N = skeys.size(0);
    auto dir = torch::empty({(1 << DB) + 1}, skeys.options().dtype(torch::kInt));
    auto dinfo = torch::empty({1}, skeys.options().dtype(torch::kInt));
    k_dir<<<((1 << DB) + 256) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const long long*)skeys.data_ptr<int64_t>(), N,
        dir.data_ptr<int>(), dinfo.data_ptr<int>());
    return {dir, dinfo};
}
void search(torch::Tensor q, int qs, int Q, torch::Tensor self, torch::Tensor selfj, torch::Tensor orow, torch::Tensor sp,
            torch::Tensor sidx, torch::Tensor skeys, torch::Tensor g, torch::Tensor dir, torch::Tensor dinfo, int K, int cap, int lmax,
            int lmax2, torch::Tensor oidx, torch::Tensor od, torch::Tensor flist, torch::Tensor fcnt, torch::Tensor blist, torch::Tensor bcnt) {
    if (Q == 0) return;
    int N = sp.size(0);
    auto st = at::cuda::getCurrentCUDAStream();
    const float4* spp = (const float4*)sp.data_ptr<float>();
    const long long* kp = (const long long*)skeys.data_ptr<int64_t>();
    long long* oi = (long long*)oidx.data_ptr<int64_t>();
    int nb = (Q + WPB - 1) / WPB;
    DISPATCH_K(K, (k_search_warp<KK><<<nb, 32 * WPB, 0, st>>>(fptr(q), qs, Q, nullptr, nullptr, iptr_opt(self), iptr_opt(selfj),
        iptr_opt(orow), spp, sidx.data_ptr<int>(), kp, N, fptr(g), dir.data_ptr<int>(), dinfo.data_ptr<int>(), lmax, oi,
        od.data_ptr<float>(), flist.data_ptr<int>(), fcnt.data_ptr<int>())));
    DISPATCH_K(K, (k_search_exact<KK><<<(Q + 127) / 128, 128, 0, st>>>(fptr(q), qs, flist.data_ptr<int>(), fcnt.data_ptr<int>(),
        iptr_opt(self), iptr_opt(selfj), iptr_opt(orow), spp, sidx.data_ptr<int>(), kp, N, fptr(g), dir.data_ptr<int>(),
        dinfo.data_ptr<int>(), cap, lmax2, oi, od.data_ptr<float>(),
        blist.data_ptr<int>(), bcnt.data_ptr<int>())));
}
void brute(torch::Tensor q, int qs, torch::Tensor self, torch::Tensor orow, torch::Tensor rp, int rs, torch::Tensor ridx, int N,
           torch::Tensor list, torch::Tensor cnt, int K, int nblocks, int bound, torch::Tensor oidx, torch::Tensor od) {
    if (nblocks <= 0) return;
    auto st = at::cuda::getCurrentCUDAStream();
    DISPATCH_K(K, (k_brute<KK><<<nblocks, BT_BRUTE, 0, st>>>(fptr(q), qs, iptr_opt(self), iptr_opt(orow), fptr(rp), rs,
        iptr_opt(ridx), N, list.data_ptr<int>(), cnt.data_ptr<int>(), bound, (long long*)oidx.data_ptr<int64_t>(), od.data_ptr<float>())));
}
"""

CPP = r"""
#include <torch/extension.h>
torch::Tensor keys(torch::Tensor p, torch::Tensor g);
std::vector<torch::Tensor> gather(torch::Tensor p, torch::Tensor perm);
std::vector<torch::Tensor> directory(torch::Tensor skeys);
void search(torch::Tensor q, int qs, int Q, torch::Tensor self, torch::Tensor selfj, torch::Tensor orow, torch::Tensor sp,
            torch::Tensor sidx, torch::Tensor skeys, torch::Tensor g, torch::Tensor dir, torch::Tensor dinfo, int K, int cap, int lmax,
            int lmax2, torch::Tensor oidx, torch::Tensor od, torch::Tensor flist, torch::Tensor fcnt, torch::Tensor blist, torch::Tensor bcnt);
void brute(torch::Tensor q, int qs, torch::Tensor self, torch::Tensor orow, torch::Tensor rp, int rs, torch::Tensor ridx, int N,
           torch::Tensor list, torch::Tensor cnt, int K, int nblocks, int bound, torch::Tensor oidx, torch::Tensor od);
"""

_EXT = None
_EMPTY = {}
_SMS = {}


def ext():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        _EXT = load_inline("knn_gpu_ext", cpp_sources=CPP, cuda_sources=CU, functions=["keys", "gather", "directory", "search", "brute"],
                           extra_cuda_cflags=["-O3", "--fmad=true"], verbose=False)
    return _EXT


def _kt(k):
    for K in (4, 8, 16):
        if k <= K:
            return K
    raise ValueError("k <= 16 supported")


def _empty_long(dev):
    if ("l", dev) not in _EMPTY:
        _EMPTY[("l", dev)] = torch.empty(0, dtype=torch.long, device=dev)
    return _EMPTY[("l", dev)]


def _empty_int(dev):
    if dev not in _EMPTY:
        _EMPTY[dev] = torch.empty(0, dtype=torch.int32, device=dev)
    return _EMPTY[dev]


def _nblocks(dev, Q):
    if dev not in _SMS:
        _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return max(1, min(Q, 4 * _SMS[dev]))


def _grid(ref, k, h, h_factor):
    """Sorted grid over the references; returns (g, skeys, sp, sidx, perm). g = [lo, h] stays on the device."""
    e = ext(); dev = ref.device; N = ref.shape[0]
    lo, hi = torch.aminmax(ref, dim=0)
    if h is None:
        # the (k/m)-th neighbour among every m-th reference has the expected distance of the k-th among all of them
        m = max(1, min(k, N // 4096)); ks = max(1, round(k / m)); Ns = (N + m - 1) // m; S = min(256, Ns)
        Ks = _kt(ks)
        sel = (torch.arange(S, device=dev, dtype=torch.int64) * Ns // S).to(torch.int32)
        cnt = torch.full((1,), S, dtype=torch.int32, device=dev)
        oi = torch.empty(S, Ks, dtype=torch.long, device=dev); od = torch.empty(S, Ks, dtype=torch.float32, device=dev)
        qs = ref[sel.long() * m].contiguous()
        e.brute(qs, 3, sel, _empty_int(dev), ref, 3 * m, _empty_int(dev), Ns, torch.arange(S, device=dev, dtype=torch.int32), cnt,
                Ks, S, 0, oi, od)
        hh = od[:, ks - 1].clamp_max(torch.finfo(torch.float32).max).median().view(1) * h_factor
    else:
        hh = torch.full((1,), float(h), device=dev)
    hh = torch.maximum(hh, (hi - lo).max().view(1) * (1.0 / 2.0e6) + 1e-30)
    g = torch.cat([lo, hh])
    keys = e.keys(ref, g)
    skeys, perm = torch.sort(keys)
    sp, sidx = e.gather(ref, perm)
    dr, dinfo = e.directory(skeys)
    return g, skeys, sp, sidx, perm, dr, dinfo


def _search(q, qs, Q, self32, selfj32, orow32, grid, k, cap, lmax, dev, nrows, lmax2=20):
    """Fast pass (all queries), exact pass (uncertified ones), brute force (the rest). Returns idx, dist and the device
    counts (exact pass, brute force)."""
    e = ext(); K = _kt(k)
    g, skeys, sp, sidx, _, dr, dinfo = grid
    oi = torch.empty(nrows, K, dtype=torch.long, device=dev)
    od = torch.empty(nrows, K, dtype=torch.float32, device=dev)
    lists = torch.empty(2, max(Q, 1), dtype=torch.int32, device=dev); cnt = torch.zeros(2, dtype=torch.int32, device=dev)
    e.search(q, qs, Q, self32, selfj32, orow32, sp, sidx, skeys, g, dr, dinfo, K, cap, lmax, lmax2, oi, od, lists[0], cnt[0:1], lists[1], cnt[1:2])
    e.brute(q, qs, self32, orow32, sp, 4, sidx, sp.shape[0], lists[1], cnt[1:2], K, _nblocks(dev, Q), 1, oi, od)
    return oi[:, :k], od[:, :k], cnt


def knn_view(query_pos, ref_pos, k=16, self_idx=None, h=None, h_factor=1.0, cap=1 << 16, lmax=20, return_fallback=False):
    """Exact k nearest references of every query (see the module docstring for the tie convention).
    query_pos (Q, 3), ref_pos (N, 3): CUDA tensors (cast to FP32); self_idx (Q,) index of each query in ref_pos to exclude
    (or None); h: grid cell size (None: estimated on the device). Returns idx (Q, k) int64 into ref_pos, dist (Q, k)
    float32 (and the device counts [exact pass, brute force] if return_fallback)."""
    dev = ref_pos.device
    q = query_pos.float().contiguous(); r = ref_pos.float().contiguous()
    Q, N = q.shape[0], r.shape[0]
    if Q == 0 or N == 0:
        oi = torch.full((Q, k), -1, dtype=torch.long, device=dev); od = torch.full((Q, k), float("inf"), device=dev)
        return (oi, od, torch.zeros(2, dtype=torch.int32, device=dev)) if return_fallback else (oi, od)
    grid = _grid(r, k, h, h_factor)
    if self_idx is not None:
        s32 = self_idx.to(torch.int32).contiguous()
        invp = torch.empty(N, dtype=torch.int32, device=dev)
        invp[grid[4]] = torch.arange(N, device=dev, dtype=torch.int32)
        sj32 = invp[self_idx.long()]
    else:
        s32 = sj32 = _empty_int(dev)
    oi, od, fc = _search(q, 3, Q, s32, sj32, _empty_int(dev), grid, k, cap, lmax, dev, Q)
    return (oi, od, fc) if return_fallback else (oi, od)


def knn_view_global(xyz, view_idx, lost_idx, k=16, h=None, h_factor=1.0, cap=1 << 16, lmax=20, return_fallback=False):
    """Neighbours of the lost anchors among the anchors of the view, as global anchor ids (Q, k) int64 (self excluded):
    the rows lost_idx of the view table of eval_cold3.py --view-nbr (tv_[idx] = idx[knn(xyz[idx], k)]).
    view_idx, lost_idx: global anchor ids (lost_idx a subset of view_idx)."""
    dev = xyz.device
    inv = torch.empty(xyz.shape[0], dtype=torch.int32, device=dev)
    inv[view_idx] = torch.arange(view_idx.numel(), device=dev, dtype=torch.int32)
    out = knn_view(xyz[lost_idx], xyz[view_idx], k, inv[lost_idx], h, h_factor, cap, lmax, return_fallback)
    oi = out[0]
    gl = torch.where(oi >= 0, view_idx[oi.clamp_min(0)], oi)
    return (gl,) + tuple(out[1:]) if return_fallback else gl


def knn_table(pos, k=16, h=None, h_factor=1.0, cap=1 << 16, lmax=20, return_fallback=False):
    """Whole-scene table: the k nearest other points of every point, (N, k) int64 and (N, k) float32 distances."""
    dev = pos.device
    r = pos.float().contiguous(); N = r.shape[0]
    if N == 0:
        e0 = torch.zeros(0, k, dtype=torch.long, device=dev)
        return (e0, e0.float(), torch.zeros(2, dtype=torch.int32, device=dev)) if return_fallback else (e0, e0.float())
    grid = _grid(r, k, h, h_factor)
    sp, sidx = grid[2], grid[3]
    # queries in key order (locality): query t is sorted point t (self at sorted position t), output row sidx[t]
    oi, od, fc = _search(sp, 4, N, sidx, torch.arange(N, device=dev, dtype=torch.int32), sidx, grid, k, cap, lmax, dev, N)
    return (oi, od, fc) if return_fallback else (oi, od)


def knn_brute64(query_pos, ref_pos, k, self_idx=None, chunk=256):
    """Reference: float64 brute force with the same distance bits and tie convention (lowest index among equal
    distances). Returns idx (Q, k) int64 and d2 (Q, k) float64."""
    q = query_pos.float().double(); r = ref_pos.float().double()
    Q, N = q.shape[0], r.shape[0]
    ar = torch.arange(N, device=r.device)
    oi, od = [], []
    for s in range(0, Q, chunk):
        qq = q[s:s + chunk]
        dx = qq[:, None, 0] - r[None, :, 0]; dy = qq[:, None, 1] - r[None, :, 1]; dz = qq[:, None, 2] - r[None, :, 2]
        d2 = (dx * dx + dy * dy) + dz * dz
        if self_idx is not None:
            d2[torch.arange(qq.shape[0], device=r.device), self_idx[s:s + chunk].long()] = float("inf")
        m = min(N, k + 32)
        while True:
            v, i = d2.topk(m, dim=1, largest=False, sorted=True)
            if m == N or bool((v[:, -1] > v[:, k - 1]).all()):
                break
            m = min(N, 2 * m)
        # lexicographic (d2, index): stable sort by index, then stable sort by distance
        o1 = torch.sort(i, dim=1, stable=True)[1]; v, i = v.gather(1, o1), i.gather(1, o1)
        o2 = torch.sort(v, dim=1, stable=True)[1]; v, i = v.gather(1, o2), i.gather(1, o2)
        oi.append(i[:, :k]); od.append(v[:, :k])
        del dx, dy, dz, d2
    return torch.cat(oi), torch.cat(od)
