"""Packetisation of the visible anchors of a slot into tokens and OAM streams, and loss application.

A token is a group of G anchors that are consecutive in Morton (Z-order) of their voxel coordinates, so a token
covers a compact region. Mapping policies (token -> stream index m = 0..M-1, where streams are sorted from the
most to the least reliable OAM mode of the configuration):
  random      : uniformly random assignment (balanced)
  contiguous  : the Morton-ordered token list is split into M consecutive blocks (regions lost as a whole)
  interleaved : token t -> stream t mod M (spatial neighbours on different streams)
  importance  : tokens ranked by importance (sum over their anchors); rank r -> stream floor(r M / T), i.e. the
                most important 1/M of the tokens on the most reliable mode; inside each importance tier the
                order is Morton, so neighbouring tokens of similar importance still alternate between tiers.
"""
import numpy as np
import torch


def morton_codes(keys):
    """keys: (A, 3) non-negative int64 voxel coordinates (< 2^21) -> (A,) int64 Morton codes."""
    def spread(v):
        v = v & 0x1FFFFF
        v = (v | v << 32) & 0x1F00000000FFFF
        v = (v | v << 16) & 0x1F0000FF0000FF
        v = (v | v << 8) & 0x100F00F00F00F00F
        v = (v | v << 4) & 0x10C30C30C30C30C3
        v = (v | v << 2) & 0x1249249249249249
        return v
    return spread(keys[:, 0]) | (spread(keys[:, 1]) << 1) | (spread(keys[:, 2]) << 2)


class Packetizer:
    def __init__(self, xyz, voxel, importance, G=64):
        keys = torch.round(xyz / voxel - 0.5).long()
        keys = keys - keys.min(0).values
        self.morton = morton_codes(keys)
        self.importance = importance.to(xyz.device)
        self.G = G

    def token_ids(self, idx):
        """Morton-sorted visible anchors and their token index (consecutive groups of G)."""
        order = torch.argsort(self.morton[idx])
        s = idx[order]
        tid = torch.arange(s.numel(), device=idx.device) // self.G
        return s, tid

    def assign(self, s, tid, M, policy, rng):
        """Stream index of every token (T = tid.max() + 1 tokens)."""
        T = int(tid[-1]) + 1 if tid.numel() else 0
        dev = s.device
        if policy == "random":
            return torch.as_tensor(rng.permutation(np.arange(T) % M), device=dev)
        if policy == "contiguous":
            return (torch.arange(T, device=dev) * M) // T
        if policy == "interleaved":
            return torch.arange(T, device=dev) % M
        if policy == "importance":
            imp = torch.zeros(T, device=dev).index_add_(0, tid, self.importance[s])
            rank = torch.empty(T, dtype=torch.long, device=dev)
            rank[torch.argsort(imp, descending=True)] = torch.arange(T, device=dev)
            return (rank * M) // T
        raise ValueError(policy)

    def lost_mask(self, idx, M, policy, failed_streams, rng):
        """Boolean mask over idx: True for anchors whose token travels on a failed stream.
        failed_streams: iterable of stream indices (0 = most reliable mode)."""
        if idx.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=idx.device)
        s, tid = self.token_ids(idx)
        st = self.assign(s, tid, M, policy, rng)
        failed = torch.zeros(M, dtype=torch.bool, device=idx.device)
        fs = list(failed_streams)
        if fs:
            failed[torch.as_tensor(fs, device=idx.device)] = True
        lost_sorted = failed[st[tid]]
        out = torch.empty_like(lost_sorted)
        out[torch.argsort(self.morton[idx])] = lost_sorted
        return out


def mode_reliability_order(per_mode_outage, tx_modes):
    """Transmitted modes sorted from the most to the least reliable (ascending marginal outage)."""
    order = np.argsort(np.asarray(per_mode_outage), kind="stable")
    return [tx_modes[i] for i in order]


def patterns_to_streams(pattern, tx_modes, rel_order):
    """Failure bit pattern over tx_modes (bit i = tx_modes[i] failed) -> failed stream indices (reliability order)."""
    failed_modes = [tx_modes[i] for i in range(len(tx_modes)) if (int(pattern) >> i) & 1]
    return [rel_order.index(m) for m in failed_modes]
