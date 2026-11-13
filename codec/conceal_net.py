"""Generative concealment network: predicts the attributes of lost anchors from received neighbours.

For each lost anchor i the input is a set of its Kn nearest anchors (positions are always known from the
protected skeleton); each neighbour token carries its relative position (Fourier features), its attributes if
received (zeros otherwise) and a received flag. A query token holds only i's own (zero) relative position.
A pre-norm transformer encoder processes the Kn + 1 tokens with attention restricted to the query and the
received neighbours; the query output is mapped to the attribute vector of i.

BitLinear (ternary weights {-1, 0, +1} with a per-tensor absmean scale, 8-bit absmax activations, straight-through
estimator; Ma et al., "The Era of 1-bit LLMs", 2024) replaces every nn.Linear when ternary=True.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        s = w.abs().mean().clamp_min(1e-5)
        wq = (torch.clamp(torch.round(w / s), -1, 1) * s - w).detach() + w
        xs = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        xq = (torch.clamp(torch.round(x * xs), -128, 127) / xs - x).detach() + x
        return F.linear(xq, wq, self.bias)


def lin(i, o, ternary):
    return BitLinear(i, o) if ternary else nn.Linear(i, o)


class Block(nn.Module):
    def __init__(self, d, heads, ternary):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = lin(d, 3 * d, ternary), lin(d, d, ternary)
        self.ff1, self.ff2 = lin(d, 4 * d, ternary), lin(4 * d, d, ternary)
        self.h = heads

    def forward(self, x, key_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff2(F.gelu(self.ff1(self.n2(x))))


class ConcealNet(nn.Module):
    def __init__(self, attr_dim, d=128, layers=4, heads=4, nfreq=6, ternary=False):
        super().__init__()
        self.nfreq = nfreq
        pe = 3 * 2 * nfreq + 3
        self.inp = lin(pe + attr_dim + 1, d, ternary)
        self.query = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, ternary) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = lin(d, attr_dim, ternary)

    def posenc(self, r):
        f = 2.0 ** torch.arange(self.nfreq, device=r.device) * math.pi
        x = r[..., None] * f
        return torch.cat([r, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], -1)

    def forward(self, rel, attrs, recv):
        """rel: (B, Kn, 3) neighbour positions relative to the lost anchor, in voxel units / Kn-radius;
        attrs: (B, Kn, attr_dim) normalised attributes (zeros where not received); recv: (B, Kn) bool."""
        tok = self.inp(torch.cat([self.posenc(rel), attrs * recv[..., None], recv[..., None].float()], -1))
        q = self.query.expand(tok.shape[0], 1, -1)
        x = torch.cat([q, tok], 1)
        key_mask = torch.cat([torch.ones_like(recv[:, :1]), recv], 1)
        for b in self.blocks:
            x = b(x, key_mask)
        return self.out(self.norm(x[:, 0]))


def knn(xyz, k, chunk=2048):
    """k nearest other anchors of every anchor (indices, (A, k))."""
    out = []
    for c in torch.split(torch.arange(xyz.shape[0], device=xyz.device), chunk):
        d = torch.cdist(xyz[c], xyz)
        d[torch.arange(c.numel()), c] = float("inf")
        out.append(d.topk(k, largest=False).indices)
    return torch.cat(out)


class LayeredConcealNet(nn.Module):
    """Concealment for the layered representation: the base layer of every anchor is always available (it travels
    on the replicated anchor pair); the enhancement layer of a lost anchor is predicted from its own base layer
    and from its Kn nearest anchors (base layer always, enhancement layer if received).
    Token input: [posenc(relative position), base attrs, enhancement attrs * received, received flag]."""

    def __init__(self, base_dim, enh_dim, d=128, layers=4, heads=4, nfreq=6, ternary=False):
        super().__init__()
        self.nfreq, self.base_dim, self.enh_dim = nfreq, base_dim, enh_dim
        pe = 3 * 2 * nfreq + 3
        self.inp = lin(pe + base_dim + enh_dim + 1, d, ternary)
        self.is_query = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, ternary) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = lin(d, enh_dim, ternary)

    def posenc(self, r):
        f = (2.0 ** torch.arange(self.nfreq, device=r.device) * math.pi).to(r.dtype)
        x = r[..., None] * f
        return torch.cat([r, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], -1)

    def forward(self, rel, base, enh, recv, own_base):
        """rel (B,Kn,3); base (B,Kn,base_dim); enh (B,Kn,enh_dim); recv (B,Kn) bool; own_base (B,base_dim).
        All attributes normalised. Returns the normalised enhancement layer of the lost anchor (B, enh_dim)."""
        r = recv[..., None].to(rel.dtype)
        tok = self.inp(torch.cat([self.posenc(rel), base, enh * r, r], -1))
        B = own_base.shape[0]
        z = dict(device=rel.device, dtype=rel.dtype)
        q_in = torch.cat([self.posenc(torch.zeros(B, 3, **z)), own_base,
                          torch.zeros(B, self.enh_dim, **z), torch.zeros(B, 1, **z)], -1)
        q = self.inp(q_in) + self.is_query
        x = torch.cat([q[:, None], tok], 1)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for b in self.blocks:
            x = b(x, mask)
        return self.out(self.norm(x[:, 0]))


class LiteConcealNet(nn.Module):
    """Deployment concealment network (query-only cross-attention).

    Neighbour tokens are encoded once by a two-layer MLP from [posenc(relative position), base, enhancement * received,
    received flag]; the lost anchor's query (from its own base layer) is refined by `layers` blocks of cross-attention
    to the fixed neighbour tokens followed by a feed-forward block. Neighbour tokens are not updated, so the cost per
    lost anchor is one MLP pass over Kn tokens plus `layers` single-query attention blocks; no attention mask is needed
    (non-received neighbours still carry their base layer and a received flag)."""

    def __init__(self, base_dim, enh_dim, d=96, layers=2, heads=4, nfreq=6, ternary=False):
        super().__init__()
        self.nfreq, self.enh_dim, self.h, self.d = nfreq, enh_dim, heads, d
        pe = 3 * 2 * nfreq + 3
        self.enc1, self.enc2 = lin(pe + base_dim + enh_dim + 1, d, ternary), lin(d, d, ternary)
        self.qin = lin(base_dim, d, ternary)
        self.nk = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.nq = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.nf = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.wq = nn.ModuleList([lin(d, d, ternary) for _ in range(layers)])
        self.wkv = nn.ModuleList([lin(d, 2 * d, ternary) for _ in range(layers)])
        self.wo = nn.ModuleList([lin(d, d, ternary) for _ in range(layers)])
        self.f1 = nn.ModuleList([lin(d, 4 * d, ternary) for _ in range(layers)])
        self.f2 = nn.ModuleList([lin(4 * d, d, ternary) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = lin(d, enh_dim, ternary)

    def posenc(self, r):
        f = (2.0 ** torch.arange(self.nfreq, device=r.device) * math.pi).to(r.dtype)
        x = r[..., None] * f
        return torch.cat([r, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], -1)

    def forward(self, rel, base, enh, recv, own_base):
        r = recv[..., None].to(rel.dtype)
        kv_tok = self.enc2(F.gelu(self.enc1(torch.cat([self.posenc(rel), base, enh * r, r], -1))))   # (B, Kn, d)
        q = self.qin(own_base)                                                                        # (B, d)
        B, Kn, D = kv_tok.shape
        h, dh = self.h, D // self.h
        for i in range(len(self.wq)):
            kv = self.wkv[i](self.nk[i](kv_tok)).reshape(B, Kn, 2, h, dh).permute(2, 0, 3, 1, 4)     # (2, B, h, Kn, dh)
            qq = self.wq[i](self.nq[i](q)).reshape(B, h, 1, dh)
            a = F.scaled_dot_product_attention(qq, kv[0], kv[1])                                     # (B, h, 1, dh)
            q = q + self.wo[i](a.reshape(B, D))
            q = q + self.f2[i](F.gelu(self.f1[i](self.nf[i](q))))
        return self.out(self.norm(q))


class EdgeConcealNet(nn.Module):
    """Deployment concealment network with per-anchor shared encodings.

    Every anchor is encoded once from its own attributes, e_j = MLP([base_j, enh_j * r_j, r_j]), independently of the
    lost anchor that queries it; at the edge e_j and its per-layer keys/values can therefore be computed when the
    anchor arrives and cached. The relative position of a neighbour enters only as an additive attention bias per
    head, b_hj = MLP_pos(Fourier(rel_j)). The lost anchor's query comes from its own base layer; `layers`
    cross-attention blocks with a feed-forward block refine it. forward() takes per-pair inputs (training and
    evaluation interface); forward_table() computes the same function from a table of anchor encodings."""

    def __init__(self, base_dim, enh_dim, d=96, layers=2, heads=4, nfreq=3, ternary=False):
        super().__init__()
        self.nfreq, self.h, self.d, self.layers = nfreq, heads, d, layers
        pe = 3 * 2 * nfreq + 3
        self.enc1, self.enc2 = lin(base_dim + enh_dim + 1, d, ternary), lin(d, d, ternary)
        self.pos1, self.pos2 = nn.Linear(pe, 32), nn.Linear(32, heads * layers)     # tiny, kept full precision
        self.qin = lin(base_dim, d, ternary)
        self.nk = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.nq = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.nf = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.wq = nn.ModuleList([lin(d, d, ternary) for _ in range(layers)])
        self.wkv = nn.ModuleList([lin(d, 2 * d, ternary) for _ in range(layers)])
        self.wo = nn.ModuleList([lin(d, d, ternary) for _ in range(layers)])
        self.f1 = nn.ModuleList([lin(d, 4 * d, ternary) for _ in range(layers)])
        self.f2 = nn.ModuleList([lin(4 * d, d, ternary) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = lin(d, enh_dim, ternary)

    def posenc(self, r):
        f = (2.0 ** torch.arange(self.nfreq, device=r.device) * math.pi).to(r.dtype)
        x = r[..., None] * f
        return torch.cat([r, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], -1)

    def encode(self, base, enh, recv):
        """Per-anchor encodings and per-layer keys/values: base (..., Db), enh (..., De), recv (...) bool."""
        r = recv[..., None].to(base.dtype)
        e = self.enc2(F.gelu(self.enc1(torch.cat([base, enh * r, r], -1))))
        return [self.wkv[i](self.nk[i](e)) for i in range(self.layers)]               # list of (..., 2d)

    def attend(self, kv_list, bias, own_base):
        """kv_list[i]: (B, Kn, 2d); bias: (B, Kn, layers*heads); own_base (B, Db)."""
        B, Kn, _ = kv_list[0].shape
        h, D = self.h, self.d; dh = D // h
        q = self.qin(own_base)
        bias = bias.reshape(B, Kn, self.layers, h).permute(2, 0, 3, 1)                 # (layers, B, h, Kn)
        for i in range(self.layers):
            k, v = kv_list[i].reshape(B, Kn, 2, h, dh).permute(2, 0, 3, 1, 4)         # (B, h, Kn, dh)
            qq = self.wq[i](self.nq[i](q)).reshape(B, h, 1, dh)
            att = torch.softmax((qq @ k.transpose(-1, -2)).squeeze(2) / math.sqrt(dh) + bias[i], -1)   # (B, h, Kn)
            q = q + self.wo[i]((att[:, :, None] @ v).reshape(B, D))
            q = q + self.f2[i](F.gelu(self.f1[i](self.nf[i](q))))
        return self.out(self.norm(q))

    def forward(self, rel, base, enh, recv, own_base):
        bias = self.pos2(F.gelu(self.pos1(self.posenc(rel))))
        return self.attend(self.encode(base, enh, recv), bias, own_base)

    def forward_table(self, kv_table, nbr_idx, rel, own_base):
        """kv_table: list over layers of (A, 2d) cached encodings; nbr_idx (B, Kn) anchor ids of the neighbours."""
        bias = self.pos2(F.gelu(self.pos1(self.posenc(rel))))
        return self.attend([t[nbr_idx] for t in kv_table], bias, own_base)
