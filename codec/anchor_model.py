"""Anchor-based scene representation (structure of Scaffold-GS, Lu et al., CVPR 2024) on the gsplat rasterizer.

Each anchor a has a fixed position x_a (voxel centre, transmitted on the protected skeleton channel) and
transmitted attributes: a latent feature f_a (F channels), k offsets o_a (k x 3) and a log-scale s_a (6).
Shared MLPs map (f_a, viewing direction, distance) to k neural Gaussians: opacity, colour, scale, rotation;
centres are x_a + o_a * exp(s_a[:3]). Attributes are quantized with learned step sizes (straight-through
estimator), so the rendered quality is that of the transmitted, quantized values.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization


class STEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step):
        return torch.round(x / step) * step

    @staticmethod
    def backward(ctx, g):
        return g, None


def quant(x, step):
    return STEQuant.apply(x, step)


class AnchorModel(nn.Module):
    ATTRS = ("feat", "offsets", "scaling")

    def __init__(self, xyz, voxel, feat_dim=32, k=10, hidden=32):
        super().__init__()
        A = xyz.shape[0]
        self.k, self.feat_dim = k, feat_dim
        self.register_buffer("xyz", xyz.clone())
        self.voxel = voxel
        self.feat = nn.Parameter(torch.zeros(A, feat_dim))
        self.offsets = nn.Parameter(torch.zeros(A, k, 3))
        self.scaling = nn.Parameter(torch.log(torch.full((A, 6), voxel)))
        # quantization steps (log), learned; fixed after training and transmitted once per scene
        self.log_qstep = nn.ParameterDict({"feat": nn.Parameter(torch.tensor(math.log(0.1))),
                                           "offsets": nn.Parameter(torch.tensor(math.log(0.05))),
                                           "scaling": nn.Parameter(torch.tensor(math.log(0.05)))})
        din = feat_dim + 3 + 1
        self.mlp_opacity = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(True), nn.Linear(hidden, k), nn.Tanh())
        self.mlp_cov = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(True), nn.Linear(hidden, 7 * k))
        self.mlp_color = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(True), nn.Linear(hidden, 3 * k), nn.Sigmoid())

    def qstep(self, name):
        return torch.exp(self.log_qstep[name])

    def attributes(self, quantized=True):
        """Transmitted attributes (quantized with the current steps)."""
        out = {}
        for n in self.ATTRS:
            v = getattr(self, n)
            out[n] = quant(v, self.qstep(n)) if quantized else v
        return out

    def neural_gaussians(self, attrs, idx, cam_center):
        """Gaussians of the anchors idx (LongTensor) from attribute dict attrs (full arrays or rows for idx)."""
        x = self.xyz[idx]
        f, o, s = attrs["feat"], attrs["offsets"], attrs["scaling"]
        if f.shape[0] != idx.shape[0]:
            f, o, s = f[idx], o[idx], s[idx]
        d = x - cam_center[None]
        dist = d.norm(dim=1, keepdim=True)
        inp = torch.cat([f, d / dist, dist], 1)
        k = self.k
        opa = self.mlp_opacity(inp).reshape(-1)                       # (A*k)
        col = self.mlp_color(inp).reshape(-1, 3)
        cov = self.mlp_cov(inp).reshape(-1, 7)
        es = torch.exp(s)
        means = (x[:, None, :] + o * es[:, None, :3]).reshape(-1, 3)
        scales = (es[:, None, 3:] * torch.sigmoid(cov[:, :3]).reshape(-1, k, 3)).reshape(-1, 3)
        quats = F.normalize(cov[:, 3:], dim=-1)
        keep = opa > 0
        return means[keep], quats[keep], scales[keep], opa[keep], col[keep]

    def visible(self, K, V, W, H, margin=1.1):
        """Anchors in front of the camera whose centres project inside the (slightly enlarged) image."""
        xc = (V[:3, :3] @ self.xyz.T + V[:3, 3:]).T
        z = xc[:, 2]
        u = K[0, 0] * xc[:, 0] / z.clamp_min(1e-6) + K[0, 2]
        v = K[1, 1] * xc[:, 1] / z.clamp_min(1e-6) + K[1, 2]
        mw, mh = (margin - 1) * W, (margin - 1) * H
        m = (z > 0.05) & (u > -mw) & (u < W + mw) & (v > -mh) & (v < H + mh)
        return torch.nonzero(m).squeeze(1)

    def render(self, attrs, idx, K, V, W, H, bg=None):
        cam = -V[:3, :3].T @ V[:3, 3]
        means, quats, scales, opa, col = self.neural_gaussians(attrs, idx, cam)
        img, alpha, info = rasterization(means, quats, scales, opa, col, V[None], K[None], W, H, packed=False,
                                         backgrounds=bg)
        return img[0], info, scales


def voxel_anchors(points, opacity, voxel, min_opacity=0.1):
    """Voxel centres occupied by Gaussians of a trained 3DGS model (opacity above a threshold)."""
    p = points[opacity > min_opacity]
    keys = torch.floor(p / voxel)
    keys = torch.unique(keys, dim=0)
    return (keys + 0.5) * voxel
