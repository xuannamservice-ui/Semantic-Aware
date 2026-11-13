"""Image quality metrics: PSNR, SSIM (11x11 Gaussian window, sigma 1.5, as in the 3DGS evaluation code), LPIPS (VGG)."""
import torch
import torch.nn.functional as F

_LPIPS = None


def psnr(a, b):
    """a, b: (H, W, 3) in [0, 1]."""
    return float(-10 * torch.log10(F.mse_loss(a, b)))


def _gauss_window(size=11, sigma=1.5, device="cuda"):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim_map(a, b):
    """a, b: (1, 3, H, W)."""
    w = _gauss_window(device=a.device)
    mu1 = F.conv2d(a, w, padding=5, groups=3)
    mu2 = F.conv2d(b, w, padding=5, groups=3)
    s11 = F.conv2d(a * a, w, padding=5, groups=3) - mu1 ** 2
    s22 = F.conv2d(b * b, w, padding=5, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=5, groups=3) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))


def ssim(a, b):
    return float(ssim_map(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]).mean())


def lpips_vgg(a, b):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="vgg", verbose=False).to(a.device).eval()
    with torch.no_grad():
        return float(_LPIPS(a.permute(2, 0, 1)[None] * 2 - 1, b.permute(2, 0, 1)[None] * 2 - 1))


def all_metrics(pred, gt):
    pred = pred.clamp(0, 1)
    return {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "lpips": lpips_vgg(pred, gt)}
