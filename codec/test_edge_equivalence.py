"""Check that EdgeConcealNet.forward_table (cached per-anchor encodings) equals forward (per-pair inputs).
usage: python test_edge_equivalence.py   (prints the maximum absolute difference; exits non-zero if it exceeds 1e-6)"""
import sys
import torch
from conceal_net import EdgeConcealNet

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
A, Kn, B = 5000, 16, 700
net = EdgeConcealNet(18, 66).to(dev).eval()
base, enh = torch.randn(A, 18, device=dev), torch.randn(A, 66, device=dev)
avail, xyz = torch.rand(A, device=dev) > 0.3, torch.rand(A, 3, device=dev)
nbr = torch.randint(0, A, (A, Kn), device=dev); lost = torch.randperm(A, device=dev)[:B]
nb = nbr[lost]; rel = xyz[nb] - xyz[lost][:, None]
with torch.no_grad():
    y1 = net(rel, base[nb], enh[nb], avail[nb], base[lost])
    y2 = net.forward_table(net.encode(base, enh, avail), nb, rel, base[lost])
d = float((y1 - y2).abs().max())
print(f"device {dev}: max |forward - forward_table| = {d:.3e}")
sys.exit(0 if d <= 1e-6 else 1)
