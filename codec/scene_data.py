"""COLMAP reader and image dataset for Mip-NeRF 360 scenes (standard 3DGS protocol: every 8th image is a test view).

Only the binary COLMAP model (sparse/0/{cameras,images,points3D}.bin) and one downsampled image folder are used.
World coordinates are recentred on the mean camera centre; scene_scale is the largest camera distance to it.
"""
import os
import struct

import imageio.v2 as imageio
import numpy as np
import torch

CAM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}   # model id -> #params


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cid, model, w, h = _read(f, "iiQQ")
            params = _read(f, "d" * CAM_PARAMS[model])
            cams[cid] = (model, w, h, np.array(params))
    return cams


def read_images(path):
    ims = []
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2d = _read(f, "Q")[0]
            f.seek(24 * n2d, 1)
            ims.append((iid, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid, name.decode()))
    return ims


def read_points(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            _, x, y, z, r, g, b, _ = _read(f, "QdddBBBd")
            tl = _read(f, "Q")[0]
            f.seek(8 * tl, 1)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


class Scene:
    """All views of one scene, images kept on the CPU as uint8; K and world-to-camera matrices per view."""

    def __init__(self, root, img_dir, test_every=8):
        sp = os.path.join(root, "sparse", "0")
        cams = read_cameras(os.path.join(sp, "cameras.bin"))
        ims = sorted(read_images(os.path.join(sp, "images.bin")), key=lambda t: t[4])
        pts, rgb = read_points(os.path.join(sp, "points3D.bin"))
        self.names, self.images, self.K, self.viewmats = [], [], [], []
        centers = []
        for iid, q, t, cid, name in ims:
            model, w, h, p = cams[cid]
            img = imageio.imread(os.path.join(root, img_dir, name))[..., :3]
            H, W = img.shape[:2]
            sx, sy = W / w, H / h
            if model in (0, 2, 3):          # SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL: f, cx, cy, ...
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:                           # PINHOLE / OPENCV ...: fx, fy, cx, cy, ...
                fx, fy, cx, cy = p[:4]
            K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float32)
            R = qvec2rot(q)
            M = np.eye(4, dtype=np.float32); M[:3, :3] = R; M[:3, 3] = t
            self.names.append(name); self.images.append(img); self.K.append(K); self.viewmats.append(M)
            centers.append(-R.T @ t)
        centers = np.array(centers)
        self.offset = centers.mean(0).astype(np.float32)
        for M in self.viewmats:               # shift world by -offset: x_c = R (x + offset) + t
            M[:3, 3] = M[:3, :3] @ self.offset + M[:3, 3]
        self.scene_scale = float(np.linalg.norm(centers - self.offset, axis=1).max() * 1.1)
        self.points = pts - self.offset
        self.points_rgb = rgb
        idx = np.arange(len(self.names))
        self.test_idx = idx[idx % test_every == 0]
        self.train_idx = idx[idx % test_every != 0]
        self.H, self.W = self.images[0].shape[:2]

    def view(self, i, device="cuda"):
        img = torch.from_numpy(self.images[i]).to(device).float() / 255.0
        K = torch.from_numpy(self.K[i]).to(device)
        V = torch.from_numpy(self.viewmats[i]).to(device)
        return img, K, V


# scene name -> (dataset folder, image folder); standard 3DGS evaluation resolutions
DATASETS = {**{s: ("360_v2", "images_4") for s in ("bicycle", "garden", "stump", "flowers", "treehill")},
            **{s: ("360_v2", "images_2") for s in ("room", "counter", "kitchen", "bonsai")},
            **{s: ("tandt", "images") for s in ("truck", "train")},
            **{s: ("db", "images") for s in ("playroom", "drjohnson")}}


def scene_location(name, data_root="data"):
    """(scene folder, image folder) of a scene; data_root may be the data folder or its 360_v2 subfolder."""
    root = os.path.dirname(data_root.rstrip("/")) if os.path.basename(data_root.rstrip("/")) == "360_v2" else data_root
    ds, img = DATASETS[name]
    return os.path.join(root, ds, name), img
