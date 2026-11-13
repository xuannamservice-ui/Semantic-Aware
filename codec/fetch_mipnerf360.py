"""Selective download of Mip-NeRF 360 scenes from the official zip (HTTP range requests, no full download).

Only the members needed for the standard 3DGS protocol are extracted: COLMAP model (sparse/0), poses_bounds.npy
and one downsampled image folder per scene (images_4 for outdoor scenes, images_2 for indoor scenes).
usage: python fetch_mipnerf360.py --out data/360_v2 [--scenes bicycle,garden,...]
"""
import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile

URLS = {"main": "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        "extra": "http://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"}
OUTDOOR = {"bicycle": "main", "garden": "main", "stump": "main", "flowers": "extra", "treehill": "extra"}
INDOOR = {"room": "main", "counter": "main", "kitchen": "main", "bonsai": "main"}


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small block cache."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self.fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, start, end):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.fetched += len(data)
                return data
            except Exception as e:                      # transient network errors
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:                          # large member: one request
            data = self._get(self.pos, self.pos + n)
        else:
            out = bytearray()
            p = self.pos
            while len(out) < n:
                b = p // self.block
                if b not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    s = b * self.block
                    self.cache[b] = self._get(s, min(s + self.block, self.size))
                chunk = self.cache[b][p - b * self.block:]
                take = chunk[:n - len(out)]
                out += take
                p += len(take)
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def wanted(name, scene, img_dir):
    parts = name.split("/")
    if scene not in parts:
        return False
    i = parts.index(scene)
    rest = "/".join(parts[i + 1:])
    return rest.startswith("sparse/0/") or rest == "poses_bounds.npy" or rest.startswith(img_dir + "/")


def main(out, scenes):
    os.makedirs(out, exist_ok=True)
    groups = {}
    for s in scenes:
        src = OUTDOOR.get(s) or INDOOR.get(s)
        if src is None:
            sys.exit(f"unknown scene {s}")
        groups.setdefault(src, []).append(s)
    for src, sc in groups.items():
        f = HTTPRangeFile(URLS[src])
        z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
        for s in sc:
            img_dir = "images_4" if s in OUTDOOR else "images_2"
            members = [m for m in z.infolist() if not m.is_dir() and wanted(m.filename, s, img_dir)]
            t0, nbytes = time.time(), 0
            for m in members:
                parts = m.filename.split("/")
                rel = "/".join(parts[parts.index(s):])
                dst = os.path.join(out, rel)
                if os.path.exists(dst) and os.path.getsize(dst) == m.file_size:
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(m) as fi, open(dst + ".part", "wb") as fo:
                    while True:
                        buf = fi.read(1 << 22)
                        if not buf:
                            break
                        fo.write(buf)
                os.replace(dst + ".part", dst)
                nbytes += m.file_size
            print(f"{s}: {len(members)} files ({img_dir}), {nbytes / 1e6:.1f} MB in {time.time() - t0:.0f}s, "
                  f"total fetched {f.fetched / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/360_v2")
    ap.add_argument("--scenes", default="bicycle,garden,stump,room,counter,kitchen,bonsai")
    a = ap.parse_args()
    main(a.out, a.scenes.split(","))
