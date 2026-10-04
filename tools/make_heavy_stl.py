"""Generate a heavy, watertight test STL (bumpy torus), streamed to disk.

    python tools/make_heavy_stl.py out.stl --triangles 15000000 [--hollow 2.0]

--hollow adds an inward-facing inner shell (wall thickness in mm), the same
layout a hollowed resin model has.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def torus_rows(nu, nv, R, r, i0, i1, inward=False):
    """Triangles of grid rows i0..i1 of a bumpy torus, as (k, 3, 3) float32."""
    def pts(i):
        u = 2 * math.pi * (i % nu) / nu
        v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
        rr = r + 1.5 * np.sin(17 * u) * np.sin(11 * v) + 0.4 * np.sin(53 * u + 3 * v)
        x = (R + rr * np.cos(v)) * math.cos(u)
        y = (R + rr * np.cos(v)) * math.sin(u)
        z = rr * np.sin(v) + 30.0
        return np.stack([x, y, z], -1)
    out = []
    prev = pts(i0)
    for i in range(i0, i1):
        nxt = pts(i + 1)
        a, d = prev, np.roll(prev, -1, axis=0)
        b, c = nxt, np.roll(nxt, -1, axis=0)
        t1 = np.stack([a, b, c], 1)
        t2 = np.stack([a, c, d], 1)
        t = np.concatenate([t1, t2])
        if inward:
            t = t[:, ::-1]
        out.append(t)
        prev = nxt
    return np.concatenate(out).astype(np.float32)


def write(path, triangles, hollow=0.0, R=60.0, r=25.0):
    shells = [(r, False)] + ([(r - hollow, True)] if hollow > 0 else [])
    per_shell = triangles // len(shells)
    nv = max(16, int(math.sqrt(per_shell / 2 / 2)))
    nu = max(16, per_shell // (2 * nv))
    total = 2 * nu * nv * len(shells)
    with open(path, "wb") as f:
        f.write(b"PHOENIX heavy test torus".ljust(80, b" "))
        f.write(np.uint32(total).tobytes())
        for rad, inward in shells:
            step = max(1, 500_000 // (2 * nv))
            for i0 in range(0, nu, step):
                tri = torus_rows(nu, nv, R, rad, i0, min(nu, i0 + step), inward)
                rec = np.zeros(len(tri), dtype=STL_DTYPE)
                rec["v"] = tri
                rec.tofile(f)
                print(f"\r{path}: {100 * (i0 + step) / nu:5.1f}%", end="", file=sys.stderr)
    print(f"\nwrote {total:,} triangles", file=sys.stderr)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--triangles", type=int, default=15_000_000)
    ap.add_argument("--hollow", type=float, default=0.0, help="wall thickness (mm) of an inner shell")
    a = ap.parse_args()
    write(a.out, a.triangles, a.hollow)
