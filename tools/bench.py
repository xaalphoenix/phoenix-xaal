"""Time every core step on a (heavy) STL and report peak memory.

    python tools/bench.py model.stl
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phoenix_stl.core.analyze import analyze  # noqa: E402
from phoenix_stl.core.cut import plane_cut  # noqa: E402
from phoenix_stl.core.io_stl import load_stl, save_stl  # noqa: E402
from phoenix_stl.core.lod import display_mesh  # noqa: E402
from phoenix_stl.core.repair import repair  # noqa: E402

proc = psutil.Process()
peak = [0]


def watch():
    while True:
        peak[0] = max(peak[0], proc.memory_info().rss)
        time.sleep(0.05)


def step(name, fn):
    peak[0] = proc.memory_info().rss
    t = time.time()
    out = fn()
    print(f"{name:<10} {time.time() - t:7.2f}s   peak {peak[0] / 2**30:5.2f} GB", flush=True)
    return out


def main(path):
    threading.Thread(target=watch, daemon=True).start()
    print(f"{os.path.getsize(path) / 2**20:.0f} MB  {path}")
    mesh = step("load", lambda: load_stl(path))
    print(f"           {mesh.n_faces:,} triangles, {mesh.n_vertices:,} vertices")
    rep, _ = step("analyze", lambda: analyze(mesh))
    print(f"           watertight={rep.watertight} shells={rep.shells} volume={rep.volume:.0f} mm3")
    step("preview", lambda: display_mesh(mesh, 2_000_000))
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2
    res = step("cut", lambda: plane_cut(mesh, (0.2, 0.1, 1.0), c))
    print(f"           parts {res.positive.n_faces:,} + {res.negative.n_faces:,} triangles, loops={res.loops}")
    step("repair", lambda: repair(mesh))
    out = os.path.join(tempfile.gettempdir(), "phoenix_bench_out.stl")
    step("export", lambda: save_stl(res.positive, out))
    os.remove(out)


if __name__ == "__main__":
    main(sys.argv[1])
