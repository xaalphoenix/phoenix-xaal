"""Headless end-to-end check, used by CI on the frozen Windows build.

    PhoenixSTLStudio.exe --selftest
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

from .core import connectors as C
from .core.analyze import analyze
from .core.io_stl import load_stl, save_stl
from .core.mesh import Mesh
from .core.decal import DecalParams
from .core.mold import MoldParams
from .engine import EngineClient


def _sphere(n=64, r=10.0) -> Mesh:
    u = np.linspace(0, np.pi, n)[1:-1]
    v = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    pts = np.stack([np.sin(U) * np.cos(Vv), np.sin(U) * np.sin(Vv), np.cos(U)], -1).reshape(-1, 3) * r
    pts = np.vstack([pts, [[0, 0, r], [0, 0, -r]]])
    rows, cols = len(u), len(v)
    faces = []
    for i in range(rows - 1):
        for j in range(cols):
            a, b = i * cols + j, i * cols + (j + 1) % cols
            c, d = a + cols, b + cols
            faces += [[a, c, b], [b, c, d]]
    top, bot = len(pts) - 2, len(pts) - 1
    for j in range(cols):
        faces.append([top, j, (j + 1) % cols])
        last = (rows - 1) * cols
        faces.append([bot, last + (j + 1) % cols, last + j])
    return Mesh(pts, np.array(faces))


def run() -> int:
    tmp = tempfile.mkdtemp(prefix="phoenix_selftest_")
    src = os.path.join(tmp, "sphere.stl")
    save_stl(_sphere(), src)
    assert analyze(load_stl(src))[0].printable, "test sphere is not printable"

    eng = EngineClient()
    try:
        t = time.time()
        ev = eng.wait(eng.submit("load", path=src, preview_faces=1000))
        assert ev.kind == "result", ev
        part = ev.result["added"][0]["part"]
        ev = eng.wait(eng.submit("cut", pid=part["id"], name="sphere", normal=(0.2, 0.1, 1), origin=(0, 0, 1)))
        assert ev.kind == "result" and len(ev.result["added"]) == 2, ev
        halves = [p["part"] for p in ev.result["added"]]
        # connectors: find the joint, place two dowels, add them (pins come out as a third part)
        ev = eng.wait(eng.submit("joint", pids=[h["id"] for h in halves]))
        assert ev.kind == "result", ev
        shape = C.JointShape.from_dict(ev.result["shape"])
        spec, fit = C.Spec("dowel", 2.0, 6.0), C.Fit.preset("resin", "snug")
        pls, _ = C.auto_place(shape, spec, fit, 2)
        assert len(pls) == 2, pls
        ev = eng.wait(eng.submit("connect", pids=[h["id"] for h in halves], names=[h["name"] for h in halves],
                                 frame=ev.result["frame"], spec=vars(spec), fit=vars(fit),
                                 placements=[vars(p) for p in pls]))
        assert ev.kind == "result" and len(ev.result["added"]) == 3 and not ev.result["warnings"], ev
        outs = []
        for p in ev.result["added"]:
            a = eng.wait(eng.submit("analyze", pid=p["part"]["id"]))
            assert a.result["report"]["printable"], a.result
            outs.append((p["part"]["id"], os.path.join(tmp, p["part"]["name"] + ".stl")))
        # silicone mother mold (draft quality) around the original sphere
        ev = eng.wait(eng.submit("mold_build", pid=part["id"], name="sphere",
                                 params=MoldParams(resolution=1.0, feet=3).to_dict()))
        assert ev.kind == "result" and len(ev.result["added"]) == 3, ev
        th = ev.result["info"]["thickness"]
        assert abs(th["min"] - 5) < 0.05 and abs(th["max"] - 5) < 0.05, th
        for p in ev.result["added"]:
            outs.append((p["part"]["id"], os.path.join(tmp, p["part"]["name"] + ".stl")))
        # a raised square on top of the sphere
        img = np.zeros((40, 40), np.uint8)
        img[8:32, 8:32] = 255
        ev = eng.wait(eng.submit("decal", pid=part["id"], name="sphere_decal", origin=(0, 0, 10), normal=(0, 0, 1),
                                 image=img, params=DecalParams(width=8, height=8, depth=0.6, resolution=0.25).to_dict()))
        assert ev.kind == "result" and ev.result["info"]["volume_change"] > 10, ev
        outs.append((ev.result["added"][0]["part"]["id"], os.path.join(tmp, "sphere_decal.stl")))
        ev = eng.wait(eng.submit("export", items=outs))
        assert ev.kind == "result", ev
        for _, path in outs:
            assert analyze(load_stl(path))[0].printable, path
        print(f"selftest OK ({time.time() - t:.1f}s)")
        return 0
    finally:
        eng.shutdown()


if __name__ == "__main__":
    sys.exit(run())
