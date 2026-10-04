import time

import numpy as np
import pytest

from phoenix_stl.core.analyze import analyze
from phoenix_stl.core.io_stl import load_stl, save_stl
from phoenix_stl.engine import EngineClient


@pytest.fixture
def engine():
    eng = EngineClient()
    yield eng
    eng.shutdown()


def test_engine_pipeline(engine, tmp_path, torus):
    src = str(tmp_path / "torus.stl")
    save_stl(torus, src)
    ev = engine.wait(engine.submit("load", path=src, preview_faces=1000))
    assert ev.kind == "result"
    part = ev.result["part"]
    assert part["n_faces"] == torus.n_faces and part["name"] == "torus"
    pv, pf = ev.result["preview"]
    assert len(pf) == torus.n_faces  # small models are drawn at full detail

    ev = engine.wait(engine.submit("analyze", pid=part["id"]))
    assert ev.result["report"]["printable"]

    ev = engine.wait(engine.submit("cut", pid=part["id"], name="torus", normal=(0, 0, 1), origin=(0, 0, 0.5)))
    assert ev.kind == "result" and ev.result["removed"] == part["id"]
    names = [p["part"]["name"] for p in ev.result["parts"]]
    assert names == ["torus_A", "torus_B"]

    # cut one of the halves again (step-by-step cutting)
    a = ev.result["parts"][0]["part"]
    ev2 = engine.wait(engine.submit("cut", pid=a["id"], name=a["name"], normal=(1, 0, 0), origin=(0, 0, 0)))
    assert len(ev2.result["parts"]) == 2

    items = [(p["part"]["id"], str(tmp_path / f"{p['part']['name']}.stl")) for p in ev2.result["parts"]]
    ev3 = engine.wait(engine.submit("export", items=items))
    assert ev3.kind == "result"
    for _, path in items:
        assert analyze(load_stl(path))[0].printable


def test_engine_repair(engine, tmp_path, sphere):
    from phoenix_stl.core.mesh import Mesh
    broken = Mesh(sphere.vertices, sphere.faces[40:])
    src = str(tmp_path / "broken.stl")
    save_stl(broken, src)
    part = engine.wait(engine.submit("load", path=src)).result["part"]
    assert not engine.wait(engine.submit("analyze", pid=part["id"])).result["report"]["watertight"]
    ev = engine.wait(engine.submit("repair", pid=part["id"], name="broken"))
    assert ev.result["log"]["filled_holes"] >= 1
    assert engine.wait(engine.submit("analyze", pid=part["id"])).result["report"]["printable"]


def test_engine_errors_and_cancel(engine, tmp_path):
    ev = engine.wait(engine.submit("load", path=str(tmp_path / "missing.stl")))
    assert ev.kind == "error" and ev.code in ("exception",)
    # cancel kills the running job, the next job still works on a fresh engine
    big = str(tmp_path / "big.stl")
    rng = np.random.default_rng(0)
    tris = rng.random((400_000, 3, 3)).astype(np.float32)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = tris
    with open(big, "wb") as f:
        f.write(b"\0" * 80 + np.uint32(len(tris)).tobytes())
        rec.tofile(f)
    job = engine.submit("load", path=big)
    t = time.monotonic()
    while engine.current is None and time.monotonic() - t < 30:
        engine.poll()
        time.sleep(0.01)
    events = engine.cancel()
    assert events and events[0].code == "cancelled" and events[0].job.id == job.id
    ev = engine.wait(engine.submit("sysinfo"))
    assert ev.kind == "result" and ev.result["cores"] >= 1


def test_selftest():
    from phoenix_stl import selftest
    assert selftest.run() == 0
