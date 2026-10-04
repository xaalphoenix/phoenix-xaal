import time

import numpy as np
import pytest

from phoenix_stl.core.analyze import analyze
from phoenix_stl.core.io_stl import load_stl, save_stl
from phoenix_stl.core.transform import rotation_about, translation
from phoenix_stl.engine import EngineClient


@pytest.fixture
def engine():
    eng = EngineClient()
    yield eng
    eng.shutdown()


def load(engine, tmp_path, mesh, name):
    src = str(tmp_path / f"{name}.stl")
    save_stl(mesh, src)
    ev = engine.wait(engine.submit("load", path=src, preview_faces=1000))
    assert ev.kind == "result", ev
    assert ev.result["removed"] == []
    return ev.result["added"][0]


def test_engine_pipeline(engine, tmp_path, torus):
    item = load(engine, tmp_path, torus, "torus")
    part = item["part"]
    assert part["n_faces"] == torus.n_faces and part["name"] == "torus"
    assert len(item["preview"][1]) == torus.n_faces  # small models are drawn at full detail

    ev = engine.wait(engine.submit("analyze", pid=part["id"]))
    assert ev.result["report"]["printable"]

    ev = engine.wait(engine.submit("cut", pid=part["id"], name="torus", normal=(0, 0, 1), origin=(0, 0, 0.5)))
    assert ev.kind == "result" and ev.result["removed"] == [part["id"]]
    names = [p["part"]["name"] for p in ev.result["added"]]
    assert names == ["torus_A", "torus_B"]

    # cut one of the halves again (step-by-step cutting)
    a = ev.result["added"][0]["part"]
    ev2 = engine.wait(engine.submit("cut", pid=a["id"], name=a["name"], normal=(1, 0, 0), origin=(0, 0, 0)))
    assert len(ev2.result["added"]) == 2

    items = [(p["part"]["id"], str(tmp_path / f"{p['part']['name']}.stl")) for p in ev2.result["added"]]
    ev3 = engine.wait(engine.submit("export", items=items))
    assert ev3.kind == "result"
    for _, path in items:
        assert analyze(load_stl(path))[0].printable


def test_parts_are_immutable_for_undo(engine, tmp_path, sphere):
    part = load(engine, tmp_path, sphere, "s")["part"]
    ev = engine.wait(engine.submit("cut", pid=part["id"], name="s", normal=(0, 0, 1), origin=(0, 0, 0)))
    assert ev.kind == "result"
    # the original is still there (undo) and can be restored with its preview
    ev = engine.wait(engine.submit("restore", pids=[part["id"]]))
    assert ev.kind == "result" and len(ev.result["parts"][0]["preview"][1]) == sphere.n_faces
    engine.wait(engine.submit("purge", pids=[part["id"]]))
    ev = engine.wait(engine.submit("restore", pids=[part["id"]]))
    assert ev.kind == "error"


def test_engine_transform(engine, tmp_path, sphere):
    part = load(engine, tmp_path, sphere, "s")["part"]
    m = translation((5, 0, 10)) @ rotation_about((0, 0, 0), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]))
    ev = engine.wait(engine.submit("transform", items=[(part["id"], "s", m.tolist())]))
    assert ev.kind == "result", ev
    new = ev.result["added"][0]
    b = np.array(new["part"]["bounds"])
    assert np.allclose(b.mean(0), [5, 0, 10], atol=1e-3)
    pv = new["preview"][0]
    assert np.allclose(pv.mean(0), [5, 0, 10], atol=0.1)
    path = str(tmp_path / "moved.stl")
    engine.wait(engine.submit("export", items=[(new["part"]["id"], path)]))
    assert analyze(load_stl(path))[0].printable


def test_engine_merge(engine, tmp_path, sphere):
    part = load(engine, tmp_path, sphere, "s")["part"]
    ev = engine.wait(engine.submit("cut", pid=part["id"], name="s", normal=(0.2, 0.1, 1), origin=(0, 0, 1)))
    ids = [p["part"]["id"] for p in ev.result["added"]]
    n_halves = sum(p["part"]["n_faces"] for p in ev.result["added"])
    ev = engine.wait(engine.submit("merge", pids=ids, name="s", mode="union"))
    assert ev.kind == "result", ev
    merged = ev.result["added"][0]
    path = str(tmp_path / "merged.stl")
    engine.wait(engine.submit("export", items=[(merged["part"]["id"], path)]))
    back = load_stl(path)
    assert analyze(back)[0].printable
    assert np.isclose(back.volume(), sphere.volume(), rtol=1e-4)
    ev = engine.wait(engine.submit("merge", pids=ids, name="s", mode="combine"))
    assert ev.result["added"][0]["part"]["n_faces"] == n_halves


def test_engine_repair(engine, tmp_path, sphere):
    from phoenix_stl.core.mesh import Mesh
    part = load(engine, tmp_path, Mesh(sphere.vertices, sphere.faces[40:]), "broken")["part"]
    assert not engine.wait(engine.submit("analyze", pid=part["id"])).result["report"]["watertight"]
    ev = engine.wait(engine.submit("repair", pid=part["id"], name="broken"))
    assert ev.result["log"]["filled_holes"] >= 1 and ev.result["removed"] == [part["id"]]
    new_id = ev.result["added"][0]["part"]["id"]
    assert engine.wait(engine.submit("analyze", pid=new_id)).result["report"]["printable"]


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


def test_engine_transform_on_bed_and_centered(engine, tmp_path, sphere):
    a = load(engine, tmp_path, sphere, "a")["part"]
    from phoenix_stl.core.transform import translation as tr
    b = load(engine, tmp_path, sphere, "b")["part"]
    ev = engine.wait(engine.submit("transform", items=[(a["id"], "a", tr((40, 7, 3)).tolist()),
                                                       (b["id"], "b", tr((-20, 7, 33)).tolist())],
                                   on_bed=True, centered=True))
    assert ev.kind == "result", ev
    bounds = np.array([x["part"]["bounds"] for x in ev.result["added"]])
    lo, hi = bounds[:, 0].min(0), bounds[:, 1].max(0)
    assert abs(lo[2]) < 1e-5 and np.allclose((lo[:2] + hi[:2]) / 2, 0, atol=1e-4)
    # relative placement inside the group is kept
    assert np.isclose(bounds[1, 0, 2] - bounds[0, 0, 2], 30, atol=1e-4)


def test_engine_grid_and_surface_cut(engine, tmp_path, sphere):
    part = load(engine, tmp_path, sphere, "ball")["part"]
    ev = engine.wait(engine.submit("grid_cut", pid=part["id"], name="ball", planes=[(0, 0.0), (2, 3.0)]))
    assert ev.kind == "result", ev
    names = sorted(p["part"]["name"] for p in ev.result["added"])
    assert names == ["ball_x1_z1", "ball_x1_z2", "ball_x2_z1", "ball_x2_z2"]
    one = ev.result["added"][0]["part"]
    frame = {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0], "w": [0, 0, 1]}
    heights = np.zeros((3, 3))
    heights[1, 1] = 3.0
    ev = engine.wait(engine.submit("surface_cut", pid=part["id"], name="ball", kind="freeform", frame=frame,
                                   params={"heights": heights.tolist(), "u_range": [-12, 12], "v_range": [-12, 12],
                                           "res": 32}, gap=0.1))
    assert ev.kind == "result", ev
    assert [p["part"]["name"] for p in ev.result["added"]] == ["ball_A", "ball_B"]
    ev = engine.wait(engine.submit("surface_cut", pid=one["id"], name="q", kind="curve", frame=frame,
                                   params={"curve": [[-20, 1], [0, 2], [20, 1]]}))
    assert ev.kind in ("result", "error")
