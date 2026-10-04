"""UI smoke test (needs a display; run under xvfb-run on Linux)."""
import os
import time

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (use xvfb-run)")


@pytest.fixture
def window(qtbot, tmp_path):
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.setOrganizationName("PHOENIX-test")
    QCoreApplication.setApplicationName("STL Studio test")
    from phoenix_stl.ui import i18n
    i18n.init("en")
    from phoenix_stl.ui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    yield win
    win.close()


def wait_idle(qtbot, win, timeout=60):
    end = time.time() + timeout
    qtbot.wait(200)
    while win.engine.busy and time.time() < end:
        qtbot.wait(50)
    qtbot.wait(200)


def test_load_cut_export_and_language(window, qtbot, tmp_path, torus):
    from phoenix_stl.core.analyze import analyze
    from phoenix_stl.core.io_stl import load_stl, save_stl
    from phoenix_stl.ui.i18n import i18n

    src = str(tmp_path / "ring.stl")
    save_stl(torus, src)
    window.open_files([src])
    wait_idle(qtbot, window)
    part = window.parts.selected()
    assert part is not None and part.status == "printable"

    window.tabs.setCurrentIndex(0)
    window.cut.offset.setValue(0.5)
    qtbot.wait(100)
    assert window.viewport._section is not None  # live section preview drawn
    window.cut.apply_btn.click()
    wait_idle(qtbot, window)
    names = sorted(p.name for p in window.parts.parts.values())
    assert names == ["ring_A", "ring_B"]

    window.export.folder.setText(str(tmp_path / "out"))
    window.export.check.setChecked(False)
    window.export.all_btn.click()
    wait_idle(qtbot, window)
    for n in names:
        assert analyze(load_stl(str(tmp_path / "out" / f"{n}.stl")))[0].printable

    i18n().set_language("fa")
    qtbot.wait(100)
    assert window.tabs.tabText(0) == "برش"
    window.help.current = window.cut.tilt1
    window.help._show_popup()
    assert window.help.popup.isVisible() and window.help.popup.title.text() == "چرخش (محور اول)"
    i18n().set_language("en")


def names(win):
    return sorted(p.name for p in win.parts.parts.values())


def test_undo_redo_move_merge(window, qtbot, tmp_path, sphere):
    import numpy as np
    from phoenix_stl.core.io_stl import load_stl, save_stl

    src = str(tmp_path / "ball.stl")
    save_stl(sphere, src)
    window.open_files([src])
    wait_idle(qtbot, window)
    assert names(window) == ["ball"]

    # cut, undo, redo
    window.tabs.setCurrentIndex(0)
    window.cut.apply_btn.click()
    wait_idle(qtbot, window)
    assert names(window) == ["ball_A", "ball_B"]
    window.undo()
    wait_idle(qtbot, window)
    assert names(window) == ["ball"]
    window.redo()
    wait_idle(qtbot, window)
    assert names(window) == ["ball_A", "ball_B"]

    # move the upper half up by 15 mm and put the group on the bed
    a = next(p for p in window.parts.parts.values() if p.name == "ball_A")
    window.parts.select(a.id)
    window.tabs.setCurrentIndex(window._tab_index[window.move])
    qtbot.wait(50)
    window.move.set_values(offset=[0, 0, 15])
    window._update_move_preview()
    window.move.apply_btn.click()
    wait_idle(qtbot, window)
    moved = next(p for p in window.parts.parts.values() if p.name == "ball_A")
    assert moved.id != a.id and np.isclose(moved.bounds[0][2], a.bounds[0][2] + 15, atol=1e-3)
    window.undo()
    wait_idle(qtbot, window)
    back = next(p for p in window.parts.parts.values() if p.name == "ball_A")
    assert back.id == a.id

    # merge both halves back into one printable ball
    window.tabs.setCurrentIndex(0)
    window.parts.list.selectAll()
    window.merge_selected("union")
    wait_idle(qtbot, window)
    assert len(window.parts.parts) == 1
    merged = window.parts.selected()
    assert merged.status == "printable"
    out = str(tmp_path / "merged.stl")
    window.engine.submit("export", tag={"folder": str(tmp_path)}, items=[(merged.id, out)])
    wait_idle(qtbot, window)
    assert np.isclose(load_stl(out).volume(), sphere.volume(), rtol=1e-4)

    # delete + undo, rename + undo
    window.parts.select(merged.id)
    from unittest import mock
    from PySide6.QtWidgets import QMessageBox
    with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
        window.delete_selected()
    assert not window.parts.parts
    window.undo()
    wait_idle(qtbot, window)
    assert len(window.parts.parts) == 1
    item = window.parts.list.item(0)
    item.setText("trophy")
    assert names(window) == ["trophy"]
    window.undo()
    assert names(window) == [merged.name]


def test_grid_curve_freeform_modes(window, qtbot, tmp_path, sphere):
    import numpy as np
    from phoenix_stl.core.io_stl import save_stl

    src = str(tmp_path / "ball.stl")
    save_stl(sphere, src)
    window.open_files([src])
    wait_idle(qtbot, window)
    window.tabs.setCurrentIndex(0)

    # grid: 2 x 1 x 2 pieces; the grid is not carried over to the pieces
    window.cut.mode.setCurrentIndex(1)
    window.cut.counts[0].setValue(2)
    window.cut.counts[2].setValue(2)
    assert len(window.cut.grid_planes) == 2
    window.cut.apply_btn.click()
    wait_idle(qtbot, window)
    assert names(window) == ["ball_x1_z1", "ball_x1_z2", "ball_x2_z1", "ball_x2_z2"]
    assert all(p.status == "printable" for p in window.parts.parts.values())
    assert window.cut.grid_planes == [] and window.cut.counts[0].value() == 1
    window.undo()
    wait_idle(qtbot, window)

    # curve drawn with clicks in the view
    window.cut.mode.setCurrentIndex(2)
    window.viewport.view("front")
    window._curve_command("draw")
    p = window.parts.selected()
    b = np.asarray(p.bounds)
    c = b.mean(0)
    for x, z in ((b[0][0] - 2, c[2] - 3), (c[0], c[2] + 4), (b[1][0] + 2, c[2] - 3)):
        sx, sy = window.viewport.world_to_display([(x, c[1], z)])[0]
        window._curve_click(sx, sy, False)
    assert window.cut.curve_info.text() == "3 points"
    window.cut.apply_btn.click()
    wait_idle(qtbot, window)
    assert names(window) == ["ball_A", "ball_B"]
    vols = [window.parts.parts[k].report for k in window.parts.parts]
    assert all(r and r["printable"] for r in vols)
    window.undo()
    wait_idle(qtbot, window)

    # free-form: raise one control point, cut with a joint gap
    window.cut.mode.setCurrentIndex(3)
    qtbot.wait(100)
    window.cut.set_free_height((1, 1), 4.0)
    window.cut.gap.setValue(0.1)
    window.cut.apply_btn.click()
    wait_idle(qtbot, window)
    assert names(window) == ["ball_A", "ball_B"]
    assert all(p.status == "printable" for p in window.parts.parts.values())


def test_connect_tab(window, qtbot, tmp_path, cube):
    import numpy as np
    from PySide6.QtCore import QSettings
    from phoenix_stl.core.io_stl import save_stl

    QSettings().remove("fit_custom")
    window.conn_tool.printer_changed()

    src = str(tmp_path / "block.stl")
    save_stl(cube, src)
    window.open_files([src])
    wait_idle(qtbot, window)
    window.tabs.setCurrentIndex(0)
    window.cut.offset.setValue(1.0)
    window.cut.apply_btn.click()
    wait_idle(qtbot, window)
    # the two halves are selected after the cut: the joint is found by itself
    window.tabs.setCurrentIndex(window._tab_index[window.conn])
    wait_idle(qtbot, window)
    tool = window.conn_tool
    assert tool.joint is not None and tool.joint["area"] > 399
    assert len(tool.placements) == 2 and all(tool.ok)
    assert window.conn.depth_gap.value() == 0.3 and window.conn.wall.value() == 1.0  # resin defaults

    # add one more by clicking on the face, drag it, then remove it
    window.conn.add_btn.setChecked(True)
    fr = tool.joint["frame"]
    window.viewport.view("top")
    sx, sy = window.viewport.world_to_display([fr.to_world(np.array([[0.0, 0.0, 0.0]]))[0]])[0]
    tool._click(sx, sy, False)
    assert len(tool.placements) == 3 and tool.selected == 2
    tool._select(2)
    tool._drag(2, 3.0 * fr.u)
    assert abs(tool.placements[2].u - 3.0) < 1e-6
    window.conn.remove_btn.click()
    assert len(tool.placements) == 2
    window.conn.add_btn.setChecked(False)

    # magnets: holes only, no pins part
    window.conn.type.setCurrentIndex(2)
    qtbot.wait(100)
    assert len(tool.placements) >= 1
    window.conn.apply_btn.click()
    wait_idle(qtbot, window)
    assert len(window.parts.parts) == 2
    assert all(p.status == "printable" for p in window.parts.parts.values())
    window.undo()
    wait_idle(qtbot, window)

    # dowels make a pins part
    window.parts.list.selectAll()
    wait_idle(qtbot, window)
    window.conn.type.setCurrentIndex(0)
    qtbot.wait(100)
    window.conn.apply_btn.click()
    wait_idle(qtbot, window)
    assert any(p.name.endswith("_pins") for p in window.parts.parts.values())
    assert all(p.status == "printable" for p in window.parts.parts.values())

    # tolerance coupon and saving the best gap
    window.conn.coupon_btn.click()
    wait_idle(qtbot, window)
    assert any(p.name.startswith("tolerance_coupon") for p in window.parts.parts.values())
    holes = window.conn._coupon
    assert len(holes) == 5 and holes[2] == pytest.approx(window.conn.clearance.value())
    window.conn.best.setValue(4)
    window.conn.use_btn.click()
    assert window.conn.clearance.value() == pytest.approx(holes[3])
    from PySide6.QtCore import QSettings
    QSettings().remove("fit_custom")
