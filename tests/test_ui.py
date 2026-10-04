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
    window.tabs.setCurrentIndex(1)
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
