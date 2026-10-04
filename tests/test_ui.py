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
