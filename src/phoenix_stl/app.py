"""Application entry point.

    phoenix-stl [files.stl ...] [--lang en|fa] [--selftest]
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys

from . import APP_NAME, __version__


def main(argv=None) -> int:
    multiprocessing.freeze_support()  # the engine process re-enters here when frozen
    ap = argparse.ArgumentParser(prog="phoenix-stl")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--lang", choices=["en", "fa"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = ap.parse_args(argv)

    if args.selftest:
        from .selftest import run
        return run()

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setOrganizationName("PHOENIX")
    QCoreApplication.setApplicationName("STL Studio")
    app = QApplication(sys.argv[:1])
    from .ui import i18n, style
    family = style.load_fonts()
    app.setFont(style.app_font(family))
    app.setStyle("Fusion")
    app.setStyleSheet(style.STYLESHEET)
    i18n.init(args.lang)

    from .ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    if args.files:
        win.open_files(args.files)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
