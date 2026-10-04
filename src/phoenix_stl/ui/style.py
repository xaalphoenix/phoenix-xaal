"""Dark theme, colors and fonts."""
from __future__ import annotations

import os

from PySide6.QtGui import QFont, QFontDatabase

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts")

BG = "#16171a"
PANEL = "#1f2125"
PANEL_2 = "#272a30"
BORDER = "#33363d"
TEXT = "#e7e8ea"
TEXT_DIM = "#9a9ea6"
ACCENT = "#ff7a1a"
ACCENT_HOVER = "#ff9445"
OK = "#4cc38a"
WARN = "#f5b23d"
BAD = "#ef5350"

VIEW_BG_TOP = "#2b2f36"
VIEW_BG_BOTTOM = "#121316"
SECTION = "#ff7a1a"
PROBLEM = "#ff3b3b"
PART_COLORS = ["#c9ccd1", "#6fa8dc", "#93c47d", "#e6b85c", "#c27ba0", "#76c7c0",
               "#e8836b", "#a58fd6", "#b6d36b", "#d98fb0"]


def load_fonts() -> str:
    family = ""
    for name in ("Vazirmatn-Regular.ttf", "Vazirmatn-Bold.ttf"):
        fid = QFontDatabase.addApplicationFont(os.path.join(FONT_DIR, name))
        if fid >= 0 and not family:
            family = QFontDatabase.applicationFontFamilies(fid)[0]
    return family


def app_font(family: str) -> QFont:
    f = QFont(family or "Segoe UI")
    f.setPointSizeF(9.5)
    return f


STYLESHEET = f"""
QMainWindow, QDialog {{ background: {BG}; color: {TEXT}; }}
QWidget {{ color: {TEXT}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; padding: 6px 10px; font-weight: bold; }}
QDockWidget > QWidget {{ background: {PANEL}; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected {{ background: {PANEL_2}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #111; }}
QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {BORDER}; spacing: 4px; padding: 4px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 4px 8px; }}
QToolButton:hover {{ background: {PANEL_2}; border-color: {BORDER}; }}
QToolButton:disabled {{ color: {TEXT_DIM}; }}
QTabWidget::pane {{ border: none; background: {PANEL}; }}
QTabBar::tab {{ background: {PANEL}; color: {TEXT_DIM}; padding: 8px 10px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; right: 10px; padding: 0 4px; color: {TEXT_DIM}; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:checked {{ background: {ACCENT}; color: #111; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {PANEL_2}; }}
QPushButton#primary {{ background: {ACCENT}; color: #111; font-weight: bold; border: none; padding: 9px 14px; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: #5a4434; color: #9a8b80; }}
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{ background: {BG}; border: 1px solid {BORDER};
    border-radius: 5px; padding: 4px 6px; selection-background-color: {ACCENT}; }}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QListWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; outline: none; }}
QListWidget::item {{ padding: 6px 4px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {PANEL_2}; border: 1px solid {ACCENT}; }}
QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QScrollArea {{ border: none; background: {PANEL}; }}
QWidget#qt_scrollarea_viewport, QWidget#toolPanel {{ background: {PANEL}; }}
QWidget#toolFooter {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#title {{ font-weight: bold; font-size: 11pt; }}
QFrame#helpPopup {{ background: #24262b; border: 1px solid {ACCENT}; border-radius: 10px; }}
QLabel#helpDots {{ background: {ACCENT}; color: #111; border-radius: 8px; font-weight: bold;
    padding: 0px 5px 2px 5px; }}
"""
