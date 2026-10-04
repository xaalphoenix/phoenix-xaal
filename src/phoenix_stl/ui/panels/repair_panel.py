"""Health report and automatic repair options."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
                               QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .. import style
from ..i18n import i18n, t

ROWS = [
    # report key, label key, kind: "bool_good" (True good) / "bool_bad" / "count_bad" / "info"
    ("watertight", "report.watertight", "bool_good"),
    ("holes", "report.holes", "count_bad"),
    ("nonmanifold_edges", "report.nonmanifold", "count_bad"),
    ("flipped_edges", "report.flipped", "count_bad"),
    ("inside_out", "report.inside_out", "bool_bad"),
    ("degenerate_faces", "report.degenerate", "count_bad"),
    ("duplicate_faces", "report.duplicates", "count_bad"),
    ("shells", "report.shells", "info"),
    ("n_faces", "report.triangles", "info"),
    ("volume", "report.volume", "volume"),
]


class RepairPanel(QWidget):
    analyze_requested = Signal()
    repair_requested = Signal(dict)
    show_edges_toggled = Signal(bool)

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        self._report = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)

        self.report_box = QGroupBox()
        grid = QGridLayout(self.report_box)
        grid.setColumnStretch(0, 1)
        self.state = QLabel()
        self.state.setObjectName("title")
        grid.addWidget(self.state, 0, 0, 1, 2)
        self.labels, self.values = {}, {}
        for i, (key, label, _) in enumerate(ROWS, start=1):
            self.labels[key] = QLabel()
            self.values[key] = QLabel()
            self.values[key].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(self.labels[key], i, 0)
            grid.addWidget(self.values[key], i, 1)
        self.analyze_btn = QPushButton()
        self.analyze_btn.clicked.connect(self.analyze_requested)
        grid.addWidget(self.analyze_btn, len(ROWS) + 1, 0, 1, 2)
        self.show_edges = QCheckBox()
        self.show_edges.setChecked(True)
        self.show_edges.toggled.connect(self.show_edges_toggled)
        grid.addWidget(self.show_edges, len(ROWS) + 2, 0, 1, 2)
        lay.addWidget(self.report_box)

        self.opt_box = QGroupBox()
        form = QFormLayout(self.opt_box)
        self.merge = QDoubleSpinBox()
        self.merge.setRange(0, 1)
        self.merge.setDecimals(4)
        self.merge.setSingleStep(0.001)
        self.merge.setSuffix(" mm")
        self.merge_label = QLabel()
        form.addRow(self.merge_label, self.merge)
        self.degenerate = QCheckBox()
        self.duplicates = QCheckBox()
        self.debris = QCheckBox()
        self.orientation = QCheckBox()
        self.holes = QCheckBox()
        for cb in (self.degenerate, self.duplicates, self.debris, self.orientation, self.holes):
            cb.setChecked(True)
            form.addRow(cb)
        self.max_hole = QSpinBox()
        self.max_hole.setRange(0, 1_000_000)
        self.max_hole.setSingleStep(50)
        self.max_hole_label = QLabel()
        form.addRow(self.max_hole_label, self.max_hole)
        lay.addWidget(self.opt_box)

        self.apply_btn = QPushButton()
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._apply)
        lay.addWidget(self.apply_btn)
        lay.addStretch(1)

        for w, k in ((self.analyze_btn, "repair.analyze"), (self.show_edges, "repair.show_edges"),
                     (self.merge, "repair.merge"), (self.degenerate, "repair.degenerate"),
                     (self.duplicates, "repair.duplicates"), (self.debris, "repair.debris"),
                     (self.orientation, "repair.orientation"), (self.holes, "repair.holes"),
                     (self.max_hole, "repair.max_hole"), (self.apply_btn, "repair.apply")):
            help_register(w, k)
        self.retranslate()
        self.set_enabled(False)

    def options(self) -> dict:
        return {"merge_distance": self.merge.value(), "remove_degenerate": self.degenerate.isChecked(),
                "remove_duplicates": self.duplicates.isChecked(), "remove_debris": self.debris.isChecked(),
                "fix_orientation": self.orientation.isChecked(), "fill_holes": self.holes.isChecked(),
                "max_hole_edges": self.max_hole.value()}

    def _apply(self):
        self.repair_requested.emit(self.options())

    def set_enabled(self, on: bool) -> None:
        self.analyze_btn.setEnabled(on)
        self.apply_btn.setEnabled(on)

    def set_report(self, report: dict | None, checking: bool = False) -> None:
        self._report = report
        n = i18n().num
        if report is None:
            self.state.setText(t("status.checking") if checking else t("report.none"))
            self.state.setStyleSheet(f"color: {style.WARN if checking else style.TEXT_DIM}")
            for key in self.values:
                self.values[key].setText("–")
                self.values[key].setStyleSheet("")
            return
        ok = report["printable"]
        self.state.setText(("✔ " if ok else "✖ ") + t("status.printable" if ok else "status.problems"))
        self.state.setStyleSheet(f"color: {style.OK if ok else style.BAD}")
        for key, _, kind in ROWS:
            v = report[key]
            if kind == "bool_good":
                text, good = t("report.yes" if v else "report.no"), bool(v)
            elif kind == "bool_bad":
                text, good = t("report.yes" if v else "report.no"), not v
            elif kind == "count_bad":
                text, good = n(v), v == 0
            elif kind == "volume":
                text, good = n(abs(v) / 1000.0, "{:,.2f}") + " cm³", None
            else:
                text, good = n(v), None
            self.values[key].setText(text)
            color = "" if good is None else (style.OK if good else style.BAD)
            self.values[key].setStyleSheet(f"color: {color}" if color else "")

    def retranslate(self):
        self.report_box.setTitle(t("repair.report"))
        for key, label, _ in ROWS:
            self.labels[key].setText(t(label))
        self.analyze_btn.setText(t("repair.analyze"))
        self.show_edges.setText(t("repair.show_edges"))
        self.opt_box.setTitle(t("repair.options"))
        self.merge_label.setText(t("repair.merge"))
        self.degenerate.setText(t("repair.degenerate"))
        self.duplicates.setText(t("repair.duplicates"))
        self.debris.setText(t("repair.debris"))
        self.orientation.setText(t("repair.orientation"))
        self.holes.setText(t("repair.holes"))
        self.max_hole_label.setText(t("repair.max_hole"))
        self.max_hole.setSpecialValueText(t("repair.max_hole_all"))
        self.apply_btn.setText(t("repair.apply"))
        self.set_report(self._report)
