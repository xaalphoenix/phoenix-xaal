"""Move / rotate / scale the selected parts (previewed live, applied on demand)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)
from scipy.spatial.transform import Rotation

from ...core.transform import scale_about, translation
from ..i18n import i18n, t


def compose(center, offset, angles_deg, scale_pct) -> np.ndarray:
    """Matrix: scale and rotate about `center`, then move by `offset`."""
    c = np.asarray(center, float)
    r = np.eye(4)
    r[:3, :3] = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
    return translation(c + np.asarray(offset, float)) @ r @ scale_about((0, 0, 0), scale_pct / 100.0) @ translation(-c)


def angles_of(rot3: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot3).as_euler("xyz", degrees=True)


class MovePanel(QWidget):
    changed = Signal()
    apply_requested = Signal()
    reset_requested = Signal()
    lay_flat_requested = Signal()
    rotate90_requested = Signal(int)  # axis index

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        self._lock = False
        self._n, self._name = 0, ""
        self._info = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.target = QLabel()
        self.target.setObjectName("title")
        self.target.setWordWrap(True)
        lay.addWidget(self.target)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setObjectName("dim")
        lay.addWidget(self.hint)

        self.move_box = QGroupBox()
        mf = QFormLayout(self.move_box)
        self.offset = [self._spin(-5000, 5000, 1.0, 2, " mm") for _ in range(3)]
        self.offset_labels = [QLabel(a) for a in "XYZ"]
        for lab, s in zip(self.offset_labels, self.offset):
            mf.addRow(lab, s)
        self.on_bed = QCheckBox()
        self.centered = QCheckBox()
        mf.addRow(self.on_bed)
        mf.addRow(self.centered)
        lay.addWidget(self.move_box)

        self.rot_box = QGroupBox()
        rf = QFormLayout(self.rot_box)
        self.angles = [self._spin(-360, 360, 5.0, 1, "°") for _ in range(3)]
        for a, s in zip("XYZ", self.angles):
            rf.addRow(QLabel(a), s)
        quick = QHBoxLayout()
        self.rot90 = []
        for i, a in enumerate("XYZ"):
            b = QPushButton(f"{a} +90°")
            b.clicked.connect(lambda _=False, k=i: self.rotate90_requested.emit(k))
            quick.addWidget(b)
            self.rot90.append(b)
        rf.addRow(quick)
        self.lay_flat = QPushButton()
        self.lay_flat.clicked.connect(self.lay_flat_requested)
        rf.addRow(self.lay_flat)
        lay.addWidget(self.rot_box)

        self.scale_box = QGroupBox()
        sf = QFormLayout(self.scale_box)
        self.scale = self._spin(1, 10000, 1.0, 2, " %")
        self.scale.setValue(100.0)
        self.scale_label = QLabel()
        sf.addRow(self.scale_label, self.scale)
        lay.addWidget(self.scale_box)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("dim")
        lay.addWidget(self.info)
        row = QHBoxLayout()
        self.reset_btn = QPushButton()
        self.reset_btn.clicked.connect(self.reset_requested)
        self.apply_btn = QPushButton()
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self.apply_requested)
        row.addWidget(self.reset_btn)
        row.addWidget(self.apply_btn, 1)
        lay.addLayout(row)
        lay.addStretch(1)

        for w in self.offset + self.angles + [self.scale]:
            w.valueChanged.connect(self._emit)
        for c in (self.on_bed, self.centered):
            c.toggled.connect(self._emit)
        for w, k in [(s, "move.offset") for s in self.offset] + [(s, "move.rotate") for s in self.angles] + [
                (self.on_bed, "move.on_bed"), (self.centered, "move.centered"), (self.lay_flat, "move.lay_flat"),
                (self.scale, "move.scale"), (self.reset_btn, "move.reset"), (self.apply_btn, "move.apply")] + [
                (b, "move.rotate90") for b in self.rot90]:
            help_register(w, k)
        self.retranslate()
        self.set_target(0)

    @staticmethod
    def _spin(lo, hi, step, decimals, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        s.setAlignment(Qt.AlignRight)
        return s

    def _emit(self, *_):
        if not self._lock:
            self.changed.emit()

    # -- state ------------------------------------------------------------------
    def values(self) -> dict:
        return {"offset": [s.value() for s in self.offset], "angles": [s.value() for s in self.angles],
                "scale": self.scale.value(), "on_bed": self.on_bed.isChecked(),
                "centered": self.centered.isChecked()}

    def set_values(self, offset=None, angles=None, scale=None, on_bed=None, centered=None) -> None:
        self._lock = True
        if offset is not None:
            for s, v in zip(self.offset, offset):
                s.setValue(float(v))
        if angles is not None:
            for s, v in zip(self.angles, angles):
                s.setValue(float(v))
        if scale is not None:
            self.scale.setValue(float(scale))
        if on_bed is not None:
            self.on_bed.setChecked(on_bed)
        if centered is not None:
            self.centered.setChecked(centered)
        self._lock = False

    def reset(self) -> None:
        self.set_values([0, 0, 0], [0, 0, 0], 100.0, False, False)

    def is_identity(self) -> bool:
        v = self.values()
        return (not any(v["offset"]) and not any(v["angles"]) and v["scale"] == 100.0
                and not v["on_bed"] and not v["centered"])

    def set_target(self, n_parts: int, name: str = "") -> None:
        self._n = n_parts
        self._name = name
        has = n_parts > 0
        for w in (self.move_box, self.rot_box, self.scale_box, self.apply_btn, self.reset_btn):
            w.setEnabled(has)
        self._update_target()

    def _update_target(self):
        if self._n == 0:
            self.target.setText(t("cut.none"))
        elif self._n == 1:
            self.target.setText(f"{t('cut.target')}: {self._name}")
        else:
            self.target.setText(t("move.n_parts", n=self._n))

    def set_info(self, size, zmin) -> None:
        self._info = (size, zmin)
        n = i18n().num
        self.info.setText(t("move.info", x=n(size[0], "{:.2f}"), y=n(size[1], "{:.2f}"),
                            z=n(size[2], "{:.2f}"), zmin=n(zmin, "{:.2f}")))

    def set_busy(self, busy: bool) -> None:
        self.apply_btn.setEnabled(not busy and self._n > 0)

    def retranslate(self):
        self.hint.setText(t("move.hint"))
        self.move_box.setTitle(t("move.move"))
        self.on_bed.setText(t("move.on_bed_label"))
        self.centered.setText(t("move.centered_label"))
        self.rot_box.setTitle(t("move.rotate_box"))
        self.lay_flat.setText(t("move.lay_flat_btn"))
        self.scale_box.setTitle(t("move.scale_box"))
        self.scale_label.setText(t("move.scale_label"))
        self.reset_btn.setText(t("move.reset_btn"))
        self.apply_btn.setText(t("move.apply_btn"))
        self._update_target()
        if self._info is not None:
            self.set_info(*self._info)
