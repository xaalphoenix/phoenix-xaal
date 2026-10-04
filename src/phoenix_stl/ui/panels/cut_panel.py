"""Plane cut controls, kept in sync with the 3D handle."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from ..i18n import t

AXES = ("x", "y", "z")
SLIDER_STEPS = 1000


def normal_from(axis: int, tilt1: float, tilt2: float) -> np.ndarray:
    """Base axis rotated by tilt1 about the next axis, then tilt2 about the one after."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    t1, t2 = math.radians(tilt1), math.radians(tilt2)
    n = np.zeros(3)
    n[a] = math.cos(t1) * math.sin(t2)
    n[b] = -math.sin(t1)
    n[axis] = math.cos(t1) * math.cos(t2)
    return n


def tilts_from(axis: int, n) -> tuple[float, float]:
    n = np.asarray(n, float) / np.linalg.norm(n)
    a, b = (axis + 1) % 3, (axis + 2) % 3
    t1 = math.degrees(math.asin(max(-1.0, min(1.0, -n[b]))))
    t2 = math.degrees(math.atan2(n[a], n[axis]))
    return t1, t2


class CutPanel(QWidget):
    plane_changed = Signal(object, object)  # normal, origin
    gizmo_toggled = Signal(bool)
    apply_requested = Signal(object, object, str)  # normal, origin, keep

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        self._center = np.zeros(3)
        self._preview_pts = None
        self._lock = False
        self._range = (-1.0, 1.0)
        self._name = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)

        self.target = QLabel()
        self.target.setObjectName("title")
        self.target.setWordWrap(True)
        lay.addWidget(self.target)

        self.plane_box = QGroupBox()
        form = QFormLayout(self.plane_box)
        form.setLabelAlignment(Qt.AlignVCenter)
        axis_row = QHBoxLayout()
        self.axis_group = QButtonGroup(self)
        self.axis_buttons = []
        for i, name in enumerate(AXES):
            b = QPushButton(name.upper())
            b.setCheckable(True)
            b.setMinimumWidth(44)
            self.axis_group.addButton(b, i)
            axis_row.addWidget(b)
            self.axis_buttons.append(b)
        self.axis_buttons[2].setChecked(True)
        self.axis_group.idClicked.connect(self._axis_clicked)
        self.axis_label = QLabel()
        form.addRow(self.axis_label, axis_row)

        self.tilt1 = self._spin(-90, 90, 0.5, 1, "°")
        self.tilt2 = self._spin(-180, 180, 0.5, 1, "°")
        self.tilt1_label, self.tilt2_label = QLabel(), QLabel()
        form.addRow(self.tilt1_label, self.tilt1)
        form.addRow(self.tilt2_label, self.tilt2)

        self.offset = self._spin(-1, 1, 0.1, 2, " mm")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.valueChanged.connect(self._slider_moved)
        self.offset_label = QLabel()
        off_box = QVBoxLayout()
        off_box.addWidget(self.offset)
        off_box.addWidget(self.slider)
        form.addRow(self.offset_label, off_box)
        self.center_btn = QPushButton()
        self.center_btn.clicked.connect(self.center_plane)
        form.addRow("", self.center_btn)
        lay.addWidget(self.plane_box)

        opts = QFormLayout()
        self.keep = QComboBox()
        self.keep_label = QLabel()
        opts.addRow(self.keep_label, self.keep)
        self.gizmo = QCheckBox()
        self.gizmo.setChecked(True)
        self.gizmo.toggled.connect(self.gizmo_toggled)
        opts.addRow(self.gizmo)
        lay.addLayout(opts)

        self.apply_btn = QPushButton()
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._apply)
        lay.addWidget(self.apply_btn)
        lay.addStretch(1)

        for w, k in ((self.axis_buttons[0], "cut.axis"), (self.axis_buttons[1], "cut.axis"),
                     (self.axis_buttons[2], "cut.axis"), (self.tilt1, "cut.tilt1"),
                     (self.tilt2, "cut.tilt2"), (self.offset, "cut.offset"), (self.slider, "cut.offset"),
                     (self.center_btn, "cut.center"), (self.keep, "cut.keep"),
                     (self.gizmo, "cut.gizmo"), (self.apply_btn, "cut.apply")):
            help_register(w, k)
        for s in (self.tilt1, self.tilt2, self.offset):
            s.valueChanged.connect(self._fields_changed)
        self.retranslate()
        self.set_part(None)

    def _spin(self, lo, hi, step, decimals, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        s.setAlignment(Qt.AlignRight)
        return s

    # -- state ------------------------------------------------------------------
    @property
    def axis(self) -> int:
        return max(0, self.axis_group.checkedId())

    def normal(self) -> np.ndarray:
        return normal_from(self.axis, self.tilt1.value(), self.tilt2.value())

    def origin(self) -> np.ndarray:
        return self._center + self.offset.value() * self.normal()

    def set_part(self, name: str | None, preview_vertices=None, bounds=None) -> None:
        has = name is not None
        self._name = name
        self.target.setText(f"{t('cut.target')}: {name}" if has else t("cut.none"))
        for w in (self.plane_box, self.keep, self.gizmo, self.apply_btn):
            w.setEnabled(has)
        if not has:
            self._preview_pts = None
            return
        b = np.asarray(bounds, float)
        self._center = (b[0] + b[1]) / 2
        self._preview_pts = np.asarray(preview_vertices, dtype=np.float32)
        self._update_range(keep_value=False)
        self._emit()

    def set_busy(self, busy: bool) -> None:
        self.apply_btn.setEnabled(not busy and self._preview_pts is not None)

    def center_plane(self):
        self._lock = True
        self.tilt1.setValue(0)
        self.tilt2.setValue(0)
        self.offset.setValue(0)
        self._lock = False
        self._update_range(keep_value=False)
        self._emit()

    def set_from_handle(self, normal, origin) -> None:
        """The user dragged the 3D handle."""
        t1, t2 = tilts_from(self.axis, normal)
        self._lock = True
        self.tilt1.setValue(t1)
        self.tilt2.setValue(t2)
        self._update_range(keep_value=True)
        n = self.normal()
        self.offset.setValue(float(np.dot(np.asarray(origin) - self._center, n)))
        self._sync_slider()
        self._lock = False
        self.plane_changed.emit(n, self.origin())

    def _update_range(self, keep_value: bool):
        if self._preview_pts is None:
            return
        d = (self._preview_pts - self._center) @ self.normal().astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        margin = 1e-3 * max(hi - lo, 1e-6)
        self._range = (lo + margin, hi - margin)
        lock = self._lock
        self._lock = True
        self.offset.setRange(self._range[0], self._range[1])
        if not keep_value:
            self.offset.setValue(min(max(0.0, self._range[0]), self._range[1]))
        self._sync_slider()
        self._lock = lock

    def _sync_slider(self):
        lo, hi = self._range
        frac = (self.offset.value() - lo) / (hi - lo) if hi > lo else 0.5
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(frac * SLIDER_STEPS)))
        self.slider.blockSignals(False)

    def _slider_moved(self, v):
        lo, hi = self._range
        self.offset.setValue(lo + (hi - lo) * v / SLIDER_STEPS)

    def _axis_clicked(self, _):
        self.retranslate()
        self.center_plane()

    def _fields_changed(self, *_):
        if self._lock:
            return
        sender = self.sender()
        if sender in (self.tilt1, self.tilt2):
            self._lock = True
            self._update_range(keep_value=True)
            self._lock = False
        self._sync_slider()
        self._emit()

    def _emit(self):
        if self._preview_pts is not None:
            self.plane_changed.emit(self.normal(), self.origin())

    def _apply(self):
        self.apply_requested.emit(self.normal(), self.origin(), self.keep.currentData())

    def retranslate(self):
        self.plane_box.setTitle(t("cut.plane"))
        self.axis_label.setText(t("cut.axis"))
        a, b = (self.axis + 1) % 3, (self.axis + 2) % 3
        self.tilt1_label.setText(t("cut.tilt", axis=AXES[a].upper()))
        self.tilt2_label.setText(t("cut.tilt", axis=AXES[b].upper()))
        self.offset_label.setText(t("cut.offset"))
        self.center_btn.setText(t("cut.center"))
        self.keep_label.setText(t("cut.keep"))
        cur = self.keep.currentData() or "both"
        self.keep.blockSignals(True)
        self.keep.clear()
        for key in ("both", "positive", "negative"):
            self.keep.addItem(t({"both": "cut.keep_both", "positive": "cut.keep_pos",
                                 "negative": "cut.keep_neg"}[key]), key)
        self.keep.setCurrentIndex(max(0, self.keep.findData(cur)))
        self.keep.blockSignals(False)
        self.gizmo.setText(t("cut.gizmo"))
        self.apply_btn.setText(t("cut.apply"))
        self.target.setText(f"{t('cut.target')}: {self._name}" if self._name else t("cut.none"))
