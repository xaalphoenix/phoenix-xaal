"""Plane cut controls, kept in sync with the 3D handle."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ..i18n import t

AXES = ("x", "y", "z")
SLIDER_STEPS = 1000
MODES = ("plane", "grid", "curve", "freeform")


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
    apply_requested = Signal(object, object, str)  # normal, origin, keep (plane mode)
    apply_mode_requested = Signal(str)  # grid / curve / freeform
    mode_changed = Signal(str)
    grid_changed = Signal()
    fit_requested = Signal()
    curve_command = Signal(str)  # "draw", "undo", "clear"
    freeform_changed = Signal()

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

        mode_row = QFormLayout()
        self.mode = QComboBox()
        for m in MODES:
            self.mode.addItem("", m)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.mode_label = QLabel()
        mode_row.addRow(self.mode_label, self.mode)
        lay.addLayout(mode_row)
        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setObjectName("dim")
        lay.addWidget(self.mode_hint)

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

        # grid mode
        self.grid_box = QGroupBox()
        gf = QFormLayout(self.grid_box)
        self.counts = []
        for a in AXES:
            sp = QSpinBox()
            sp.setRange(1, 30)
            sp.valueChanged.connect(self._counts_changed)
            self.counts.append(sp)
            gf.addRow(QLabel(t("cut.grid_count", axis=a.upper())), sp)
        self.count_labels = [gf.labelForField(sp) for sp in self.counts]
        self.fit_btn = QPushButton()
        self.fit_btn.clicked.connect(self.fit_requested)
        gf.addRow(self.fit_btn)
        self.plane_list = QListWidget()
        self.plane_list.setMaximumHeight(130)
        self.plane_list.currentRowChanged.connect(self._plane_row)
        gf.addRow(self.plane_list)
        self.plane_pos = self._spin(-1e5, 1e5, 0.5, 2, " mm")
        self.plane_pos.valueChanged.connect(self._plane_pos_changed)
        self.plane_pos_label = QLabel()
        gf.addRow(self.plane_pos_label, self.plane_pos)
        self.grid_info = QLabel()
        self.grid_info.setWordWrap(True)
        self.grid_info.setObjectName("dim")
        gf.addRow(self.grid_info)
        lay.addWidget(self.grid_box)
        self.grid_planes: list[tuple[int, float]] = []

        # curve mode
        self.curve_box = QGroupBox()
        cf = QVBoxLayout(self.curve_box)
        row = QHBoxLayout()
        self.draw_btn = QPushButton()
        self.draw_btn.clicked.connect(lambda: self.curve_command.emit("draw"))
        self.undo_pt_btn = QPushButton()
        self.undo_pt_btn.clicked.connect(lambda: self.curve_command.emit("undo"))
        row.addWidget(self.draw_btn, 1)
        row.addWidget(self.undo_pt_btn)
        cf.addLayout(row)
        self.smooth = QCheckBox()
        self.smooth.setChecked(True)
        self.smooth.toggled.connect(lambda _: self.curve_command.emit("redraw"))
        cf.addWidget(self.smooth)
        self.curve_info = QLabel()
        self.curve_info.setObjectName("dim")
        self.curve_info.setWordWrap(True)
        cf.addWidget(self.curve_info)
        lay.addWidget(self.curve_box)

        # free-form mode (uses the plane above as its base)
        self.free_box = QGroupBox()
        ff = QFormLayout(self.free_box)
        self.free_size = QSpinBox()
        self.free_size.setRange(3, 8)
        self.free_size.setValue(4)
        self.free_size.valueChanged.connect(self._free_size_changed)
        self.free_size_label = QLabel()
        ff.addRow(self.free_size_label, self.free_size)
        self.free_point = QLabel()
        ff.addRow(self.free_point)
        self.free_height = self._spin(-1e4, 1e4, 0.5, 2, " mm")
        self.free_height.valueChanged.connect(self._free_height_changed)
        self.free_height_label = QLabel()
        ff.addRow(self.free_height_label, self.free_height)
        self.free_reset = QPushButton()
        self.free_reset.clicked.connect(self._free_reset)
        ff.addRow(self.free_reset)
        lay.addWidget(self.free_box)
        self.heights = np.zeros((4, 4))
        self.free_selected = None

        lay.addStretch(1)

        # footer: shown below the scroll area so Cut is always reachable
        self.footer = QWidget()
        foot = QVBoxLayout(self.footer)
        foot.setContentsMargins(10, 6, 10, 10)
        gap_row = QFormLayout()
        gap_row.setContentsMargins(0, 0, 0, 0)
        self.gap = self._spin(0, 2, 0.01, 2, " mm")
        self.gap_label = QLabel()
        gap_row.addRow(self.gap_label, self.gap)
        self.gap_widget = QWidget()
        self.gap_widget.setLayout(gap_row)
        foot.addWidget(self.gap_widget)

        opts = QFormLayout()
        self.keep = QComboBox()
        self.keep_label = QLabel()
        opts.addRow(self.keep_label, self.keep)
        self.gizmo = QCheckBox()
        self.gizmo.setChecked(True)
        self.gizmo.toggled.connect(self.gizmo_toggled)
        opts.addRow(self.gizmo)
        foot.addLayout(opts)

        self.apply_btn = QPushButton()
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._apply)
        foot.addWidget(self.apply_btn)

        for w, k in ((self.axis_buttons[0], "cut.axis"), (self.axis_buttons[1], "cut.axis"),
                     (self.axis_buttons[2], "cut.axis"), (self.tilt1, "cut.tilt1"),
                     (self.tilt2, "cut.tilt2"), (self.offset, "cut.offset"), (self.slider, "cut.offset"),
                     (self.center_btn, "cut.center"), (self.keep, "cut.keep"),
                     (self.gizmo, "cut.gizmo"), (self.apply_btn, "cut.apply"), (self.mode, "cut.mode"),
                     (self.fit_btn, "cut.fit"), (self.plane_list, "cut.grid_planes"),
                     (self.plane_pos, "cut.grid_pos"), (self.draw_btn, "cut.curve_draw"),
                     (self.undo_pt_btn, "cut.curve_undo"), (self.smooth, "cut.curve_smooth"),
                     (self.free_size, "cut.free_size"), (self.free_height, "cut.free_height"),
                     (self.free_reset, "cut.free_reset"), (self.gap, "cut.gap")) + tuple(
                        (sp, "cut.grid_count_help") for sp in self.counts):
            help_register(w, k)
        for s in (self.tilt1, self.tilt2, self.offset):
            s.valueChanged.connect(self._fields_changed)
        self.retranslate()
        self.set_part(None)
        self._mode_changed()

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
        for w in (self.plane_box, self.keep, self.gizmo, self.apply_btn, self.grid_box, self.curve_box,
                  self.free_box):
            w.setEnabled(has)
        if not has:
            self._preview_pts = None
            return
        b = np.asarray(bounds, float)
        self._bounds = b
        self._center = (b[0] + b[1]) / 2
        self._preview_pts = np.asarray(preview_vertices, dtype=np.float32)
        self._update_range(keep_value=False)
        self._counts_changed()
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

    def current_mode(self) -> str:
        return self.mode.currentData() or "plane"

    def _apply(self):
        if self.current_mode() == "plane":
            self.apply_requested.emit(self.normal(), self.origin(), self.keep.currentData())
        else:
            self.apply_mode_requested.emit(self.current_mode())

    def _mode_changed(self, *_):
        m = self.current_mode()
        self.plane_box.setVisible(m in ("plane", "freeform"))
        self.gizmo.setVisible(m == "plane")
        self.grid_box.setVisible(m == "grid")
        self.curve_box.setVisible(m == "curve")
        self.free_box.setVisible(m == "freeform")
        self.gap_widget.setVisible(m in ("curve", "freeform"))
        self.keep.setEnabled(m != "grid")
        self.mode_hint.setText(t("cut.hint_" + m))
        self.mode_changed.emit(m)

    # -- grid --------------------------------------------------------------------
    def _counts_changed(self, *_):
        if getattr(self, "_bounds", None) is None:
            return
        from ...core.grid import even_planes
        b = self._bounds
        planes = [(k, p) for k in range(3) for p in even_planes(b[0][k], b[1][k], self.counts[k].value())]
        self.set_grid_planes(planes, emit=True)

    def set_grid_planes(self, planes, counts=None, emit: bool = True) -> None:
        self.grid_planes = [(int(a), float(p)) for a, p in planes]
        if counts is not None:
            for sp, c in zip(self.counts, counts):
                sp.blockSignals(True)
                sp.setValue(int(c))
                sp.blockSignals(False)
        self.plane_list.blockSignals(True)
        self.plane_list.clear()
        for a, p in self.grid_planes:
            self.plane_list.addItem(f"{AXES[a].upper()}  {p:.2f} mm")
        self.plane_list.blockSignals(False)
        self.plane_pos.setEnabled(False)
        if emit:
            self.grid_changed.emit()

    def set_grid_info(self, text: str) -> None:
        self.grid_info.setText(text)

    def _plane_row(self, row: int):
        ok = 0 <= row < len(self.grid_planes)
        self.plane_pos.setEnabled(ok)
        if ok:
            self.plane_pos.blockSignals(True)
            self.plane_pos.setValue(self.grid_planes[row][1])
            self.plane_pos.blockSignals(False)

    def _plane_pos_changed(self, value):
        row = self.plane_list.currentRow()
        if 0 <= row < len(self.grid_planes):
            a = self.grid_planes[row][0]
            self.grid_planes[row] = (a, float(value))
            self.plane_list.item(row).setText(f"{AXES[a].upper()}  {value:.2f} mm")
            self.grid_changed.emit()

    # -- curve ---------------------------------------------------------------------
    def set_curve_info(self, n_points: int, drawing: bool) -> None:
        self.curve_info.setText(t("cut.curve_info", n=n_points))
        self.draw_btn.setText(t("cut.curve_stop") if drawing else t("cut.curve_draw"))
        self.undo_pt_btn.setEnabled(n_points > 0)

    # -- free-form -----------------------------------------------------------------
    def _free_size_changed(self, k):
        self.heights = np.zeros((k, k))
        self.select_free_point(None)
        self.freeform_changed.emit()

    def _free_reset(self):
        self.heights[:] = 0.0
        self.select_free_point(self.free_selected)
        self.freeform_changed.emit()

    def select_free_point(self, ij) -> None:
        self.free_selected = ij
        self.free_height.setEnabled(ij is not None)
        self.free_height.blockSignals(True)
        if ij is not None:
            self.free_height.setValue(float(self.heights[ij]))
            self.free_point.setText(t("cut.free_point", i=ij[0] + 1, j=ij[1] + 1))
        else:
            self.free_height.setValue(0.0)
            self.free_point.setText(t("cut.free_none"))
        self.free_height.blockSignals(False)

    def set_free_height(self, ij, value: float) -> None:
        self.heights[ij] = value
        if ij == self.free_selected:
            self.free_height.blockSignals(True)
            self.free_height.setValue(float(value))
            self.free_height.blockSignals(False)
        self.freeform_changed.emit()

    def _free_height_changed(self, value):
        if self.free_selected is not None:
            self.heights[self.free_selected] = value
            self.freeform_changed.emit()

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
        self.mode_label.setText(t("cut.mode"))
        for i, m in enumerate(MODES):
            self.mode.setItemText(i, t("cut.mode_" + m))
        self.mode_hint.setText(t("cut.hint_" + self.current_mode()))
        self.grid_box.setTitle(t("cut.mode_grid"))
        for lab, a in zip(self.count_labels, AXES):
            lab.setText(t("cut.grid_count", axis=a.upper()))
        self.fit_btn.setText(t("cut.fit_btn"))
        self.plane_pos_label.setText(t("cut.grid_pos"))
        self.curve_box.setTitle(t("cut.mode_curve"))
        self.smooth.setText(t("cut.curve_smooth"))
        self.undo_pt_btn.setText(t("cut.curve_undo"))
        self.free_box.setTitle(t("cut.mode_freeform"))
        self.free_size_label.setText(t("cut.free_size"))
        self.free_height_label.setText(t("cut.free_height"))
        self.free_reset.setText(t("cut.free_reset"))
        self.gap_label.setText(t("cut.gap"))
        self.select_free_point(self.free_selected)
        self.target.setText(f"{t('cut.target')}: {self._name}" if self._name else t("cut.none"))
