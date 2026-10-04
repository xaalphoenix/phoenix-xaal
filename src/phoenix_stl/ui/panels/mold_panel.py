"""Silicone mother mold settings."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ...core.mold import MoldParams
from ..i18n import i18n, t

QUALITY = (("draft", 1.0), ("normal", 0.5), ("fine", 0.3))


class MoldPanel(QWidget):
    changed = Signal()  # any setting: the preview is refreshed
    auto_vents_requested = Signal()
    add_vent_toggled = Signal(bool)
    remove_requested = Signal()
    base_auto_requested = Signal()
    preview_requested = Signal()
    build_requested = Signal()

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        self._lock = False
        self._name = None
        self.base_auto = True  # base plane at the model's bottom until the user sets it
        self.angle_auto = True  # parting plane along the longer side until the user sets it
        self._info = None
        self._counts = (0, 0, 0)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.target = QLabel()
        self.target.setObjectName("title")
        self.target.setWordWrap(True)
        lay.addWidget(self.target)
        self.hint = QLabel()
        self.hint.setObjectName("dim")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        d = MoldParams()
        # silicone and shell
        self.shell_box = QGroupBox()
        f = QFormLayout(self.shell_box)
        self.thickness = self._spin(0.5, 40, 0.5, 2, " mm", d.thickness)
        self.wall = self._spin(0.8, 15, 0.5, 2, " mm", d.wall)
        self.quality = QComboBox()
        for key, h in QUALITY:
            self.quality.addItem("", h)
        self.quality.setCurrentIndex(1)
        self.thickness_label, self.wall_label, self.quality_label = QLabel(), QLabel(), QLabel()
        f.addRow(self.thickness_label, self.thickness)
        f.addRow(self.wall_label, self.wall)
        f.addRow(self.quality_label, self.quality)
        lay.addWidget(self.shell_box)

        # base
        self.base_box = QGroupBox()
        f = QFormLayout(self.base_box)
        row = QHBoxLayout()
        self.base_z = self._spin(-5000, 5000, 0.5, 2, " mm", 0.0)
        self.base_reset = QPushButton()
        self.base_reset.clicked.connect(self.base_auto_requested)
        row.addWidget(self.base_z, 1)
        row.addWidget(self.base_reset)
        self.base_z_label = QLabel()
        f.addRow(self.base_z_label, row)
        self.plate = QCheckBox()
        self.plate.setChecked(d.base_plate)
        f.addRow(self.plate)
        self.plate_margin = self._spin(2, 50, 1, 1, " mm", d.plate_margin)
        self.plate_thickness = self._spin(1, 20, 0.5, 1, " mm", d.plate_thickness)
        self.groove = self._spin(0.0, 10, 0.25, 2, " mm", d.groove_depth)
        self.plate_margin_label, self.plate_thickness_label, self.groove_label = QLabel(), QLabel(), QLabel()
        f.addRow(self.plate_margin_label, self.plate_margin)
        f.addRow(self.plate_thickness_label, self.plate_thickness)
        f.addRow(self.groove_label, self.groove)
        self.locator = QCheckBox()
        self.locator.setChecked(d.locator)
        f.addRow(self.locator)
        rrow = QHBoxLayout()
        self.rod = QCheckBox()
        self.rod_d = self._spin(1, 20, 0.5, 1, " mm", d.rod_diameter)
        rrow.addWidget(self.rod, 1)
        rrow.addWidget(self.rod_d)
        f.addRow(rrow)
        lay.addWidget(self.base_box)

        # halves
        self.split_box = QGroupBox()
        f = QFormLayout(self.split_box)
        self.split = QCheckBox()
        self.split.setChecked(d.split)
        f.addRow(self.split)
        self.angle = self._spin(-180, 180, 5, 1, "°", 0.0)
        self.offset = self._spin(-1000, 1000, 0.5, 2, " mm", 0.0)
        self.flange_w = self._spin(3, 40, 1, 1, " mm", d.flange_width)
        self.flange_t = self._spin(1.5, 15, 0.5, 1, " mm", d.flange_thickness)
        self.bolt = QComboBox()
        for b in ("M3", "M4", "none"):
            self.bolt.addItem("", b)
        self.bolt_spacing = self._spin(10, 200, 5, 0, " mm", d.bolt_spacing)
        self.angle_label, self.offset_label, self.flange_w_label = QLabel(), QLabel(), QLabel()
        self.flange_t_label, self.bolt_label, self.bolt_spacing_label = QLabel(), QLabel(), QLabel()
        f.addRow(self.angle_label, self.angle)
        f.addRow(self.offset_label, self.offset)
        f.addRow(self.flange_w_label, self.flange_w)
        f.addRow(self.flange_t_label, self.flange_t)
        f.addRow(self.bolt_label, self.bolt)
        f.addRow(self.bolt_spacing_label, self.bolt_spacing)
        self.nuts = QCheckBox()
        self.nuts.setChecked(d.nut_traps)
        self.keys = QCheckBox()
        self.keys.setChecked(d.keys)
        f.addRow(self.nuts)
        f.addRow(self.keys)
        lay.addWidget(self.split_box)

        # pouring
        self.pour_box = QGroupBox()
        f = QFormLayout(self.pour_box)
        self.sprue_d = self._spin(2, 40, 0.5, 1, " mm", d.sprue_diameter)
        self.chimney = self._spin(0, 60, 1, 1, " mm", d.chimney)
        self.vent_d = self._spin(0.8, 10, 0.25, 2, " mm", d.vent_diameter)
        self.sprue_d_label, self.chimney_label, self.vent_d_label = QLabel(), QLabel(), QLabel()
        f.addRow(self.sprue_d_label, self.sprue_d)
        f.addRow(self.chimney_label, self.chimney)
        self.funnel = QCheckBox()
        self.funnel.setChecked(d.funnel)
        f.addRow(self.funnel)
        f.addRow(self.vent_d_label, self.vent_d)
        vrow = QHBoxLayout()
        self.auto_vents = QPushButton()
        self.auto_vents.clicked.connect(self.auto_vents_requested)
        self.add_vent = QPushButton()
        self.add_vent.setCheckable(True)
        self.add_vent.toggled.connect(self.add_vent_toggled)
        self.remove = QPushButton()
        self.remove.clicked.connect(self.remove_requested)
        vrow.addWidget(self.auto_vents)
        vrow.addWidget(self.add_vent)
        vrow.addWidget(self.remove)
        f.addRow(vrow)
        self.marker_info = QLabel()
        self.marker_info.setObjectName("dim")
        self.marker_info.setWordWrap(True)
        f.addRow(self.marker_info)
        lay.addWidget(self.pour_box)

        # feet
        self.feet_box = QGroupBox()
        f = QFormLayout(self.feet_box)
        self.feet = QSpinBox()
        self.feet.setRange(0, 8)
        self.feet.setValue(d.feet)
        self.feet_h = self._spin(0, 100, 1, 1, " mm", d.feet_height)
        self.feet_d = self._spin(2, 40, 1, 1, " mm", d.feet_diameter)
        self.feet_spread = QSpinBox()
        self.feet_spread.setRange(10, 95)
        self.feet_spread.setSuffix(" %")
        self.feet_spread.setValue(int(d.feet_spread * 100))
        self.feet_label, self.feet_h_label, self.feet_d_label, self.feet_spread_label = (QLabel(), QLabel(), QLabel(),
                                                                                       QLabel())
        f.addRow(self.feet_label, self.feet)
        f.addRow(self.feet_h_label, self.feet_h)
        f.addRow(self.feet_d_label, self.feet_d)
        f.addRow(self.feet_spread_label, self.feet_spread)
        lay.addWidget(self.feet_box)

        # silicone
        self.sil_box = QGroupBox()
        f = QFormLayout(self.sil_box)
        self.density = self._spin(0.8, 2.5, 0.01, 2, " g/ml", d.density)
        self.density_label = QLabel()
        f.addRow(self.density_label, self.density)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("dim")
        f.addRow(self.info)
        lay.addWidget(self.sil_box)
        lay.addStretch(1)

        self.footer = QWidget()
        foot = QHBoxLayout(self.footer)
        foot.setContentsMargins(10, 6, 10, 10)
        self.preview_btn = QPushButton()
        self.preview_btn.clicked.connect(self.preview_requested)
        self.build_btn = QPushButton()
        self.build_btn.setObjectName("primary")
        self.build_btn.clicked.connect(self.build_requested)
        foot.addWidget(self.preview_btn)
        foot.addWidget(self.build_btn, 1)

        spins = (self.thickness, self.wall, self.plate_margin, self.plate_thickness, self.groove, self.rod_d,
                 self.flange_w, self.flange_t, self.bolt_spacing, self.sprue_d, self.chimney, self.vent_d,
                 self.feet_h, self.feet_d, self.density)
        for w in spins:
            w.valueChanged.connect(self._emit)
        for w in (self.feet, self.feet_spread):
            w.valueChanged.connect(self._emit)
        for w in (self.plate, self.locator, self.rod, self.split, self.nuts, self.keys, self.funnel):
            w.toggled.connect(self._emit)
        for w in (self.quality, self.bolt):
            w.currentIndexChanged.connect(self._emit)
        self.base_z.valueChanged.connect(self._base_edited)
        self.angle.valueChanged.connect(self._angle_edited)
        self.offset.valueChanged.connect(self._emit)
        self.split.toggled.connect(self._update_enabled)
        self.plate.toggled.connect(self._update_enabled)
        self.rod.toggled.connect(self._update_enabled)
        for w, k in ((self.thickness, "mold.thickness"), (self.wall, "mold.wall"), (self.quality, "mold.quality"),
                     (self.base_z, "mold.base_z"), (self.base_reset, "mold.base_z"), (self.plate, "mold.plate"),
                     (self.plate_margin, "mold.plate_margin"), (self.plate_thickness, "mold.plate_thickness"),
                     (self.groove, "mold.groove"), (self.locator, "mold.locator"), (self.rod, "mold.rod"),
                     (self.rod_d, "mold.rod"), (self.split, "mold.split"), (self.angle, "mold.angle"),
                     (self.offset, "mold.offset"), (self.flange_w, "mold.flange"), (self.flange_t, "mold.flange"),
                     (self.bolt, "mold.bolt"), (self.bolt_spacing, "mold.bolt"), (self.nuts, "mold.nuts"),
                     (self.keys, "mold.keys"), (self.sprue_d, "mold.sprue"), (self.chimney, "mold.chimney"),
                     (self.funnel, "mold.funnel"), (self.vent_d, "mold.vents"), (self.auto_vents, "mold.vents"),
                     (self.add_vent, "mold.add_vent"), (self.remove, "mold.add_vent"), (self.feet, "mold.feet"),
                     (self.feet_h, "mold.feet"), (self.feet_d, "mold.feet"), (self.feet_spread, "mold.feet"),
                     (self.density, "mold.density"), (self.preview_btn, "mold.preview"),
                     (self.build_btn, "mold.build")):
            help_register(w, k)
        self.retranslate()
        self._update_enabled()
        self.set_target(None)

    @staticmethod
    def _spin(lo, hi, step, decimals, suffix, value):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        s.setAlignment(Qt.AlignRight)
        s.setValue(value)
        return s

    def _emit(self, *_):
        if not self._lock:
            self.changed.emit()

    def _base_edited(self, *_):
        if not self._lock:
            self.base_auto = False
            self.changed.emit()

    def _angle_edited(self, *_):
        if not self._lock:
            self.angle_auto = False
            self.changed.emit()

    def _update_enabled(self, *_):
        for w in (self.angle, self.offset, self.flange_w, self.flange_t, self.bolt, self.bolt_spacing, self.nuts,
                  self.keys):
            w.setEnabled(self.split.isChecked())
        for w in (self.plate_margin, self.plate_thickness, self.groove, self.locator):
            w.setEnabled(self.plate.isChecked())
        self.rod_d.setEnabled(self.rod.isChecked())

    # -- state -------------------------------------------------------------------------
    def params(self, sprue=None, vents=None) -> MoldParams:
        return MoldParams(
            thickness=self.thickness.value(), wall=self.wall.value(), resolution=self.quality.currentData(),
            base_z=None if self.base_auto else self.base_z.value(),
            split=self.split.isChecked(), split_angle=None if self.angle_auto else self.angle.value(),
            split_offset=self.offset.value(), flange_width=self.flange_w.value(),
            flange_thickness=self.flange_t.value(), bolt=self.bolt.currentData(),
            bolt_spacing=self.bolt_spacing.value(), nut_traps=self.nuts.isChecked(), keys=self.keys.isChecked(),
            sprue=sprue, sprue_diameter=self.sprue_d.value(), chimney=self.chimney.value(),
            funnel=self.funnel.isChecked(), vents=vents, vent_diameter=self.vent_d.value(),
            feet=self.feet.value(), feet_height=self.feet_h.value(), feet_diameter=self.feet_d.value(),
            feet_spread=self.feet_spread.value() / 100.0, base_plate=self.plate.isChecked(),
            plate_margin=self.plate_margin.value(), plate_thickness=self.plate_thickness.value(),
            groove_depth=self.groove.value(), locator=self.locator.isChecked(), rod=self.rod.isChecked(),
            rod_diameter=self.rod_d.value(), density=self.density.value())

    def set_auto_values(self, base_z: float, angle: float) -> None:
        """Show the automatic base height and parting direction (without leaving auto mode)."""
        self._lock = True
        if self.base_auto:
            self.base_z.setValue(base_z)
        if self.angle_auto:
            self.angle.setValue(angle)
        self._lock = False

    def set_target(self, name: str | None) -> None:
        self._name = name
        self.target.setText(t("mold.target", name=name) if name else t("mold.none"))
        for w in (self.shell_box, self.base_box, self.split_box, self.pour_box, self.feet_box, self.sil_box,
                  self.preview_btn, self.build_btn):
            w.setEnabled(name is not None)

    def set_markers(self, sprue: bool, vents: int, feet: int) -> None:
        self._counts = (int(sprue), vents, feet)
        self.marker_info.setText(t("mold.markers", sprue=int(sprue), vents=vents, feet=feet))

    def set_info(self, info: dict | None) -> None:
        """info: {"silicone_ml", optional "silicone_g", "thickness", "size", "built"}"""
        self._info = info
        if not info:
            self.info.setText("")
            return
        n = i18n().num
        lines = []
        ml = info.get("silicone_ml_with_tubes", info.get("silicone_ml", 0.0))
        g = ml * self.density.value()
        lines.append(t("mold.silicone", ml=n(ml, "{:.1f}"), g=n(g, "{:.0f}")))
        th = info.get("thickness")
        if th:
            lines.append(t("mold.measured", min=n(th["min"], "{:.3f}"), max=n(th["max"], "{:.3f}")))
        size = info.get("size")
        if size:
            lines.append(t("mold.size", x=n(size[0], "{:.0f}"), y=n(size[1], "{:.0f}"), z=n(size[2], "{:.0f}")))
        if not info.get("built"):
            lines.append(t("mold.preview_note"))
        self.info.setText("\n".join(lines))

    def set_add_mode(self, on: bool) -> None:
        self.add_vent.blockSignals(True)
        self.add_vent.setChecked(on)
        self.add_vent.blockSignals(False)

    def set_busy(self, busy: bool) -> None:
        self.build_btn.setEnabled(not busy and self._name is not None)
        self.preview_btn.setEnabled(not busy and self._name is not None)

    def retranslate(self):
        self.hint.setText(t("mold.hint"))
        self.shell_box.setTitle(t("mold.shell_box"))
        self.thickness_label.setText(t("mold.thickness"))
        self.wall_label.setText(t("mold.wall"))
        self.quality_label.setText(t("mold.quality"))
        for i, (key, h) in enumerate(QUALITY):
            self.quality.setItemText(i, t("mold.quality_" + key, h=f"{h:g}"))
        self.base_box.setTitle(t("mold.base_box"))
        self.base_z_label.setText(t("mold.base_z"))
        self.base_reset.setText(t("mold.base_reset"))
        self.plate.setText(t("mold.plate"))
        self.plate_margin_label.setText(t("mold.plate_margin"))
        self.plate_thickness_label.setText(t("mold.plate_thickness"))
        self.groove_label.setText(t("mold.groove"))
        self.locator.setText(t("mold.locator"))
        self.rod.setText(t("mold.rod"))
        self.split_box.setTitle(t("mold.split_box"))
        self.split.setText(t("mold.split"))
        self.angle_label.setText(t("mold.angle"))
        self.offset_label.setText(t("mold.offset"))
        self.flange_w_label.setText(t("mold.flange_w"))
        self.flange_t_label.setText(t("mold.flange_t"))
        self.bolt_label.setText(t("mold.bolt"))
        for i, b in enumerate(("M3", "M4", "none")):
            self.bolt.setItemText(i, b if b != "none" else t("mold.bolt_none"))
        self.bolt_spacing_label.setText(t("mold.bolt_spacing"))
        self.nuts.setText(t("mold.nuts"))
        self.keys.setText(t("mold.keys"))
        self.pour_box.setTitle(t("mold.pour_box"))
        self.sprue_d_label.setText(t("mold.sprue"))
        self.chimney_label.setText(t("mold.chimney"))
        self.funnel.setText(t("mold.funnel"))
        self.vent_d_label.setText(t("mold.vent_d"))
        self.auto_vents.setText(t("mold.auto_vents"))
        self.add_vent.setText(t("mold.add_vent"))
        self.remove.setText(t("mold.remove"))
        self.feet_box.setTitle(t("mold.feet_box"))
        self.feet_label.setText(t("mold.feet"))
        self.feet_h_label.setText(t("mold.feet_h"))
        self.feet_d_label.setText(t("mold.feet_d"))
        self.feet_spread_label.setText(t("mold.feet_spread"))
        self.sil_box.setTitle(t("mold.sil_box"))
        self.density_label.setText(t("mold.density"))
        self.preview_btn.setText(t("mold.preview"))
        self.build_btn.setText(t("mold.build"))
        self.set_target(self._name)
        self.set_markers(*[bool(self._counts[0])] + list(self._counts[1:]))
        self.set_info(self._info)
