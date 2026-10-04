"""Connectors between cut parts: type, size, fit and placement."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ...core import connectors as C
from ..i18n import i18n, t

# label key per type for the two size fields
SIZE_LABELS = {
    "dowel": ("conn.diameter", "conn.length"),
    "peg": ("conn.diameter", "conn.height"),
    "magnet": ("conn.diameter", "conn.thickness"),
    "rod": ("conn.diameter", "conn.length"),
    "key": ("conn.width", "conn.height"),
    "tongue": ("conn.width", "conn.height"),
    "dovetail": ("conn.narrow", "conn.height"),
}


class ConnectPanel(QWidget):
    find_requested = Signal()
    pick_requested = Signal()
    changed = Signal()  # type, size or fit changed: placements need checking
    auto_requested = Signal()
    add_mode_toggled = Signal(bool)
    remove_requested = Signal()
    clear_requested = Signal()
    angle_changed = Signal(float)
    apply_requested = Signal()
    coupon_requested = Signal()
    coupon_use_requested = Signal(int)  # best hole index (0-based)

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        self._lock = False
        self._names = []
        self._kind = None  # printer kind the fit defaults were last set for
        self._printer_name = ""
        self._joint_info = None
        self._coupon = None  # clearances of the last coupon
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

        row = QHBoxLayout()
        self.find_btn = QPushButton()
        self.find_btn.clicked.connect(self.find_requested)
        self.pick_btn = QPushButton()
        self.pick_btn.clicked.connect(self.pick_requested)
        row.addWidget(self.find_btn)
        row.addWidget(self.pick_btn)
        lay.addLayout(row)
        self.joint_label = QLabel()
        self.joint_label.setObjectName("dim")
        self.joint_label.setWordWrap(True)
        lay.addWidget(self.joint_label)

        # type and size
        self.type_box = QGroupBox()
        tf = QFormLayout(self.type_box)
        self.type = QComboBox()
        for k in C.TYPES:
            self.type.addItem("", k)
        self.type.currentIndexChanged.connect(self._type_changed)
        self.type_label = QLabel()
        tf.addRow(self.type_label, self.type)
        self.magnet = QComboBox()
        for d, h in C.MAGNETS:
            self.magnet.addItem(f"{d} × {h} mm", (d, h))
        self.magnet.addItem("", None)
        self.magnet.setCurrentIndex(C.MAGNETS.index((6, 3)))
        self.magnet.currentIndexChanged.connect(self._magnet_changed)
        self.magnet_label = QLabel()
        tf.addRow(self.magnet_label, self.magnet)
        self.size1 = self._spin(0.5, 100, 0.5, 2, " mm")
        self.size2 = self._spin(0.3, 200, 0.5, 2, " mm")
        self.edge = self._spin(0.0, 50, 0.25, 2, " mm")
        self.flare = self._spin(5, 35, 1, 1, "°")
        self.angle = self._spin(-180, 180, 5, 1, "°")
        self.size1_label, self.size2_label = QLabel(), QLabel()
        self.edge_label, self.flare_label, self.angle_label = QLabel(), QLabel(), QLabel()
        tf.addRow(self.size1_label, self.size1)
        tf.addRow(self.size2_label, self.size2)
        tf.addRow(self.edge_label, self.edge)
        tf.addRow(self.flare_label, self.flare)
        tf.addRow(self.angle_label, self.angle)
        self.male = QComboBox()
        self.male.addItem("", "A")
        self.male.addItem("", "B")
        self.male.currentIndexChanged.connect(self._emit)
        self.male_label = QLabel()
        tf.addRow(self.male_label, self.male)
        self.type_note = QLabel()
        self.type_note.setObjectName("dim")
        self.type_note.setWordWrap(True)
        tf.addRow(self.type_note)
        lay.addWidget(self.type_box)

        # fit
        self.fit_box = QGroupBox()
        ff = QFormLayout(self.fit_box)
        self.fit = QComboBox()
        for k in C.FITS:
            self.fit.addItem("", k)
        self.fit.setCurrentIndex(C.FITS.index("snug"))
        self.fit_label = QLabel()
        ff.addRow(self.fit_label, self.fit)
        self.clearance = self._spin(0.0, 1.0, 0.01, 2, " mm")
        self.depth_gap = self._spin(0.0, 3.0, 0.05, 2, " mm")
        self.chamfer = self._spin(0.0, 2.0, 0.05, 2, " mm")
        self.wall = self._spin(0.4, 10.0, 0.1, 2, " mm")
        self.clearance_label, self.depth_gap_label = QLabel(), QLabel()
        self.chamfer_label, self.wall_label = QLabel(), QLabel()
        ff.addRow(self.clearance_label, self.clearance)
        ff.addRow(self.depth_gap_label, self.depth_gap)
        ff.addRow(self.chamfer_label, self.chamfer)
        ff.addRow(self.wall_label, self.wall)
        self.fit_note = QLabel()
        self.fit_note.setObjectName("dim")
        self.fit_note.setWordWrap(True)
        ff.addRow(self.fit_note)
        lay.addWidget(self.fit_box)

        # placement
        self.place_box = QGroupBox()
        pf = QVBoxLayout(self.place_box)
        cf = QFormLayout()
        self.count = QSpinBox()
        self.count.setRange(1, 12)
        self.count.setValue(2)
        self.count_label = QLabel()
        cf.addRow(self.count_label, self.count)
        pf.addLayout(cf)
        self.auto_btn = QPushButton()
        self.auto_btn.clicked.connect(self.auto_requested)
        pf.addWidget(self.auto_btn)
        r2 = QHBoxLayout()
        self.add_btn = QPushButton()
        self.add_btn.setCheckable(True)
        self.add_btn.toggled.connect(self.add_mode_toggled)
        self.remove_btn = QPushButton()
        self.remove_btn.clicked.connect(self.remove_requested)
        self.clear_btn = QPushButton()
        self.clear_btn.clicked.connect(self.clear_requested)
        r2.addWidget(self.add_btn, 1)
        r2.addWidget(self.remove_btn)
        r2.addWidget(self.clear_btn)
        pf.addLayout(r2)
        self.boss = QCheckBox()
        self.boss.setChecked(True)
        self.boss.toggled.connect(self._emit)
        pf.addWidget(self.boss)
        self.place_info = QLabel()
        self.place_info.setObjectName("dim")
        self.place_info.setWordWrap(True)
        pf.addWidget(self.place_info)
        lay.addWidget(self.place_box)

        # tolerance coupon
        self.coupon_box = QGroupBox()
        kf = QVBoxLayout(self.coupon_box)
        self.coupon_btn = QPushButton()
        self.coupon_btn.clicked.connect(self.coupon_requested)
        kf.addWidget(self.coupon_btn)
        self.coupon_info = QLabel()
        self.coupon_info.setObjectName("dim")
        self.coupon_info.setWordWrap(True)
        kf.addWidget(self.coupon_info)
        r3 = QHBoxLayout()
        self.best_label = QLabel()
        self.best = QSpinBox()
        self.best.setRange(1, 5)
        self.best.setValue(3)
        self.use_btn = QPushButton()
        self.use_btn.clicked.connect(lambda: self.coupon_use_requested.emit(self.best.value() - 1))
        r3.addWidget(self.best_label)
        r3.addWidget(self.best)
        r3.addWidget(self.use_btn, 1)
        kf.addLayout(r3)
        lay.addWidget(self.coupon_box)
        lay.addStretch(1)

        self.footer = QWidget()
        foot = QVBoxLayout(self.footer)
        foot.setContentsMargins(10, 6, 10, 10)
        self.apply_btn = QPushButton()
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self.apply_requested)
        foot.addWidget(self.apply_btn)

        for w in (self.size1, self.size2, self.edge, self.flare, self.clearance, self.depth_gap, self.chamfer,
                  self.wall):
            w.valueChanged.connect(self._emit)
        self.count.valueChanged.connect(lambda _: self.auto_requested.emit())
        self.angle.valueChanged.connect(lambda v: None if self._lock else self.angle_changed.emit(v))
        self.fit.currentIndexChanged.connect(self._fit_changed)
        for w, k in ((self.find_btn, "conn.find"), (self.pick_btn, "conn.pick"), (self.type, "conn.type"),
                     (self.magnet, "conn.magnet"), (self.size1, "conn.size1"), (self.size2, "conn.size2"),
                     (self.edge, "conn.edge"), (self.flare, "conn.flare"), (self.angle, "conn.angle"),
                     (self.male, "conn.male"), (self.fit, "conn.fit"), (self.clearance, "conn.clearance"),
                     (self.depth_gap, "conn.depth_gap"), (self.chamfer, "conn.chamfer"), (self.wall, "conn.wall"),
                     (self.count, "conn.count"), (self.auto_btn, "conn.auto"), (self.add_btn, "conn.add"),
                     (self.remove_btn, "conn.remove"), (self.clear_btn, "conn.clear"), (self.boss, "conn.boss"),
                     (self.coupon_btn, "conn.coupon"), (self.best, "conn.best"), (self.use_btn, "conn.best"),
                     (self.apply_btn, "conn.apply")):
            help_register(w, k)
        self._custom_fits = {}
        self._load_type_defaults()
        self.retranslate()
        self.set_target([])

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
    def current_type(self) -> str:
        return self.type.currentData() or "dowel"

    def spec(self) -> C.Spec:
        return C.Spec(self.current_type(), self.size1.value(), self.size2.value(), self.edge.value(),
                      self.flare.value())

    def fit_values(self) -> C.Fit:
        return C.Fit(self.clearance.value(), self.depth_gap.value(), self.chamfer.value(), self.wall.value())

    def male_side(self) -> str:
        return self.male.currentData() or "A"

    def set_target(self, names: list[str]) -> None:
        """names: [] (nothing usable), [a] (one part) or [a, b] (a joint)."""
        self._names = list(names)
        self._joint_error = None
        self._update_target_text()
        self.find_btn.setEnabled(len(names) == 2)
        self.pick_btn.setEnabled(len(names) >= 1)
        self._retranslate_male()
        self.set_joint(None)

    def _update_target_text(self):
        names = self._names
        if not names:
            self.target.setText(t("conn.none"))
        elif len(names) == 1:
            self.target.setText(t("conn.target_one", a=names[0]))
        else:
            self.target.setText(t("conn.target_two", a=names[0], b=names[1]))

    def set_joint(self, info: dict | None) -> None:
        """info: {"area", "hollow"} of the found joint face, or None."""
        self._joint_info = info
        has = info is not None
        for w in (self.type_box, self.fit_box, self.place_box, self.apply_btn):
            w.setEnabled(has)
        if not has:
            self.joint_label.setText(t(getattr(self, "_joint_error", None) or "conn.no_joint"))
            self.place_info.setText("")
            return
        self._joint_error = None
        n = i18n().num
        self.joint_label.setText(t("conn.joint_info", area=n(info["area"], "{:.0f}"))
                                 + (t("conn.hollow") if info.get("hollow") else ""))

    def set_joint_error(self, key: str) -> None:
        self._joint_error = key
        self.set_joint(None)

    def set_printer(self, kind: str, name: str, key: str, custom: dict) -> None:
        """Printer kind (resin/fdm) sets the fit defaults; custom = {fit: clearance} from the coupon."""
        changed_kind = kind != self._kind
        self._kind, self._printer_name, self._custom_fits = kind, name, dict(custom or {})
        self._lock = True
        if changed_kind:
            d = C.KIND_DEFAULTS.get(kind, C.KIND_DEFAULTS["resin"])
            self.depth_gap.setValue(d["depth_gap"])
            self.chamfer.setValue(d["chamfer"])
            self.wall.setValue(d["wall"])
        self._apply_fit_preset()
        self._lock = False
        self._update_fit_note()
        self.changed.emit()

    def _apply_fit_preset(self):
        fit = self.fit.currentData() or "snug"
        value = self._custom_fits.get(fit, C.FIT_PRESETS.get(self._kind, C.FIT_PRESETS["resin"])[fit])
        self.clearance.setValue(float(value))

    def _fit_changed(self, *_):
        self._lock = True
        self._apply_fit_preset()
        self._lock = False
        self._update_fit_note()
        self.changed.emit()

    def _update_fit_note(self):
        fit = self.fit.currentData() or "snug"
        kind = t("printer.kind_" + (self._kind or "resin"))
        note = t("conn.fit_note", kind=kind, name=self._printer_name)
        if fit in self._custom_fits:
            note += " " + t("conn.fit_custom")
        self.fit_note.setText(note)

    def _load_type_defaults(self):
        k = self.current_type()
        d = C.DEFAULTS[k]
        self._lock = True
        self.size1.setValue(d["diameter"])
        self.size2.setValue(d["length"])
        self.edge.setValue(d.get("edge", 1.5))
        self.flare.setValue(d.get("flare", 15.0))
        if k == "magnet":
            data = self.magnet.currentData()
            if data:
                self.size1.setValue(data[0])
                self.size2.setValue(data[1])
        self._lock = False

    def _type_changed(self, *_):
        self._load_type_defaults()
        self._update_type_ui()
        self.changed.emit()
        self.auto_requested.emit()

    def _magnet_changed(self, *_):
        data = self.magnet.currentData()
        if data:
            self._lock = True
            self.size1.setValue(data[0])
            self.size2.setValue(data[1])
            self._lock = False
            self.changed.emit()
            self.auto_requested.emit()

    def _update_type_ui(self):
        k = self.current_type()
        l1, l2 = SIZE_LABELS[k]
        self.size1_label.setText(t(l1))
        self.size2_label.setText(t(l2))
        for w, show in ((self.magnet, k == "magnet"), (self.magnet_label, k == "magnet"),
                        (self.edge, k in ("tongue", "dovetail")), (self.edge_label, k in ("tongue", "dovetail")),
                        (self.flare, k == "dovetail"), (self.flare_label, k == "dovetail"),
                        (self.angle, k in ("key", "dovetail")), (self.angle_label, k in ("key", "dovetail")),
                        (self.male, k in C.MALE_TYPES), (self.male_label, k in C.MALE_TYPES)):
            w.setVisible(show)
        self.place_box.setVisible(k != "tongue")
        self.type_note.setText(t("conn.note_" + k))

    def _retranslate_male(self):
        names = self._names + ["", ""]
        self.male.setItemText(0, t("conn.male_a", name=names[0]) if names[0] else "A")
        self.male.setItemText(1, t("conn.male_b", name=names[1]) if names[1] else "B")
        self.male.setEnabled(len(self._names) >= 1)

    def set_angle(self, value: float) -> None:
        self._lock = True
        self.angle.setValue(value)
        self._lock = False

    def set_add_mode(self, on: bool) -> None:
        self.add_btn.blockSignals(True)
        self.add_btn.setChecked(on)
        self.add_btn.blockSignals(False)

    def set_place_info(self, text: str) -> None:
        self.place_info.setText(text)

    def set_coupon(self, clearances) -> None:
        self._coupon = list(clearances) if clearances else None
        self.best.setRange(1, len(self._coupon) if self._coupon else 5)
        self.use_btn.setEnabled(self._coupon is not None)
        self._update_coupon_info()

    def _update_coupon_info(self):
        if not self._coupon:
            self.coupon_info.setText(t("conn.coupon_hint"))
            return
        n = i18n().num
        items = "، ".join if i18n().rtl else ", ".join
        self.coupon_info.setText(t("conn.coupon_info", list=items(n(c, "{:.2f}") for c in self._coupon)))

    def set_busy(self, busy: bool) -> None:
        self.apply_btn.setEnabled(not busy and self._joint_info is not None)
        self.coupon_btn.setEnabled(not busy)

    def retranslate(self):
        self.hint.setText(t("conn.hint"))
        self.find_btn.setText(t("conn.find"))
        self.pick_btn.setText(t("conn.pick"))
        self.type_box.setTitle(t("conn.type_box"))
        self.type_label.setText(t("conn.type"))
        for i, k in enumerate(C.TYPES):
            self.type.setItemText(i, t("conn.type_" + k))
        self.magnet_label.setText(t("conn.magnet"))
        self.magnet.setItemText(len(C.MAGNETS), t("conn.custom"))
        self.edge_label.setText(t("conn.edge"))
        self.flare_label.setText(t("conn.flare"))
        self.angle_label.setText(t("conn.angle"))
        self.male_label.setText(t("conn.male"))
        self._retranslate_male()
        self.fit_box.setTitle(t("conn.fit_box"))
        self.fit_label.setText(t("conn.fit"))
        for i, k in enumerate(C.FITS):
            self.fit.setItemText(i, t("conn.fit_" + k))
        self.clearance_label.setText(t("conn.clearance"))
        self.depth_gap_label.setText(t("conn.depth_gap"))
        self.chamfer_label.setText(t("conn.chamfer"))
        self.wall_label.setText(t("conn.wall"))
        self.place_box.setTitle(t("conn.place_box"))
        self.count_label.setText(t("conn.count"))
        self.auto_btn.setText(t("conn.auto"))
        self.add_btn.setText(t("conn.add"))
        self.remove_btn.setText(t("conn.remove"))
        self.clear_btn.setText(t("conn.clear"))
        self.boss.setText(t("conn.boss"))
        self.coupon_box.setTitle(t("conn.coupon_box"))
        self.coupon_btn.setText(t("conn.coupon"))
        self.best_label.setText(t("conn.best"))
        self.use_btn.setText(t("conn.use_best"))
        self.apply_btn.setText(t("conn.apply"))
        self._update_type_ui()
        self._update_fit_note()
        self._update_coupon_info()
        self._update_target_text()
        self.set_joint(self._joint_info)
