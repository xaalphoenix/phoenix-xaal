"""Image or text on the model's surface: source, relief, size and placement."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFontComboBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ...core.decal import DecalParams
from ..i18n import i18n, t

DETAIL = (("fine", 0.05), ("normal", 0.1), ("draft", 0.2))


class DecalPanel(QWidget):
    place_toggled = Signal(bool)
    open_image_requested = Signal()
    source_changed = Signal()  # text, font or image choice: the image must be made again
    changed = Signal()  # size, relief settings: preview only
    apply_requested = Signal()

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        self._lock = False
        self._name = None
        self._placed = False
        self._image_name = ""
        self._aspect = None  # rows / cols of the current image
        self._ready = False  # placed, with something to put there
        self._busy = False
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
        self.place = QPushButton()
        self.place.setCheckable(True)
        self.place.toggled.connect(self.place_toggled)
        lay.addWidget(self.place)
        self.place_info = QLabel()
        self.place_info.setObjectName("dim")
        lay.addWidget(self.place_info)

        d = DecalParams()
        # source
        self.src_box = QGroupBox()
        f = QVBoxLayout(self.src_box)
        row = QHBoxLayout()
        self.use_text = QRadioButton()
        self.use_image = QRadioButton()
        self.use_text.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.use_text)
        grp.addButton(self.use_image)
        row.addWidget(self.use_text)
        row.addWidget(self.use_image)
        f.addLayout(row)
        self.text = QLineEdit()
        self.text.setText("PHOENIX")
        f.addWidget(self.text)
        frow = QHBoxLayout()
        self.font_box = QFontComboBox()
        self.font_box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.font_box.setMinimumContentsLength(8)
        self.bold = QCheckBox()
        self.bold.setChecked(True)
        frow.addWidget(self.font_box, 1)
        frow.addWidget(self.bold)
        f.addLayout(frow)
        irow = QHBoxLayout()
        self.open_btn = QPushButton()
        self.open_btn.clicked.connect(self.open_image_requested)
        self.image_label = QLabel()
        self.image_label.setObjectName("dim")
        irow.addWidget(self.open_btn)
        irow.addWidget(self.image_label, 1)
        f.addLayout(irow)
        lay.addWidget(self.src_box)

        # relief
        self.relief_box = QGroupBox()
        f = QFormLayout(self.relief_box)
        self.mode = QComboBox()
        self.mode.addItem("", "emboss")
        self.mode.addItem("", "deboss")
        self.depth_spin = self._spin(0.05, 20, 0.1, 2, " mm", d.depth)
        self.kind = QComboBox()
        self.kind.addItem("", "logo")
        self.kind.addItem("", "texture")
        self.threshold = QSpinBox()
        self.threshold.setRange(1, 99)
        self.threshold.setSuffix(" %")
        self.threshold.setValue(int(d.threshold * 100))
        self.invert = QCheckBox()
        self.blur = self._spin(0, 20, 0.5, 1, " px", d.blur)
        self.tile = QSpinBox()
        self.tile.setRange(1, 20)
        self.mode_label, self.depth_label, self.kind_label = QLabel(), QLabel(), QLabel()
        self.threshold_label, self.blur_label, self.tile_label = QLabel(), QLabel(), QLabel()
        f.addRow(self.mode_label, self.mode)
        f.addRow(self.depth_label, self.depth_spin)
        f.addRow(self.kind_label, self.kind)
        f.addRow(self.threshold_label, self.threshold)
        f.addRow(self.invert)
        f.addRow(self.blur_label, self.blur)
        f.addRow(self.tile_label, self.tile)
        lay.addWidget(self.relief_box)

        # size and placement
        self.size_box = QGroupBox()
        f = QFormLayout(self.size_box)
        self.size_w = self._spin(1, 1000, 1, 1, " mm", d.width)
        self.size_h = self._spin(1, 1000, 1, 1, " mm", d.height)
        self.keep_aspect = QCheckBox()
        self.keep_aspect.setChecked(True)
        self.rotation = self._spin(-180, 180, 5, 1, "°", 0.0)
        self.projection = QComboBox()
        self.projection.addItem("", "flat")
        self.projection.addItem("", "cylinder")
        self.radius = self._spin(0, 5000, 1, 1, " mm", 0.0)
        self.detail = QComboBox()
        for key, h in DETAIL:
            self.detail.addItem("", h)
        self.detail.setCurrentIndex(1)
        self.width_label, self.height_label, self.rotation_label = QLabel(), QLabel(), QLabel()
        self.projection_label, self.radius_label, self.detail_label = QLabel(), QLabel(), QLabel()
        f.addRow(self.width_label, self.size_w)
        f.addRow(self.height_label, self.size_h)
        f.addRow(self.keep_aspect)
        f.addRow(self.rotation_label, self.rotation)
        f.addRow(self.projection_label, self.projection)
        f.addRow(self.radius_label, self.radius)
        f.addRow(self.detail_label, self.detail)
        self.estimate = QLabel()
        self.estimate.setObjectName("dim")
        self.estimate.setWordWrap(True)
        f.addRow(self.estimate)
        lay.addWidget(self.size_box)
        lay.addStretch(1)

        self.footer = QWidget()
        foot = QVBoxLayout(self.footer)
        foot.setContentsMargins(10, 6, 10, 10)
        self.apply_btn = QPushButton()
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self.apply_requested)
        foot.addWidget(self.apply_btn)

        for w in (self.use_text, self.use_image, self.bold, self.invert):
            w.toggled.connect(self._source)
        self.text.textChanged.connect(self._source)
        self.font_box.currentFontChanged.connect(self._source)
        for w in (self.depth_spin, self.rotation, self.radius):
            w.valueChanged.connect(self._emit)
        self.size_w.valueChanged.connect(self._width_changed)
        self.size_h.valueChanged.connect(self._height_changed)
        for w in (self.threshold, self.tile, self.blur):
            w.valueChanged.connect(self._source)
        for w in (self.mode, self.projection, self.detail):
            w.currentIndexChanged.connect(self._emit)
        self.kind.currentIndexChanged.connect(self._source)
        self.kind.currentIndexChanged.connect(self._sync_enabled)
        self.projection.currentIndexChanged.connect(self._sync_enabled)
        for w in (self.use_text, self.use_image):
            w.toggled.connect(self._sync_enabled)
        self.keep_aspect.toggled.connect(self._width_changed)
        for w, k in ((self.place, "decal.place"), (self.use_text, "decal.text"), (self.text, "decal.text"),
                     (self.font_box, "decal.font"), (self.bold, "decal.font"), (self.use_image, "decal.image"),
                     (self.open_btn, "decal.image"), (self.mode, "decal.mode"), (self.depth_spin, "decal.depth"),
                     (self.kind, "decal.kind"), (self.threshold, "decal.threshold"), (self.invert, "decal.invert"),
                     (self.blur, "decal.blur"), (self.tile, "decal.tile"), (self.size_w, "decal.size"),
                     (self.size_h, "decal.size"), (self.keep_aspect, "decal.size"), (self.rotation, "decal.rotation"),
                     (self.projection, "decal.projection"), (self.radius, "decal.projection"),
                     (self.detail, "decal.detail"), (self.apply_btn, "decal.apply")):
            help_register(w, k)
        self.retranslate()
        self.set_target(None)
        self._sync_enabled()

    def _sync_enabled(self, *_):
        text = self.use_text.isChecked()
        for w in (self.text, self.font_box, self.bold):
            w.setEnabled(text)
        self.threshold.setEnabled(self.kind.currentData() == "logo")
        self.radius.setEnabled(self.projection.currentData() == "cylinder")

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
            self._update_estimate()
            self.changed.emit()

    def _source(self, *_):
        if not self._lock:
            self.source_changed.emit()

    def _width_changed(self, *_):
        if self._lock:
            return
        if self.keep_aspect.isChecked() and self._aspect:
            self._lock = True
            self.size_h.setValue(max(1.0, self.size_w.value() * self._aspect))
            self._lock = False
        self._emit()

    def _height_changed(self, *_):
        if self._lock:
            return
        if self.keep_aspect.isChecked() and self._aspect:
            self._lock = True
            self.size_w.setValue(max(1.0, self.size_h.value() / self._aspect))
            self._lock = False
        self._emit()

    # -- state ------------------------------------------------------------------------
    def params(self) -> DecalParams:
        return DecalParams(width=self.size_w.value(), height=self.size_h.value(), rotation=self.rotation.value(),
                           depth=self.depth_spin.value(), mode=self.mode.currentData(), kind=self.kind.currentData(),
                           threshold=self.threshold.value() / 100.0, invert=self.invert.isChecked(),
                           blur=self.blur.value(), tile=self.tile.value(), projection=self.projection.currentData(),
                           radius=self.radius.value(), resolution=self.detail.currentData())

    def text_font(self) -> QFont:
        f = QFont(self.font_box.currentFont())
        f.setBold(self.bold.isChecked())
        return f

    def set_aspect(self, aspect: float | None) -> None:
        """rows / cols of the image; with Keep proportions the height follows the width."""
        self._aspect = aspect
        self._width_changed()

    def set_image_name(self, name: str) -> None:
        self._image_name = name
        self.image_label.setText(name or t("decal.no_image"))

    def set_target(self, name: str | None) -> None:
        self._name = name
        self.target.setText(t("decal.target", name=name) if name else t("decal.none"))
        for w in (self.place, self.src_box, self.relief_box, self.size_box):
            w.setEnabled(name is not None)
        self.set_placed(False)

    def set_placed(self, placed: bool, where: str = "") -> None:
        self._placed = placed
        self.place_info.setText(where if placed else t("decal.not_placed"))
        self._update_apply()

    def set_ready(self, ready: bool) -> None:
        """Placed on a part and the text or image is not empty."""
        self._ready = ready
        self._update_apply()

    def _update_apply(self):
        self.apply_btn.setEnabled(self._ready and self._placed and self._name is not None and not self._busy)

    def set_place_mode(self, on: bool) -> None:
        self.place.blockSignals(True)
        self.place.setChecked(on)
        self.place.blockSignals(False)
        self.place.setText(t("decal.placing") if on else t("decal.place"))

    def _update_estimate(self):
        p = self.params()
        from ...core.decal import voxel_count
        n = voxel_count(p)
        secs = max(2, int(n / 250_000))
        self.estimate.setText(t("decal.estimate", m=i18n().num(n / 1e6, "{:.1f}"), s=i18n().num(secs)))

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_apply()

    def retranslate(self):
        self.hint.setText(t("decal.hint"))
        self.set_place_mode(self.place.isChecked())
        self.src_box.setTitle(t("decal.src_box"))
        self.use_text.setText(t("decal.use_text"))
        self.use_image.setText(t("decal.use_image"))
        self.bold.setText(t("decal.bold"))
        self.open_btn.setText(t("decal.open"))
        self.text.setPlaceholderText(t("decal.text_placeholder"))
        self.relief_box.setTitle(t("decal.relief_box"))
        self.mode_label.setText(t("decal.mode"))
        self.mode.setItemText(0, t("decal.emboss"))
        self.mode.setItemText(1, t("decal.deboss"))
        self.depth_label.setText(t("decal.depth"))
        self.kind_label.setText(t("decal.kind"))
        self.kind.setItemText(0, t("decal.kind_logo"))
        self.kind.setItemText(1, t("decal.kind_texture"))
        self.threshold_label.setText(t("decal.threshold"))
        self.invert.setText(t("decal.invert"))
        self.blur_label.setText(t("decal.blur"))
        self.tile_label.setText(t("decal.tile"))
        self.size_box.setTitle(t("decal.size_box"))
        self.width_label.setText(t("decal.width"))
        self.height_label.setText(t("decal.height"))
        self.keep_aspect.setText(t("decal.keep_aspect"))
        self.rotation_label.setText(t("decal.rotation"))
        self.projection_label.setText(t("decal.projection"))
        self.projection.setItemText(0, t("decal.flat"))
        self.projection.setItemText(1, t("decal.cylinder"))
        self.radius_label.setText(t("decal.radius"))
        self.radius.setSpecialValueText(t("decal.auto"))
        self.detail_label.setText(t("decal.detail"))
        for i, (key, h) in enumerate(DETAIL):
            self.detail.setItemText(i, t("decal.detail_" + key, h=f"{h:g}"))
        self.apply_btn.setText(t("decal.apply"))
        self.set_image_name(self._image_name)
        self.target.setText(t("decal.target", name=self._name) if self._name else t("decal.none"))
        if not self._placed:
            self.place_info.setText(t("decal.not_placed"))
        self._update_estimate()
