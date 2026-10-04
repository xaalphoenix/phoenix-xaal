"""Parts list (visibility, rename, multi-select, merge, delete) plus printer and fit check."""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QStackedLayout, QVBoxLayout, QWidget)

from . import style
from .i18n import i18n, t
from .printers import PrinterDialog, load_printers


@dataclass
class Part:
    id: str
    name: str
    n_faces: int
    bounds: list
    color: str
    visible: bool = True
    status: str = "unknown"  # unknown | checking | printable | problems
    report: dict | None = None
    segments: object = None
    preview: object = None  # core Mesh used for live previews
    extra: dict = field(default_factory=dict)

    @property
    def size(self):
        return [self.bounds[1][k] - self.bounds[0][k] for k in range(3)]


def fit_state(size, volume) -> str:
    sx, sy, sz = size
    w, d, h = volume
    if sz > h:
        return "too_big"
    if sx <= w and sy <= d:
        return "fits"
    if sy <= w and sx <= d:
        return "fits_rotated"
    return "too_big"


def _swatch(color: str, status: str) -> QIcon:
    pm = QPixmap(22, 22)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(1, 1, 18, 18, 4, 4)
    dot = {"printable": style.OK, "problems": style.BAD, "checking": style.WARN}.get(status)
    if dot:
        p.setBrush(QColor(dot))
        p.setPen(QColor(style.PANEL))
        p.drawEllipse(12, 12, 9, 9)
    p.end()
    return QIcon(pm)


class PartsPanel(QWidget):
    selection_changed = Signal(object)  # current part id or None
    visibility_changed = Signal(str, bool)
    renamed = Signal(str, str, str)  # id, old name, new name
    delete_requested = Signal()
    merge_requested = Signal(str)  # "union" or "combine"
    printer_changed = Signal()
    volume_toggled = Signal(bool)

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        self.parts: dict[str, Part] = {}
        self._updating = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.printer_box = QGroupBox()
        form = QFormLayout(self.printer_box)
        prow = QHBoxLayout()
        self.printer = QComboBox()
        self.printer.currentIndexChanged.connect(self._printer_changed)
        self.printer_edit = QPushButton("…")
        self.printer_edit.setFixedWidth(32)
        self.printer_edit.clicked.connect(self._edit_printers)
        prow.addWidget(self.printer, 1)
        prow.addWidget(self.printer_edit)
        self.show_volume = QCheckBox()
        self.show_volume.setChecked(QSettings().value("show_volume", True, type=bool))
        self.show_volume.toggled.connect(self._volume_toggled)
        self.printer_label = QLabel()
        form.addRow(self.printer_label, prow)
        form.addRow(self.show_volume)
        lay.addWidget(self.printer_box)
        help_register(self.printer, "printer.profile")
        help_register(self.printer_edit, "printer.edit")
        help_register(self.show_volume, "printer.volume")
        self._load_printers(QSettings().value("printer", "saturn4u16k"))

        holder = QWidget()
        self.stack = QStackedLayout(holder)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setObjectName("dim")
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setEditTriggers(QListWidget.DoubleClicked | QListWidget.EditKeyPressed)
        self.list.currentItemChanged.connect(self._current_changed)
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.itemChanged.connect(self._item_changed)
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.list)
        lay.addWidget(holder, 1)
        help_register(self.list, "parts.list")

        row = QHBoxLayout()
        self.merge_btn = QPushButton()
        self.combine_btn = QPushButton()
        self.delete_btn = QPushButton()
        self.merge_btn.clicked.connect(lambda: self.merge_requested.emit("union"))
        self.combine_btn.clicked.connect(lambda: self.merge_requested.emit("combine"))
        self.delete_btn.clicked.connect(self.delete_requested)
        for b, key in ((self.merge_btn, "parts.merge"), (self.combine_btn, "parts.combine"),
                       (self.delete_btn, "parts.delete")):
            row.addWidget(b)
            help_register(b, key)
        lay.addLayout(row)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("dim")
        lay.addWidget(self.info)
        self.retranslate()

    # -- printer ------------------------------------------------------------------
    def _load_printers(self, select_key: str | None = None):
        self.printers = load_printers()
        self.printer.blockSignals(True)
        self.printer.clear()
        for p in self.printers:
            self.printer.addItem(p.name, p.key)
        self.printer.setCurrentIndex(max(0, self.printer.findData(select_key)))
        self.printer.blockSignals(False)

    def current_printer(self):
        return self.printers[max(0, self.printer.currentIndex())]

    def printer_volume(self):
        return self.current_printer().volume

    def _printer_changed(self, *_):
        QSettings().setValue("printer", self.printer.currentData())
        self._update_info()
        self.printer_changed.emit()

    def _edit_printers(self):
        key = self.printer.currentData()
        PrinterDialog(self, key).exec()
        self._load_printers(key)
        self._printer_changed()

    def _volume_toggled(self, on):
        QSettings().setValue("show_volume", on)
        self.volume_toggled.emit(on)

    # -- parts --------------------------------------------------------------------
    def next_color(self) -> str:
        used = {p.color for p in self.parts.values()}
        for c in style.PART_COLORS:
            if c not in used:
                return c
        return style.PART_COLORS[len(self.parts) % len(style.PART_COLORS)]

    def add(self, part: Part, select: bool = True, index: int | None = None) -> None:
        self.parts[part.id] = part
        item = QListWidgetItem()
        item.setData(Qt.UserRole, part.id)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
        if index is None or index >= self.list.count():
            self.list.addItem(item)
        else:
            self.list.insertItem(max(0, index), item)
        self._refresh_item(item)
        self.stack.setCurrentIndex(1)
        if select:
            self.list.setCurrentItem(item)

    def remove(self, pid: str) -> int:
        """Remove a part; returns the row it had (-1 if unknown)."""
        item = self._item(pid)
        row = -1
        if item is not None:
            row = self.list.row(item)
            self.list.takeItem(row)
        self.parts.pop(pid, None)
        if not self.parts:
            self.stack.setCurrentIndex(0)
            self._update_info()
        self._update_buttons()
        return row

    def row_of(self, pid: str) -> int:
        item = self._item(pid)
        return self.list.row(item) if item is not None else -1

    def update_part(self, pid: str) -> None:
        item = self._item(pid)
        if item is not None:
            self._refresh_item(item)
        if pid == self.selected_id():
            self._update_info()

    def select(self, pid: str, extend: bool = False) -> None:
        item = self._item(pid)
        if item is None:
            return
        if extend:
            item.setSelected(True)
        else:
            self.list.setCurrentItem(item)

    def select_many(self, pids: list[str]) -> None:
        self.list.clearSelection()
        for i, pid in enumerate(pids):
            item = self._item(pid)
            if item is not None:
                if i == 0:
                    self.list.setCurrentItem(item)
                item.setSelected(True)

    def selected_id(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def selected(self) -> Part | None:
        pid = self.selected_id()
        return self.parts.get(pid) if pid else None

    def selected_parts(self) -> list[Part]:
        """All selected parts in list order (the current part counts as selected)."""
        rows = sorted({self.list.row(i) for i in self.list.selectedItems()})
        out = [self.parts[self.list.item(r).data(Qt.UserRole)] for r in rows
               if self.list.item(r).data(Qt.UserRole) in self.parts]
        cur = self.selected()
        if cur is not None and cur not in out:
            out.insert(0, cur)
        return out

    def visible_parts(self) -> list[Part]:
        out = []
        for i in range(self.list.count()):
            p = self.parts.get(self.list.item(i).data(Qt.UserRole))
            if p is not None and p.visible:
                out.append(p)
        return out

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Delete and self.selected_id():
            self.delete_requested.emit()
            return
        super().keyPressEvent(ev)

    def _item(self, pid):
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == pid:
                return self.list.item(i)
        return None

    def _refresh_item(self, item):
        p = self.parts[item.data(Qt.UserRole)]
        self._updating = True
        item.setText(p.name)
        item.setIcon(_swatch(p.color, p.status))
        item.setCheckState(Qt.Checked if p.visible else Qt.Unchecked)
        item.setToolTip(f"{t('parts.triangles', n=i18n().num(p.n_faces))}\n{t('status.' + p.status)}")
        self._updating = False

    def _current_changed(self, cur, prev):
        self._update_info()
        self._update_buttons()
        self.selection_changed.emit(cur.data(Qt.UserRole) if cur is not None else None)

    def _update_buttons(self):
        n = len(self.selected_parts())
        self.merge_btn.setEnabled(n >= 2)
        self.combine_btn.setEnabled(n >= 2)
        self.delete_btn.setEnabled(n >= 1)

    def set_busy(self, busy: bool) -> None:
        if busy:
            for b in (self.merge_btn, self.combine_btn, self.delete_btn):
                b.setEnabled(False)
        else:
            self._update_buttons()

    def _item_changed(self, item):
        if self._updating:
            return
        pid = item.data(Qt.UserRole)
        p = self.parts.get(pid)
        if p is None:
            return
        visible = item.checkState() == Qt.Checked
        if visible != p.visible:
            p.visible = visible
            self.visibility_changed.emit(pid, visible)
        name = item.text().strip()
        if name and name != p.name:
            old, p.name = p.name, name
            self.renamed.emit(pid, old, name)
            self._update_info()

    def _update_info(self):
        p = self.selected()
        if p is None:
            self.info.setText("")
            return
        sx, sy, sz = p.size
        n = i18n().num
        fit = fit_state(p.size, self.printer_volume())
        color = {"fits": style.OK, "fits_rotated": style.WARN, "too_big": style.BAD}[fit]
        status_color = {"printable": style.OK, "problems": style.BAD}.get(p.status, style.TEXT_DIM)
        self.info.setText(
            f"<b>{p.name}</b><br>"
            f"{t('parts.size', x=n(sx, '{:.1f}'), y=n(sy, '{:.1f}'), z=n(sz, '{:.1f}'))}<br>"
            f"{t('parts.triangles', n=n(p.n_faces))}<br>"
            f"<span style='color:{status_color}'>● {t('status.' + p.status)}</span><br>"
            f"<span style='color:{color}'>● {t('printer.' + fit)}</span>")

    def retranslate(self):
        self.printer_box.setTitle(t("printer.label"))
        self.printer_label.setText(t("printer.label"))
        self.show_volume.setText(t("action.show_volume"))
        self.empty.setText(t("parts.empty"))
        self.merge_btn.setText(t("parts.merge_btn"))
        self.combine_btn.setText(t("parts.combine_btn"))
        self.delete_btn.setText(t("parts.delete_btn"))
        for i in range(self.list.count()):
            self._refresh_item(self.list.item(i))
        self._update_info()
        self._update_buttons()
