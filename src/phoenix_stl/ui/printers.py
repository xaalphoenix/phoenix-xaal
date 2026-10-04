"""Printer profiles: built-in ones plus the user's own (saved in settings)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLineEdit, QListWidget, QPushButton, QVBoxLayout)

from .i18n import t


@dataclass
class Printer:
    key: str
    name: str
    kind: str  # "resin" or "fdm"
    w: float
    d: float
    h: float
    margin: float  # kept free around parts (rafts, supports, brims)
    builtin: bool = False

    @property
    def volume(self):
        return (self.w, self.d, self.h)


BUILTIN = [
    Printer("saturn4u16k", "Elegoo Saturn 4 Ultra 16K", "resin", 211.68, 118.37, 220.0, 5.0, True),
    Printer("fdm_generic", "Generic FDM 220 × 220 × 250", "fdm", 220.0, 220.0, 250.0, 2.0, True),
]


def load_printers() -> list[Printer]:
    raw = QSettings().value("printers_custom", "[]")
    try:
        custom = [Printer(**{**d, "builtin": False}) for d in json.loads(raw)]
    except (TypeError, ValueError):
        custom = []
    return BUILTIN + custom


def save_custom(printers: list[Printer]) -> None:
    data = [{k: v for k, v in asdict(p).items() if k != "builtin"} for p in printers if not p.builtin]
    QSettings().setValue("printers_custom", json.dumps(data))


class PrinterDialog(QDialog):
    """Add, edit and delete custom printer profiles."""

    def __init__(self, parent=None, current_key: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(t("printer.edit_title"))
        self.setMinimumWidth(680)
        self.printers = load_printers()
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.list = QListWidget()
        left.addWidget(self.list)
        row = QHBoxLayout()
        self.add_btn = QPushButton(t("printer.add"))
        self.del_btn = QPushButton(t("printer.delete"))
        row.addWidget(self.add_btn)
        row.addWidget(self.del_btn)
        left.addLayout(row)
        lay.addLayout(left, 1)

        right = QVBoxLayout()
        form = QFormLayout()
        self.name = QLineEdit()
        self.kind = QComboBox()
        self.kind.addItem(t("printer.kind_resin"), "resin")
        self.kind.addItem(t("printer.kind_fdm"), "fdm")
        self.w, self.d, self.h = (self._spin(1, 2000) for _ in range(3))
        self.margin = self._spin(0, 50)
        form.addRow(t("printer.name"), self.name)
        form.addRow(t("printer.kind"), self.kind)
        form.addRow(t("printer.width"), self.w)
        form.addRow(t("printer.depth"), self.d)
        form.addRow(t("printer.height"), self.h)
        form.addRow(t("printer.margin"), self.margin)
        right.addLayout(form)
        right.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        right.addWidget(buttons)
        lay.addLayout(right, 1)

        self._loading = False
        self.list.currentRowChanged.connect(self._show)
        self.add_btn.clicked.connect(self._add)
        self.del_btn.clicked.connect(self._delete)
        for w in (self.w, self.d, self.h, self.margin):
            w.valueChanged.connect(self._edit)
        self.name.textEdited.connect(self._edit)
        self.kind.currentIndexChanged.connect(self._edit)
        self._refresh(next((i for i, p in enumerate(self.printers) if p.key == current_key), 0))

    @staticmethod
    def _spin(lo, hi):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(2)
        s.setSuffix(" mm")
        return s

    def _refresh(self, row: int):
        self.list.clear()
        for p in self.printers:
            self.list.addItem(p.name + ("" if not p.builtin else "  🔒"))
        self.list.setCurrentRow(max(0, min(row, len(self.printers) - 1)))

    def _show(self, row: int):
        if row < 0:
            return
        p = self.printers[row]
        self._loading = True
        self.name.setText(p.name)
        self.kind.setCurrentIndex(max(0, self.kind.findData(p.kind)))
        self.w.setValue(p.w)
        self.d.setValue(p.d)
        self.h.setValue(p.h)
        self.margin.setValue(p.margin)
        for w in (self.name, self.kind, self.w, self.d, self.h, self.margin, self.del_btn):
            w.setEnabled(not p.builtin)
        self._loading = False

    def _edit(self, *_):
        row = self.list.currentRow()
        if self._loading or row < 0 or self.printers[row].builtin:
            return
        p = self.printers[row]
        p.name = self.name.text().strip() or p.name
        p.kind = self.kind.currentData()
        p.w, p.d, p.h, p.margin = self.w.value(), self.d.value(), self.h.value(), self.margin.value()
        self.list.item(row).setText(p.name)
        save_custom(self.printers)

    def _add(self):
        base = self.printers[max(0, self.list.currentRow())]
        n = sum(1 for p in self.printers if not p.builtin) + 1
        self.printers.append(Printer(f"custom{n}_{len(self.printers)}", f"{t('printer.custom')} {n}", base.kind,
                                     base.w, base.d, base.h, base.margin, False))
        save_custom(self.printers)
        self._refresh(len(self.printers) - 1)

    def _delete(self):
        row = self.list.currentRow()
        if row >= 0 and not self.printers[row].builtin:
            del self.printers[row]
            save_custom(self.printers)
            self._refresh(row - 1)
