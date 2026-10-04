"""Export parts as binary STL files."""
from __future__ import annotations

import os

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout, QWidget)

from ..i18n import t


class ExportPanel(QWidget):
    export_selected = Signal(str)
    export_all = Signal(str)

    def __init__(self, help_register, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.folder_label = QLabel()
        lay.addWidget(self.folder_label)
        row = QHBoxLayout()
        self.folder = QLineEdit(QSettings().value("export_folder", os.path.expanduser("~")))
        self.browse = QPushButton()
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(self.browse)
        lay.addLayout(row)
        self.check = QCheckBox()
        self.check.setChecked(True)
        lay.addWidget(self.check)
        self.sel_btn = QPushButton()
        self.sel_btn.setObjectName("primary")
        self.sel_btn.clicked.connect(lambda: self._go(self.export_selected))
        self.all_btn = QPushButton()
        self.all_btn.clicked.connect(lambda: self._go(self.export_all))
        lay.addWidget(self.sel_btn)
        lay.addWidget(self.all_btn)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("dim")
        lay.addWidget(self.note)
        lay.addStretch(1)
        for w, k in ((self.folder, "export.folder"), (self.browse, "export.folder"),
                     (self.check, "export.check"), (self.sel_btn, "export.selected"),
                     (self.all_btn, "export.all")):
            help_register(w, k)
        self.retranslate()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, t("export.folder"), self.folder.text())
        if d:
            self.folder.setText(d)

    def _go(self, signal):
        folder = self.folder.text().strip()
        QSettings().setValue("export_folder", folder)
        signal.emit(folder)

    def set_busy(self, busy: bool) -> None:
        self.sel_btn.setEnabled(not busy)
        self.all_btn.setEnabled(not busy)

    def retranslate(self):
        self.folder_label.setText(t("export.folder"))
        self.browse.setText(t("export.browse"))
        self.check.setText(t("export.check"))
        self.sel_btn.setText(t("export.selected"))
        self.all_btn.setText(t("export.all"))
        self.note.setText(t("export.note"))
        self.folder.setAlignment(Qt.AlignLeft)
