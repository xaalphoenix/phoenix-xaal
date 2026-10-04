"""UI strings and hover-help texts in English and Persian, switchable live."""
from __future__ import annotations

import json
import os

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtWidgets import QApplication

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "i18n")
LANGS = {"en": "English", "fa": "فارسی"}
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


class I18n(QObject):
    changed = Signal(str)

    def __init__(self, lang: str | None = None):
        super().__init__()
        self._ui = {k: self._load(f"{k}.json") for k in LANGS}
        self._help = {k: self._load(f"help_{k}.json") for k in LANGS}
        saved = QSettings().value("language", "en")
        self.lang = lang if lang in LANGS else (saved if saved in LANGS else "en")

    @staticmethod
    def _load(name):
        with open(os.path.join(BASE, name), encoding="utf-8") as f:
            return json.load(f)

    @property
    def rtl(self) -> bool:
        return self.lang == "fa"

    def set_language(self, lang: str) -> None:
        if lang not in LANGS or lang == self.lang:
            return
        self.lang = lang
        QSettings().setValue("language", lang)
        self.apply_direction()
        self.changed.emit(lang)

    def apply_direction(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setLayoutDirection(Qt.RightToLeft if self.rtl else Qt.LeftToRight)

    def t(self, key: str, **kw) -> str:
        s = self._ui[self.lang].get(key) or self._ui["en"].get(key) or key
        return s.format(**kw) if kw else s

    def progress(self, text: str) -> str:
        return self.t("p:" + text) if ("p:" + text) in self._ui["en"] else text

    def help(self, key: str) -> dict | None:
        return self._help[self.lang].get(key) or self._help["en"].get(key)

    def num(self, value, fmt: str = "{:,.0f}") -> str:
        s = fmt.format(value)
        return s.replace(",", "٬").replace(".", "٫").translate(FA_DIGITS) if self.rtl else s


_INSTANCE: I18n | None = None


def i18n() -> I18n:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = I18n()
    return _INSTANCE


def init(lang: str | None = None) -> I18n:
    global _INSTANCE
    _INSTANCE = I18n(lang)
    _INSTANCE.apply_direction()
    return _INSTANCE


def t(key: str, **kw) -> str:
    return i18n().t(key, **kw)
