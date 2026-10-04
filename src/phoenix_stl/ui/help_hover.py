"""Hover help: rest the mouse on a setting, a ⋯ badge appears beside it, and
a moment later a guide explains what the setting does (current language).
Clicking the badge opens the guide immediately; F1 also works.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

from . import style
from .i18n import i18n

DOTS_DELAY_MS = 450
POPUP_DELAY_MS = 1100
HIDE_DELAY_MS = 350


class HelpPopup(QFrame):
    def __init__(self):
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setObjectName("helpPopup")
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        self.title = QLabel()
        self.title.setObjectName("title")
        self.body = QLabel()
        self.rec = QLabel()
        self.tip = QLabel()
        for w in (self.body, self.rec, self.tip):
            w.setWordWrap(True)
            w.setTextFormat(Qt.RichText)
        for w in (self.title, self.body, self.rec, self.tip):
            lay.addWidget(w)
        self.setMaximumWidth(360)

    def show_for(self, key: str, anchor: QWidget) -> bool:
        h = i18n().help(key)
        if not h:
            return False
        rtl = i18n().rtl
        self.setLayoutDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)
        align = Qt.AlignRight if rtl else Qt.AlignLeft
        self.title.setText(h.get("title", ""))
        self.body.setText(h.get("body", ""))
        rec_label = "پیشنهاد" if rtl else "Recommended"
        tip_label = "نکته" if rtl else "Tip"
        self.rec.setText(f"<b style='color:{style.OK}'>{rec_label}:</b> {h['recommended']}" if h.get("recommended") else "")
        self.tip.setText(f"<b style='color:{style.WARN}'>{tip_label}:</b> {h['tip']}" if h.get("tip") else "")
        self.rec.setVisible(bool(h.get("recommended")))
        self.tip.setVisible(bool(h.get("tip")))
        for w in (self.title, self.body, self.rec, self.tip):
            w.setAlignment(align)
        self.adjustSize()
        top_left = anchor.mapToGlobal(QPoint(0, anchor.height() + 6))
        if rtl:
            top_left.setX(anchor.mapToGlobal(QPoint(anchor.width(), 0)).x() - self.width())
        screen = QApplication.screenAt(top_left) or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = min(max(top_left.x(), geo.left() + 4), geo.right() - self.width() - 4)
        y = top_left.y()
        if y + self.height() > geo.bottom():
            y = anchor.mapToGlobal(QPoint(0, 0)).y() - self.height() - 6
        self.move(x, y)
        self.show()
        self.raise_()
        return True


class HoverHelp(QObject):
    def __init__(self, window: QWidget):
        super().__init__(window)
        self.window = window
        self.keys: dict[QWidget, str] = {}
        self.popup = HelpPopup()
        self.dots = QLabel("⋯", window)
        self.dots.setObjectName("helpDots")
        self.dots.setCursor(Qt.PointingHandCursor)
        self.dots.hide()
        self.dots.installEventFilter(self)
        self.current: QWidget | None = None
        self._dots_timer = self._timer(DOTS_DELAY_MS, self._show_dots)
        self._popup_timer = self._timer(POPUP_DELAY_MS, self._show_popup)
        self._hide_timer = self._timer(HIDE_DELAY_MS, self._hide_if_away)
        QApplication.instance().installEventFilter(self)

    def _timer(self, ms, slot):
        tm = QTimer(self)
        tm.setSingleShot(True)
        tm.setInterval(ms)
        tm.timeout.connect(slot)
        return tm

    def register(self, widget: QWidget, key: str) -> None:
        self.keys[widget] = key
        widget.installEventFilter(self)
        widget.destroyed.connect(lambda *_: self.keys.pop(widget, None))

    def eventFilter(self, obj, ev):
        et = ev.type()
        if obj is self.dots:
            if et == QEvent.MouseButtonPress and self.current is not None:
                self._show_popup()
                return True
            if et == QEvent.Enter:
                self._hide_timer.stop()
            elif et == QEvent.Leave:
                self._hide_timer.start()
            return False
        if et == QEvent.KeyPress and ev.key() == Qt.Key_F1:
            w = QApplication.widgetAt(QCursor.pos())
            while w is not None and w not in self.keys:
                w = w.parentWidget()
            if w is not None:
                self.current = w
                self._show_dots()
                self._show_popup()
                return True
        if obj in self.keys:
            if et == QEvent.Enter:
                if self.current is not obj:
                    self._hide_now()
                self.current = obj
                self._hide_timer.stop()
                self._dots_timer.start()
                self._popup_timer.start()
            elif et == QEvent.Leave:
                self._dots_timer.stop()
                self._popup_timer.stop()
                self._hide_timer.start()
            elif et in (QEvent.MouseButtonPress, QEvent.Wheel):
                self._popup_timer.stop()
                self.popup.hide()
            elif et in (QEvent.Hide, QEvent.EnabledChange) and obj is self.current:
                self._hide_now()
        return False

    def _show_dots(self):
        w = self.current
        if w is None or not w.isVisible():
            return
        rtl = i18n().rtl
        self.dots.adjustSize()
        anchor = w.mapTo(self.window, QPoint(0, 0))
        y = anchor.y() + (w.height() - self.dots.height()) // 2
        x = anchor.x() - self.dots.width() - 2 if rtl else anchor.x() + w.width() + 2
        x = min(max(x, 0), self.window.width() - self.dots.width())
        if not rtl and x + self.dots.width() > self.window.width() - 2:
            x = anchor.x() + w.width() - self.dots.width()  # no room: overlay the edge
        self.dots.move(x, y)
        self.dots.show()
        self.dots.raise_()

    def _show_popup(self):
        if self.current is not None and self.current.isVisible():
            self._show_dots()
            self.popup.show_for(self.keys[self.current], self.current)

    def _hide_if_away(self):
        w = QApplication.widgetAt(QCursor.pos())
        if w is self.dots or (w is not None and self.current is not None
                              and (w is self.current or self.current.isAncestorOf(w))):
            return
        self._hide_now()

    def _hide_now(self):
        self.dots.hide()
        self.popup.hide()
        self.current = None
