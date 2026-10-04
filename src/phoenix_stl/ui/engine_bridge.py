"""Qt signals on top of the engine process client."""
from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from ..engine import EngineClient, Event, Job

POLL_MS = 30


class EngineBridge(QObject):
    started = Signal(object)  # Job
    progress = Signal(object, float, str)  # Job, fraction, text
    finished = Signal(object, dict)  # Job, result
    failed = Signal(object, str, str)  # Job, code, detail
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = EngineClient()
        self._busy = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(POLL_MS)

    @property
    def busy(self) -> bool:
        return self.client.busy

    def submit(self, op: str, tag=None, **kwargs) -> Job:
        job = self.client.submit(op, tag, **kwargs)
        self._set_busy(True)
        return job

    def cancel(self) -> None:
        self._dispatch(self.client.cancel())
        self._set_busy(self.client.busy)

    def shutdown(self) -> None:
        self._timer.stop()
        self.client.shutdown()

    def _poll(self):
        self._dispatch(self.client.poll())
        self._set_busy(self.client.busy)

    def _dispatch(self, events: list[Event]):
        for ev in events:
            if ev.kind == "started":
                self.started.emit(ev.job)
            elif ev.kind == "progress":
                self.progress.emit(ev.job, ev.frac, ev.text)
            elif ev.kind == "result":
                self.finished.emit(ev.job, ev.result)
            elif ev.kind == "error":
                self.failed.emit(ev.job, ev.code, ev.detail)

    def _set_busy(self, busy: bool):
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)
