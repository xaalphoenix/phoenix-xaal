"""Talks to the engine process. Qt-free so tests and --selftest can use it.

Heavy work never runs in the UI process: it can't freeze the window, Cancel
simply kills the engine (parts are on disk, so a fresh engine carries on),
and running out of memory takes down only the engine.
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import tempfile
from collections import deque
from dataclasses import dataclass, field

from .server import serve
from .store import PartStore


@dataclass
class Job:
    id: int
    op: str
    kwargs: dict
    tag: object = None  # caller data handed back with the result


@dataclass
class Event:
    kind: str  # started | progress | result | error
    job: Job
    frac: float = 0.0
    text: str = ""
    result: dict = field(default_factory=dict)
    code: str = ""
    detail: str = ""


class EngineClient:
    def __init__(self, root: str | None = None):
        self.root = root or tempfile.mkdtemp(prefix="phoenix_stl_")
        self._ctx = mp.get_context("spawn")
        self._ids = itertools.count(1)
        self._queue: deque[Job] = deque()
        self._current: Job | None = None
        self._proc = None
        self._conn = None
        self._ready = False
        self._start()

    # -- process lifecycle -------------------------------------------------
    def _start(self) -> None:
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=serve, args=(child, self.root), daemon=True,
                                       name="phoenix-engine")
        self._proc.start()
        child.close()
        self._conn = parent
        self._ready = False

    def _restart(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(5)
        if self._conn is not None:
            self._conn.close()
        self._start()

    def shutdown(self, remove_files: bool = True) -> None:
        try:
            if self._proc is not None and self._proc.is_alive():
                self._conn.send(("quit",))
                self._proc.join(2)
                if self._proc.is_alive():
                    self._proc.kill()
        except (OSError, BrokenPipeError):
            pass
        if remove_files:
            PartStore(self.root).destroy()

    # -- jobs ----------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._current is not None or bool(self._queue)

    @property
    def current(self) -> Job | None:
        return self._current

    def submit(self, op: str, tag=None, **kwargs) -> Job:
        job = Job(next(self._ids), op, kwargs, tag)
        self._queue.append(job)
        return job

    def cancel(self) -> list[Event]:
        """Kill the running job (and drop queued ones)."""
        events = []
        if self._current is not None:
            events.append(Event("error", self._current, code="cancelled"))
            self._current = None
            self._restart()
        while self._queue:
            events.append(Event("error", self._queue.popleft(), code="cancelled"))
        return events

    def poll(self) -> list[Event]:
        """Non-blocking: deliver whatever happened since the last call."""
        events: list[Event] = []
        try:
            while self._conn.poll():
                msg = self._conn.recv()
                kind = msg[0]
                if kind == "ready":
                    self._ready = True
                    continue
                job = self._current
                if job is None or msg[1] != job.id:
                    continue
                if kind == "progress":
                    events.append(Event("progress", job, frac=msg[2], text=msg[3]))
                elif kind == "result":
                    events.append(Event("result", job, result=msg[2]))
                    self._current = None
                elif kind == "error":
                    events.append(Event("error", job, code=msg[2], detail=msg[3]))
                    self._current = None
        except (EOFError, OSError):
            pass
        if not self._proc.is_alive():
            if self._current is not None:
                events.append(Event("error", self._current, code="crashed",
                                    detail=f"exit code {self._proc.exitcode}"))
                self._current = None
            self._restart()
        if self._current is None and self._queue and self._ready:
            job = self._queue.popleft()
            self._current = job
            self._conn.send(("job", job.id, job.op, job.kwargs))
            events.append(Event("started", job))
        return events

    def wait(self, job: Job, timeout: float = 600.0) -> Event:
        """Blocking helper for scripts/tests: run until job finishes."""
        import time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for ev in self.poll():
                if ev.job.id == job.id and ev.kind in ("result", "error"):
                    return ev
            time.sleep(0.01)
        raise TimeoutError(f"job {job.op} timed out")
