"""Engine process main loop: one job at a time, progress streamed back."""
from __future__ import annotations

import time
import traceback

from .ops import OPS, EngineError
from .store import PartStore

PROGRESS_INTERVAL = 0.1


def serve(conn, root: str) -> None:
    store = PartStore(root)
    conn.send(("ready",))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] == "quit":
            return
        _, job_id, op, kwargs = msg
        last = [0.0]

        def progress(frac, text="", _job=job_id):
            now = time.monotonic()
            if now - last[0] >= PROGRESS_INTERVAL or frac >= 1.0:
                last[0] = now
                conn.send(("progress", _job, float(frac), text))

        try:
            result = OPS[op](store, progress, **kwargs)
            conn.send(("result", job_id, result))
        except EngineError as e:
            conn.send(("error", job_id, e.code, e.detail))
        except MemoryError:
            store.drop_cache()
            conn.send(("error", job_id, "memory", ""))
        except Exception as e:  # report anything else instead of dying
            conn.send(("error", job_id, "exception", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
