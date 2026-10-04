"""Undo/redo of part changes.

A step records which parts an action removed and added (metadata only, no
meshes). The engine keeps every part's files, so undo just swaps parts back;
files are purged once no step can bring them back any more.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

MAX_STEPS = 30


def snapshot(part):
    """Metadata copy of a part (drops the heavy preview mesh)."""
    return replace(part, preview=None, segments=None, extra=dict(part.extra))


@dataclass
class Step:
    label: str  # i18n key of the action, e.g. "action.cut"
    removed: list = field(default_factory=list)  # part snapshots
    added: list = field(default_factory=list)
    renames: list = field(default_factory=list)  # (pid, old name, new name)


class History:
    def __init__(self, purge):
        self._purge = purge  # callback(list of part ids)
        self.undo_steps: list[Step] = []
        self.redo_steps: list[Step] = []

    @property
    def can_undo(self) -> bool:
        return bool(self.undo_steps)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_steps)

    def push(self, step: Step) -> None:
        dropped = [p.id for s in self.redo_steps for p in s.added]
        self.redo_steps.clear()
        self.undo_steps.append(step)
        while len(self.undo_steps) > MAX_STEPS:
            dropped += [p.id for p in self.undo_steps.pop(0).removed]
        if dropped:
            self._purge(dropped)

    def take_undo(self) -> Step:
        step = self.undo_steps.pop()
        self.redo_steps.append(step)
        return step

    def take_redo(self) -> Step:
        step = self.redo_steps.pop()
        self.undo_steps.append(step)
        return step

    def label(self, redo: bool = False) -> str | None:
        steps = self.redo_steps if redo else self.undo_steps
        return steps[-1].label if steps else None
