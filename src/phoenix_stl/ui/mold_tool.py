"""Mold tab logic: live coarse preview, pour/vent markers, building the mold parts."""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from .. import APP_NAME
from . import style
from .i18n import i18n, t

OVERLAYS = ("mold_shell", "mold_split", "mold_feet")
VENT = "#4aa3ff"
FOOT = "#3fb68b"


class MoldTool:
    def __init__(self, win):
        self.win = win
        self.panel = win.mold
        self.vp = win.viewport
        self.pid = None
        self.data = None  # last preview result
        self.sprue = None  # (x, y, z) or None
        self.sprue_user = False
        self.vents = []  # [(x, y, z)]
        self.vents_user = False
        self.selected = None
        self._drag_base = None
        self._busy = False
        self._again = False
        self._shown = False
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(450)
        self._timer.timeout.connect(self.request_preview)
        self._built = False  # the panel shows the measured numbers of the last build
        p = self.panel
        p.changed.connect(self._changed)
        p.preview_requested.connect(self.request_preview)
        p.build_requested.connect(self.build)
        p.auto_vents_requested.connect(self.auto_vents)
        p.add_vent_toggled.connect(self.set_add_mode)
        p.remove_requested.connect(self.remove)
        p.base_auto_requested.connect(self._base_auto)

    def _changed(self):
        self._built = False
        self._timer.start()

    # -- activation -----------------------------------------------------------------
    def active(self) -> bool:
        return self.win._tab_is(self.panel)

    def refresh(self) -> None:
        if not self.active():
            if self._shown:
                self._hide()
            return
        self._shown = True
        part = self.win.parts.selected()
        if part is None or part.extra.get("mold"):
            # nothing to mold (or a mold part is selected)
            if part is None or self.pid not in self.win.parts.parts:
                self.pid = None
                self.data = None
                self.panel.set_target(None)
                self._draw()
            return
        if part.id != self.pid:
            self.pid = part.id
            self._built = False
            self.sprue, self.sprue_user, self.vents, self.vents_user = None, False, [], False
            self.selected = None
            self.data = None
            self.panel.base_auto = True
            self.panel.angle_auto = True
            self.panel.set_target(part.name)
            self.panel.set_info(None)
            self.request_preview()
        self._draw()

    def _hide(self):
        self._shown = False
        self.set_add_mode(False)
        for name in OVERLAYS:
            self.vp.set_overlay(name, None)
        if self.vp._handles is not None and self.vp._handles.get("owner") == "mold":
            self.vp.set_handles(None)

    # -- preview ------------------------------------------------------------------------
    def _params(self, for_build: bool = False):
        sprue = self.sprue[:2] if (self.sprue_user and self.sprue) else None
        vents = [v[:2] for v in self.vents] if self.vents_user else None
        p = self.panel.params(sprue, vents)
        if not for_build:
            p.resolution = 0.0  # engine picks a coarse size for the preview
        return p

    def request_preview(self):
        if self.pid is None or not self.active() or self.pid not in self.win.parts.parts:
            return
        if self._busy:
            self._again = True
            return
        self._busy = True
        self.win.engine.submit("mold_preview", tag={"pid": self.pid, "auto": True}, pid=self.pid,
                               params=self._params().to_dict())

    def done_preview(self, job, r):
        self._busy = False
        if job.tag["pid"] != self.pid:
            self._again = False
            self.request_preview()
            return
        self.data = r
        if not self.sprue_user:
            self.sprue = tuple(r["sprue"]) if r.get("sprue") else None
        if not self.vents_user:
            self.vents = [tuple(v) for v in r.get("vents", [])]
        self.panel.set_auto_values(r["z0"], r["split_angle"])
        if not self._built:
            self.panel.set_info({"silicone_ml": r["silicone_ml"], "size": r["size"]})
        self._draw()
        if self._again:
            self._again = False
            self.request_preview()

    def preview_failed(self, job):
        self._busy = False
        self._again = False

    # -- markers ----------------------------------------------------------------------------
    def _markers(self):
        out = []
        if self.sprue:
            out.append(("sprue", self.sprue))
        out.extend(("vent", v) for v in self.vents)
        return out

    def _select(self, i):
        self.selected = i
        kind, pos = self._markers()[i]
        self._drag_base = tuple(pos)
        self._draw()

    def _drag(self, i, delta):
        if self._drag_base is None:
            return
        kind, _ = self._markers()[i]
        x, y, z = self._drag_base
        new = (x + float(delta[0]), y + float(delta[1]), z)
        if kind == "sprue":
            self.sprue, self.sprue_user = new, True
        else:
            self.vents[i - (1 if self.sprue else 0)] = new
            self.vents_user = True
        self._draw()

    def _drag_done(self):
        if self.selected is not None:
            self._drag_base = tuple(self._markers()[self.selected][1])
            self._timer.start()

    def auto_vents(self):
        self.vents_user = False
        self.vents = []
        self.request_preview()

    def set_add_mode(self, on: bool):
        self.panel.set_add_mode(on)
        if on and self.active() and self.pid:
            self.vp.pick_surface(self._picked)
        else:
            self.vp.cancel_pick()

    def _picked(self, pid, point, normal):
        z = self.sprue[2] if self.sprue else float(point[2])
        self.vents.append((float(point[0]), float(point[1]), z))
        self.vents_user = True
        self.selected = len(self._markers()) - 1
        self._draw()
        self._timer.start()
        if self.panel.add_vent.isChecked():
            self.vp.pick_surface(self._picked)  # keep adding until the button is released

    def remove(self):
        if self.selected is None:
            return
        kind, _ = self._markers()[self.selected]
        if kind == "sprue":
            self.sprue, self.sprue_user = None, True
        else:
            del self.vents[self.selected - (1 if self.sprue else 0)]
            self.vents_user = True
        self.selected = None
        self._draw()
        self._timer.start()

    def _base_auto(self):
        self.panel.base_auto = True
        self.request_preview()

    # -- drawing -------------------------------------------------------------------------------
    def _draw(self):
        if not self.active() or self.data is None or self.pid not in self.win.parts.parts:
            for name in OVERLAYS:
                self.vp.set_overlay(name, None)
            if self.vp._handles is not None and self.vp._handles.get("owner") == "mold":
                self.vp.set_handles(None)
            self.panel.set_markers(False, 0, 0)
            return
        v, f = self.data["shell"]
        if len(f) and not self._built:  # once built, the real parts show the mold
            self.vp.set_overlay("mold_shell", pv.PolyData.from_regular_faces(np.asarray(v, np.float32),
                                                                              np.asarray(f)),
                                color=style.ACCENT, opacity=0.25, show_edges=False)
        else:
            self.vp.set_overlay("mold_shell", None)
        params = self.panel.params()
        if params.split:
            ang = math.radians(self.data["split_angle"] if params.split_angle is None else params.split_angle)
            n = np.array([math.cos(ang), math.sin(ang), 0.0])
            b = np.asarray(self.win.parts.parts[self.pid].bounds)
            c = b.mean(0) + params.split_offset * n
            size = np.asarray(self.data["size"], float)
            plane = pv.Plane(center=c, direction=n, i_size=float(size[2]) + 10, j_size=float(size[:2].max()) + 10)
            self.vp.set_overlay("mold_split", plane, color="white", opacity=0.12, show_edges=False)
        else:
            self.vp.set_overlay("mold_split", None)
        feet = self.data.get("feet") or []
        if feet:
            pts = np.array(feet, np.float32).reshape(-1, 3)
            self.vp.set_overlay("mold_feet", pv.PolyData(pts), on_top=True, color=FOOT, point_size=12,
                                render_points_as_spheres=True)
        else:
            self.vp.set_overlay("mold_feet", None)
        marks = self._markers()
        pts = np.array([m[1] for m in marks], float).reshape(-1, 3)
        if self.vp._handles is None or self.vp._handles.get("owner") != "mold":
            self.vp.set_handles(pts, on_drag=self._drag, on_select=self._select, on_done=self._drag_done,
                                plane=(0, 0, 1))
            self.vp._handles["owner"] = "mold"
        colors = [style.ACCENT if k == "sprue" else VENT for k, _ in marks]
        self.vp.update_handles(pts, self.selected, colors)
        self.panel.set_markers(self.sprue is not None, len(self.vents), len(feet))

    # -- build ---------------------------------------------------------------------------------
    def build(self):
        part = self.win.parts.parts.get(self.pid)
        if part is None:
            return
        self.set_add_mode(False)
        self._timer.stop()
        p = self._params(for_build=True)
        self.win.engine.submit("mold_build", tag={"pid": self.pid}, pid=self.pid, name=part.name,
                               params=p.to_dict(), preview_faces=self.win.preview_faces(3))

    def done_build(self, job, r):
        colors = {"A": "#e8a33d", "B": "#d9773a", "shell": "#e8a33d", "base": "#9aa0a6"}
        self.win._apply_change(r, "action.mold", meta=lambda i, item, old: {
            "color": colors.get(item.get("mold")), "status": "unknown", "extra": {"mold": item.get("mold")}})
        info = dict(r.get("info") or {})
        info["built"] = True
        self._built = True
        if self.data:
            info["size"] = self.data.get("size")
        self.panel.set_info(info)
        self._draw()
        n = i18n().num
        self.win.status("mold.done", ml=n(info.get("silicone_ml_with_tubes", 0.0), "{:.1f}"))
        for p in self.win.parts.parts.values():
            if p.extra.get("mold") and p.status == "unknown":
                self.win._analyze(p)
        notes = []
        for w in r.get("warnings") or []:
            notes.append(t("mold.warn_" + w["code"]))
        if notes:
            QMessageBox.information(self.win, APP_NAME, "\n".join(dict.fromkeys(notes)))
        # keep working on the model
        if self.pid in self.win.parts.parts:
            self.win.parts.select(self.pid)
