"""Connect tab logic: finds the joint, shows and edits connector positions, applies them."""
from __future__ import annotations

import json
import math

import manifold3d as m3d
import numpy as np
import pyvista as pv
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from .. import APP_NAME
from ..core import connectors as C
from ..core.section import polygons
from ..core.surface_cut import Frame
from . import style
from .i18n import i18n, t

OVERLAYS = ("conn_outline", "conn_safe", "conn_items")
BAD = "#e5484d"
OK_SAFE = "#3fb68b"


def load_custom_fits() -> dict:
    try:
        return json.loads(QSettings().value("fit_custom", "{}"))
    except (TypeError, ValueError):
        return {}


def save_custom_fit(printer_key: str, fit: str, value: float) -> None:
    data = load_custom_fits()
    data.setdefault(printer_key, {})[fit] = round(float(value), 3)
    QSettings().setValue("fit_custom", json.dumps(data))


def _fill(polys2d, frame: Frame, lift: float = 0.0) -> pv.PolyData | None:
    """Filled outline (with holes) as a flat mesh on the joint face."""
    polys = [np.asarray(p, float) for p in polys2d if len(p) >= 3]
    if not polys:
        return None
    try:
        tri = np.asarray(m3d.triangulate(polys))
    except Exception:
        return None
    if not len(tri):
        return None
    pts = np.vstack(polys)
    verts = frame.to_world(np.c_[pts, np.full(len(pts), lift)]).astype(np.float32)
    return pv.PolyData.from_regular_faces(verts, tri)


def _lines(polys2d, frame: Frame) -> pv.PolyData | None:
    polys = [np.asarray(p, float) for p in polys2d if len(p) >= 2]
    if not polys:
        return None
    out = []
    for p in polys:
        loop = np.vstack([p, p[:1]])
        out.append(pv.lines_from_points(frame.to_world(np.c_[loop, np.zeros(len(loop))]).astype(np.float32)))
    return pv.merge(out) if len(out) > 1 else out[0]


class ConnectTool:
    def __init__(self, win):
        self.win = win
        self.panel = win.conn
        self.vp = win.viewport
        self.joint = None  # {"pids", "names", "frame", "shape", "area", "hollow"}
        self.placements: list[C.Placement] = []
        self.ok: list[bool] = []
        self.notes: list[int] = []
        self.selected = None
        self._drag_base = None
        self._pending = None  # pids of a joint request in flight
        self._shown = False
        self._target = []  # pids the panel currently targets
        self._no_auto = False  # just applied: don't propose new connectors on the same joint
        p = self.panel
        p.find_requested.connect(self.find)
        p.pick_requested.connect(self.pick)
        p.changed.connect(self._panel_changed)
        p.auto_requested.connect(self.auto)
        p.add_mode_toggled.connect(self.set_add_mode)
        p.remove_requested.connect(self.remove)
        p.clear_requested.connect(self.clear)
        p.angle_changed.connect(self.set_angle)
        p.apply_requested.connect(self.apply)
        p.coupon_requested.connect(self.coupon)
        p.coupon_use_requested.connect(self.use_coupon)

    # -- activation ------------------------------------------------------------------
    def active(self) -> bool:
        return self.win._tab_is(self.panel)

    def _selected_pair(self):
        cur = self.win.parts.selected()
        sel = [p for p in self.win.parts.selected_parts() if p.visible]
        if cur is None or not sel:
            return []
        others = [p for p in sel if p.id != cur.id]
        return [cur] + others[:1] if len(sel) <= 2 else []

    def refresh(self) -> None:
        """Selection or tab changed."""
        if not self.active():
            if self._shown:
                self._hide()
            return
        self._shown = True
        parts = self._selected_pair()
        pids = [p.id for p in parts]
        if self.joint and self.joint["pids"] == pids:
            self._show()
            return
        if self.joint and len(self.joint["pids"]) == 1 and self.joint["pids"][0] in pids:
            self._show()  # single-face mode stays while that part is selected
            return
        self.joint = None
        self.placements = []
        self.selected = None
        self._target = pids
        self.panel.set_target([p.name for p in parts])
        self._show()
        if len(parts) == 2 and self._pending != pids:
            self.find(auto=True)

    def _hide(self):
        self._shown = False
        self.set_add_mode(False)
        self.vp.cancel_pick()
        for name in OVERLAYS:
            self.vp.set_overlay(name, None)
        if self.vp._handles is not None and self.vp._handles.get("owner") == "conn":
            self.vp.set_handles(None)

    # -- joint -----------------------------------------------------------------------
    def find(self, auto: bool = False):
        parts = self._selected_pair()
        if len(parts) != 2:
            return
        pids = [p.id for p in parts]
        self._pending = pids
        self.win.engine.submit("joint", tag={"pids": pids, "auto": auto}, pids=pids)

    def pick(self):
        if not self._selected_pair():
            return
        self.win.status("conn.picking")

        def picked(pid, point, normal):
            part = self.win.parts.parts.get(pid)
            if part is None:
                return
            self._pending = [pid]
            self.win.engine.submit("joint", tag={"pids": [pid]}, pids=[pid], point=point.tolist(),
                                   normal=normal.tolist())
        self.vp.pick_surface(picked)

    def done_joint(self, job, r):
        self._pending = None
        pids = r["pids"]
        parts = [self.win.parts.parts.get(pid) for pid in pids]
        if any(p is None for p in parts):
            return
        fr = Frame(*(np.asarray(r["frame"][k], float) for k in ("origin", "u", "v", "w")))
        self.joint = {"pids": pids, "names": [p.name for p in parts], "frame": fr,
                      "shape": C.JointShape.from_dict(r["shape"]), "area": r["area"], "hollow": r["hollow"]}
        self._target = pids
        self.panel.set_target(self.joint["names"])
        self.panel.set_joint({"area": r["area"], "hollow": r["hollow"]})
        if self._no_auto:  # stays set until the user changes a setting or places again
            self.placements, self.notes, self.selected = [], [], None
            self.validate()
        else:
            self.win.status("conn.found", area=i18n().num(r["area"], "{:.0f}"))
            self.auto()

    def joint_failed(self, job, code: str = ""):
        self._pending = None
        self._no_auto = False
        if code in ("no_joint", "no_face"):
            self.panel.set_joint_error("err." + code)

    # -- placements --------------------------------------------------------------------
    def _panel_changed(self):
        self._no_auto = False  # the user is setting up the next connectors
        self.validate()

    def auto(self):
        self._no_auto = False
        if not self.joint:
            return
        spec, fit = self.panel.spec(), self.panel.fit_values()
        self.placements, self.notes = C.auto_place(self.joint["shape"], spec, fit, self.panel.count.value(),
                                                   self.panel.boss.isChecked(), self.panel.male_side())
        angle = self.panel.angle.value()
        for p in self.placements:
            p.angle = angle
        self.selected = None
        self.validate()

    def validate(self):
        if not self.joint:
            self._draw()
            return
        spec, fit = self.panel.spec(), self.panel.fit_values()
        if spec.type == "tongue":
            self.ok = []
        else:
            self.ok = C.placement_ok(self.joint["shape"], spec, fit, self.placements, self.panel.boss.isChecked(),
                                     self.panel.male_side())
        self._update_info()
        self._draw()

    def _update_info(self):
        spec = self.panel.spec()
        if spec.type == "tongue":
            self.panel.set_place_info("")
            return
        text = t("conn.place_info", n=len(self.placements))
        bad = sum(1 for k in self.ok if not k)
        if bad:
            text += t("conn.place_bad", k=bad)
        if self.notes:
            text += t("conn.place_empty", k=len(self.notes))
        if spec.type == "dowel" and self.placements:
            text += "\n" + t("conn.pins_note", n=len(self.placements))
        self.panel.set_place_info(text)

    def set_add_mode(self, on: bool):
        self.panel.set_add_mode(on)
        if on and self.joint and self.active():
            self.vp.set_click_capture(self._click)
        elif self.vp._click_cb == self._click:
            self.vp.set_click_capture(None)

    def _click(self, x, y, shift):
        if not self.joint:
            return
        fr = self.joint["frame"]
        p0, d = self.vp.display_ray(x, y)
        denom = float(d @ fr.w)
        if abs(denom) < 1e-12:
            return
        hit = p0 + d * (float((fr.origin - p0) @ fr.w) / denom)
        u, v, _ = fr.to_local(hit[None])[0]
        shape = self.joint["shape"]
        inside = shape.levels[0][1] if shape.levels else shape.solid  # material or cavity
        if not C.contains(inside, [(u, v)])[0]:
            self.win.status("conn.outside")
            return
        self.placements.append(C.Placement(float(u), float(v), self.panel.angle.value()))
        self.selected = len(self.placements) - 1
        self.validate()

    def remove(self):
        if self.selected is not None and 0 <= self.selected < len(self.placements):
            del self.placements[self.selected]
            self.selected = None
            self.validate()

    def clear(self):
        self.placements = []
        self.selected = None
        self.validate()

    def set_angle(self, value: float):
        targets = [self.selected] if self.selected is not None else range(len(self.placements))
        for i in targets:
            self.placements[i].angle = value
        self._draw()

    def _select(self, i):
        self.selected = i
        p = self.placements[i]
        self._drag_base = (p.u, p.v)
        self.panel.set_angle(p.angle)
        self._draw()

    def _drag(self, i, delta):
        if self._drag_base is None or not self.joint:
            return
        fr = self.joint["frame"]
        du, dv = float(np.asarray(delta) @ fr.u), float(np.asarray(delta) @ fr.v)
        p = self.placements[i]
        p.u, p.v = self._drag_base[0] + du, self._drag_base[1] + dv
        self.validate()

    def _drag_done(self):
        if self.selected is not None:
            p = self.placements[self.selected]
            self._drag_base = (p.u, p.v)

    # -- drawing ------------------------------------------------------------------------
    def _show(self):
        self._draw()
        if self.panel.add_btn.isChecked():
            self.set_add_mode(True)

    def _draw(self):
        if not self.active() or not self.joint:
            for name in OVERLAYS:
                self.vp.set_overlay(name, None)
            if self.vp._handles is not None and self.vp._handles.get("owner") == "conn":
                self.vp.set_handles(None)
            return
        fr, shape = self.joint["frame"], self.joint["shape"]
        spec, fit = self.panel.spec(), self.panel.fit_values()
        self.vp.set_overlay("conn_outline", _lines(polygons(shape.solid), fr), on_top=True,
                            color=style.ACCENT, line_width=2)
        items = self._footprints(spec, fit)
        self.vp.set_overlay("conn_items", items, on_top=True, scalars="rgb", rgb=True, opacity=0.75,
                            show_scalar_bar=False) if items is not None else self.vp.set_overlay("conn_items", None)
        safe = None
        if spec.type != "tongue":
            r = C.footprint(spec, fit)
            regions = [reg for _, reg, _ in C.safe_regions(shape, r, self.panel.boss.isChecked(),
                                                             max(0.8, 0.35 * r),
                                                             C.boss_depth(spec, fit, self.panel.male_side()))]
            polys = [q for reg in regions for q in polygons(reg)]
            safe = _lines(polys, fr)
        self.vp.set_overlay("conn_safe", safe, on_top=True, color=OK_SAFE, line_width=1)
        pts = fr.to_world(np.array([[p.u, p.v, 0.0] for p in self.placements]).reshape(-1, 3))
        if self.vp._handles is None or self.vp._handles.get("owner") != "conn":
            self.vp.set_handles(pts, on_drag=self._drag, on_select=self._select, on_done=self._drag_done,
                                plane=fr.w)
            self.vp._handles["owner"] = "conn"
        colors = [style.ACCENT if (i >= len(self.ok) or self.ok[i]) else BAD for i in range(len(self.placements))]
        self.vp.update_handles(pts, self.selected, colors)

    def _footprints(self, spec: C.Spec, fit: C.Fit):
        """Coloured shapes of every connector on the joint face."""
        fr, shape = self.joint["frame"], self.joint["shape"]
        meshes = []
        if spec.type == "tongue" and not self._no_auto:
            band = C._band(shape, spec.edge, spec.diameter, depth=spec.length + fit.depth_gap)
            m = _fill(polygons(band), fr)
            if m is not None:
                m["rgb"] = np.tile(np.array(pv.Color(style.ACCENT).int_rgb, np.uint8), (m.n_points, 1))
                meshes.append(m)
        else:
            for i, p in enumerate(self.placements):
                if spec.type == "key":
                    cs = m3d.CrossSection.square((spec.diameter, spec.diameter), True).rotate(p.angle)
                elif spec.type == "dovetail":
                    reach = 2 * math.sqrt(max(shape.solid.area(), 1.0)) + 50
                    clip = shape.material(spec.length + fit.depth_gap).offset(-spec.edge, m3d.JoinType.Round)
                    cs = (m3d.CrossSection.square((2 * reach, spec.diameter), True).rotate(p.angle)
                          ^ clip.translate((-p.u, -p.v)))
                else:
                    cs = m3d.CrossSection.circle(C.hole_radius(spec, fit), 48)
                cs = cs.translate((p.u, p.v))
                m = _fill(polygons(cs), fr)
                if m is None:
                    continue
                color = "white" if i == self.selected else (style.ACCENT if (i >= len(self.ok) or self.ok[i])
                                                            else BAD)
                m["rgb"] = np.tile(np.array(pv.Color(color).int_rgb, np.uint8), (m.n_points, 1))
                meshes.append(m)
        if not meshes:
            return None
        return pv.merge(meshes) if len(meshes) > 1 else meshes[0]

    # -- apply ----------------------------------------------------------------------------
    def apply(self):
        if not self.joint:
            return
        spec, fit = self.panel.spec(), self.panel.fit_values()
        if (spec.type != "tongue" and not self.placements) or (spec.type == "tongue" and self._no_auto):
            QMessageBox.information(self.win, APP_NAME, t("conn.no_room" if spec.type != "tongue" else "conn.again"))
            return
        pids = self.joint["pids"]
        names = [self.win.parts.parts[pid].name for pid in pids if pid in self.win.parts.parts]
        if len(names) != len(pids):
            return
        self.set_add_mode(False)
        self.win.engine.submit(
            "connect", tag={"pids": pids, "type": spec.type, "n": len(self.placements)}, pids=pids, names=names,
            frame={k: np.asarray(getattr(self.joint["frame"], k), float).tolist() for k in ("origin", "u", "v", "w")},
            spec=vars(spec), fit=vars(fit),
            placements=[vars(p) for p in self.placements] if spec.type != "tongue" else [],
            male=self.panel.male_side(), boss=self.panel.boss.isChecked(),
            preview_faces=self.win.preview_faces(len(pids)))

    def done_connect(self, job, r):
        old = {pid: self.win.parts.parts.get(pid) for pid in r["removed"]}
        new = self.win._apply_change(r, "action.connect", meta=lambda i, item, o: {
            "color": o[i].color if i < len(o) else None})
        for p in new:
            self.win._analyze(p)
        self.joint = None
        self.placements = []
        self._no_auto = True
        if job.tag["type"] == "tongue":
            self.win.status("conn.done_tongue")
        else:
            self.win.status("conn.done", n=job.tag["n"])
        if r.get("warnings"):
            names = {s: (old[pid].name if old.get(pid) else "?") for s, pid in zip("AB", r["removed"])}
            lines = [t("conn.warn_line", part=names.get(w["side"], w["side"]),
                       items=", ".join(str(i + 1) for i in w["connectors"])) for w in r["warnings"]]
            QMessageBox.warning(self.win, APP_NAME, t("conn.warn_breakout", list="\n".join(lines)))
        self.refresh()

    # -- printer and coupon ------------------------------------------------------------------
    def printer_changed(self):
        pr = self.win.parts.current_printer()
        self.panel.set_printer(pr.kind, pr.name, pr.key, load_custom_fits().get(pr.key, {}))

    def coupon(self):
        spec, fit = self.panel.spec(), self.panel.fit_values()
        d = spec.diameter if spec.type in ("dowel", "peg", "rod", "magnet") else 4.0
        pr = self.win.parts.current_printer()
        step = 0.05 if pr.kind == "resin" else 0.1
        cl = C.coupon_clearances(fit.clearance, step)
        self._coupon = cl
        self.win.engine.submit("coupon", tag={"clearances": cl}, diameter=d, clearances=cl, chamfer=fit.chamfer,
                               name=t("conn.coupon_name", d=f"{d:g}"), preview_faces=self.win.preview_faces(2))

    def done_coupon(self, job, r):
        self.win._apply_change(r, "action.coupon", meta=lambda i, item, o: {"status": "printable"})
        self.panel.set_coupon(job.tag["clearances"])
        QSettings().setValue("last_coupon", json.dumps(job.tag["clearances"]))
        self.win.status("conn.coupon_done")

    def use_coupon(self, index: int):
        cl = self.panel._coupon
        if not cl or not 0 <= index < len(cl):
            return
        pr = self.win.parts.current_printer()
        fit = self.panel.fit.currentData()
        save_custom_fit(pr.key, fit, cl[index])
        self.printer_changed()
        self.win.status("conn.saved", c=i18n().num(cl[index], "{:.2f}"), fit="@conn.fit_" + fit, printer=pr.name)
