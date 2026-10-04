"""Decal tab logic: text or image source, placing the frame on the surface, live preview, apply."""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter
from PySide6.QtWidgets import QFileDialog, QMessageBox
from scipy.spatial import cKDTree

from .. import APP_NAME
from ..core.decal import DecalParams, oriented_frame, prepare_image
from ..core.sdf import closest_on_triangles
from . import style
from .i18n import i18n, t

OVERLAYS = ("decal_patch", "decal_frame")
MAX_IMAGE = 1600  # px, longest side
TEXT_PX = 220
DEBOSS = "#4aa3ff"


def _to_array(img: QImage, channels: int) -> np.ndarray:
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes()).reshape(h, img.bytesPerLine())
    a = buf[:, :w * channels].reshape(h, w, channels) if channels > 1 else buf[:, :w]
    return a.copy()


def render_text(text: str, font: QFont) -> np.ndarray | None:
    """The text as white on black (uint8, rows top to bottom), cropped to the ink with a small margin."""
    text = text.strip()
    if not text:
        return None
    f = QFont(font)
    f.setPixelSize(TEXT_PX)
    fm = QFontMetricsF(f)
    pad = TEXT_PX // 2
    w = int(fm.horizontalAdvance(text) + fm.averageCharWidth()) + 2 * pad
    h = int(fm.height()) + 2 * pad
    img = QImage(w, h, QImage.Format_Grayscale8)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.setFont(f)
    p.setPen(QColor(255, 255, 255))
    p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, text)  # Qt joins Persian letters and orders RTL text
    p.end()
    a = _to_array(img, 1)
    rows = np.nonzero(a.max(axis=1) > 8)[0]
    cols = np.nonzero(a.max(axis=0) > 8)[0]
    if not len(rows):
        return None
    m = max(4, TEXT_PX // 20)
    r0, r1 = max(0, rows[0] - m), min(h, rows[-1] + 1 + m)
    c0, c1 = max(0, cols[0] - m), min(w, cols[-1] + 1 + m)
    return a[r0:r1, c0:c1]


def load_image(path: str):
    """(luminance, alpha) of an image file as float arrays 0..1, at most MAX_IMAGE px; None if unreadable."""
    img = QImage(path)
    if img.isNull():
        return None
    if max(img.width(), img.height()) > MAX_IMAGE:
        img = img.scaled(MAX_IMAGE, MAX_IMAGE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    a = _to_array(img.convertToFormat(QImage.Format_RGBA8888), 4).astype(np.float32) / 255.0
    lum = a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    return lum, a[..., 3]


def image_source(lum: np.ndarray, alpha: np.ndarray, kind: str) -> np.ndarray:
    """What gets raised (or pressed), 0..255.

    Logo: the dark parts on a light background, or the opaque parts of an image with
    transparency. Texture: brightness is height (as in height maps).
    """
    if kind == "texture":
        v = lum * alpha
    elif (alpha < 0.98).mean() > 0.01:
        v = alpha
    else:
        v = 1.0 - lum
    return np.round(np.clip(v, 0, 1) * 255).astype(np.uint8)


class DecalTool:
    def __init__(self, win):
        self.win = win
        self.panel = win.decal
        self.vp = win.viewport
        self.pid = None
        self.origin = None  # frame centre on the surface
        self.normal = None
        self._placements = {}  # pid -> (origin, normal), so undo brings the frame back
        self._file = None  # (luminance, alpha) of the opened image
        self._file_name = ""
        self.src = None  # uint8 image sent to the engine
        self._texture = None
        self._geo = None  # (pid, centroids, normals, areas, tree) of the part's preview
        self._drag_base = None
        self._shown = False
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._draw)
        p = self.panel
        p.place_toggled.connect(self.set_place_mode)
        p.open_image_requested.connect(self.open_image)
        p.source_changed.connect(self.update_source)
        p.changed.connect(self._timer.start)
        p.apply_requested.connect(self.apply)
        self.update_source()

    # -- activation -------------------------------------------------------------------
    def active(self) -> bool:
        return self.win._tab_is(self.panel)

    def refresh(self) -> None:
        if not self.active():
            if self._shown:
                self._hide()
            return
        self._shown = True
        part = self.win.parts.selected()
        pid = part.id if part is not None else None
        if pid != self.pid:
            self.pid = pid
            self._geo = None
            self.origin, self.normal = self._placements.get(pid, (None, None))
            self.panel.set_target(part.name if part is not None else None)
            self._placed_changed()
        self._draw()

    def _hide(self):
        self._shown = False
        self.set_place_mode(False)
        for name in OVERLAYS:
            self.vp.set_overlay(name, None)
        if self.vp._handles is not None and self.vp._handles.get("owner") == "decal":
            self.vp.set_handles(None)

    def retranslate(self):
        self._placed_changed()

    # -- source -----------------------------------------------------------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self.win, t("decal.open"), "",
                                              t("decal.image_filter") + " (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if not path:
            return
        data = load_image(path)
        if data is None:
            QMessageBox.warning(self.win, APP_NAME, t("decal.bad_image"))
            return
        self._file, self._file_name = data, os.path.basename(path)
        self.panel.set_image_name(self._file_name)
        if self.panel.use_image.isChecked():
            self.update_source()
        else:
            self.panel.use_image.setChecked(True)  # updates the source

    def set_image(self, lum: np.ndarray, alpha: np.ndarray | None = None, name: str = "image") -> None:
        """Use an image array directly (tests, paste)."""
        self._file = (np.asarray(lum, np.float32), np.ones_like(lum, np.float32) if alpha is None else alpha)
        self._file_name = name
        self.panel.set_image_name(name)
        self.panel.use_image.setChecked(True)
        self.update_source()

    def update_source(self):
        p = self.panel
        if p.use_text.isChecked():
            self.src = render_text(p.text.text(), p.text_font())
        elif self._file is not None:
            self.src = image_source(*self._file, p.kind.currentData())
        else:
            self.src = None
        self._texture = None
        if self.src is not None:
            p.set_aspect(self.src.shape[0] / self.src.shape[1])
        self._placed_changed()
        self._timer.start()

    def _make_texture(self, prm: DecalParams):
        """RGBA texture of what will be raised/pressed, with a transparent border."""
        a = prepare_image(self.src, prm)
        if prm.tile > 1:
            a = np.tile(a, (prm.tile, prm.tile))
        step = int(np.ceil(max(a.shape) / 1024))
        a = a[::step, ::step]
        a = np.pad(a, 2)
        rgba = np.zeros(a.shape + (4,), np.uint8)
        rgba[..., :3] = pv.Color(DEBOSS if prm.mode == "deboss" else style.ACCENT).int_rgb
        rgba[..., 3] = np.round(a * 235).astype(np.uint8)
        tex = pv.Texture(rgba)
        tex.interpolate = True
        tex.repeat = False
        tex.SetEdgeClamp(True)
        return tex, a.shape

    # -- placing ------------------------------------------------------------------------
    def set_place_mode(self, on: bool):
        on = bool(on and self.active() and self.pid is not None)
        self.panel.set_place_mode(on)
        if on:
            self.vp.pick_surface(self._picked)
        else:
            self.vp.cancel_pick()

    def _geometry(self):
        """Face centroids, unit normals and areas of the selected part's preview mesh."""
        part = self.win.parts.parts.get(self.pid)
        if part is None:
            return None
        if self._geo is None or self._geo[0] != self.pid:
            v = np.asarray(part.preview.vertices, np.float64)
            f = np.asarray(part.preview.faces)
            tri = v[f]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = np.linalg.norm(n, axis=1)
            n /= np.maximum(area, 1e-300)[:, None]
            c = tri.mean(axis=1)
            size = np.linalg.norm(tri - c[:, None], axis=2).max(axis=1)  # reach of each triangle from its centre
            self._geo = (self.pid, c, n, 0.5 * area, cKDTree(c), tri.astype(np.float32), size)
        return self._geo

    def _surface_frame(self, point, hint):
        """(point on the surface, averaged outward normal) near `point`.

        The normal is averaged over a patch about a third of the frame across, so a
        single odd triangle (scans, faceted models) does not tilt the frame.
        """
        g = self._geometry()
        if g is None:
            return None
        _, c, n, area, tree, tri, size = g
        point = np.asarray(point, float)
        _, idx = tree.query(point, k=min(12, len(c)))
        idx = np.atleast_1d(idx)
        q = closest_on_triangles(np.repeat(point[None], len(idx), 0), tri[idx, 0], tri[idx, 1], tri[idx, 2])
        j = int(np.argmin(np.linalg.norm(q - point, axis=1)))
        on = q[j]
        hint = np.asarray(hint if hint is not None else n[idx[j]], float)
        prm = self.panel.params()
        r = max(0.5, 0.3 * min(prm.width, prm.height))
        # triangles that come within r of the point (big CAD facets included)
        near = tree.query_ball_point(on, r + float(size.max()))
        if len(near) > 200_000:
            near = tree.query_ball_point(on, r)
        near = np.asarray(near, dtype=np.int64)
        near = near[np.linalg.norm(c[near] - on, axis=1) <= r + size[near]]
        if len(near):
            q = closest_on_triangles(np.repeat(on[None], len(near), 0), tri[near, 0], tri[near, 1], tri[near, 2])
            near = near[(np.linalg.norm(q - on, axis=1) <= r) & (n[near] @ hint > 0.2)]
        if len(near):
            s = (n[near] * area[near, None]).sum(axis=0)
            if np.linalg.norm(s) > 1e-12:
                return on, s / np.linalg.norm(s)
        return on, hint / max(np.linalg.norm(hint), 1e-12)

    def _picked(self, pid, point, normal):
        self.panel.set_place_mode(False)
        if pid != self.pid:
            self.win.parts.select(pid)  # refresh() follows the new part
            if pid != self.pid:
                return
        res = self._surface_frame(point, normal)
        if res is None:
            return
        self._set_placement(*res)

    def _set_placement(self, origin, normal):
        self.origin, self.normal = np.asarray(origin, float), np.asarray(normal, float)
        self._placements[self.pid] = (self.origin, self.normal)
        if self.vp._handles is not None and self.vp._handles.get("owner") == "decal":
            self.vp.set_handles(None)  # dragging works in the new tangent plane
        self._placed_changed()
        self._draw()

    def _placed_changed(self):
        placed = self.origin is not None and self.pid is not None
        n = i18n().num
        where = t("decal.placed", x=n(self.origin[0], "{:.1f}"), y=n(self.origin[1], "{:.1f}"),
                  z=n(self.origin[2], "{:.1f}")) if placed else ""
        self.panel.set_placed(placed, where)
        self.panel.set_ready(self.src is not None)

    def _select(self, i):
        self._drag_base = None if self.origin is None else self.origin.copy()

    def _drag(self, i, delta):
        if self._drag_base is None:
            return
        res = self._surface_frame(self._drag_base + np.asarray(delta, float), self.normal)
        if res is not None:
            self.origin, self.normal = res
            self._draw(handles=False)  # the handle stays in its drag plane until released

    def _drag_done(self):
        if self._drag_base is not None and self.origin is not None:
            self._set_placement(self.origin, self.normal)
        self._drag_base = None

    # -- preview -------------------------------------------------------------------------
    def frame(self, prm: DecalParams | None = None):
        prm = prm or self.panel.params()
        return oriented_frame(self.origin, self.normal, prm.rotation)

    def _draw(self, handles: bool = True):
        if not self.active() or self.origin is None or self.pid not in self.win.parts.parts:
            for name in OVERLAYS:
                self.vp.set_overlay(name, None)
            if self.vp._handles is not None and self.vp._handles.get("owner") == "decal":
                self.vp.set_handles(None)
            return
        prm = self.panel.params()
        fr = self.frame(prm)
        radius = prm.radius if prm.projection == "cylinder" else 0.0
        if prm.projection == "cylinder" and radius <= 0:
            radius = self._estimate_radius(fr, prm)
        self._draw_patch(fr, prm, radius)
        self._draw_outline(fr, prm, radius)
        if handles:
            if self.vp._handles is None or self.vp._handles.get("owner") != "decal":
                self.vp.set_handles(self.origin[None], on_drag=self._drag, on_select=self._select,
                                    on_done=self._drag_done, plane=self.normal)
                self.vp._handles["owner"] = "decal"
            self.vp.update_handles(self.origin[None], None, [style.ACCENT])

    def _estimate_radius(self, fr, prm) -> float:
        from ..core.decal import estimate_radius
        g = self._geometry()
        if g is None:
            return 0.0
        return estimate_radius(fr.to_local(g[1]), prm.width)

    @staticmethod
    def _unwrap(loc, radius):
        if radius <= 0:
            return loc[:, 0], loc[:, 1], loc[:, 2]
        rho = np.hypot(loc[:, 0], loc[:, 2] + radius)
        return radius * np.arctan2(loc[:, 0], loc[:, 2] + radius), loc[:, 1], rho - radius

    def _draw_patch(self, fr, prm, radius):
        """The part's own surface inside the frame, painted with the image."""
        g = self._geometry()
        if g is None or self.src is None:
            self.vp.set_overlay("decal_patch", None)
            return
        _, c, n, _, _, tri, size = g
        hw, hh = 0.5 * prm.width, 0.5 * prm.height
        reach = max(hw, hh) + prm.depth + 1.0
        # triangles that can reach the frame (big CAD facets too), then their extent in the frame
        cand = np.nonzero(np.linalg.norm(c - fr.origin, axis=1) <= np.hypot(hw, hh) + reach + size)[0]
        loc = fr.to_local(c[cand])
        nl = n[cand] @ np.stack([fr.u, fr.v, fr.w], axis=1)
        if radius > 0:
            out = np.stack([loc[:, 0], np.zeros(len(loc)), loc[:, 2] + radius], axis=1)
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
            facing = np.sum(nl * out, axis=1)
        else:
            facing = nl[:, 2]
        cand, loc = cand[facing > 0.15], loc[facing > 0.15]
        corners = fr.to_local(tri[cand].reshape(-1, 3).astype(np.float64))
        x, y, z = (a.reshape(-1, 3) for a in self._unwrap(corners, radius))
        pad = 0.02 * max(hw, hh)
        ok = ((x.max(1) >= -hw - pad) & (x.min(1) <= hw + pad) & (y.max(1) >= -hh - pad) & (y.min(1) <= hh + pad)
              & (z.max(1) >= -reach) & (z.min(1) <= reach))
        sel = cand[ok]
        if not len(sel):
            self.vp.set_overlay("decal_patch", None)
            return
        if self._texture is None or self._texture[0] != (prm.kind, prm.threshold, prm.invert, prm.blur,
                                                         prm.tile, prm.mode):
            self._texture = ((prm.kind, prm.threshold, prm.invert, prm.blur, prm.tile, prm.mode),
                             *self._make_texture(prm))
        _, tex, shape = self._texture
        pts = tri[sel].reshape(-1, 3)
        faces = np.arange(len(pts)).reshape(-1, 3)
        ux, uy, _ = self._unwrap(fr.to_local(pts), radius)
        bx, by = 2.0 / shape[1], 2.0 / shape[0]  # transparent border of the texture
        tc = np.stack([bx + (ux / prm.width + 0.5) * (1 - 2 * bx), by + (uy / prm.height + 0.5) * (1 - 2 * by)], 1)
        mesh = pv.PolyData.from_regular_faces(pts.astype(np.float32), faces)
        mesh.active_texture_coordinates = tc.astype(np.float32)  # outside the frame: the clear border
        self.vp.set_overlay("decal_patch", mesh, texture=tex, show_edges=False, smooth_shading=False)
        actor = self.vp._overlays.get("decal_patch")
        if actor is not None:
            m = actor.GetMapper()
            m.SetResolveCoincidentTopologyToPolygonOffset()
            m.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -60)
            self.vp.render()

    def _draw_outline(self, fr, prm, radius):
        hw, hh = 0.5 * prm.width, 0.5 * prm.height
        k = 24
        s = np.linspace(-1, 1, k)
        edge = np.concatenate([np.stack([s * hw, -hh + 0 * s], 1), np.stack([hw + 0 * s, s * hh], 1),
                               np.stack([-s * hw, hh + 0 * s], 1), np.stack([-hw + 0 * s, -s * hh], 1)])
        if radius > 0:
            th = edge[:, 0] / radius
            loc = np.stack([radius * np.sin(th), edge[:, 1], radius * np.cos(th) - radius], 1)
        else:
            loc = np.column_stack([edge, np.zeros(len(edge))])
        pts = fr.to_world(loc)
        line = pv.lines_from_points(np.vstack([pts, pts[:1]]))
        color = DEBOSS if prm.mode == "deboss" else style.ACCENT
        self.vp.set_overlay("decal_frame", line, on_top=True, color=color, line_width=2)

    # -- apply ---------------------------------------------------------------------------
    def apply(self):
        part = self.win.parts.parts.get(self.pid)
        if part is None or self.origin is None or self.src is None:
            return
        self.set_place_mode(False)
        prm = self.panel.params()
        self.win.engine.submit("decal", tag={"pid": self.pid}, pid=self.pid, name=part.name,
                               origin=[float(v) for v in self.origin], normal=[float(v) for v in self.normal],
                               image=self.src, params=prm.to_dict(), preview_faces=self.win.preview_faces())

    def done(self, job, r):
        def meta(i, item, old):
            if not old:
                return {"status": "unknown"}
            o = old[0]
            return {"color": o.color, "visible": True, "status": "unknown", "extra": dict(o.extra)}
        new = self.win._apply_change(r, "action.decal", meta=meta)
        for p in new:
            self.win._analyze(p)
        info = r.get("info") or {}
        n = i18n().num
        self.win.status("decal.done", v=n(abs(info.get("volume_change", 0.0)) / 1000.0, "{:.2f}"))
