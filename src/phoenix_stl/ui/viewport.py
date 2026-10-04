"""3D view: parts, cutting-plane handle, live cross-section, printer box."""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkPropPicker, vtkRenderer

from . import style

pv.global_theme.allow_empty_mesh = True


def _blend(hex_color: str, toward: str, k: float):
    a = np.array(pv.Color(hex_color).float_rgb)
    b = np.array(pv.Color(toward).float_rgb)
    return tuple(a * (1 - k) + b * k)


class Viewport(QWidget):
    plane_moved = Signal(object, object)  # normal, origin (from the 3D handle)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        # No periodic auto-render: on weak GPUs every frame of a big model is costly,
        # so we only render when something changed (VTK renders interaction itself).
        self.plotter = QtInteractor(self, auto_update=False)
        lay.addWidget(self.plotter.interactor)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self.plotter.render)
        p = self.plotter
        p.set_background(style.VIEW_BG_BOTTOM, top=style.VIEW_BG_TOP)
        p.enable_anti_aliasing("fxaa")
        p.add_axes(line_width=2, labels_off=False)
        p.enable_lightkit()
        self.actors: dict[str, object] = {}
        self.colors: dict[str, str] = {}
        self._plane = None
        self._plane_lock = False
        self._section = None
        self._volume = None
        self._problems = None
        self._mouse_style = None  # interactor style our mouse observers sit on
        self._drag_targets: set[str] = set()
        self._drag_cb = None
        self._drag = None
        self._pick_cb = None
        self._click_cb = None  # capture left clicks (drawing)
        self._handles = None  # {"points", "axis", "on_drag", "on_select", "on_done"}
        self._handle_drag = None
        self._overlays: dict[str, object] = {}
        self._top = None  # renderer drawn over the model (curves and handles stay visible)

    # -- info -----------------------------------------------------------------
    def renderer_string(self) -> str:
        try:
            for line in self.plotter.ren_win.ReportCapabilities().splitlines():
                if "renderer string" in line.lower():
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return ""

    def render(self):
        """Request a render; several requests in one event become one frame."""
        if self._top is not None:
            self._top.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
        self._render_timer.start()

    def _top_renderer(self):
        if self._top is None:
            ren = vtkRenderer()
            rw = self.plotter.render_window
            layer = rw.GetNumberOfLayers()
            rw.SetNumberOfLayers(layer + 1)
            ren.SetLayer(layer)
            ren.SetInteractive(0)
            ren.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
            rw.AddRenderer(ren)
            self._top = ren
        return self._top

    # -- parts ------------------------------------------------------------------
    def set_part(self, pid: str, vertices, faces, color: str) -> None:
        self.remove_part(pid, render=False)
        poly = pv.PolyData.from_regular_faces(np.asarray(vertices, dtype=np.float32),
                                              np.asarray(faces, dtype=np.int64))
        actor = self.plotter.add_mesh(poly, color=color, name=f"part-{pid}", specular=0.25,
                                      specular_power=20, smooth_shading=False,
                                      reset_camera=False, pickable=True)
        self.actors[pid] = actor
        self.colors[pid] = color
        self.render()

    def remove_part(self, pid: str, render: bool = True) -> None:
        actor = self.actors.pop(pid, None)
        self.colors.pop(pid, None)
        if actor is not None:
            self.plotter.remove_actor(actor, render=False)
        if render:
            self.render()

    def set_visible(self, pid: str, visible: bool) -> None:
        if pid in self.actors:
            self.actors[pid].SetVisibility(visible)
            self.render()

    def set_focus(self, pids) -> None:
        """Dim every part except the selected ones (None: dim nothing)."""
        keep = None if pids is None else set(pids)
        for k, actor in self.actors.items():
            dim = keep is not None and k not in keep
            actor.prop.color = _blend(self.colors[k], "#3a3d44", 0.65) if dim else self.colors[k]
        self.render()

    def set_matrix(self, pid: str, matrix) -> None:
        """Show a part moved by a 4x4 matrix (None = as stored)."""
        actor = self.actors.get(pid)
        if actor is not None:
            actor.user_matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
            self.render()

    # -- mouse modes: drag parts, pick a surface ----------------------------------
    def set_drag(self, pids, on_move=None, on_done=None) -> None:
        """Left-drag on one of `pids` slides it on the bed (Shift: up/down).

        on_move(delta xyz, vertical) is called while dragging, on_done() at the
        end. Clicks elsewhere still orbit the camera. pids=None turns it off.
        """
        self._drag_targets = set(pids or [])
        self._drag_cb = (on_move, on_done)
        self._install_mouse()
        cursor = Qt.OpenHandCursor if self._drag_targets else Qt.ArrowCursor
        self.plotter.interactor.setCursor(cursor)

    def pick_surface(self, callback) -> None:
        """The next left click on a part calls callback(pid, point, normal)."""
        self._pick_cb = callback
        self._install_mouse()
        self.plotter.interactor.setCursor(Qt.CrossCursor)

    def cancel_pick(self) -> None:
        self._pick_cb = None
        self.set_drag(self._drag_targets, *(self._drag_cb or (None, None)))

    def _install_mouse(self) -> None:
        style = self.plotter.iren.interactor.GetInteractorStyle()
        if style is self._mouse_style:
            return
        # Observers on the style replace its default handlers; we call those
        # ourselves whenever the click is not ours.
        style.AddObserver("LeftButtonPressEvent", self._on_press)
        style.AddObserver("MouseMoveEvent", self._on_move)
        style.AddObserver("LeftButtonReleaseEvent", self._on_release)
        self._mouse_style = style

    def _pid_of(self, actor):
        for pid, a in self.actors.items():
            if a is actor:
                return pid
        return None

    def _on_press(self, style, _event):
        x, y = self.plotter.iren.interactor.GetEventPosition()
        if self._pick_cb is not None:
            cb, self._pick_cb = self._pick_cb, None
            self.set_drag(self._drag_targets, *(self._drag_cb or (None, None)))
            picker = vtkCellPicker()
            picker.SetTolerance(0.0005)
            if picker.Pick(x, y, 0, self.plotter.renderer):
                pid = self._pid_of(picker.GetActor())
                if pid is not None:
                    cb(pid, np.array(picker.GetPickPosition()), np.array(picker.GetPickNormal()))
            return
        if self._click_cb is not None:
            self._click_cb(x, y, bool(self.plotter.iren.interactor.GetShiftKey()))
            return
        if self._handles is not None:
            i = self._nearest_handle(x, y)
            if i is not None:
                h = self._handles
                t0 = self._axis_param(x, y, h["points"][i], h["axis"])
                self._handle_drag = {"i": i, "t0": t0}
                if h.get("on_select"):
                    h["on_select"](i)
                return
        if self._drag_targets:
            picker = vtkPropPicker()
            if picker.Pick(x, y, 0, self.plotter.renderer):
                pid = self._pid_of(picker.GetActor())
                if pid in self._drag_targets:
                    vertical = bool(self.plotter.iren.interactor.GetShiftKey())
                    start = np.array(picker.GetPickPosition())
                    self._drag = {"start": start, "vertical": vertical}
                    hit = self._plane_hit(x, y)
                    self._drag["start"] = hit if hit is not None else start
                    self.plotter.interactor.setCursor(Qt.ClosedHandCursor)
                    return
        style.OnLeftButtonDown()

    def _on_move(self, style, _event):
        if self._handle_drag is not None:
            x, y = self.plotter.iren.interactor.GetEventPosition()
            h, d = self._handles, self._handle_drag
            t = self._axis_param(x, y, h["points"][d["i"]], h["axis"])
            if t is not None and d["t0"] is not None and h.get("on_drag"):
                h["on_drag"](d["i"], t - d["t0"])
            return
        if self._drag is None:
            style.OnMouseMove()
            return
        x, y = self.plotter.iren.interactor.GetEventPosition()
        hit = self._plane_hit(x, y)
        if hit is not None and self._drag_cb and self._drag_cb[0]:
            delta = hit - self._drag["start"]
            if self._drag["vertical"]:
                delta[:2] = 0.0
            else:
                delta[2] = 0.0
            self._drag_cb[0](delta, self._drag["vertical"])

    def _on_release(self, style, _event):
        if self._handle_drag is not None:
            self._handle_drag = None
            if self._handles and self._handles.get("on_done"):
                self._handles["on_done"]()
            return
        if self._drag is None:
            style.OnLeftButtonUp()
            return
        self._drag = None
        self.plotter.interactor.setCursor(Qt.OpenHandCursor)
        if self._drag_cb and self._drag_cb[1]:
            self._drag_cb[1]()

    def set_click_capture(self, callback) -> None:
        """Left clicks call callback(x, y, shift) instead of orbiting (None: off)."""
        self._click_cb = callback
        self._install_mouse()
        self.plotter.interactor.setCursor(Qt.CrossCursor if callback else Qt.ArrowCursor)

    def set_handles(self, points, axis=None, on_drag=None, on_select=None, on_done=None) -> None:
        """Draggable handle points; dragging moves along `axis` (None: off)."""
        if points is None:
            self._handles = None
            self.set_overlay("handles", None)
            return
        self._handles = {"points": np.asarray(points, float), "axis": np.asarray(axis, float),
                         "on_drag": on_drag, "on_select": on_select, "on_done": on_done}
        self._install_mouse()

    def update_handles(self, points, selected=None) -> None:
        if self._handles is None:
            return
        self._handles["points"] = np.asarray(points, float)
        cloud = pv.PolyData(np.asarray(points, np.float32))
        colors = np.tile(np.array(pv.Color(style.ACCENT).int_rgb, np.uint8), (len(points), 1))
        if selected is not None:
            colors[selected] = (255, 255, 255)
        cloud["rgb"] = colors
        self.set_overlay("handles", cloud, on_top=True, scalars="rgb", rgb=True, point_size=14,
                         render_points_as_spheres=True)

    def display_ray(self, x, y):
        """World-space (origin, direction) of the mouse ray at display (x, y)."""
        ren = self.plotter.renderer
        pts = []
        for z in (0.0, 1.0):
            ren.SetDisplayPoint(x, y, z)
            ren.DisplayToWorld()
            w = np.array(ren.GetWorldPoint(), float)
            pts.append(w[:3] / (w[3] or 1.0))
        return pts[0], pts[1] - pts[0]

    def world_to_display(self, pts) -> np.ndarray:
        ren = self.plotter.renderer
        out = []
        for p in np.asarray(pts, float):
            ren.SetWorldPoint(*p, 1.0)
            ren.WorldToDisplay()
            out.append(ren.GetDisplayPoint()[:2])
        return np.array(out)

    def camera_frame(self):
        """(view direction, right, up) of the current camera."""
        cam = self.plotter.camera
        d = np.array(cam.focal_point) - np.array(cam.position)
        d /= np.linalg.norm(d)
        up = np.array(cam.up)
        right = np.cross(d, up)
        right /= np.linalg.norm(right)
        return d, right, np.cross(right, d)

    def set_parallel(self, on: bool) -> None:
        if on:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.render()

    def _nearest_handle(self, x, y, radius: float = 14.0):
        pts = self._handles["points"]
        if not len(pts):
            return None
        disp = self.world_to_display(pts)
        d = np.linalg.norm(disp - np.array([x, y]), axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= radius else None

    def _axis_param(self, x, y, point, axis):
        """Parameter along the line point + t*axis closest to the mouse ray."""
        p0, d = self.display_ray(x, y)
        a = axis / np.linalg.norm(axis)
        dn = d / np.linalg.norm(d)
        w0 = point - p0
        b = float(a @ dn)
        denom = 1.0 - b * b
        if denom < 1e-9:
            return None
        return float((b * (dn @ w0) - (a @ w0)) / denom)

    def set_overlay(self, name: str, dataset, on_top: bool = False, **kw) -> None:
        """Show (or with None remove) a named helper object.

        on_top draws it over the model, so lines and handles behind it stay visible.
        """
        old = self._overlays.pop(name, None)
        if old is not None:
            self.plotter.remove_actor(old, render=False)
            if self._top is not None:
                self._top.RemoveActor(old)
        if dataset is not None and dataset.n_points:
            kw.setdefault("pickable", False)
            kw.setdefault("reset_camera", False)
            actor = self.plotter.add_mesh(dataset, name=f"overlay-{name}", **kw)
            if on_top:
                self.plotter.remove_actor(actor, render=False)
                self._top_renderer().AddActor(actor)
            self._overlays[name] = actor
        self.render()

    def _plane_hit(self, x, y):
        """Mouse ray hit on the drag plane (horizontal, or vertical facing the camera)."""
        ren = self.plotter.renderer
        pts = []
        for z in (0.0, 1.0):
            ren.SetDisplayPoint(x, y, z)
            ren.DisplayToWorld()
            w = np.array(ren.GetWorldPoint(), float)
            pts.append(w[:3] / (w[3] or 1.0))
        p0, d = pts[0], pts[1] - pts[0]
        start = self._drag["start"] if self._drag else None
        if start is None:
            return None
        if self._drag["vertical"]:
            n = np.array([d[0], d[1], 0.0])
            if np.linalg.norm(n) < 1e-12:
                return None
            n /= np.linalg.norm(n)
        else:
            n = np.array([0.0, 0.0, 1.0])
        denom = float(d @ n)
        if abs(denom) < 1e-12:
            return None
        t = float((start - p0) @ n) / denom
        return p0 + t * d

    def reset_view(self, pid: str | None = None) -> None:
        """Frame one part, or all visible parts (overlays like the printer box don't count)."""
        actors = [self.actors[pid]] if pid in self.actors else \
            [a for a in self.actors.values() if a.GetVisibility()]
        if not actors:
            return
        b = np.array([a.GetBounds() for a in actors])
        bounds = [b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max(), b[:, 4].min(), b[:, 5].max()]
        self.plotter.reset_camera(bounds=bounds)
        self.render()

    def view(self, which: str) -> None:
        {"top": self.plotter.view_xy, "front": self.plotter.view_xz,
         "side": self.plotter.view_yz, "iso": self.plotter.view_isometric}[which]()
        self.render()

    # -- cutting plane ------------------------------------------------------------
    def show_plane(self, normal, origin, bounds) -> None:
        self.hide_plane(render=False)
        self._plane = self.plotter.add_plane_widget(
            self._on_plane, normal=tuple(normal), origin=tuple(origin), bounds=bounds,
            factor=1.15, color=style.ACCENT, implicit=True, normal_rotation=True,
            outline_translation=False, origin_translation=True, interaction_event="always")
        w = self._plane
        accent = pv.Color(style.ACCENT).float_rgb
        for prop in (w.GetPlaneProperty(), w.GetSelectedPlaneProperty()):
            prop.SetColor(accent)
            prop.SetOpacity(0.22)
        w.GetEdgesProperty().SetOpacity(0.0)
        w.GetOutlineProperty().SetOpacity(0.0)
        # pyvista hides the plane fill when a drag ends; keep it always visible
        w.SetDrawPlane(True)
        w.AddObserver("EndInteractionEvent", lambda obj, ev: obj.SetDrawPlane(True))
        w.GetSelectedOutlineProperty().SetOpacity(0.0)
        self.render()

    def set_plane(self, normal, origin) -> None:
        if self._plane is None:
            return
        self._plane_lock = True
        self._plane.SetNormal(*map(float, normal))
        self._plane.SetOrigin(*map(float, origin))
        self._plane_lock = False
        self.render()

    def hide_plane(self, render: bool = True) -> None:
        if self._plane is not None:
            self.plotter.clear_plane_widgets()
            self._plane = None
        self.set_section(None, render=render)

    def _on_plane(self, normal, origin):
        if not self._plane_lock:
            self.plane_moved.emit(np.array(normal, float), np.array(origin, float))

    def set_section(self, segments, render: bool = True) -> None:
        if self._section is not None:
            self.plotter.remove_actor(self._section, render=False)
            self._section = None
        if segments is not None and len(segments):
            lines = pv.line_segments_from_points(np.asarray(segments, dtype=np.float32).reshape(-1, 3))
            self._section = self.plotter.add_mesh(lines, color=style.SECTION, line_width=3,
                                                  name="section", reset_camera=False, pickable=False)
        if render:
            self.render()

    # -- overlays ---------------------------------------------------------------------
    def show_volume(self, box_bounds, fits: bool | None) -> None:
        if self._volume is not None:
            self.plotter.remove_actor(self._volume, render=False)
            self._volume = None
        if box_bounds is not None:
            color = style.OK if fits else style.BAD
            self._volume = self.plotter.add_mesh(pv.Box(bounds=box_bounds), style="wireframe",
                                                 color=color, line_width=1.5, opacity=0.6, name="volume",
                                                 reset_camera=False, pickable=False)
        self.render()

    def show_problems(self, segments) -> None:
        if self._problems is not None:
            self.plotter.remove_actor(self._problems, render=False)
            self._problems = None
        if segments is not None and len(segments):
            lines = pv.line_segments_from_points(np.asarray(segments, dtype=np.float32).reshape(-1, 3))
            self._problems = self.plotter.add_mesh(lines, color=style.PROBLEM, line_width=4,
                                                   name="problems", reset_camera=False, pickable=False)
        self.render()

    def close(self):
        self._render_timer.stop()
        self.plotter.close()
        super().close()
