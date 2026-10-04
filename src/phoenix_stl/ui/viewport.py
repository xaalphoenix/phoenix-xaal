"""3D view: parts, cutting-plane handle, live cross-section, printer box."""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

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
        self._render_timer.start()

    # -- parts ------------------------------------------------------------------
    def set_part(self, pid: str, vertices, faces, color: str) -> None:
        self.remove_part(pid, render=False)
        poly = pv.PolyData.from_regular_faces(np.asarray(vertices, dtype=np.float32),
                                              np.asarray(faces, dtype=np.int64))
        actor = self.plotter.add_mesh(poly, color=color, name=f"part-{pid}", specular=0.25,
                                      specular_power=20, smooth_shading=False,
                                      reset_camera=False, pickable=False)
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

    def set_focus(self, pid: str | None) -> None:
        """Dim every part except the selected one."""
        for k, actor in self.actors.items():
            dim = pid is not None and k != pid
            actor.prop.color = _blend(self.colors[k], "#3a3d44", 0.65) if dim else self.colors[k]
        self.render()

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
