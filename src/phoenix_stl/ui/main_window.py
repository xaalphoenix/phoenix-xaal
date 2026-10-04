"""Main window: wires parts, viewport, tool panels and the engine together."""
from __future__ import annotations

import os
import re

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QStyle,
                               QTabWidget, QWidget)

from .. import APP_NAME, __version__
from ..core import hardware
from ..core.cut import section_segments
from ..core.mesh import Mesh
from .engine_bridge import EngineBridge
from .help_hover import HoverHelp
from .i18n import LANGS, i18n, t
from .panels.cut_panel import CutPanel
from .panels.export_panel import ExportPanel
from .panels.repair_panel import RepairPanel
from .parts_panel import Part, PartsPanel, fit_state
from .viewport import Viewport

MIN_PREVIEW = 300_000


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    widget.setObjectName("toolPanel")
    area.setWidget(widget)
    return area


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "part"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, int(screen.width() * 0.92)), min(900, int(screen.height() * 0.9)))
        self.setAcceptDrops(True)
        self.help = HoverHelp(self)
        self.engine = EngineBridge(self)
        self._budget = None
        self._first_load = True
        self._last_msg = ("progress.idle", {})

        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)

        self.parts = PartsPanel(self.help.register)
        self.parts_dock = QDockWidget(self)
        self.parts_dock.setObjectName("parts")
        self.parts_dock.setWidget(self.parts)
        self.parts_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.parts_dock.setMinimumWidth(270)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.parts_dock)

        self.cut = CutPanel(self.help.register)
        self.repair = RepairPanel(self.help.register)
        self.export = ExportPanel(self.help.register)
        self.tabs = QTabWidget()
        self.tabs.addTab(_scroll(self.cut), "")
        self.tabs.addTab(_scroll(self.repair), "")
        self.tabs.addTab(_scroll(self.export), "")
        self.tools_dock = QDockWidget(self)
        self.tools_dock.setObjectName("tools")
        self.tools_dock.setWidget(self.tabs)
        self.tools_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.tools_dock.setMinimumWidth(330)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tools_dock)

        self._build_status()
        self._build_actions()
        self._connect()
        self._section_timer = QTimer(self)
        self._section_timer.setSingleShot(True)
        self._section_timer.setInterval(25)
        self._section_timer.timeout.connect(self._update_section)
        self._pending_plane = None
        i18n().changed.connect(self.retranslate)
        self.retranslate()
        self._set_busy_ui(False)

    # -- construction -----------------------------------------------------------
    def _build_status(self):
        sb = self.statusBar()
        self.msg = QLabel()
        sb.addWidget(self.msg, 1)
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self.progress_text = QLabel()
        self.progress_text.setObjectName("dim")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(False)
        self.cancel_btn = QPushButton()
        self.cancel_btn.clicked.connect(self.engine.cancel)
        for w in (self.progress_text, self.progress, self.cancel_btn):
            lay.addWidget(w)
        sb.addPermanentWidget(box)

    def _action(self, slot, shortcut=None, icon=None, checkable=False):
        a = QAction(self)
        if icon is not None:
            a.setIcon(self.style().standardIcon(icon))
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.triggered.connect(slot)
        return a

    def _build_actions(self):
        S = QStyle.StandardPixmap
        self.a_open = self._action(self.open_dialog, "Ctrl+O", S.SP_DialogOpenButton)
        self.a_export_sel = self._action(lambda: self.export_parts(selected_only=True), "Ctrl+E",
                                         S.SP_DialogSaveButton)
        self.a_export_all = self._action(lambda: self.export_parts(selected_only=False), "Ctrl+Shift+E")
        self.a_quit = self._action(self.close, "Ctrl+Q")
        self.a_analyze = self._action(self.analyze_selected, "Ctrl+I", S.SP_FileDialogContentsView)
        self.a_repair = self._action(lambda: self.repair_selected(self.repair.options()), "Ctrl+R",
                                     S.SP_BrowserReload)
        self.a_reset = self._action(lambda: self.viewport.reset_view(), "Home", S.SP_DesktopIcon)
        self.a_top = self._action(lambda: self.viewport.view("top"), "Ctrl+1")
        self.a_front = self._action(lambda: self.viewport.view("front"), "Ctrl+2")
        self.a_side = self._action(lambda: self.viewport.view("side"), "Ctrl+3")
        self.a_volume = self._action(lambda on: self.parts.show_volume.setChecked(on), checkable=True)
        self.a_volume.setChecked(self.parts.show_volume.isChecked())
        self.a_edges = self._action(lambda on: self.repair.show_edges.setChecked(on), checkable=True)
        self.a_edges.setChecked(True)
        self.a_delete = self._action(self.delete_selected, None)
        self.a_about = self._action(self.about)
        self.a_sysinfo = self._action(self.sysinfo)
        self.lang_actions = {}
        group = QActionGroup(self)
        for code, label in LANGS.items():
            a = QAction(label, self, checkable=True)
            a.setChecked(code == i18n().lang)
            a.triggered.connect(lambda _=False, c=code: i18n().set_language(c))
            group.addAction(a)
            self.lang_actions[code] = a

        mb = self.menuBar()
        self.m_file = mb.addMenu("")
        for a in (self.a_open, self.a_export_sel, self.a_export_all, None, self.a_delete, None, self.a_quit):
            self.m_file.addSeparator() if a is None else self.m_file.addAction(a)
        self.m_view = mb.addMenu("")
        for a in (self.a_reset, self.a_top, self.a_front, self.a_side, None, self.a_volume, self.a_edges):
            self.m_view.addSeparator() if a is None else self.m_view.addAction(a)
        self.m_lang = mb.addMenu("")
        for a in self.lang_actions.values():
            self.m_lang.addAction(a)
        self.m_help = mb.addMenu("")
        self.m_help.addAction(self.a_sysinfo)
        self.m_help.addAction(self.a_about)

        tb = self.addToolBar("main")
        tb.setObjectName("main")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_open, self.a_export_sel, None, self.a_analyze, self.a_repair, None, self.a_reset):
            tb.addSeparator() if a is None else tb.addAction(a)

    def _connect(self):
        e = self.engine
        e.started.connect(self._job_started)
        e.progress.connect(self._job_progress)
        e.finished.connect(self._job_finished)
        e.failed.connect(self._job_failed)
        e.busy_changed.connect(self._set_busy_ui)
        p = self.parts
        p.selection_changed.connect(self._selection_changed)
        p.visibility_changed.connect(self._visibility_changed)
        p.delete_requested.connect(lambda pid: self.delete_selected())
        p.printer_changed.connect(self._update_volume)
        p.volume_toggled.connect(self._volume_toggled)
        self.cut.plane_changed.connect(self._plane_changed)
        self.cut.gizmo_toggled.connect(lambda _: self._update_tool_overlays())
        self.cut.apply_requested.connect(self.cut_selected)
        self.viewport.plane_moved.connect(self.cut.set_from_handle)
        self.repair.analyze_requested.connect(self.analyze_selected)
        self.repair.repair_requested.connect(self.repair_selected)
        self.repair.show_edges_toggled.connect(self._edges_toggled)
        self.export.export_selected.connect(lambda folder: self.export_parts(True, folder))
        self.export.export_all.connect(lambda folder: self.export_parts(False, folder))
        self.tabs.currentChanged.connect(lambda _: self._update_tool_overlays())

    # -- helpers --------------------------------------------------------------------
    def budget(self) -> int:
        if self._budget is None:
            self._budget = hardware.display_budget(self.viewport.renderer_string())
        return self._budget

    def preview_faces(self, extra_parts: int = 1) -> int:
        n = len(self.parts.visible_parts()) + extra_parts
        return max(MIN_PREVIEW, self.budget() // max(1, n))

    def status(self, key: str, **kw) -> None:
        """Show a translated status message; it follows language switches."""
        self._last_msg = (key, kw)
        self.msg.setText(t(key, **kw))

    def _set_busy_ui(self, busy: bool):
        for w in (self.progress, self.progress_text, self.cancel_btn):
            w.setVisible(busy)
        if not busy:
            self.progress.setValue(0)
            self.progress_text.setText("")
        self.cut.set_busy(busy)
        self.export.set_busy(busy)
        has = self.parts.selected() is not None
        self.repair.set_enabled(has and not busy)
        for a in (self.a_analyze, self.a_repair, self.a_export_sel, self.a_export_all, self.a_delete):
            a.setEnabled(not busy and bool(self.parts.parts))

    def _error(self, code: str, detail: str = ""):
        if code == "memory" and "|" in detail:
            need, avail = detail.split("|")
            text = t("err.memory", need=need, avail=avail)
        elif code in ("crashed", "exception"):
            text = t("err." + code, detail=detail.splitlines()[0] if detail else "")
        else:
            text = t("err." + code)
        QMessageBox.warning(self, APP_NAME, text)

    # -- files ----------------------------------------------------------------------
    def open_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, t("action.open"), "", "STL (*.stl *.STL)")
        self.open_files(files)

    def open_files(self, paths):
        for path in paths:
            if not path.lower().endswith(".stl"):
                QMessageBox.information(self, APP_NAME, t("err.not_stl"))
                continue
            self.engine.submit("load", tag={"path": path}, path=path, preview_faces=self.preview_faces())

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        self.open_files([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])

    # -- actions ----------------------------------------------------------------------
    def analyze_selected(self, auto: bool = False):
        p = self.parts.selected()
        if p is not None:
            self._analyze(p, auto)

    def _analyze(self, p: Part, auto: bool = False):
        p.status = "checking"
        self.parts.update_part(p.id)
        if p.id == self.parts.selected_id():
            self.repair.set_report(None, checking=True)
        self.engine.submit("analyze", tag={"pid": p.id, "auto": auto}, pid=p.id)

    def repair_selected(self, options: dict):
        p = self.parts.selected()
        if p is not None:
            self.engine.submit("repair", tag={"pid": p.id}, pid=p.id, name=p.name, options=options,
                               preview_faces=self.preview_faces(0))

    def cut_selected(self, normal, origin, keep: str):
        p = self.parts.selected()
        if p is not None:
            self.engine.submit("cut", tag={"pid": p.id}, pid=p.id, name=p.name, normal=list(normal),
                               origin=list(origin), keep=keep, preview_faces=self.preview_faces(1))

    def delete_selected(self):
        p = self.parts.selected()
        if p is None or self.engine.busy:
            return
        if QMessageBox.question(self, APP_NAME, f"{t('action.delete')}: {p.name}?") != QMessageBox.Yes:
            return
        self.engine.submit("delete", tag={"pid": p.id}, pid=p.id)
        self._remove_part(p.id)
        self.status("msg.deleted", name=p.name)

    def export_parts(self, selected_only: bool, folder: str | None = None):
        folder = folder or self.export.folder.text().strip()
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, t("export.folder"))
            if not folder:
                return
            self.export.folder.setText(folder)
        os.makedirs(folder, exist_ok=True)
        parts = ([self.parts.selected()] if selected_only else self.parts.visible_parts())
        parts = [p for p in parts if p is not None]
        if not parts:
            QMessageBox.information(self, APP_NAME, t("export.nothing"))
            return
        if self.export.check.isChecked():
            unchecked = [p.name for p in parts if p.status != "printable"]
            if unchecked and QMessageBox.question(
                    self, APP_NAME, t("export.unchecked", names="\n".join(unchecked))) != QMessageBox.Yes:
                return
        items, used = [], set()
        for p in parts:
            base = safe_filename(p.name)
            name, k = base, 2
            while name.lower() in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name.lower())
            items.append((p.id, os.path.join(folder, name + ".stl")))
        existing = [path for _, path in items if os.path.exists(path)]
        if existing and QMessageBox.question(
                self, APP_NAME, t("export.overwrite", n=len(existing))) != QMessageBox.Yes:
            return
        self.engine.submit("export", tag={"folder": folder}, items=items)

    # -- engine events -------------------------------------------------------------
    def _job_started(self, job):
        self.progress.setValue(0)
        self.progress_text.setText("")

    def _job_progress(self, job, frac, text):
        self.progress.setValue(int(frac * 1000))
        self.progress_text.setText(f"{i18n().progress(text)}  {i18n().num(frac * 100)}%")

    def _job_finished(self, job, r: dict):
        handler = getattr(self, f"_done_{job.op}", None)
        if handler is not None:
            handler(job, r)

    def _job_failed(self, job, code, detail):
        if code == "cancelled":
            self.status("progress.cancelled")
        else:
            self._error(code, detail)
        if job.op == "analyze":
            p = self.parts.parts.get(job.tag["pid"])
            if p is not None:
                p.status = "unknown"
                self.parts.update_part(p.id)
                if p.id == self.parts.selected_id():
                    self.repair.set_report(None)

    def _add_part(self, info: dict, preview, color: str | None = None, status="unknown") -> Part:
        v, f = preview
        p = Part(id=info["id"], name=info["name"], n_faces=info["n_faces"], bounds=info["bounds"],
                 color=color or self.parts.next_color(), status=status, preview=Mesh(v, f))
        self.viewport.set_part(p.id, v, f, p.color)
        self.parts.add(p, select=False)
        return p

    def _remove_part(self, pid: str):
        self.viewport.remove_part(pid)
        self.parts.remove(pid)
        if not self.parts.parts:
            self._selection_changed(None)

    def _done_load(self, job, r):
        p = self._add_part(r["part"], r["preview"])
        self.parts.select(p.id)
        if self._first_load:
            self.viewport.view("iso")
            self._first_load = False
        self.viewport.reset_view(p.id)
        self.status("msg.loaded", name=p.name, n=i18n().num(p.n_faces))
        self._analyze(p, auto=True)

    def _done_analyze(self, job, r):
        p = self.parts.parts.get(r["id"])
        if p is None:
            return
        p.report, p.segments = r["report"], r["segments"]
        p.status = "printable" if p.report["printable"] else "problems"
        self.parts.update_part(p.id)
        if p.id == self.parts.selected_id():
            self.repair.set_report(p.report)
            self._update_problems()
        if p.status == "problems":
            self.status("msg.problems", name=p.name)
            if job.tag.get("auto"):
                self.tabs.setCurrentIndex(1)
        else:
            self.status("msg.printable", name=p.name)

    def _done_repair(self, job, r):
        p = self.parts.parts.get(r["part"]["id"])
        if p is None:
            return
        info = r["part"]
        p.n_faces, p.bounds = info["n_faces"], info["bounds"]
        v, f = r["preview"]
        p.preview = Mesh(v, f)
        self.viewport.set_part(p.id, v, f, p.color)
        self._show_repair_log({k: v for k, v in r["log"].items() if v})
        self._selection_changed(p.id)
        self._analyze(p)

    def _show_repair_log(self, log: dict):
        summary = ", ".join(t("log." + k, n=i18n().num(v)) for k, v in log.items()) or t("repair.nothing")
        self._last_msg = ("repair.done", {"log": log})
        self.msg.setText(t("repair.done", summary=summary))

    def _done_cut(self, job, r):
        parent = self.parts.parts.get(r["removed"])
        inherit = parent is not None and parent.status == "printable" and not r["open_loops"]
        parent_color = parent.color if parent is not None else None
        self._remove_part(r["removed"])
        new = []
        for i, item in enumerate(r["parts"]):
            color = parent_color if i == 0 else None
            new.append(self._add_part(item["part"], item["preview"], color,
                                      "printable" if inherit else "unknown"))
        if new:
            self.parts.select(new[0].id)
        if r["open_loops"]:
            QMessageBox.warning(self, APP_NAME, t("cut.open_warning"))
        elif not inherit:
            for p in new:
                self._analyze(p)
        self.status("cut.done", n=len(new))

    def _done_export(self, job, r):
        self.status("export.done", n=len(r["paths"]), folder=job.tag["folder"])

    # -- selection & overlays ---------------------------------------------------------
    def _selection_changed(self, pid):
        p = self.parts.parts.get(pid) if pid else None
        self.viewport.set_focus(pid if len(self.parts.parts) > 1 else None)
        if p is None:
            self.cut.set_part(None)
            self.repair.set_report(None)
        else:
            self.cut.set_part(p.name, p.preview.vertices, p.bounds)
            self.repair.set_report(p.report, checking=p.status == "checking")
        self._set_busy_ui(self.engine.busy)
        self._update_tool_overlays()
        self._update_volume()
        self._update_problems()

    def _visibility_changed(self, pid, visible):
        self.viewport.set_visible(pid, visible)

    def _plane_changed(self, normal, origin):
        p = self.parts.selected()
        if p is None:
            return
        if self.tabs.currentIndex() == 0 and self.cut.gizmo.isChecked():
            self.viewport.set_plane(normal, origin)
        self._pending_plane = (normal, origin)
        self._section_timer.start()

    def _update_section(self):
        p = self.parts.selected()
        if p is None or self._pending_plane is None or self.tabs.currentIndex() != 0:
            self.viewport.set_section(None)
            return
        n, o = self._pending_plane
        self.viewport.set_section(section_segments(p.preview, n, o))

    def _update_tool_overlays(self):
        p = self.parts.selected()
        if p is not None and self.tabs.currentIndex() == 0:
            n, o = self.cut.normal(), self.cut.origin()
            if self.cut.gizmo.isChecked():
                b = np.asarray(p.bounds)
                pad = 0.1 * (b[1] - b[0]).max()
                bounds = [b[0][0] - pad, b[1][0] + pad, b[0][1] - pad, b[1][1] + pad,
                          b[0][2] - pad, b[1][2] + pad]
                self.viewport.show_plane(n, o, bounds)
            else:
                self.viewport.hide_plane(render=False)
            self._pending_plane = (n, o)
            self._section_timer.start()
        else:
            self.viewport.hide_plane()

    def _volume_toggled(self, on):
        self.a_volume.setChecked(on)
        self._update_volume()

    def _update_volume(self):
        p = self.parts.selected()
        if p is None or not self.parts.show_volume.isChecked():
            self.viewport.show_volume(None, None)
            return
        w, d, h = self.parts.printer_volume()
        fit = fit_state(p.size, (w, d, h))
        if fit == "fits_rotated":
            w, d = d, w
        b = np.asarray(p.bounds)
        cx, cy = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
        z0 = b[0][2]
        self.viewport.show_volume([cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2, z0, z0 + h], fit != "too_big")

    def _edges_toggled(self, on):
        self.a_edges.setChecked(on)
        self._update_problems()

    def _update_problems(self):
        p = self.parts.selected()
        show = p is not None and self.repair.show_edges.isChecked() and p.segments is not None
        self.viewport.show_problems(p.segments if show else None)

    # -- dialogs ------------------------------------------------------------------------
    def sysinfo(self):
        info = hardware.system_info()
        QMessageBox.information(self, t("action.sysinfo"), t(
            "sysinfo.text", os=info["os"], cpu=info["cpu"], cores=info["cores"],
            ram=f"{info['ram_total'] / 2**30:.1f}", avail=f"{info['ram_available'] / 2**30:.1f}",
            gpu=self.viewport.renderer_string() or "?", budget=i18n().num(self.budget())))

    def about(self):
        QMessageBox.about(self, t("action.about"), t("about.text", version=__version__))

    # -- language -------------------------------------------------------------------------
    def retranslate(self, *_):
        self.setWindowTitle(t("app.title"))
        self.parts_dock.setWindowTitle(t("dock.parts"))
        self.tools_dock.setWindowTitle(t("dock.tools"))
        for i, key in enumerate(("tab.cut", "tab.repair", "tab.export")):
            self.tabs.setTabText(i, t(key))
        self.m_file.setTitle(t("menu.file"))
        self.m_view.setTitle(t("menu.view"))
        self.m_lang.setTitle(t("menu.language"))
        self.m_help.setTitle(t("menu.help"))
        for a, key in ((self.a_open, "action.open"), (self.a_export_sel, "action.export_selected"),
                       (self.a_export_all, "action.export_all"), (self.a_quit, "action.quit"),
                       (self.a_analyze, "action.analyze"), (self.a_repair, "action.repair"),
                       (self.a_reset, "action.reset_view"), (self.a_top, "action.view_top"),
                       (self.a_front, "action.view_front"), (self.a_side, "action.view_side"),
                       (self.a_volume, "action.show_volume"), (self.a_edges, "action.show_edges"),
                       (self.a_delete, "action.delete"), (self.a_about, "action.about"),
                       (self.a_sysinfo, "action.sysinfo")):
            a.setText(t(key))
        self.cancel_btn.setText(t("progress.cancel"))
        for code, a in self.lang_actions.items():
            a.setChecked(code == i18n().lang)
        self.parts.retranslate()
        self.cut.retranslate()
        self.repair.retranslate()
        self.export.retranslate()
        key, kw = self._last_msg
        if key == "repair.done":
            self._show_repair_log(kw["log"])
        else:
            self.status(key, **kw)

    def closeEvent(self, ev):
        self.engine.shutdown()
        self.viewport.close()
        super().closeEvent(ev)
