"""Main window: wires parts, viewport, tool panels, undo history and the engine."""
from __future__ import annotations

import os
import re

import numpy as np
import pyvista as pv
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QStyle,
                               QTabWidget, QVBoxLayout, QWidget)
from scipy.spatial.transform import Rotation

from .. import APP_NAME, __version__
from ..core import hardware
from ..core.cut import section_segments
from ..core.grid import fit_planes
from ..core.surface_cut import (Frame, height_function, heightfield_values, level_segments,
                                smooth_curve)
from ..core.mesh import Mesh
from ..core.transform import rotation_to, transform_points, translation
from . import style
from .engine_bridge import EngineBridge
from .help_hover import HoverHelp
from .history import History, Step, snapshot
from .i18n import LANGS, i18n, t
from .panels.cut_panel import CutPanel
from .connect_tool import ConnectTool
from .mold_tool import MoldTool
from .panels.connect_panel import ConnectPanel
from .panels.export_panel import ExportPanel
from .panels.mold_panel import MoldPanel
from .panels.move_panel import MovePanel, angles_of, compose
from .panels.repair_panel import RepairPanel
from .parts_panel import Part, PartsPanel, fit_state
from .viewport import Viewport

MIN_PREVIEW = 300_000


def _scroll(widget: QWidget) -> QWidget:
    """Scrollable tool page; a panel's `footer` (apply buttons) stays visible below it."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    widget.setObjectName("toolPanel")
    area.setWidget(widget)
    footer = getattr(widget, "footer", None)
    if footer is None:
        return area
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(area, 1)
    footer.setObjectName("toolFooter")
    lay.addWidget(footer)
    return page


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "part"


def merged_name(names: list[str]) -> str:
    prefix = os.path.commonprefix(names).rstrip("_- ")
    return (prefix if len(prefix) >= 2 else names[0]) + "_merged"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, int(screen.width() * 0.92)), min(900, int(screen.height() * 0.9)))
        self.setAcceptDrops(True)
        self.help = HoverHelp(self)
        self.engine = EngineBridge(self)
        self.history = History(self._purge)
        self._budget = None
        self._first_load = True
        self._last_msg = ("progress.idle", {})
        self._pending = None  # move tool session: {"pids", "center", "matrix"}
        self._drag_base = None
        self._curve = None  # curve drawing: {"frame", "points", "corners", "drawing"}
        self._free = None  # free-form session: {"frame", "u_range", "v_range", "base"}

        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)

        self.parts = PartsPanel(self.help.register)
        self.parts_dock = QDockWidget(self)
        self.parts_dock.setObjectName("parts")
        self.parts_dock.setWidget(self.parts)
        self.parts_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.parts_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.parts_dock)

        self.cut = CutPanel(self.help.register)
        self.conn = ConnectPanel(self.help.register)
        self.move = MovePanel(self.help.register)
        self.mold = MoldPanel(self.help.register)
        self.repair = RepairPanel(self.help.register)
        self.export = ExportPanel(self.help.register)
        self.tabs = QTabWidget()
        self._tab_keys = []
        self._tab_index = {}
        for panel, key in ((self.cut, "tab.cut"), (self.conn, "tab.connect"), (self.move, "tab.move"),
                           (self.mold, "tab.mold"), (self.repair, "tab.repair"), (self.export, "tab.export")):
            self._tab_index[panel] = self.tabs.addTab(_scroll(panel), "")
            self._tab_keys.append(key)
        self.tools_dock = QDockWidget(self)
        self.tools_dock.setObjectName("tools")
        self.tools_dock.setWidget(self.tabs)
        self.tools_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.tools_dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tools_dock)

        self._section_timer = QTimer(self)
        self._section_timer.setSingleShot(True)
        self._section_timer.setInterval(25)
        self._section_timer.timeout.connect(self._update_section)
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(30)
        self._overlay_timer.timeout.connect(self._update_mode_preview)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, lambda: self._curve_command("undo"))
        QShortcut(QKeySequence(Qt.Key_Return), self, lambda: self._curve_command("stop"))
        self._build_status()
        self._build_actions()
        self.conn_tool = ConnectTool(self)
        self.mold_tool = MoldTool(self)
        self._connect()
        self._pending_plane = None
        self._current_tab = self.tabs.currentIndex()
        self.conn_tool.printer_changed()
        i18n().changed.connect(self.retranslate)
        self.retranslate()
        self._set_busy_ui(False)

    # -- construction -----------------------------------------------------------
    def _tab_is(self, panel) -> bool:
        return self.tabs.currentIndex() == self._tab_index[panel]

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
            a.setShortcuts([QKeySequence(s) for s in (shortcut if isinstance(shortcut, tuple) else (shortcut,))])
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
        self.a_undo = self._action(self.undo, ("Ctrl+Z",), S.SP_ArrowBack)
        self.a_redo = self._action(self.redo, ("Ctrl+Y", "Ctrl+Shift+Z"), S.SP_ArrowForward)
        self.a_merge = self._action(lambda: self.merge_selected("union"), "Ctrl+M")
        self.a_combine = self._action(lambda: self.merge_selected("combine"), "Ctrl+Shift+M")
        self.a_select_all = self._action(lambda: self.parts.list.selectAll(), "Ctrl+A")
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
        for a in (self.a_open, self.a_export_sel, self.a_export_all, None, self.a_quit):
            self.m_file.addSeparator() if a is None else self.m_file.addAction(a)
        self.m_edit = mb.addMenu("")
        for a in (self.a_undo, self.a_redo, None, self.a_select_all, self.a_merge, self.a_combine, self.a_delete):
            self.m_edit.addSeparator() if a is None else self.m_edit.addAction(a)
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
        for a in (self.a_open, self.a_export_sel, None, self.a_undo, self.a_redo, None, self.a_analyze,
                  self.a_repair, None, self.a_reset):
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
        p.list.itemSelectionChanged.connect(self._multi_selection_changed)
        p.visibility_changed.connect(self._visibility_changed)
        p.renamed.connect(self._renamed)
        p.delete_requested.connect(self.delete_selected)
        p.merge_requested.connect(self.merge_selected)
        p.printer_changed.connect(self._update_volume)
        p.printer_changed.connect(self.conn_tool.printer_changed)
        p.volume_toggled.connect(self._volume_toggled)
        self.cut.plane_changed.connect(self._plane_changed)
        self.cut.gizmo_toggled.connect(lambda _: self._update_tool_overlays())
        self.cut.apply_requested.connect(self.cut_selected)
        self.cut.apply_mode_requested.connect(self.cut_mode_selected)
        self.cut.mode_changed.connect(lambda _: self._update_tool_overlays())
        self.cut.grid_changed.connect(self._overlay_timer.start)
        self.cut.freeform_changed.connect(self._overlay_timer.start)
        self.cut.fit_requested.connect(self._fit_to_printer)
        self.cut.curve_command.connect(self._curve_command)
        self.viewport.plane_moved.connect(self.cut.set_from_handle)
        self.move.changed.connect(self._update_move_preview)
        self.move.apply_requested.connect(lambda: self.commit_move())
        self.move.reset_requested.connect(self._reset_move)
        self.move.lay_flat_requested.connect(self._lay_flat)
        self.move.rotate90_requested.connect(self._rotate90)
        self.repair.analyze_requested.connect(self.analyze_selected)
        self.repair.repair_requested.connect(self.repair_selected)
        self.repair.show_edges_toggled.connect(self._edges_toggled)
        self.export.export_selected.connect(lambda folder: self.export_parts(True, folder))
        self.export.export_all.connect(lambda folder: self.export_parts(False, folder))
        self.tabs.currentChanged.connect(self._tab_changed)

    # -- helpers --------------------------------------------------------------------
    def budget(self) -> int:
        if self._budget is None:
            self._budget = hardware.display_budget(self.viewport.renderer_string())
        return self._budget

    def preview_faces(self, extra_parts: int = 1) -> int:
        n = len(self.parts.visible_parts()) + extra_parts
        return max(MIN_PREVIEW, self.budget() // max(1, n))

    def status(self, key: str, **kw) -> None:
        """Show a translated status message; it follows language switches.

        A value written as "@some.key" is itself translated when shown.
        """
        self._last_msg = (key, kw)
        shown = {k: t(v[1:]) if isinstance(v, str) and v.startswith("@") else v for k, v in kw.items()}
        self.msg.setText(t(key, **shown))

    def _set_busy_ui(self, busy: bool):
        for w in (self.progress, self.progress_text, self.cancel_btn):
            w.setVisible(busy)
        if not busy:
            self.progress.setValue(0)
            self.progress_text.setText("")
        self.cut.set_busy(busy)
        self.conn.set_busy(busy)
        self.mold.set_busy(busy)
        self.move.set_busy(busy)
        self.export.set_busy(busy)
        self.parts.set_busy(busy)
        has = self.parts.selected() is not None
        self.repair.set_enabled(has and not busy)
        any_parts = bool(self.parts.parts)
        for a in (self.a_analyze, self.a_repair, self.a_export_sel, self.a_export_all, self.a_delete):
            a.setEnabled(not busy and any_parts)
        n_sel = len(self.parts.selected_parts())
        self.a_merge.setEnabled(not busy and n_sel >= 2)
        self.a_combine.setEnabled(not busy and n_sel >= 2)
        self._refresh_undo(busy)

    def _refresh_undo(self, busy: bool | None = None):
        busy = self.engine.busy if busy is None else busy
        self.a_undo.setEnabled(not busy and self.history.can_undo)
        self.a_redo.setEnabled(not busy and self.history.can_redo)
        u, r = self.history.label(), self.history.label(redo=True)
        self.a_undo.setText(t("action.undo") + (f": {t(u)}" if u else ""))
        self.a_redo.setText(t("action.redo") + (f": {t(r)}" if r else ""))

    def _error(self, code: str, detail: str = ""):
        if code == "memory" and "|" in detail:
            need, avail = detail.split("|")
            text = t("err.memory", need=need, avail=avail)
        elif code in ("crashed", "exception"):
            text = t("err." + code, detail=detail.splitlines()[0] if detail else "")
        else:
            text = t("err." + code)
        QMessageBox.warning(self, APP_NAME, text)

    def _purge(self, pids):
        self.engine.submit("purge", pids=list(pids))

    # -- parts bookkeeping ------------------------------------------------------------
    def _add_part(self, info: dict, preview, color: str | None = None, status="unknown", index=None,
                  report=None, visible=True, extra=None) -> Part:
        v, f = preview
        p = Part(id=info["id"], name=info["name"], n_faces=info["n_faces"], bounds=info["bounds"],
                 color=color or self.parts.next_color(), status=status, report=report,
                 visible=visible, preview=Mesh(v, f), extra=dict(extra or {}))
        self.viewport.set_part(p.id, v, f, p.color)
        if not visible:
            self.viewport.set_visible(p.id, False)
        self.parts.add(p, select=False, index=index)
        return p

    def _remove_part(self, pid: str) -> int:
        self.viewport.remove_part(pid)
        row = self.parts.remove(pid)
        if not self.parts.parts:
            self._selection_changed(None)
        return row

    def _apply_change(self, r: dict, label: str, meta=None) -> list[Part]:
        """Swap the change set's removed parts for its added ones; record an undo step."""
        old = [self.parts.parts[pid] for pid in r["removed"] if pid in self.parts.parts]
        snaps = []
        index = None
        for p in old:
            s = snapshot(p)
            s.extra["row"] = self._remove_part(p.id)
            snaps.append(s)
            index = s.extra["row"] if index is None else min(index, s.extra["row"])
        new = []
        for i, item in enumerate(r["added"]):
            kw = meta(i, item, old) if meta else {}
            new.append(self._add_part(item["part"], item["preview"],
                                      index=None if index is None or index < 0 else index + i, **kw))
        self.history.push(Step(label, snaps, [snapshot(p) for p in new]))
        if new:
            self.parts.select_many([p.id for p in new])
        self._refresh_undo()
        return new

    def _same_as_old(self, i, item, old):
        """Keep look and state of the part a result replaces (move, repair)."""
        if i < len(old):
            o = old[i]
            return {"color": o.color, "status": o.status, "report": o.report, "visible": o.visible}
        return {}

    def _remember_status(self, p: Part):
        """Analysis finished later than the undo step: update its snapshots too."""
        for step in self.history.undo_steps + self.history.redo_steps:
            for s in step.removed + step.added:
                if s.id == p.id:
                    s.status, s.report = p.status, p.report

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

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.viewport.cancel_pick()
        super().keyPressEvent(ev)

    # -- actions ----------------------------------------------------------------------
    def _after_commit(self, fn):
        """Run fn once any pending move has been applied to the real parts."""
        if self._move_dirty():
            self.commit_move(then=fn)
        else:
            fn()

    def analyze_selected(self, auto: bool = False):
        def go():
            p = self.parts.selected()
            if p is not None:
                self._analyze(p, auto)
        self._after_commit(go)

    def _analyze(self, p: Part, auto: bool = False):
        p.status = "checking"
        self.parts.update_part(p.id)
        if p.id == self.parts.selected_id():
            self.repair.set_report(None, checking=True)
        self.engine.submit("analyze", tag={"pid": p.id, "auto": auto}, pid=p.id)

    def repair_selected(self, options: dict):
        def go():
            p = self.parts.selected()
            if p is not None:
                self.engine.submit("repair", tag={"pid": p.id}, pid=p.id, name=p.name, options=options,
                                   preview_faces=self.preview_faces(0))
        self._after_commit(go)

    def cut_selected(self, normal, origin, keep: str):
        p = self.parts.selected()
        if p is not None:
            self.engine.submit("cut", tag={"pid": p.id}, pid=p.id, name=p.name, normal=list(normal),
                               origin=list(origin), keep=keep, preview_faces=self.preview_faces(1))

    def merge_selected(self, mode: str = "union"):
        def go():
            parts = self.parts.selected_parts()
            if len(parts) < 2 or self.engine.busy:
                return
            self.engine.submit("merge", tag={"mode": mode}, pids=[p.id for p in parts],
                               name=merged_name([p.name for p in parts]), mode=mode,
                               preview_faces=self.preview_faces(0))
        self._after_commit(go)

    def delete_selected(self):
        parts = self.parts.selected_parts()
        if not parts or self.engine.busy:
            return
        names = ", ".join(p.name for p in parts)
        if QMessageBox.question(self, APP_NAME, f"{t('action.delete')}: {names}?") != QMessageBox.Yes:
            return
        self._end_move_session()
        snaps = []
        for p in parts:
            s = snapshot(p)
            s.extra["row"] = self._remove_part(p.id)
            snaps.append(s)
        self.history.push(Step("action.delete", removed=snaps))
        self._refresh_undo()
        self.status("msg.deleted", name=names)

    def _renamed(self, pid, old, new):
        self.history.push(Step("action.rename", renames=[(pid, old, new)]))
        self._refresh_undo()
        if pid == self.parts.selected_id():
            self._selection_changed(pid)

    def export_parts(self, selected_only: bool, folder: str | None = None):
        self._after_commit(lambda: self._export(selected_only, folder))

    def _export(self, selected_only: bool, folder: str | None):
        folder = folder or self.export.folder.text().strip()
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, t("export.folder"))
            if not folder:
                return
            self.export.folder.setText(folder)
        os.makedirs(folder, exist_ok=True)
        parts = self.parts.selected_parts() if selected_only else self.parts.visible_parts()
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

    # -- undo / redo ------------------------------------------------------------------
    def undo(self):
        if not self.engine.busy and self.history.can_undo:
            self._after_commit(lambda: self._swap(self.history.take_undo(), undo=True))

    def redo(self):
        if not self.engine.busy and self.history.can_redo:
            self._after_commit(lambda: self._swap(self.history.take_redo(), undo=False))

    def _swap(self, step: Step, undo: bool):
        self.conn_tool._no_auto = False  # after undo/redo, propose connectors again
        self._end_move_session()
        remove, restore = (step.added, step.removed) if undo else (step.removed, step.added)
        for s in remove:
            if s.id in self.parts.parts:
                self._remove_part(s.id)
        for pid, old, new in step.renames:
            p = self.parts.parts.get(pid)
            if p is not None:
                p.name = old if undo else new
                self.parts.update_part(pid)
        key = "history.undone" if undo else "history.redone"
        if restore:
            self.engine.submit("restore", tag={"parts": restore, "key": key, "label": step.label},
                               pids=[s.id for s in restore])
        else:
            self.status(key, what="@" + step.label)
            self._selection_changed(self.parts.selected_id())
        self._refresh_undo()

    def _done_restore(self, job, r):
        previews = {x["id"]: x["preview"] for x in r["parts"]}
        restored = []
        for s in sorted(job.tag["parts"], key=lambda s: s.extra.get("row", 1 << 30)):
            info = {"id": s.id, "name": s.name, "n_faces": s.n_faces, "bounds": s.bounds}
            row = s.extra.get("row")
            restored.append(self._add_part(info, previews[s.id], s.color, s.status,
                                           index=row if row is not None and row >= 0 else None,
                                           report=s.report, visible=s.visible,
                                           extra={k: v for k, v in s.extra.items() if k != "row"}))
        if restored:
            self.parts.select_many([p.id for p in restored])
        self.status(job.tag["key"], what="@" + job.tag["label"])

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
        elif not (job.op == "joint" and job.tag.get("auto") and code == "no_joint"):
            self._error(code, detail)
        if job.op == "joint":
            self.conn_tool.joint_failed(job, code)
        if job.op == "mold_preview":
            self.mold_tool.preview_failed(job)
        if job.op == "analyze":
            p = self.parts.parts.get(job.tag["pid"])
            if p is not None:
                p.status = "unknown"
                self.parts.update_part(p.id)
                if p.id == self.parts.selected_id():
                    self.repair.set_report(None)
        elif job.op == "transform":
            for pid, _, _ in job.kwargs["items"]:
                self.viewport.set_matrix(pid, None)
            if self._tab_is(self.move):
                self._start_move_session()

    def _done_load(self, job, r):
        new = self._apply_change(r, "action.open")
        p = new[0]
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
        self._remember_status(p)
        self.parts.update_part(p.id)
        if p.id == self.parts.selected_id():
            self.repair.set_report(p.report)
            self._update_problems()
        if p.status == "problems":
            self.status("msg.problems", name=p.name)
            if job.tag.get("auto"):
                self.tabs.setCurrentIndex(self._tab_index[self.repair])
        else:
            self.status("msg.printable", name=p.name)

    def _done_repair(self, job, r):
        new = self._apply_change(r, "action.repair", meta=lambda i, item, old: {
            "color": old[0].color if old else None, "visible": old[0].visible if old else True})
        self._show_repair_log({k: v for k, v in r["log"].items() if v})
        for p in new:
            self._analyze(p)

    def _show_repair_log(self, log: dict):
        summary = ", ".join(t("log." + k, n=i18n().num(v)) for k, v in log.items()) or t("repair.nothing")
        self._last_msg = ("repair.done", {"log": log})
        self.msg.setText(t("repair.done", summary=summary))

    def _done_cut(self, job, r):
        parent = self.parts.parts.get(r["removed"][0]) if r["removed"] else None
        inherit = parent is not None and parent.status == "printable" and not r["open_loops"]

        def meta(i, item, old):
            return {"color": old[0].color if (old and i == 0) else None,
                    "status": "printable" if inherit else "unknown"}

        new = self._apply_change(r, "action.cut", meta)  # both pieces stay selected (ready to connect)
        if r["open_loops"]:
            QMessageBox.warning(self, APP_NAME, t("cut.open_warning"))
        elif not inherit:
            for p in new:
                self._analyze(p)
        self.status("cut.done", n=len(new))

    def _done_transform(self, job, r):
        self._pending = None
        self._apply_change(r, "action.move", self._same_as_old)
        self.move.reset()
        self.status("move.done")
        then = job.tag.get("then") if job.tag else None
        if self._tab_is(self.move):
            self._start_move_session()
        if then:
            then()

    def _done_merge(self, job, r):
        new = self._apply_change(r, "action.merge" if job.tag["mode"] == "union" else "action.combine",
                                 meta=lambda i, item, old: {"color": old[0].color if old else None})
        for p in new:
            self._analyze(p)
        self.status("msg.merged", name=new[0].name if new else "")

    def _done_export(self, job, r):
        self.status("export.done", n=len(r["paths"]), folder=job.tag["folder"])

    # -- move tool ------------------------------------------------------------------
    def _move_dirty(self) -> bool:
        return bool(self._pending and self._pending["pids"]) and not self.move.is_identity()

    def _start_move_session(self):
        parts = [p for p in self.parts.selected_parts() if p.visible]
        if not parts:
            self._pending = None
            self.move.set_target(0)
            self.viewport.set_drag(None)
            return
        b = np.array([p.bounds for p in parts])
        center = (b[:, 0].min(0) + b[:, 1].max(0)) / 2
        self._pending = {"pids": [p.id for p in parts], "center": center, "matrix": np.eye(4)}
        self.move.reset()
        self.move.set_target(len(parts), parts[0].name)
        self.viewport.set_drag(self._pending["pids"], self._on_drag, self._on_drag_done)
        self._update_move_preview()

    def _end_move_session(self):
        """Drop an unapplied move (the parts snap back)."""
        if self._pending:
            for pid in self._pending["pids"]:
                self.viewport.set_matrix(pid, None)
        self._pending = None
        self.viewport.set_drag(None)

    def _reset_move(self):
        self.move.reset()
        self._update_move_preview()

    def _preview_points(self):
        pts = [self.parts.parts[pid].preview.vertices for pid in self._pending["pids"] if pid in self.parts.parts]
        return np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)

    def _update_move_preview(self):
        if not self._pending:
            return
        v = self.move.values()
        m = compose(self._pending["center"], v["offset"], v["angles"], v["scale"])
        self._pending["matrix"] = m
        shown = m
        pts = self._preview_points()
        if len(pts):
            moved = transform_points(pts, m)
            lo, hi = moved.min(0).astype(float), moved.max(0).astype(float)
            shift = np.zeros(3)
            if v["centered"]:
                shift[:2] = -(lo[:2] + hi[:2]) / 2
            if v["on_bed"]:
                shift[2] = -lo[2]
            shown = translation(shift) @ m
            self.move.set_info(hi - lo, lo[2] + shift[2])
        for pid in self._pending["pids"]:
            self.viewport.set_matrix(pid, shown)

    def _on_drag(self, delta, vertical):
        if self._drag_base is None:
            self._drag_base = list(self.move.values()["offset"])
        off = list(self._drag_base)
        if vertical:
            off[2] += float(delta[2])
            self.move.set_values(offset=off, on_bed=False)
        else:
            off[0] += float(delta[0])
            off[1] += float(delta[1])
            self.move.set_values(offset=off, centered=False)
        self._update_move_preview()

    def _on_drag_done(self):
        self._drag_base = None

    def _rotate90(self, axis: int):
        rot = Rotation.from_euler("xyz", self.move.values()["angles"], degrees=True).as_matrix()
        step = Rotation.from_euler("xyz", np.eye(3)[axis] * 90, degrees=True).as_matrix()
        self.move.set_values(angles=angles_of(step @ rot))
        self._update_move_preview()

    def _lay_flat(self):
        if not self._pending:
            return
        self.status("move.pick_face")

        def picked(pid, point, normal):
            if pid not in self._pending["pids"]:
                return
            rot = Rotation.from_euler("xyz", self.move.values()["angles"], degrees=True).as_matrix()
            new = rotation_to(normal, (0, 0, -1)) @ rot
            self.move.set_values(angles=angles_of(new), on_bed=True)
            self._update_move_preview()
            self.status("move.laid_flat")

        self.viewport.pick_surface(picked)

    def commit_move(self, then=None):
        if not self._move_dirty():
            if then:
                then()
            return
        v = self.move.values()
        m = self._pending["matrix"]
        items = [(pid, self.parts.parts[pid].name, m.tolist()) for pid in self._pending["pids"]
                 if pid in self.parts.parts]
        self.viewport.set_drag(None)
        self.engine.submit("transform", tag={"then": then}, items=items, on_bed=v["on_bed"],
                           centered=v["centered"])
        self._pending = {**self._pending, "pids": []}  # applied; don't apply twice

    # -- selection & overlays ---------------------------------------------------------
    def _tab_changed(self, index):
        prev, self._current_tab = self._current_tab, index
        move = self._tab_index[self.move]
        if prev == move and index != move:
            if self._move_dirty():
                self.commit_move()
            else:
                self._end_move_session()
        if index == move:
            self._start_move_session()
        self._update_tool_overlays()
        self.conn_tool.refresh()
        self.mold_tool.refresh()

    def _multi_selection_changed(self):
        sel = [p.id for p in self.parts.selected_parts()]
        self.viewport.set_focus(sel if len(self.parts.parts) > 1 and sel else None)
        self._set_busy_ui(self.engine.busy)
        self.conn_tool.refresh()
        if self._tab_is(self.move):
            current = set(self._pending["pids"]) if self._pending else set()
            if current != {p.id for p in self.parts.selected_parts() if p.visible}:
                if self._move_dirty():
                    self.commit_move(then=self._start_move_session)
                else:
                    self._end_move_session()
                    self._start_move_session()

    def _selection_changed(self, pid):
        p = self.parts.parts.get(pid) if pid else None
        if p is None:
            self.cut.set_part(None)
            self.repair.set_report(None)
        else:
            self.cut.set_part(p.name, p.preview.vertices, p.bounds)
            self.repair.set_report(p.report, checking=p.status == "checking")
        self._multi_selection_changed()
        self._update_tool_overlays()
        self.conn_tool.refresh()
        self.mold_tool.refresh()
        self._update_volume()
        self._update_problems()

    def _visibility_changed(self, pid, visible):
        self.viewport.set_visible(pid, visible)

    def _plane_changed(self, normal, origin):
        p = self.parts.selected()
        if p is None:
            return
        if self._tab_is(self.cut) and self.cut.current_mode() == "freeform":
            self._free = None
            self._overlay_timer.start()
            return
        if self._tab_is(self.cut) and self.cut.gizmo.isChecked():
            self.viewport.set_plane(normal, origin)
        self._pending_plane = (normal, origin)
        self._section_timer.start()

    def _update_section(self):
        p = self.parts.selected()
        if (p is None or self._pending_plane is None or not self._tab_is(self.cut)
                or self.cut.current_mode() != "plane"):
            self.viewport.set_section(None)
            return
        n, o = self._pending_plane
        self.viewport.set_section(section_segments(p.preview, n, o))

    def _update_tool_overlays(self):
        p = self.parts.selected()
        mode = self.cut.current_mode()
        on_cut = p is not None and self._tab_is(self.cut)
        if not (on_cut and mode == "curve"):
            self._stop_curve_drawing()
        if not (on_cut and mode == "freeform"):
            self.viewport.set_handles(None)
            self._free = None
        for name in ("grid_planes", "curve", "curve_pts", "ribbon", "free_surface"):
            self.viewport.set_overlay(name, None)
        if on_cut and mode == "plane":
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
            if on_cut:
                self._update_mode_preview()

    def _update_mode_preview(self):
        p = self.parts.selected()
        if p is None or not self._tab_is(self.cut):
            return
        mode = self.cut.current_mode()
        if mode == "grid":
            self._update_grid_preview(p)
        elif mode == "curve":
            self._update_curve_preview(p)
        elif mode == "freeform":
            self._update_freeform_preview(p)

    # -- grid cut ---------------------------------------------------------------------
    def _update_grid_preview(self, p: Part):
        planes = self.cut.grid_planes
        b = np.asarray(p.bounds)
        size = b[1] - b[0]
        segs, quads = [], []
        for axis, pos in planes:
            normal = np.eye(3)[axis]
            origin = (b[0] + b[1]) / 2
            origin[axis] = pos
            segs.append(section_segments(p.preview, normal, origin))
            others = [k for k in range(3) if k != axis]
            quads.append(pv.Plane(center=origin, direction=normal, i_size=size[others[1]] * 1.1 + 1,
                                  j_size=size[others[0]] * 1.1 + 1))
        self.viewport.set_section(np.concatenate(segs) if segs else None)
        merged = pv.merge(quads) if quads else None
        self.viewport.set_overlay("grid_planes", merged, color=style.ACCENT, opacity=0.18, show_edges=False)
        if not planes:
            self.cut.set_grid_info(t("cut.grid_none"))
            return
        # largest piece vs printer
        biggest = []
        for k in range(3):
            cuts = sorted([b[0][k]] + [q for a, q in planes if a == k] + [b[1][k]])
            biggest.append(max(np.diff(cuts)))
        n_pieces = int(np.prod([1 + sum(1 for a, _ in planes if a == k) for k in range(3)]))
        fit = fit_state(biggest, self.parts.printer_volume())
        n = i18n().num
        self.cut.set_grid_info(t("cut.grid_info", n=n_pieces, x=n(biggest[0], "{:.1f}"),
                                 y=n(biggest[1], "{:.1f}"), z=n(biggest[2], "{:.1f}"),
                                 fit=t("printer." + fit)))

    def _fit_to_printer(self):
        p = self.parts.selected()
        if p is None:
            return
        printer = self.parts.current_printer()
        planes, counts, rotated = fit_planes(p.bounds, printer.volume, printer.margin)
        self.cut.set_grid_planes(planes, counts)
        self.status("cut.fit_done", n=int(np.prod(counts)))

    # -- curve cut --------------------------------------------------------------------
    def _curve_command(self, cmd: str):
        p = self.parts.selected()
        if p is None or not self._tab_is(self.cut) or self.cut.current_mode() != "curve":
            return
        if cmd == "draw":
            if self._curve and self._curve["drawing"]:
                self._stop_curve_drawing()
            else:
                d, right, up = self.viewport.camera_frame()
                b = np.asarray(p.bounds)
                frame = Frame((b[0] + b[1]) / 2, right, np.cross(d, right), d)
                self._curve = {"frame": frame, "points": [], "corners": set(), "drawing": True}
                self.viewport.set_parallel(True)
                self.viewport.set_click_capture(self._curve_click)
        elif cmd == "stop":
            self._stop_curve_drawing()
        elif cmd == "undo" and self._curve and self._curve["points"]:
            self._curve["points"].pop()
            self._curve["corners"].discard(len(self._curve["points"]))
        self._refresh_curve_ui()
        self._overlay_timer.start()

    def _stop_curve_drawing(self):
        if self._curve and self._curve["drawing"]:
            self._curve["drawing"] = False
            self.viewport.set_click_capture(None)
            self.viewport.set_parallel(False)
        self._refresh_curve_ui()

    def _refresh_curve_ui(self):
        c = self._curve
        self.cut.set_curve_info(len(c["points"]) if c else 0, bool(c and c["drawing"]))

    def _curve_click(self, x, y, shift):
        c = self._curve
        if not c:
            return
        p0, d = self.viewport.display_ray(x, y)
        fr = c["frame"]
        denom = float(d @ fr.w)
        if abs(denom) < 1e-12:
            return
        hit = p0 + d * (float((fr.origin - p0) @ fr.w) / denom)
        loc = fr.to_local(hit[None])[0]
        if shift:
            c["corners"].add(len(c["points"]))
        c["points"].append((float(loc[0]), float(loc[1])))
        self._refresh_curve_ui()
        self._overlay_timer.start()

    def _curve_polyline(self):
        c = self._curve
        if not c or len(c["points"]) < 2:
            return None
        pts = np.array(c["points"], float)
        return smooth_curve(pts, c["corners"]) if self.cut.smooth.isChecked() and len(pts) >= 3 else pts

    def _update_curve_preview(self, p: Part):
        c = self._curve
        self.viewport.set_section(None)
        if not c or not c["points"]:
            for name in ("curve", "curve_pts", "ribbon"):
                self.viewport.set_overlay(name, None)
            return
        fr = c["frame"]
        pts3 = fr.to_world(np.c_[np.array(c["points"]), np.zeros(len(c["points"]))])
        self.viewport.set_overlay("curve_pts", pv.PolyData(pts3.astype(np.float32)), on_top=True, color="white",
                                  point_size=10, render_points_as_spheres=True)
        line = self._curve_polyline()
        if line is None:
            return
        b = np.asarray(p.bounds)
        L = float(np.linalg.norm(b[1] - b[0]))
        t0, t1 = line[1] - line[0], line[-1] - line[-2]
        ext = np.vstack([line[0] - t0 / np.linalg.norm(t0) * L, line, line[-1] + t1 / np.linalg.norm(t1) * L])
        top = fr.to_world(np.c_[ext, np.full(len(ext), L)])
        bot = fr.to_world(np.c_[ext, np.full(len(ext), -L)])
        n = len(ext)
        verts = np.vstack([top, bot]).astype(np.float32)
        i = np.arange(n - 1)
        faces = np.concatenate([np.stack([i, i + 1, i + n + 1], 1), np.stack([i, i + n + 1, i + n], 1)])
        self.viewport.set_overlay("ribbon", pv.PolyData.from_regular_faces(verts, faces), color=style.ACCENT,
                                  opacity=0.3)
        line3 = fr.to_world(np.c_[line, np.zeros(len(line))]).astype(np.float32)
        self.viewport.set_overlay("curve", pv.lines_from_points(line3), on_top=True, color=style.ACCENT,
                                  line_width=4)

    # -- free-form cut ------------------------------------------------------------------
    def _free_session(self, p: Part):
        if self._free is not None and self._free["k"] != self.cut.heights.shape[0]:
            self._free = None
        if self._free is None:
            fr = Frame.from_normal(self.cut.origin(), self.cut.normal())
            b = np.asarray(p.bounds)
            corners = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            loc = fr.to_local(corners)
            pad = 0.05 * float(np.ptp(loc[:, :2], axis=0).max())
            self._free = {"frame": fr, "u_range": (loc[:, 0].min() - pad, loc[:, 0].max() + pad),
                          "v_range": (loc[:, 1].min() - pad, loc[:, 1].max() + pad), "base": None,
                          "k": self.cut.heights.shape[0]}
            self.viewport.set_handles(self._free_points(), fr.w, on_drag=self._free_drag,
                                      on_select=self._free_select, on_done=self._free_drag_done)
        return self._free

    def _free_points(self):
        f = self._free
        h = self.cut.heights
        k = h.shape[0]
        us = np.linspace(*f["u_range"], k)
        vs = np.linspace(*f["v_range"], k)
        U, V = np.meshgrid(us, vs, indexing="ij")
        return f["frame"].to_world(np.stack([U, V, h], -1).reshape(-1, 3))

    def _update_freeform_preview(self, p: Part):
        f = self._free_session(p)
        h = self.cut.heights
        hf = height_function(h, f["u_range"], f["v_range"])
        res = 40
        us = np.linspace(*f["u_range"], res)
        vs = np.linspace(*f["v_range"], res)
        U, V = np.meshgrid(us, vs, indexing="ij")
        verts = f["frame"].to_world(np.stack([U, V, hf(U, V)], -1).reshape(-1, 3)).astype(np.float32)
        idx = np.arange(res * res).reshape(res, res)
        a, b2, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        faces = np.concatenate([np.stack([a, b2, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        self.viewport.set_overlay("free_surface", pv.PolyData.from_regular_faces(verts, faces),
                                  color=style.ACCENT, opacity=0.28)
        sel = self.cut.free_selected
        k = h.shape[0]
        self.viewport.update_handles(self._free_points(), None if sel is None else sel[0] * k + sel[1])
        vals = heightfield_values(p.preview.vertices, f["frame"], hf)
        self.viewport.set_section(level_segments(p.preview, vals))

    def _free_select(self, i):
        k = self.cut.heights.shape[0]
        ij = divmod(i, k)
        self.cut.select_free_point(ij)
        self._free["base"] = float(self.cut.heights[ij])
        self._overlay_timer.start()

    def _free_drag(self, i, delta):
        k = self.cut.heights.shape[0]
        ij = divmod(i, k)
        base = self._free.get("base") or 0.0
        self.cut.set_free_height(ij, base + delta)

    def _free_drag_done(self):
        self._free["base"] = None

    # -- applying the other cut modes ------------------------------------------------------
    def cut_mode_selected(self, mode: str):
        p = self.parts.selected()
        if p is None:
            return
        gap, keep = self.cut.gap.value(), self.cut.keep.currentData()
        if mode == "grid":
            if not self.cut.grid_planes:
                QMessageBox.information(self, APP_NAME, t("cut.grid_empty"))
                return
            self.engine.submit("grid_cut", tag={"pid": p.id}, pid=p.id, name=p.name,
                               planes=self.cut.grid_planes, preview_faces=self.preview_faces(4))
        elif mode == "curve":
            line = self._curve_polyline()
            if line is None:
                QMessageBox.information(self, APP_NAME, t("cut.curve_empty"))
                return
            fr = self._curve["frame"]
            self._stop_curve_drawing()
            self.engine.submit("surface_cut", tag={"pid": p.id}, pid=p.id, name=p.name, kind="curve",
                               frame=self._frame_dict(fr), params={"curve": line.tolist()}, gap=gap, keep=keep,
                               preview_faces=self.preview_faces(1))
        elif mode == "freeform":
            f = self._free_session(p)
            self.engine.submit("surface_cut", tag={"pid": p.id}, pid=p.id, name=p.name, kind="freeform",
                               frame=self._frame_dict(f["frame"]),
                               params={"heights": self.cut.heights.tolist(), "u_range": list(f["u_range"]),
                                       "v_range": list(f["v_range"]), "res": 128},
                               gap=gap, keep=keep, preview_faces=self.preview_faces(1))

    @staticmethod
    def _frame_dict(fr: Frame) -> dict:
        return {k: np.asarray(getattr(fr, k), float).tolist() for k in ("origin", "u", "v", "w")}

    def _done_grid_cut(self, job, r):
        parent = self.parts.parts.get(r["removed"][0]) if r["removed"] else None
        inherit = parent is not None and parent.status == "printable"
        self.cut.set_grid_planes([], (1, 1, 1), emit=False)  # don't carry the grid over to the pieces
        new = self._apply_change(r, "action.cut", lambda i, item, old: {
            "status": "printable" if inherit else "unknown"})
        if not inherit:
            for p in new:
                self._analyze(p)
        self.status("cut.done", n=len(new))

    def _done_surface_cut(self, job, r):
        new = self._apply_change(r, "action.cut", lambda i, item, old: {
            "color": old[0].color if (old and i == 0) else None})
        self._curve = None
        self._refresh_curve_ui()
        for p in new:
            self._analyze(p)
        self.status("cut.done", n=len(new))

    def _done_joint(self, job, r):
        self.conn_tool.done_joint(job, r)

    def _done_connect(self, job, r):
        self.conn_tool.done_connect(job, r)

    def _done_coupon(self, job, r):
        self.conn_tool.done_coupon(job, r)

    def _done_mold_preview(self, job, r):
        self.mold_tool.done_preview(job, r)

    def _done_mold_build(self, job, r):
        self.mold_tool.done_build(job, r)

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
        for i, key in enumerate(self._tab_keys):
            self.tabs.setTabText(i, t(key))
        for m, key in ((self.m_file, "menu.file"), (self.m_edit, "menu.edit"), (self.m_view, "menu.view"),
                       (self.m_lang, "menu.language"), (self.m_help, "menu.help")):
            m.setTitle(t(key))
        for a, key in ((self.a_open, "action.open"), (self.a_export_sel, "action.export_selected"),
                       (self.a_export_all, "action.export_all"), (self.a_quit, "action.quit"),
                       (self.a_analyze, "action.analyze"), (self.a_repair, "action.repair"),
                       (self.a_reset, "action.reset_view"), (self.a_top, "action.view_top"),
                       (self.a_front, "action.view_front"), (self.a_side, "action.view_side"),
                       (self.a_volume, "action.show_volume"), (self.a_edges, "action.show_edges"),
                       (self.a_delete, "action.delete"), (self.a_about, "action.about"),
                       (self.a_sysinfo, "action.sysinfo"), (self.a_merge, "action.merge"),
                       (self.a_combine, "action.combine"), (self.a_select_all, "action.select_all")):
            a.setText(t(key))
        self._refresh_undo()
        self.cancel_btn.setText(t("progress.cancel"))
        for code, a in self.lang_actions.items():
            a.setChecked(code == i18n().lang)
        for panel in (self.parts, self.cut, self.conn, self.move, self.mold, self.repair, self.export):
            panel.retranslate()
        self.conn_tool.validate()
        key, kw = self._last_msg
        if key == "repair.done":
            self._show_repair_log(kw["log"])
        else:
            self.status(key, **kw)

    def closeEvent(self, ev):
        self.engine.shutdown()
        self.viewport.close()
        super().closeEvent(ev)
