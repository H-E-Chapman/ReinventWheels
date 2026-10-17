"""HCR Cells -- group experiments: transcripts per image and channel, for
worms in several groups imaged on several days (one image per worm).

    pixi run gui-groups <folder>        # or: pixi run gui <folder> (detected)

1. Images -- the folder is scanned: day (the date folder), group (the file
   name's start; misspellings matched to the nearest group), worm number. The
   images window lists every image; fix a group or leave an image out there.
2. Check images -- drift across each z-stack (small: corrected; big: left
   out) and the same worm saved twice (only the first is used); every
   problem is listed as a warning.
3. Run -- cells and spots in every image used (parallel).
4. Review cells -- per image: which cells count, which neuron each is.
5. Export CSV -- transcripts_per_image.csv, the result; cells.csv and
   images_qc.csv with it. Rewritten whenever a review choice changes.

Statistics, batch correction and group comparisons are left to the analysis
in R. The image recipe (recipes/image.toml) and every check are Qt-free
(:mod:`hcrcells.groups`, :mod:`hcrcells.qc`).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from vanua.ui import theme  # noqa: I001
from vanua.blocks.plot import label_colors
from vanua.ui.blocks import show_image
from vanua.ui.forms import ParamsForm
from vanua.ui.shell import MainWindow, WorkflowSpec, WorkflowStep
from vanua.ui.viewers.multi_view import MultiViewHost
from vanua.ui.workers import OpWorkerThread
from vanua.core.layers import ImageLayer, LayerStack

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QSlider,
                             QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .app import LAYERS, SPOT_CHANNELS, _column, _label, _row, default_contrast, theme_color
from .display import (CellLabelOverlay, ChannelPanel, MipCompositor, OrthoCompositor, SliceCompositor, Volume3D, RENDER_MODES,
                      frame_bounds,
                      install_checkmark_style, label_slice_slider)
from .groups import (GroupSettings, cell_results, check_images, check_warnings, decide, find_duplicates,
                     load_runs, naming_warnings, rule_warnings, run_images, scan, write_results)
from .lazy import LazyOutlines, LazyPositions, open_lazy_channels, sample_for_contrast, tiff_frame
from .ops import DriftParams
from .pipeline import CHANNEL_INDEX, GENE, IDENTITIES, RoiSettings, Settings, default_workers, shown_name
from .plotwindow import ResultsWindow
from .qc import DriftSettings

APP_NAME = "HCR Cells — Groups"
APP_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = APP_DIR / "out_groups"
STATE_FILE = "groups_state.json"
IMAGE_COLUMNS = ["Day", "File", "Group", "Worm", "Drift (µm)", "Drift check", "Same worm as", "Use", "Reason"]
GROUP_COL, USE_COL = 2, 7
IDENTITY_COL, INCLUDE_COL = 1, 6  # review table
TAB_3D = 3  # MultiViewHost tabs: Slice, MIP, Ortho, 3D


def detect_experiment(folder: "str | Path") -> str:
    """'bleaching' when the folder holds '<series> - T=<n>.tif' series of
    several timepoints, else 'groups'."""
    from .series import sample_series

    try:
        series = sample_series(Path(folder))
    except Exception:  # noqa: BLE001 -- not a bleaching folder
        return "groups"
    return "bleaching" if any(len(files) > 1 for files in series.values()) else "groups"


class ImagesWindow(QWidget):
    """Every image of the experiment: its day, group (fix it here), worm
    number, the checks' results, whether it is used (tick to change) and why
    not; the warnings below. Selecting a row shows that image."""

    def __init__(self, owner: "GroupsApp", parent=None) -> None:
        super().__init__(parent, Qt.Window)
        self.owner = owner
        self.setWindowTitle("HCR Cells — Images")
        self.resize(1250, 700)
        self.table = QTableWidget(0, len(IMAGE_COLUMNS))
        self.table.setHorizontalHeaderLabels(IMAGE_COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemChanged.connect(self._on_item)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.summary = _label("")
        self.warnings = _label("")
        self.warnings.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 3)
        layout.addWidget(QLabel("<b>Warnings</b>"))
        layout.addWidget(self.warnings, 1)
        self._filling = False

    def fill(self, decided, warnings: list, groups: list) -> None:
        self._filling = True
        try:
            self.table.setRowCount(len(decided))
            for i, r in enumerate(decided.itertuples()):
                cells = [r.day, r.file, None, "" if r.worm is None or r.worm != r.worm else str(r.worm),
                         "" if not np.isfinite(r.drift_um) else f"{r.drift_um:.2f}", r.drift_decision,
                         r.duplicate_of, None, r.reason]
                for j, text in enumerate(cells):
                    if text is None:
                        continue
                    item = QTableWidgetItem(str(text))
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    if j == 0:
                        item.setData(Qt.UserRole, r.image)
                    self.table.setItem(i, j, item)
                box = QComboBox()
                box.addItem("— (unknown)", "")
                for g in groups:
                    box.addItem(g, g)
                box.setCurrentIndex(max(0, box.findData(r.group)))
                box.setToolTip(f"Written as '{r.group_raw}' ({r.group_match})")
                box.currentIndexChanged.connect(lambda _i, b=box, iid=r.image: self.owner.set_group(iid, b.currentData()))
                self.table.setCellWidget(i, GROUP_COL, box)
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                use.setCheckState(Qt.Checked if r.use else Qt.Unchecked)
                self.table.setItem(i, USE_COL, use)
            self.table.resizeColumnsToContents()
        finally:
            self._filling = False
        n_use = int(decided["use"].sum()) if len(decided) else 0
        self.summary.setText(f"<b>{len(decided)}</b> images, <b>{decided['day'].nunique() if len(decided) else 0}"
                             f"</b> days; <b>{n_use}</b> used. Fix a group in its drop-down; tick or untick "
                             "<b>Use</b> to include or leave out an image (your choice overrides the checks).")
        self.warnings.setText("<br>".join(f"<b>{w['kind']}</b>: {w['message']}" for w in warnings)
                              or "<i>No warnings.</i>")

    def _on_item(self, item) -> None:
        if self._filling or item.column() != USE_COL:
            return
        image = self.table.item(item.row(), 0).data(Qt.UserRole)
        self.owner.set_use(image, item.checkState() == Qt.Checked)

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.owner.show_image(self.table.item(rows[0].row(), 0).data(Qt.UserRole))


class GroupsApp(QObject):
    """Builds the window and owns the experiment's state. `synchronous=True`
    runs the workers' functions inline (offscreen tests)."""

    scanned = pyqtSignal()
    checked = pyqtSignal()
    ran = pyqtSignal()
    exported = pyqtSignal()

    def __init__(self, synchronous: bool = False) -> None:
        super().__init__()
        self.synchronous = synchronous
        install_checkmark_style(QApplication.instance())
        self.folder: "Path | None" = None
        self.table = None                # scan(): one row per image
        self.checks: dict = {}           # image -> drift and fingerprint
        self.duplicates = None
        self.decided = None              # decide(): which images are used and why not
        self.runs: dict = {}             # image -> its run (cells table, drift, ROI)
        self.cells = None                # cell_results(): one row per cell
        self.user: dict = {}             # image -> {"use": bool, "group": str}: your choices
        self.review: dict = {}           # image -> {"overrides": {cell: bool}, "identities": {cell: neuron}}
        self.warnings: list = []
        self.current: "str | None" = None  # the image shown
        self.channels: dict = {}
        self.contrast: dict = {}
        self.frame = None
        self._threads: list = []
        self._view = {}                  # the shown image: labels, outlines, positions, spots, table

        self.viewer = MultiViewHost()
        steps = (
            WorkflowStep("1  Images", self._build_images()),
            WorkflowStep("2  Check images", self._build_check()),
            WorkflowStep("3  Run", self._build_run()),
            WorkflowStep("4  Review cells", self._build_review()),
            WorkflowStep("5  Export CSV", self._build_export()),
        )
        self.window = MainWindow(WorkflowSpec(title=APP_NAME, steps=steps, viewer=self.viewer,
                                              rendering_panel=self._build_display()))
        self.window.rendering_dock.setWindowTitle("Display")
        self.images_window = ImagesWindow(self, self.window)
        self.results_window = ResultsWindow(self.window, out_dir=lambda: self.out_dir)
        for i in (1, 2, 3, 4):
            self.window.set_step_enabled(i, False)
        self.window.cancel_button.clicked.connect(self._cancel)

    # -- pages ------------------------------------------------------------------------

    def _build_images(self) -> QWidget:
        self.folder_edit = QLineEdit("")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_folder)
        self.groups_edit = QLineEdit("")
        self.groups_edit.setPlaceholderText("e.g. N2, ins-6, ins-6rescue (empty: from the names)")
        self.groups_edit.setToolTip("The groups. File names are matched to the nearest one (typos up to "
                                    "2 letters, names of 4+ letters); empty: learnt from the file names")
        self.out_edit = QLineEdit(str(DEFAULT_OUT))
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._pick_out)
        self.scan_button = QPushButton("Scan folder")
        theme.style_button_role(self.scan_button, "primary")
        self.scan_button.clicked.connect(lambda: self.scan(self.folder_edit.text()))
        self.images_button = QPushButton("Open image table")
        self.images_button.clicked.connect(self.open_images_window)
        self.images_status = _label("A folder with one sub-folder per imaging day (its name has the date), in "
                                    "each one TIFF per worm (ZCYX: C0 daf-28, C1 ins-4, C2 ins-6, C3 DAPI), "
                                    "named &lt;group&gt;_…_&lt;worm number&gt;.tif.")
        page, _ = _column(QLabel("Image folder"), _row(self.folder_edit, browse), QLabel("Groups"),
                          self.groups_edit, QLabel("Output folder"), _row(self.out_edit, out_browse),
                          self.scan_button, self.images_button, self.images_status)
        return page

    def _build_check(self) -> QWidget:
        self.drift_form = ParamsForm(DriftParams())
        self.dup_form = ParamsForm(_DuplicateSettings())
        self.check_button = QPushButton("Check drift and duplicates")
        theme.style_button_role(self.check_button, "primary")
        self.check_button.clicked.connect(self.check)
        self.check_status = _label("Each image: how far its slices drifted sideways through the stack (DAPI), "
                                   "and whether it shows the same worm as another image of that day (DAPI "
                                   "nuclei pattern). Only DAPI is read, so this is quick; results are cached.")
        page, _ = _column(self.check_button, QLabel("<b>Drift</b>"), self.drift_form, QLabel("<b>Same worm</b>"),
                          self.dup_form, self.check_status)
        return page

    def _build_run(self) -> QWidget:
        self.settings_form = ParamsForm(Settings())
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.workers_spin.setValue(default_workers())
        self.workers_spin.setPrefix("Parallel workers: ")
        self.workers_spin.setToolTip("Images processed at the same time, each in its own process (~2.5 GB of "
                                     "memory each); the results are identical either way.")
        self.run_button = QPushButton("Run all images")
        theme.style_button_role(self.run_button, "primary")
        self.run_button.clicked.connect(self.run)
        self.run_status = _label("Every image used: the drift corrected where needed, the somata segmented "
                                 "on daf-28, spots in every HCR channel for every cell. Which cells count is "
                                 "decided afterwards (rule, review), without a rerun. <b>Cells without</b> sets "
                                 "the rule: in a group that lacks that transcript (e.g. ins-6 in an ins-6 "
                                 "mutant) choose another channel or review the cells.")
        page, _ = _column(self.run_button, self.workers_spin, self.settings_form, self.run_status)
        return page

    def _build_review(self) -> QWidget:
        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(lambda _i: self.show_image(self.image_combo.currentData()))
        prev_b, next_b = QPushButton("◀ Previous"), QPushButton("Next ▶")
        prev_b.clicked.connect(lambda: self._step_image(-1))
        next_b.clicked.connect(lambda: self._step_image(1))
        self.cell_table = QTableWidget(0, 7)
        self.cell_table.setHorizontalHeaderLabels(["Cell", "Identity", "z", "Volume (µm³)", "Rule channel − bg",
                                                   "Rule", "Include"])
        self.cell_table.verticalHeader().hide()
        self.cell_table.setMinimumHeight(220)
        self.cell_table.itemChanged.connect(self._on_cell_item)
        self.cell_table.itemSelectionChanged.connect(self._jump_to_cell)
        self.review_note = _label("Per image: tick or untick <b>Include</b>, and name each cell of interest in "
                                  "<b>Identity</b> (ASIL, ASIR, ASJL, ASJR; each once per worm). The CSV follows "
                                  "at once.")
        page, _ = _column(self.image_combo, _row(prev_b, next_b, stretch_first=False), self.cell_table,
                          self.review_note)
        return page

    def _build_export(self) -> QWidget:
        self.export_button = QPushButton("Write CSV")
        theme.style_button_role(self.export_button, "primary")
        self.export_button.clicked.connect(lambda: self.export())
        self.table_button = QPushButton("Open results table")
        self.table_button.clicked.connect(self.open_results_window)
        self.export_status = _label("<b>transcripts_per_image.csv</b>: one row per image used -- date, group, "
                                    "worm, file, number of cells of interest, and per channel the dots (spot "
                                    "peaks) and the transcript estimate (Big-FISH) summed over them; per named "
                                    "neuron too (ASIL_ins6_dots, …). With it: cells.csv (one row per cell) and "
                                    "images_qc.csv (every image, used or why not).")
        page, _ = _column(self.export_button, self.table_button, self.export_status)
        return page

    def _build_display(self) -> QWidget:
        self.layers = LayerStack()
        self.layer_panel = ChannelPanel(self.layers)
        self.compositor = SliceCompositor(self.layers, self.viewer.slice_viewer)
        tabs = self.viewer.as_widget()
        self.mip_compositor = MipCompositor(self.layers, self.viewer.mip_viewer, tabs)
        self.ortho_compositor = OrthoCompositor(self.layers, self.viewer.ortho_viewer, tabs,
                                                slice_viewer=self.viewer.slice_viewer)
        self.volume_3d = Volume3D(self.layers, self.viewer, lambda: self.frame, tab_index=TAB_3D)
        # 3D: the composite of the channels above (their LUT, min / max, gamma, opacity, visibility).
        self.volume_check = QCheckBox("Image in 3D")
        self.volume_check.setChecked(True)
        self.volume_check.setToolTip("The image in the 3D tab as the same colour composite as the 2D views, set "
                                     "by the channel controls above; below a channel's min it is clear")
        self.volume_check.toggled.connect(self.volume_3d.set_enabled)
        self.render_mode = QComboBox()
        for key, text in RENDER_MODES.items():
            self.render_mode.addItem(text, key)
        self.render_mode.setToolTip("Additive: each channel's brightest point along the view, the channels' "
                                    "colours added (napari's look). Composite: see-through, near voxels hide "
                                    "far ones (depth).")
        self.render_mode.currentIndexChanged.connect(lambda _i: self.volume_3d.set_mode(self.render_mode.currentData()))
        self.volume_density = QSlider(Qt.Horizontal)
        self.volume_density.setRange(5, 100)
        self.volume_density.setValue(50)
        self.volume_density.setToolTip("3D opacity: how opaque the brightest voxels are (lower: see deeper "
                                       "into the volume and the cells / spots inside it)")
        self.volume_density.valueChanged.connect(lambda v: self.volume_3d.set_density(v / 100.0))
        self.surface_opacity = QSlider(Qt.Horizontal)
        self.surface_opacity.setRange(0, 100)
        self.surface_opacity.setValue(50)
        self.surface_opacity.setToolTip("How opaque the segmented cells' surfaces are in 3D (0: hidden)")
        self.surface_opacity.valueChanged.connect(lambda v: self._set_surface_opacity(v / 100.0))
        self._3d_actors: list = []
        label_slice_slider(self.viewer.slice_viewer, "Z", lambda: self._step_um(0))
        self.cell_labels = CellLabelOverlay(self.viewer.slice_viewer)
        self.compositor.z_listeners.append(self.cell_labels.refresh)
        self.show_outlines = QCheckBox("Cell outlines")
        self.show_outlines.setChecked(True)
        self.show_outlines.toggled.connect(lambda _: self._show_overlays())
        self.show_cell_labels = QCheckBox("Cell labels")
        self.show_cell_labels.setChecked(True)
        self.show_cell_labels.toggled.connect(lambda _: self._show_overlays())
        self.show_spots = QCheckBox("Spot markers")
        self.show_spots.setChecked(True)
        self.show_spots.toggled.connect(lambda _: self._show_overlays())
        self.spot_combo = QComboBox()
        self.spot_combo.addItems(list(SPOT_CHANNELS))
        self.spot_combo.currentIndexChanged.connect(lambda _: self._show_overlays())
        self.display_status = _label("")
        page, _ = _column(self.display_status, QLabel("<b>Channels</b>"), self.layer_panel, QLabel("<b>Overlays</b>"),
                          self.show_outlines, self.show_cell_labels, self.show_spots, QLabel("Spots of channel"),
                          self.spot_combo, QLabel("<b>3D</b>"), self.volume_check, QLabel("Rendering"),
                          self.render_mode, QLabel("Image opacity in 3D"),
                          self.volume_density, QLabel("Cell surface opacity"), self.surface_opacity)
        return page

    # -- paths and state ----------------------------------------------------------------

    @property
    def out_dir(self) -> Path:
        return Path(self.out_edit.text())

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self.window, "Image folder (one sub-folder per day)",
                                                self.folder_edit.text())
        if path:
            self.folder_edit.setText(path)
            self.scan(path)

    def _pick_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self.window, "Output folder", self.out_edit.text())
        if path:
            self.out_edit.setText(path)

    def _group_settings(self) -> GroupSettings:
        d = self.dup_form.values()
        return GroupSettings(groups=self.groups_edit.text(), duplicate_above=d.duplicate_above,
                             possible_above=d.possible_above)

    def _drift_settings(self) -> DriftSettings:
        d = self.drift_form.values()
        return DriftSettings(d.register_above, d.jump_above, d.exclude_above)

    def _save_state(self) -> None:
        if self.folder is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        state = {"folder": str(self.folder), "groups": self.groups_edit.text(), "user": self.user,
                 "review": self.review, "settings": asdict(self.settings_form.values()),
                 "drift": asdict(self.drift_form.values()), "duplicates": asdict(self.dup_form.values())}
        (self.out_dir / STATE_FILE).write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _load_state(self) -> dict:
        path = self.out_dir / STATE_FILE
        if not path.exists():
            return {}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if Path(state.get("folder", "")).resolve() != Path(self.folder).resolve():
            return {}  # another experiment's choices
        return state

    # -- workers ----------------------------------------------------------------------

    def _start(self, fn, on_success, label: str, buttons=()) -> None:
        for b in buttons:
            b.setEnabled(False)

        def _end():
            for b in buttons:
                b.setEnabled(True)
            QApplication.restoreOverrideCursor()

        QApplication.setOverrideCursor(Qt.BusyCursor)
        if self.synchronous:
            import threading

            try:
                result = fn(None, None, lambda *a: None, threading.Event(), self.window.append_log)
            finally:
                _end()
            on_success(result)
            return
        thread = OpWorkerThread(fn, None, None)
        thread.progress.connect(lambda e: self.window.show_progress(e.fraction, f"{label}: {e.message}"))
        thread.log.connect(self.window.append_log)
        thread.succeeded.connect(lambda r: (_end(), self.window.clear_progress(), on_success(r)))
        thread.failed.connect(lambda msg: (_end(), self.window.clear_progress("Failed"),
                                           self.window.append_log(f"{label} failed: {msg}")))
        thread.cancelled.connect(lambda: (_end(), self.window.clear_progress("Cancelled")))
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)
        self._threads.append(thread)
        self.window.show_progress(0.0, f"{label}…")
        thread.start()

    def _cancel(self) -> None:
        for thread in list(self._threads):
            thread.cancel()

    # -- 1 images ---------------------------------------------------------------------

    def scan(self, folder: "str | Path") -> int:
        folder = Path(folder)
        if not folder.is_dir():
            self.images_status.setText(f"No such folder: {folder}")
            return 0
        self.folder = folder
        self.folder_edit.setText(str(folder))
        state = self._load_state()
        if state:
            self.groups_edit.setText(state.get("groups", ""))
            self.user = state.get("user", {})
            self.review = state.get("review", {})
            for form, cls, key in ((self.settings_form, Settings, "settings"), (self.drift_form, DriftParams, "drift"),
                                   (self.dup_form, _DuplicateSettings, "duplicates")):
                known = {f.name for f in fields(cls)}
                if state.get(key):
                    form.set_values(cls(**{k: v for k, v in state[key].items() if k in known}))
        else:
            self.user, self.review = {}, {}
        self.table = scan(folder, self._group_settings())
        self.checks, self.duplicates, self.runs, self.cells = {}, None, {}, None
        self._decide()
        self.images_status.setText(
            f"<b>{len(self.table)}</b> images in <b>{self.table['day'].nunique()}</b> day folder(s); groups: "
            + ", ".join(f"{g or '(unknown)'} ({n})" for g, n in self.table["group"].value_counts().items())
            + ("<br>Your earlier choices for this folder were restored." if state else ""))
        self.window.set_step_enabled(1, len(self.table) > 0)
        self.window.set_current_step(1)
        self.window.append_log(f"Scanned {folder}: {len(self.table)} images")
        self.open_images_window()
        self.scanned.emit()
        # Checks and runs done before (same files): reuse them.
        cache = self.out_dir / "qc_cache"
        if cache.exists() and len(self.table):
            self.check()
        return len(self.table)

    def _decide(self) -> None:
        import pandas as pd

        dups = self.duplicates if self.duplicates is not None else pd.DataFrame(
            columns=["day", "image_a", "image_b", "similarity", "verdict"])
        self.decided = decide(self.table, self.checks, dups, self._drift_settings(), self.user)
        self.warnings = naming_warnings(self.table) + (check_warnings(self.decided, dups) if self.checks else [])
        groups = sorted(g for g in set(self.table["group"]) | {g.strip() for g in self.groups_edit.text().split(",")}
                        if g)
        self.images_window.fill(self.decided, self.warnings + self._rule_warnings(), groups)

    def _rule_warnings(self) -> list:
        return rule_warnings(self.cells, self.settings_form.values()) if self.cells is not None else []

    def set_group(self, image: str, group: str) -> None:
        self.user.setdefault(image, {})["group"] = group
        self._choices_changed()

    def set_use(self, image: str, use: bool) -> None:
        self.user.setdefault(image, {})["use"] = bool(use)
        self._choices_changed()

    def _choices_changed(self) -> None:
        QTimer.singleShot(0, self._apply_choices)  # not while the table emits

    def _apply_choices(self) -> None:
        self._decide()
        self._save_state()
        if self.runs:
            self._results_changed()

    def open_images_window(self) -> None:
        self.images_window.show()
        self.images_window.raise_()

    # -- 2 check ----------------------------------------------------------------------

    def check(self) -> None:
        if self.table is None:
            return
        table, cache, workers = self.table, self.out_dir / "qc_cache", self.workers_spin.value()
        channel = self._drift_settings().channel

        def _job(params, ctx, progress, cancel, log):
            return check_images(table, cache, channel, workers, progress=progress, cancel=cancel.is_set)

        self.check_status.setText("Checking…")
        self._start(_job, self._on_checked, "Check", [self.check_button, self.run_button])

    def _on_checked(self, checks: dict) -> None:
        self.checks = checks
        self.duplicates = find_duplicates(self.table, checks, self._group_settings())
        self._decide()
        d = self.decided
        n = {k: int((d["drift_decision"] == k).sum()) for k in ("ok", "registered", "excluded")}
        n_dup = int((d["duplicate_of"] != "").sum())
        self.check_status.setText(f"Drift: <b>{n['ok']}</b> fine, <b>{n['registered']}</b> corrected, "
                                  f"<b>{n['excluded']}</b> left out. Same worm saved twice: <b>{n_dup}</b> image(s) "
                                  f"not used. <b>{int(d['use'].sum())}</b> of {len(d)} images will be analysed. "
                                  "Details and every warning: the image table.")
        self.window.set_step_enabled(2, True)
        self.window.set_current_step(2)
        self._save_state()
        self.checked.emit()
        # Images run before with the same settings: their results are reused.
        done = load_runs(self.out_dir / "images", list(d.loc[d["use"], "image"]))
        key = self._run_key()
        if done and (self.out_dir / "images" / "settings.json").exists() and \
                json.loads((self.out_dir / "images" / "settings.json").read_text(encoding="utf-8")) == key:
            self.runs = done
            self._results_changed()
            self.window.append_log(f"Reused the results of {len(done)} image(s) run before with these settings")

    # -- 3 run ------------------------------------------------------------------------

    def _run_key(self) -> dict:
        return {"settings": asdict(self.settings_form.values()), "drift": asdict(self.drift_form.values())}

    def run(self) -> None:
        if self.decided is None:
            return
        decided, settings, drift = self.decided, self.settings_form.values(), self._drift_settings()
        save, workers, key = self.out_dir / "images", self.workers_spin.value(), self._run_key()
        old = {}
        settings_file = save / "settings.json"
        if settings_file.exists() and json.loads(settings_file.read_text(encoding="utf-8")) == key:
            old = load_runs(save, list(decided.loc[decided["use"], "image"]))
        todo = decided[decided["use"] & ~decided["image"].isin(list(old))]

        def _job(params, ctx, progress, cancel, log):
            save.mkdir(parents=True, exist_ok=True)
            settings_file.write_text(json.dumps(key, indent=2), encoding="utf-8")
            new = run_images(todo, settings, RoiSettings(), drift, save, workers, progress=progress,
                             cancel=cancel.is_set, log=log)
            return old | new

        self.run_status.setText(f"Running {len(todo)} image(s)" + (f" ({len(old)} reused)…" if old else "…"))
        self._start(_job, self._on_run, "Run", [self.run_button, self.check_button])

    def _on_run(self, runs: dict) -> None:
        self.runs = runs
        self._save_state()
        self._results_changed()
        self.run_status.setText(f"<b>{len(runs)}</b> image(s) analysed. Review the cells (step 4) or write the "
                                "CSV (step 5) -- it is written already and follows every review change.")
        self.window.set_current_step(3)
        self.ran.emit()

    # -- results ----------------------------------------------------------------------

    def _overrides(self) -> dict:
        return {(img, c): bool(v) for img, r in self.review.items() for c, v in r.get("overrides", {}).items()}

    def _identities(self) -> dict:
        return {(img, c): v for img, r in self.review.items() for c, v in r.get("identities", {}).items() if v}

    def _results_changed(self) -> None:
        """The cell table from the runs, the rule and your review; the CSVs; the review list."""
        self.cells = cell_results(self.decided, self.runs, self.settings_form.values(), self._overrides(),
                                  self._identities())
        self.export()
        self.window.set_step_enabled(3, bool(self.runs))
        self.window.set_step_enabled(4, bool(self.runs))
        keep = self.image_combo.currentData()
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for image in sorted(self.runs):
            if image in set(self.decided.loc[self.decided["use"], "image"]):
                self.image_combo.addItem(image, image)
        if keep is not None and self.image_combo.findData(keep) >= 0:
            self.image_combo.setCurrentIndex(self.image_combo.findData(keep))
        self.image_combo.blockSignals(False)
        if self.current in self.runs:
            self._fill_cells()
            self._show_overlays()

    def export(self) -> "dict | None":
        if self.decided is None:
            return None
        import pandas as pd

        cells = self.cells if self.cells is not None else pd.DataFrame()
        doc = {"folder": str(self.folder), "groups": asdict(self._group_settings()),
               "settings": asdict(self.settings_form.values()), "drift": asdict(self._drift_settings()),
               "roi": asdict(RoiSettings()), "review": self.review, "your_choices": self.user}
        paths = write_results(self.out_dir, self.decided, cells, self.warnings + self._rule_warnings(), doc)
        self.paths = paths
        self.export_status.setText(f"Wrote <b>{paths['transcripts_per_image'].name}</b> "
                                   f"({int(self.decided['use'].sum())} images), cells.csv and images_qc.csv in "
                                   f"{self.out_dir}")
        if self.results_window.isVisible():
            self.open_results_window()
        self.exported.emit()
        return paths

    def open_results_window(self) -> None:
        import pandas as pd

        paths = getattr(self, "paths", None) or self.export()
        if not paths:
            return
        tables = {"transcripts per image": pd.read_csv(paths["transcripts_per_image"]),
                  "cells": pd.read_csv(paths["cells"]), "images (QC)": pd.read_csv(paths["images_qc"])}
        sources = {"transcripts per image": paths["transcripts_per_image"], "cells": paths["cells"],
                   "images (QC)": paths["images_qc"]}
        self.results_window.set_results(tables, sources, "transcripts per image")
        self.results_window.show()
        self.results_window.raise_()

    # -- 4 review and the display ---------------------------------------------------------

    def _step_image(self, delta: int) -> None:
        n = self.image_combo.count()
        if n:
            self.image_combo.setCurrentIndex((self.image_combo.currentIndex() + delta) % n)

    def _step_um(self, axis: int):
        return float(np.linalg.norm(np.asarray(self.frame["affine"])[:3, axis])) if self.frame else None

    def show_image(self, image: "str | None") -> None:
        """Show one image: its channels (moved back into line if its drift was
        corrected, as the analysis saw it), cells and spots once it has run."""
        if not image or self.table is None or image not in set(self.table["image"]):
            return
        row = self.decided[self.decided["image"] == image].iloc[0]
        path, registered = Path(row["path"]), row["drift_decision"] in ("registered", "excluded") and \
            image in self.runs
        run_dir = self.out_dir / "images" / image

        def _job(params, ctx, progress, cancel, log):
            import pandas as pd

            frame = tiff_frame(path)
            if registered and (run_dir / "drift.csv").exists():
                from .lazy import TiffStack
                from .qc import apply_z_drift

                drift = pd.read_csv(run_dir / "drift.csv")
                stack = TiffStack(path)
                try:
                    channels = {name: apply_z_drift(np.stack([stack.read(z, c) for z in range(stack.n_z)]), drift)
                                for name, c in CHANNEL_INDEX.items()}
                finally:
                    stack.close()
                contrast = {k: default_contrast(v) for k, v in channels.items()}
            else:
                channels = open_lazy_channels(path, CHANNEL_INDEX)
                z = next(iter(channels.values())).shape[0] // 2
                contrast = {}
                for k, ch in channels.items():
                    sample_for_contrast(ch, z)
                    contrast[k] = default_contrast(ch)
            view = {}
            if image in self.runs and (run_dir / "cells.npz").exists():
                with np.load(run_dir / "cells.npz") as f:
                    labels = f["labels"]
                view = {"labels": labels, "outlines": LazyOutlines(labels), "positions": LazyPositions(labels),
                        "spots": {ch: pd.read_csv(run_dir / f"spots_{ch}.csv") for ch in SPOT_CHANNELS.values()
                                  if (run_dir / f"spots_{ch}.csv").exists()}}
            return image, channels, contrast, frame, view

        self._start(_job, self._on_image, "Image")

    def _on_image(self, result) -> None:
        image, channels, contrast, frame, view = result
        self.current, self.channels, self.contrast = image, channels, contrast
        self.frame = frame.as_document()
        self._view = view
        self.build_surfaces()
        spacing = tuple(float(s) for s in frame.spacing()[:3])
        sv = self.viewer.slice_viewer
        sv.set_overlay_labels(None, {}, {})
        sv.set_skeleton_points(None)
        show_image(sv, channels["daf28"], self.frame)
        keep = {layer.name: layer for layer in self.layers} if len(self.layers) else {}
        self.layer_panel.set_layers([
            ImageLayer(name=name, data=channels[key], spacing_um_zyx=spacing,
                       visible=keep[name].visible if name in keep else visible,
                       lut_name=keep[name].lut_name if name in keep else lut, contrast_range=contrast[key])
            for name, key, lut, visible in LAYERS])
        sv.fit_to_view()
        self.compositor.refresh()
        row = self.decided[self.decided["image"] == image].iloc[0]
        self.display_status.setText(f"<b>{row['group'] or '(unknown group)'}</b> worm {row['worm']}, {row['day']}"
                                    f"<br>{row['file']}<br>drift {row['drift_um']:.2f} µm: {row['drift_decision']}"
                                    + ("" if row["use"] else f"<br><b>not used</b>: {row['reason']}"))
        if image in self.runs and self.image_combo.findData(image) >= 0:
            self.image_combo.blockSignals(True)
            self.image_combo.setCurrentIndex(self.image_combo.findData(image))
            self.image_combo.blockSignals(False)
        self._fill_cells()
        self._show_overlays()

    # -- 3D: the image as a volume (Volume3D), the cells as surfaces, the spots ---------------

    def initialize_3d(self) -> None:
        """Create the 3D view (needs a real display) the first time its tab is opened."""
        if self.viewer.viewer_3d is not None:
            return
        tabs = self.viewer.as_widget()
        tabs.blockSignals(True)
        try:
            self.viewer.initialize()
            tabs.setCurrentIndex(TAB_3D)
        finally:
            tabs.blockSignals(False)
        if self.viewer.viewer_3d is not None:
            self.viewer.viewer_3d.set_orientation_labels("X+", "X-", "Y+", "Y-", "Z+", "Z-")
            self.volume_3d.refresh()
            self.build_surfaces()

    def _set_surface_opacity(self, opacity: float) -> None:
        """The cell surfaces in 3D: more or less see-through (the slider)."""
        v3 = self.viewer.viewer_3d
        for actor in self._3d_actors:
            actor.GetProperty().SetOpacity(opacity)
            actor.SetVisibility(opacity > 0)
        if v3 is not None:
            v3.render()

    def _clear_3d(self) -> None:
        from vanua.ui.blocks import clear_label_surfaces

        if self.viewer.viewer_3d is not None and self._3d_actors:
            clear_label_surfaces(self.viewer.viewer_3d, self._3d_actors)
        self._3d_actors = []
        self._3d_spots = None

    def build_surfaces(self) -> None:
        v3 = self.viewer.viewer_3d
        if v3 is None:
            return
        self._clear_3d()
        if "labels" not in self._view or self.frame is None:
            v3.render()
            return
        from vanua.blocks.mesh import label_surfaces

        labels, frame, image = self._view["labels"], self.frame, self.current

        def _job(params, ctx, progress, cancel, log):
            mesh, _table = label_surfaces(labels, frame, step_size=1, cancel=cancel, progress=progress)
            return image, mesh

        self._start(_job, self._on_surfaces, "Surfaces")

    def _on_surfaces(self, result) -> None:
        from vanua.ui.blocks import add_label_surfaces

        image, mesh = result
        if image != self.current or self.viewer.viewer_3d is None:
            return  # another image is shown by now
        self._clear_3d()
        colors = label_colors(np.unique(self._view["labels"]))
        self._3d_actors = add_label_surfaces(self.viewer.viewer_3d, mesh, colors=colors,
                                             opacity=self.surface_opacity.value() / 100.0)
        self._set_surface_opacity(self.surface_opacity.value() / 100.0)
        frame_bounds(self.viewer.viewer_3d, self.viewer.viewer_3d.bounds)
        self._show_3d_spots(self._view_overlays()[2])

    def _show_3d_spots(self, points) -> None:
        v3 = self.viewer.viewer_3d
        if v3 is None:
            return
        from .app import _points_actor

        old = getattr(self, "_3d_spots", None)
        if old is not None:
            v3.renderer.RemoveActor(old)
            self._3d_spots = None
        if points is not None and len(points) and self.frame is not None:
            spacing = np.array([self._step_um(i) for i in range(3)], float)
            actor = _points_actor((points * spacing)[:, ::-1], (255, 255, 0))
            v3.renderer.AddActor(actor)
            self._3d_spots = actor
        v3.render()

    def _image_cells(self):
        if self.cells is None or not len(self.cells) or self.current is None:
            return None
        return self.cells[self.cells["image"] == self.current].sort_values("cell").reset_index(drop=True)

    def _fill_cells(self) -> None:
        t = self._image_cells()
        self.cell_table.blockSignals(True)
        self.cell_table.setRowCount(0 if t is None else len(t))
        if t is not None:
            s = self.settings_form.values()
            rule_col = f"{s.exclude_channel}_mean_bgsub_au" if s.exclude_channel != "none" else None
            colors = label_colors(np.unique(self._view.get("labels", np.zeros(1, int))))
            names = self.review.get(self.current, {}).get("identities", {})
            for i, row in t.iterrows():
                z = row["centroid_z_um"] / (self._step_um(0) or 1.0)
                name = QTableWidgetItem(row["cell"])
                name.setForeground(theme_color(colors.get(int(row["label"]), (200, 200, 200))))
                name.setData(Qt.UserRole, int(row["label"]))
                rule = "excluded" if row["excluded_by_rule"] else "kept"
                if rule_col:
                    rule += f": {GENE[s.exclude_channel]} {row[rule_col]:.1f}"
                include = QTableWidgetItem()
                include.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                include.setCheckState(Qt.Checked if row["included"] else Qt.Unchecked)
                items = [name, QTableWidgetItem(""), QTableWidgetItem(f"{z:.0f}"),
                         QTableWidgetItem(f"{row['volume_um3']:.1f}"),
                         QTableWidgetItem(f"{row[rule_col]:.1f}" if rule_col else ""), QTableWidgetItem(rule), include]
                for j, item in enumerate(items):
                    if j < INCLUDE_COL:
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self.cell_table.setItem(i, j, item)
                box = QComboBox()
                box.addItem("—", "")
                for n in IDENTITIES:
                    box.addItem(n, n)
                box.setCurrentIndex(max(0, box.findData(names.get(row["cell"], ""))))
                box.currentIndexChanged.connect(lambda _i, b=box, c=row["cell"]: self.set_identity(c, b.currentData()))
                self.cell_table.setCellWidget(i, IDENTITY_COL, box)
        self.cell_table.blockSignals(False)
        self.cell_table.resizeColumnsToContents()

    def _on_cell_item(self, item) -> None:
        if item.column() != INCLUDE_COL or self.current is None:
            return
        cell = self.cell_table.item(item.row(), 0).text()
        self.set_include(cell, item.checkState() == Qt.Checked)

    def set_include(self, cell: str, include: bool) -> None:
        """Your choice for a cell of the shown image (stored only where it differs from the rule)."""
        t = self._image_cells()
        by_rule = not bool(t.loc[t["cell"] == cell, "excluded_by_rule"].iat[0]) if t is not None else True
        over = self.review.setdefault(self.current, {}).setdefault("overrides", {})
        if include == by_rule:
            over.pop(cell, None)
        else:
            over[cell] = bool(include)
        self._review_changed()

    def set_identity(self, cell: str, identity: str) -> None:
        """Name a cell of the shown image; each neuron goes to one cell per worm."""
        names = self.review.setdefault(self.current, {}).setdefault("identities", {})
        if identity:
            for other in [c for c, v in names.items() if v == identity and c != cell]:
                names.pop(other)
        if identity:
            names[cell] = identity
        else:
            names.pop(cell, None)
        self._review_changed()

    def _review_changed(self) -> None:
        self._save_state()
        QTimer.singleShot(0, self._results_changed)

    def _jump_to_cell(self) -> None:
        rows = self.cell_table.selectionModel().selectedRows() or [
            self.cell_table.model().index(i.row(), 0) for i in self.cell_table.selectedIndexes()[:1]]
        if not rows or "labels" not in self._view:
            return
        label = self.cell_table.item(rows[0].row(), 0).data(Qt.UserRole)
        zs = np.nonzero((self._view["labels"] == label).any(axis=(1, 2)))[0]
        if len(zs):
            self.viewer.slice_viewer.set_z(int(zs[len(zs) // 2]))

    def _view_overlays(self):
        """(labels or None, colors, spot points or None) of the image shown."""
        view, run = self._view, self.runs.get(self.current, {})
        if not view:
            return None, {}, None
        colors = label_colors(np.unique(view["labels"]))
        labels = view["labels"] if self.show_outlines.isChecked() else None
        spots = view["spots"].get(SPOT_CHANNELS[self.spot_combo.currentText()])
        points = None
        if spots is not None and self.show_spots.isChecked() and len(spots):
            points = spots[["z_index", "y_index", "x_index"]].to_numpy(float)
            box = run.get("ref_roi") or run.get("roi")
            if box:
                points = points + np.array([box["z"][0], box["y"][0], box["x"][0]], float)
        return labels, colors, points

    def _show_overlays(self) -> None:
        labels, colors, points = self._view_overlays()
        for v in (self.mip_compositor, self.ortho_compositor):
            v.set_overlays(labels, colors, points, (255, 255, 0))
        self._show_3d_spots(points)
        sv = self.viewer.slice_viewer
        view = self._view
        t = self._image_cells()
        if not view or t is None:
            sv.set_overlay_labels(None, {}, {})
            sv.set_skeleton_points(None)
            self.cell_labels.clear()
            return
        colors = label_colors(np.unique(view["labels"]))
        names = self.review.get(self.current, {}).get("identities", {})
        shown = {int(lbl): shown_name(c, names.get(c, "")) for lbl, c in zip(t["label"], t["cell"])}
        excluded = {shown[int(lbl)] for lbl, inc in zip(t["label"], t["included"]) if not inc}
        if self.show_outlines.isChecked():
            sv.set_overlay_labels(view["outlines"], colors, {i: 255 for i in colors})
        else:
            sv.set_overlay_labels(None, {}, {})
        self.cell_labels.visible = self.show_cell_labels.isChecked()
        self.cell_labels.set_cells(view["positions"], shown, colors, excluded)
        ch = SPOT_CHANNELS[self.spot_combo.currentText()]
        spots = view["spots"].get(ch)
        run = self.runs.get(self.current, {})
        if spots is None or not self.show_spots.isChecked() or not len(spots):
            sv.set_skeleton_points(None)
            return
        box = run.get("ref_roi") or run.get("roi")
        points = spots[["z_index", "y_index", "x_index"]].to_numpy(float)
        if box:
            points = points + np.array([box["z"][0], box["y"][0], box["x"][0]], float)
        sv.set_skeleton_points(points, z_tolerance=1.0, color=(255, 255, 0))


@dataclass
class _DuplicateSettings:
    """The same-worm thresholds of GroupSettings, as a form."""

    duplicate_above: float = field(default=0.90, metadata={
        "label": "Same worm above", "range": (0.0, 1.0), "step": 0.01,
        "help": "Similarity of two DAPI nuclei patterns above which two images of one day are the same worm "
                "(only the first is used). Same worm re-imaged: 0.93-1.00; two worms: about 0"})
    possible_above: float = field(default=0.75, metadata={
        "label": "Flag as possibly the same above", "range": (0.0, 1.0), "step": 0.01,
        "help": "Between this and 'same worm' both images are kept and a warning is given"})


def build_app(synchronous: bool = False) -> GroupsApp:
    return GroupsApp(synchronous=synchronous)


def show(app_obj: GroupsApp, init_3d: bool = False) -> None:
    app_obj.window.resize(1500, 950)
    app_obj.window.show()
    theme.apply_dark_titlebar(app_obj.window)
    # The 3D view is created the first time its tab is opened (needs the window shown).
    tabs = app_obj.viewer.as_widget()
    tabs.currentChanged.connect(lambda i: app_obj.initialize_3d() if i == TAB_3D else None)
    if init_3d:
        app_obj.initialize_3d()


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    theme.enable_high_dpi()
    qapp = QApplication(argv)
    qapp.setOrganizationName("hcrcells")
    qapp.setApplicationName(APP_NAME)
    theme.install(qapp, app_id="org.vanua.hcrcells", app_name="HCR Cells")
    hcr = build_app()
    show(hcr)
    if len(argv) > 1:
        hcr.scan(argv[1])
    return qapp.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
