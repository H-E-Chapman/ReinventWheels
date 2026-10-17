"""HCR Cells GUI on the vanua shell:
Load -> Segment -> Review cells -> Run series -> Plots, with a Tutorial / Demo
mode (vanua.ui.guide.GuidedTour; View menu).

    PYTHONPATH=src pixi run --manifest-path <vanua>/pixi.toml python -m hcrcells.app [FOLDER]

Segment runs the timepoint recipe (recipes/timepoint.toml via
:mod:`hcrcells.pipeline`) on the sample's first timepoint, off the GUI thread;
Run series runs the same recipe on every timepoint (:func:`hcrcells.series.track_series`),
so the GUI and the headless runner compute the same numbers.

The Display panel (right dock) picks the channel on the slice view and
toggles cell outlines, spot markers (that channel's spots) and the 3D spots.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

# vanua.ui before anything that imports VTK (building.md §5).
from vanua.ui import theme  # noqa: I001
from vanua.blocks.io import read_volume
from vanua.blocks.mesh import label_surfaces
from vanua.blocks.plot import label_colors
from vanua.ui.blocks import (add_label_surfaces, clear_label_surfaces, show_image,
                             world_to_display)
from vanua.ui.forms import ParamsForm
from vanua.ui.guide import GuidedTour, GuideStep
from vanua.ui.shell import MainWindow, WorkflowSpec, WorkflowStep
from vanua.ui.viewers.multi_view import MultiViewHost
from vanua.ui.workers import OpWorkerThread
from vanua.core.layers import ImageLayer, LayerStack

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal

from .lazy import LazyOutlines, LazyPositions, open_lazy_channels, sample_for_contrast, tiff_frame
from .display import (CellLabelOverlay, ChannelPanel, RoiOverlay, SliceCompositor, SpotColorPicker, install_checkmark_style,
                      label_positions, label_slice_slider, lut_color)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QSlider,
                             QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .ops import format_targets, roi_box
from .pipeline import (CELLS_RECIPE_PATH, CHANNEL_INDEX, CHANNELS, GENE, IDENTITIES, RECIPE_PATH, RoiSettings,
                       Settings, apply_exclusion, apply_identities, shown_name, configured, default_workers, load_cells_recipe, load_recipe, registry,
                       run_timepoint, segment_cells)
from .session import (SessionStore, clear_timepoints, file_sha256, fingerprint, load_timepoint, restore_bundle,
                      stage_index)
from .plotwindow import PlotWindow, ResultsWindow
from .series import (CENTROID, export_cells, name_first, sample_id, save_timepoint_view, spots_in_volume,
                     sample_series, track_series)

IDENTITY_COL, INCLUDE_COL = 1, 6  # Select cells table: the Identity drop-down, the Include tick

APP_NAME = "HCR Cells"
APP_ID = "org.vanua.hcrcells"
APP_DIR = Path(__file__).resolve().parents[2]
SAMPLE_DIR = APP_DIR / "originals"
DEFAULT_OUT = APP_DIR / "out"
DEMO_KEY = "hcrcells/demoMode"
DEMO_LAST_T = 4          # demo mode runs T=0..4 so a visitor waits minutes, not an hour
# Composite layers: (layer name, channel key, default LUT, visible at load).
RECIPE_FILES = [CELLS_RECIPE_PATH, RECIPE_PATH]  # hashed together for the saved state
LAYERS = (("daf-28", "daf28", "Magenta", True), ("ins-4", "ins4", "Green", True),
          ("ins-6", "ins6", "Cyan", True), ("DAPI", "dapi", "Grays", False))
SPOT_CHANNELS = {"daf-28": "daf28", "ins-4": "ins4", "ins-6": "ins6"}


def _column(*widgets) -> "tuple[QWidget, QVBoxLayout]":
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setAlignment(Qt.AlignTop)
    for w in widgets:
        layout.addWidget(w)
    return page, layout


def _row(*widgets, stretch_first: bool = True) -> QWidget:
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        h.addWidget(w, 1 if (i == 0 and stretch_first) else 0)
    return row


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.RichText)
    return label


def default_contrast(volume: np.ndarray) -> "tuple[float, float]":
    """Initial min / max of a channel's display.

    Min: the *elbow* of the channel's brightness -- the same automatic cut the
    spot detector uses (:func:`hcrcells.ops.elbow_threshold`), applied here to
    the voxels' own brightness: how many voxels are brighter than each value
    falls steeply through the dim background, then flattens into the signal;
    the turn is the min, so the background shows as black.
    Max: the 99.95th percentile (HCR puncta are a tiny fraction of the voxels;
    daf-28 saturates, so the true maximum is useless) -- raised to at least
    1.5 x the min (or min + 10) when the two come too close (ins-4: elbow 35,
    99.95th percentile 38), so the window never collapses."""
    from .ops import elbow_threshold

    if hasattr(volume, "loaded_sample"):  # a lazy channel: the slices read so far
        sample = np.asarray(volume.loaded_sample(), np.float64).ravel()
    else:
        sample = np.asarray(volume)[:, ::4, ::4].astype(np.float64).ravel()  # every 16th voxel: plenty
    hi = float(np.percentile(sample, 99.95)) or float(sample.max() or 1.0)
    lo = float(elbow_threshold(sample)) if sample.size else 0.0
    if not np.isfinite(lo) or lo < 0:
        lo = 0.0
    hi = max(hi, 1.5 * lo, lo + 10.0)
    return lo, hi


def outline_labels(labels: np.ndarray) -> np.ndarray:
    """Per z-plane inner boundaries of each label (the label id on the edge, 0 elsewhere)."""
    from skimage.segmentation import find_boundaries

    out = np.zeros_like(labels)
    for z in range(labels.shape[0]):
        edge = find_boundaries(labels[z], mode="inner")
        out[z][edge] = labels[z][edge]
    return out


class HcrCellsApp(QObject):
    """Builds the window and owns the workflow state. `synchronous=True` runs
    the workers' functions inline (offscreen tests, screenshots)."""

    loaded = pyqtSignal()
    segmented = pyqtSignal()
    reviewed = pyqtSignal()
    roi_done = pyqtSignal()
    series_done = pyqtSignal()

    def __init__(self, synchronous: bool = False) -> None:
        super().__init__()
        self.synchronous = synchronous
        install_checkmark_style(QApplication.instance())  # ticks, not filled boxes
        self.recipe = load_recipe()                # timepoint: cells + spots in the ROI
        self.cells_recipe = load_cells_recipe()    # cells only: Segment and Select
        self.reg = registry()
        self.samples: dict = {}          # sample id -> [(t, path)]
        self.sample: "str | None" = None
        self.first = None                # (configured cells recipe, values): first timepoint's cells
        self.roi_run = None              # (configured timepoint recipe, values): first timepoint, spots in the ROI
        self.interest = None             # the cells of interest (names + first-timepoint centroids)
        self.tp_view = None              # the timepoint shown in the Display, when not the first one
        self._t0_state = None            # the first timepoint's display data, while another is shown
        self._tp_loading = False
        self._tp_pending = None
        self._preload_cancel = None      # stops the background slice loading of the previous view
        self._preload_threads: list = []
        self._quick_contrast: dict = {}  # first min / max from a few slices, refined when all are in
        self.first_table = None
        self.identities: dict = {}       # cell name -> 'ASIL' / 'ASIR' / 'ASJL' / 'ASJR' (this sample)
        self.channels: dict = {}         # first timepoint, every channel (read at Load)
        self.results: dict = {}          # sample id -> per-cell table over T (this session)
        self.surface_actors: list = []
        self.spot_actors: list = []
        self.label_actors: list = []
        self.label_colors: dict = {}
        self._threads: list = []
        self.session: "SessionStore | None" = None   # saved state of the loaded sample
        self._restoring = False                        # applying a saved state: do not save over it
        # Display changes (a slider drag, z scrolling) are saved 2 s after they stop.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(2000)
        self._save_timer.timeout.connect(self._save_display)

        self.viewer = MultiViewHost()
        steps = (
            WorkflowStep("1  Load", self._build_load()),
            WorkflowStep("2  Segment cells", self._build_segment()),
            WorkflowStep("3  Select cells", self._build_review()),
            WorkflowStep("4  ROI and spots", self._build_roi()),
            WorkflowStep("5  Run series", self._build_series()),
            WorkflowStep("6  Plots", self._build_plots()),
        )
        self.window = MainWindow(WorkflowSpec(title=APP_NAME, steps=steps, viewer=self.viewer,
                                              rendering_panel=self._build_display()))
        self.window.rendering_dock.setWindowTitle("Display")
        # Plots in their own window (closes with the main window: it is its child).
        self.plot_window = PlotWindow(self.window, out_dir=lambda: Path(self.out_edit.text()))
        self.results_window = ResultsWindow(self.window, out_dir=lambda: Path(self.out_edit.text()))
        for i in (1, 2, 3, 4, 5):
            self.window.set_step_enabled(i, False)
        self.window.cancel_button.clicked.connect(self._cancel)
        self.tour = GuidedTour(self.window, [
            GuideStep(self.sample_button, "Click <b>Load sample</b>: the first timepoint of a photobleaching "
                      "series opens (daf-28 channel).", page=0, done=self.loaded),
            GuideStep(self.segment_button, "Click <b>Segment cells</b>: the HCR-positive somata of the first "
                      "timepoint are outlined and lettered.", page=1, done=self.segmented),
            GuideStep(self.review_button, "Select the cells of interest: a cell with no ins-6 signal is unticked "
                      "by the rule; tick or untick <b>Include</b> to change it. Then click <b>Use these cells</b>.",
                      page=2, done=self.reviewed),
            GuideStep(self.roi_button, "The dashed amber box is the ROI around the cells of interest. Click "
                      "<b>Detect spots in ROI</b>: spots are found only inside it.", page=3, done=self.roi_done),
            GuideStep(self.run_button, f"Click <b>Run series and export</b> (demo: T=0..{DEMO_LAST_T}); each "
                      "timepoint follows the cells and detects spots in their ROI.", page=4, done=self.series_done),
            GuideStep(self.open_plots_button, "The plots open in their own window: pick per cell, the mean "
                      "of the cells, or the batch-effect check; <b>Advanced plot settings</b> has every option. "
                      "Click <b>Open plot window</b> to bring it back.", page=5),
        ], settings_key=DEMO_KEY)
        self.tour.mode_changed.connect(self._apply_demo_presets)
        self._apply_demo_presets(self.tour.enabled)
        self._watch_display()

    # -- pages ------------------------------------------------------------------------

    def _build_load(self) -> QWidget:
        self.sample_button = QPushButton("Load sample")
        theme.style_button_role(self.sample_button, "primary")
        self.sample_button.setToolTip(f"One click: the first sample in {SAMPLE_DIR.name}/")
        self.sample_button.clicked.connect(self.load_sample)
        self.folder_edit = QLineEdit(str(SAMPLE_DIR))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_folder)
        scan = QPushButton("Scan folder")
        scan.clicked.connect(lambda: self.scan(self.folder_edit.text()))
        self.sample_combo = QComboBox()
        self.load_button = load_button = QPushButton("Load selected sample")
        theme.style_button_role(load_button, "primary")
        load_button.clicked.connect(lambda: self.load(self.sample_combo.currentData()))
        self.fresh_button = QPushButton("Start fresh")
        self.fresh_button.setToolTip("Discard this sample's saved processing and display state, then reload it")
        self.fresh_button.clicked.connect(self.start_fresh)
        self.load_status = _label("Folder of '&lt;series&gt; - T=&lt;n&gt;.tif' files (ZCYX: C0 daf-28, C1 ins-4, "
                                  "C2 ins-6, C3 DAPI). Each series is one sample. A sample processed before "
                                  "reopens where it was left (state saved in &lt;out&gt;/&lt;sample&gt;/session/).")
        page, _ = _column(self.sample_button, QLabel("…or another folder"), _row(self.folder_edit, browse),
                          scan, self.sample_combo, load_button, self.fresh_button, self.load_status)
        return page

    def _build_segment(self) -> QWidget:
        self.settings_form = ParamsForm(Settings())
        self.segment_button = QPushButton("Segment cells")
        theme.style_button_role(self.segment_button, "primary")
        self.segment_button.clicked.connect(self.segment)
        self.segment_status = _label()
        # The button above the form: it is the tour's control and must be visible without scrolling.
        page, _ = _column(self.segment_button, self.segment_status, self.settings_form)
        return page

    def _build_review(self) -> QWidget:
        self.review_note = _label()
        self.review_button = QPushButton("Use these cells")
        theme.style_button_role(self.review_button, "primary")
        self.review_button.clicked.connect(self.accept_review)
        self.cell_table = QTableWidget(0, 7)
        self.cell_table.setHorizontalHeaderLabels(["Cell", "Identity", "z", "Volume (µm³)", "ins-6 − bg (a.u.)",
                                                   "Rule", "Include"])
        self.cell_table.horizontalHeaderItem(IDENTITY_COL).setToolTip(
            "Which neuron this cell is (ASI or ASJ, left or right). Each name can go to one cell; it is "
            "shown on the cell's label, in the results table and in the plots.")
        self.cell_table.verticalHeader().hide()
        self.cell_table.setMinimumHeight(220)
        self.cell_table.itemSelectionChanged.connect(self._jump_to_cell)
        # Ticking / unticking Include greys or restores that cell's label at once.
        self.cell_table.itemChanged.connect(lambda item: self._show_cell_labels() if item.column() == INCLUDE_COL else None)
        page, _ = _column(self.review_note, self.review_button, self.cell_table)
        return page

    def _build_roi(self) -> QWidget:
        self.roi_form = ParamsForm(RoiSettings())
        self.roi_form.changed.connect(self.update_roi_preview)
        self.roi_button = QPushButton("Detect spots in ROI")
        theme.style_button_role(self.roi_button, "primary")
        self.roi_button.clicked.connect(self.detect_spots_in_roi)
        self.roi_status = _label("A loose box around the cells of interest (dashed amber on the slice view). "
                                 "Spots are detected, and cells measured, only inside it -- at every timepoint, "
                                 "around the cells' positions at that timepoint.")
        page, _ = _column(self.roi_button, self.roi_status, self.roi_form)
        return page

    def _build_series(self) -> QWidget:
        self.last_t = QSpinBox()
        self.last_t.setRange(0, 999)
        self.last_t.setPrefix("up to T = ")
        self.all_t = QCheckBox("All timepoints")
        self.all_t.setChecked(True)
        self.all_t.toggled.connect(lambda on: self.last_t.setEnabled(not on))
        self.last_t.setEnabled(False)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.workers_spin.setValue(default_workers())
        self.workers_spin.setPrefix("Parallel workers: ")
        self.workers_spin.setToolTip("Timepoints processed at the same time, each in its own process (~1.6 GB "
                                     "of memory each). 1 = one after the other. The default comes from this "
                                     "computer's CPUs and free memory; the results are identical either way.")
        self.out_edit = QLineEdit(str(DEFAULT_OUT))
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._pick_out)
        self.run_button = QPushButton("Run series and export")
        theme.style_button_role(self.run_button, "primary")
        self.run_button.clicked.connect(self.run_series)
        self.series_status = _label("Each timepoint is segmented with the same settings (whole volume); the "
                                    "cells of interest are followed and spots detected in their ROI. Writes "
                                    "&lt;out&gt;/&lt;sample&gt;/cells_per_timepoint.csv with its provenance file.")
        page, _ = _column(self.run_button, self.all_t, self.last_t, self.workers_spin, QLabel("Output folder"),
                          _row(self.out_edit, out_browse), self.series_status)
        return page

    def _build_plots(self) -> QWidget:
        self.open_plots_button = QPushButton("Open plot window")
        theme.style_button_role(self.open_plots_button, "primary")
        self.open_plots_button.clicked.connect(self.open_plot_window)
        self.open_results_button = QPushButton("Open results table")
        self.open_results_button.clicked.connect(self.open_results_window)
        self.plot_note = _label("The plots and the results table open in their own windows after a series run. "
                                "Plots: per cell, mean of cells, or the batch-effect check; normalized or "
                                "absolute values; export (PNG, TIFF, PDF, SVG, …). Table: sort by any column; "
                                "export CSV or Excel.")
        self.result_paths: dict = {}   # sample id -> its cells_per_timepoint.csv (provenance source)
        page, _ = _column(self.open_plots_button, self.open_results_button, self.plot_note)
        return page

    def _build_display(self) -> QWidget:
        # Composite image: one layer per channel, each with its own visibility,
        # LUT, gamma, opacity and min/max; one combined histogram with each
        # channel's draggable handles (ChannelPanel); SliceCompositor paints the current z.
        self.layers = LayerStack()
        self.layer_panel = ChannelPanel(self.layers)
        self.compositor = SliceCompositor(self.layers, self.viewer.slice_viewer)
        self._label_sliders()
        self.show_outlines = QCheckBox("Cell outlines")
        self.show_outlines.setChecked(True)
        self.show_outlines.toggled.connect(lambda _: self._show_overlay())
        self.show_cell_labels = QCheckBox("Cell labels")
        self.show_cell_labels.setChecked(True)
        self.show_cell_labels.setToolTip("Each cell's letter at its centre on the current slice, and in 3D")
        self.cell_labels = CellLabelOverlay(self.viewer.slice_viewer)
        self.compositor.z_listeners.append(self.cell_labels.refresh)
        self.roi_overlay = RoiOverlay(self.viewer.slice_viewer)
        self.compositor.z_listeners.append(self.roi_overlay.refresh)
        self.show_cell_labels.toggled.connect(lambda _: self._show_cell_labels())
        self.spot_combo = QComboBox()
        self.spot_combo.addItems(list(SPOT_CHANNELS))
        self.spot_combo.currentIndexChanged.connect(lambda _: self._on_spot_channel())
        self.show_spots = QCheckBox("Spot markers")
        self.show_spots.setChecked(True)
        self.show_spots.toggled.connect(lambda _: self._show_spots())
        self.show_spots_3d = QCheckBox("Spots in 3D")
        self.show_spots_3d.setChecked(True)
        self.show_spots_3d.toggled.connect(lambda _: self._show_spots_3d())
        # Spot marker colour, remembered per spot channel; 'Auto' follows the
        # channel's LUT colour (highest-contrast palette colour).
        self.spot_color_choice: dict = {}
        self.spot_color = SpotColorPicker()
        self.spot_color.colorChanged.connect(lambda _rgb: self._on_spot_color())
        self.layers.add_listener(self._update_auto_spot_color)
        self.display_note = _label("Spot markers: spots detected in the chosen HCR channel, shown on "
                                   "the slice they sit in (±1). Auto colour: the one that stands out most "
                                   "against that channel's colour.")
        # Channels (combined histogram + per-channel controls) on top, overlays
        # below. One control per line: side by side, the themed checkbox and
        # a combo were wider than the dock (horizontal scroll).
        # Timepoint browser: after Run series, show any timepoint with its segmentation.
        self.tp_slider = QSlider(Qt.Horizontal)
        self.tp_spin = QSpinBox()
        self.tp_spin.setPrefix("T = ")
        self.tp_slider.valueChanged.connect(lambda v: self._on_tp_changed(v, "slider"))
        self.tp_spin.valueChanged.connect(lambda v: self._on_tp_changed(v, "spin"))
        self.tp_auto = QCheckBox("Auto contrast per timepoint")
        self.tp_auto.setToolTip("Off (default): every timepoint keeps the first timepoint's min / max, so "
                                "bleaching shows as dimming. On: each timepoint gets its own min / max.")
        self.tp_auto.toggled.connect(lambda _: self._reload_timepoint())
        self.tp_status = _label("Run the series to browse the timepoints.")
        self._tp_timer = QTimer(self)
        self._tp_timer.setSingleShot(True)
        self._tp_timer.setInterval(300)
        self._tp_timer.timeout.connect(lambda: self.show_timepoint(self.tp_spin.value()))
        self._set_tp_enabled(False)
        page, _ = _column(QLabel("<b>Timepoint</b>"), _row(self.tp_slider, self.tp_spin), self.tp_auto,
                          self.tp_status, QLabel("<b>Channels</b>"), self.layer_panel, QLabel("<b>Overlays</b>"),
                          self.show_outlines, self.show_cell_labels, self.show_spots, self.show_spots_3d,
                          QLabel("Spots of channel"), self.spot_combo, QLabel("Spot colour"), self.spot_color,
                          self.display_note)
        self._update_auto_spot_color()
        return page

    def _spot_channel_rgb(self) -> "tuple[int, int, int]":
        """The current spot channel's display colour (its layer's LUT)."""
        name = self.spot_combo.currentText()
        try:
            return lut_color(self.layers.by_name(name).lut_name)
        except (KeyError, ValueError):
            return lut_color(dict((n, lut) for n, _k, lut, _v in LAYERS)[name])

    def _update_auto_spot_color(self) -> None:
        """Layers changed (e.g. a LUT): refresh the Auto colour, and the markers if Auto is on."""
        before = self.spot_color.color()
        self.spot_color.set_auto_color(self._spot_channel_rgb())
        if self.spot_color.color() != before:
            self._on_spot_color()

    def _on_spot_channel(self) -> None:
        self.spot_color.set_choice(self.spot_color_choice.get(self._spot_channel()))
        self.spot_color.set_auto_color(self._spot_channel_rgb())
        self._show_spots()
        self._show_spots_3d()

    def _on_spot_color(self) -> None:
        self.spot_color_choice[self._spot_channel()] = self.spot_color.choice()
        self._show_spots()
        self._show_spots_3d()

    def _step_um(self, axis: int):
        """Voxel size along z (0), y (1) or x (2) of the loaded stack, or None."""
        frame = getattr(self, "frame", None)
        return float(np.linalg.norm(np.asarray(frame["affine"])[:3, axis])) if frame else None

    def _label_sliders(self) -> None:
        """Axis labels and position readouts on the slice scroll bars: Z on the
        Slice tab; Z, Y, X on the Ortho tab's three panes (XY, XZ, YZ)."""
        label_slice_slider(self.viewer.slice_viewer, "Z", lambda: self._step_um(0))
        ortho = getattr(self.viewer, "ortho_viewer", None)
        for pane, axis, i in (("xy", "Z", 0), ("xz", "Y", 1), ("yz", "X", 2)):
            if ortho is not None and hasattr(ortho, pane):
                label_slice_slider(getattr(ortho, pane), axis, lambda i=i: self._step_um(i))

    def _set_layers(self, channels: dict, frame) -> None:
        """One composite layer per channel, with its default LUT and min/max."""
        spacing = tuple(float(s) for s in frame.spacing()[:3])
        self.layer_panel.set_layers([
            ImageLayer(name=name, data=channels[key], spacing_um_zyx=spacing, visible=visible, lut_name=lut,
                       contrast_range=self.contrast[key])
            for name, key, lut, visible in LAYERS])

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self.window, "Folder of timepoint TIFFs", self.folder_edit.text())
        if path:
            self.folder_edit.setText(path)
            self.scan(path)

    def _pick_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self.window, "Output folder", self.out_edit.text())
        if path:
            self.out_edit.setText(path)

    def _apply_demo_presets(self, on: bool) -> None:
        self.all_t.setChecked(not on)
        if on:
            self.last_t.setValue(DEMO_LAST_T)

    # -- workers ----------------------------------------------------------------------

    # Steps whose job shows a running state on its buttons (not the background
    # 3D surface build): the button text while running.
    BUSY_TEXT = {"Load": "Loading…", "Segment": "Segmenting…", "ROI": "Detecting spots…",
                 "Series": "Running series…"}

    def _step_buttons(self) -> dict:
        """{job label: (buttons that show its progress, buttons only disabled)}."""
        return {"Load": ([self.sample_button, self.load_button], [self.fresh_button]),
                "Segment": ([self.segment_button], []),
                "ROI": ([self.roi_button], []),
                "Series": ([self.run_button], [])}

    def _busy_begin(self, label: str) -> None:
        """While a step runs: its buttons are disabled -- which also stops the
        tutorial's pulse on them (HeroPulse glows only on an enabled button),
        so nothing blinks as if waiting for a click -- and say "Segmenting… 45%".
        Every step's run buttons are disabled: overlapping jobs would conflict."""
        self._busy_label = label
        self._busy_texts = {}
        for name, (shown, others) in self._step_buttons().items():
            for button in shown + others:
                button.setEnabled(False)
                if name == label and button in shown:
                    self._busy_texts[button] = button.text()
                    button.setText(self.BUSY_TEXT[label])
        QApplication.setOverrideCursor(Qt.BusyCursor)  # arrow + hourglass: the app stays usable

    def _busy_progress(self, fraction: float) -> None:
        for button in getattr(self, "_busy_texts", {}):
            button.setText(f"{self.BUSY_TEXT[self._busy_label]} {int(round(100 * fraction))}%")

    def _busy_end(self) -> None:
        if getattr(self, "_busy_label", None) is None:
            return
        for button, text in self._busy_texts.items():
            button.setText(text)
        for shown, others in self._step_buttons().values():
            for button in shown + others:
                button.setEnabled(True)
        self._busy_label, self._busy_texts = None, {}
        QApplication.restoreOverrideCursor()

    def _start(self, fn, params, ctx, on_success, label: str) -> None:
        """Run `fn(params, ctx, progress, cancel, log)` on a worker thread (or inline)."""
        busy = label in self.BUSY_TEXT
        if busy:
            self._busy_begin(label)
        if self.synchronous:
            import threading

            try:
                result = fn(params, ctx, lambda *a: None, threading.Event(), self.window.append_log)
            finally:
                if busy:
                    self._busy_end()
            on_success(result)
            return
        thread = OpWorkerThread(fn, params, ctx)

        def _progress(e) -> None:
            self.window.show_progress(e.fraction, f"{label}: {e.message}")
            if busy:
                self._busy_progress(e.fraction)

        def _done() -> None:
            if busy:
                self._busy_end()

        thread.progress.connect(_progress)
        thread.log.connect(self.window.append_log)
        # The running state ends before the result is shown, so the next step's
        # button is enabled (and pulses, in tutorial mode) when it appears.
        thread.succeeded.connect(lambda result: (_done(), self.window.clear_progress(), on_success(result)))
        thread.failed.connect(lambda msg: (_done(), self.window.clear_progress("Failed"),
                                           self.window.append_log(f"{label} failed: {msg}")))
        thread.cancelled.connect(lambda: (_done(), self.window.clear_progress("Cancelled")))
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)
        self._threads.append(thread)  # keep a reference (building.md §5)
        self.window.show_progress(0.0, f"{label}…")
        thread.start()

    def _cancel(self) -> None:
        for thread in list(self._threads):
            thread.cancel()

    # -- 1 load -----------------------------------------------------------------------

    def scan(self, folder: "str | Path") -> int:
        folder = Path(folder)
        if not folder.is_dir():
            self.load_status.setText(f"No such folder: {folder}")
            return 0
        self.samples = {sample_id(name): files for name, files in sample_series(folder).items()}
        self.sample_combo.clear()
        for sid, files in self.samples.items():
            self.sample_combo.addItem(f"{sid}  ({len(files)} timepoints)", sid)
        self.load_status.setText(f"{len(self.samples)} sample(s) in {folder.name}/" if self.samples
                                 else "No '&lt;series&gt; - T=&lt;n&gt;.tif' files found.")
        return len(self.samples)

    def load_sample(self) -> None:
        self.folder_edit.setText(str(SAMPLE_DIR))
        if self.scan(SAMPLE_DIR):
            self.load(next(iter(self.samples)))

    def load(self, sid: "str | None") -> None:
        """Open the first timepoint lazily: the slice shown first (the middle one)
        of every channel is read now, the rest in the background after the image
        is on screen (:meth:`_start_preload`)."""
        if not sid or sid not in self.samples:
            return
        t, path = self.samples[sid][0]
        files, out = self.samples[sid], Path(self.out_edit.text())

        def _job(params, ctx, progress, cancel, log):
            frame = tiff_frame(path)
            channels = open_lazy_channels(path, CHANNEL_INDEX)
            z = next(iter(channels.values())).shape[0] // 2
            # First min / max from a few slices around the one shown; refined later.
            contrast = {name: default_contrast(_sample(ch, z)) for name, ch in channels.items()}
            progress(0.5, "first slices read")
            # A saved state (segmentation, series results) is read here, off the GUI thread.
            bundle = restore_bundle(out, sid, files, RECIPE_FILES, channels, frame.as_document())
            progress(1.0, "saved state checked")
            return channels, contrast, frame, bundle

        def _sample(channel, z):
            sample_for_contrast(channel, z)
            return channel

        self.load_status.setText(f"Reading {path.name}…")
        self._start(_job, None, None, lambda result: self._on_loaded(sid, t, path, *result), "Load")

    def _on_loaded(self, sid, t, path, channels, contrast, frame, bundle=None) -> None:
        # Detach the previous sample's saved state first: the display resets
        # below would otherwise be saved into it.
        self._flush_display_save()
        self.session = None
        self.sample, self.first, self.first_table = sid, None, None
        self.roi_run, self.interest = None, None
        self.tp_view, self._t0_state = None, None
        self._set_tp_enabled(False)
        self.roi_overlay.set_box(None)
        self.cell_labels.clear()
        self.cell_table.setRowCount(0)
        self.channels, self.contrast = channels, contrast
        self._quick_contrast = dict(contrast)
        self.frame = frame.as_document()
        image = channels["daf28"]
        self._clear_3d()
        self.viewer.slice_viewer.set_overlay_labels(None, {}, {})
        self.viewer.slice_viewer.set_skeleton_points(None)
        # daf-28 is the carrier of z navigation; the composite is painted over it.
        show_image(self.viewer.slice_viewer, image, self.frame)
        self._set_layers(channels, frame)
        self.viewer.slice_viewer.fit_to_view()
        self.compositor.refresh()
        z, y, x = frame.spacing()[:3]
        self.load_status.setText(f"<b>{sid}</b>: {len(self.samples[sid])} timepoints; first T={t}<br>"
                                 f"{image.shape[0]} x {image.shape[1]} x {image.shape[2]} (ZYX), "
                                 f"voxel {z:.3g} x {y:.3g} x {x:.3g} {frame.unit}")
        self.window.append_log(f"Loaded {sid} T={t}: {path.name} (other slices loading in the background)")
        for i in (2, 3, 4, 5):
            self.window.set_step_enabled(i, sid in self.results if i == 5 else False)
        self.window.set_step_enabled(1, True)
        self.window.set_current_step(1)
        self.loaded.emit()
        self.session = SessionStore(self.out_edit.text(), sid)
        for note in (bundle or {}).get("notes", ()):
            self.window.append_log(note)
        self._start_preload(channels, refine_contrast=True)
        if bundle and bundle.get("state"):
            self._restore(bundle)
        else:
            self.session.clear()  # nothing usable (none, or the source files changed)
            self.session.update(stage="loaded", sources=fingerprint(self.samples[sid]),
                                recipe_sha256=file_sha256(RECIPE_FILES),
                                settings=asdict(self.settings_form.values()), overrides=[], series=None,
                                roi=asdict(self.roi_form.values()), display=self._display_state())

    # -- 2 segment --------------------------------------------------------------------

    def segment(self) -> None:
        if self.sample is None:
            return
        errors = self.settings_form.errors()
        if errors:
            self.segment_status.setText("<br>".join(errors))
            return
        settings = self.settings_form.values()
        path = self.samples[self.sample][0][1]
        recipe, reg, store = self.cells_recipe, self.reg, self.session

        def _job(params, ctx, progress, cancel, log):
            # Cells only (whole volume, no spots). The recipe fills 0-90 % of the
            # bar; outlines and label positions the rest.
            result = segment_cells(path, params, recipe=recipe, reg=reg, progress=progress, cancel=cancel, log=log,
                                   span=(0.0, 0.9))
            cells = result[1]["cells"]
            # Outlines and label positions here too: not on the GUI thread.
            progress(0.9, "cell outlines")
            outlines = outline_labels(cells)
            progress(0.95, "label positions")
            positions = label_positions(cells)
            if store is not None:
                progress(0.97, "saving state")
                store.save_segmentation(result[1], outlines, positions)
            progress(1.0, "done")
            return result, outlines, positions

        self.segment_status.setText("Running…")
        self._start(_job, settings, None, self._on_segmented, "Segment")

    def _on_segmented(self, result) -> None:
        self.first, self.outlines, self.label_positions = result
        self.identities = {}
        self.roi_run, self.interest = None, None
        self.tp_view, self._t0_state = None, None
        self._set_tp_enabled(False)
        self.roi_overlay.set_box(None)
        values = self.first[1]
        table = name_first(values["cell_table"])
        table = table.assign(sample=self.sample, timepoint=self.samples[self.sample][0][0])
        self.first_table = apply_exclusion(table, self.settings_form.values())
        self.label_colors = label_colors(np.unique(values["cells"]))
        self.refresh_display()
        self._show_cell_labels()
        n_in = int(self.first_table["included"].sum())
        self.segment_status.setText(f"<b>{len(table)}</b> cells found; <b>{n_in}</b> pass the selection rule.")
        self._fill_review()
        self.window.set_step_enabled(2, True)
        self.window.set_current_step(2)
        self.build_surfaces()
        if not self._restoring and self.session is not None:
            clear_timepoints(self.session.dir)  # saved timepoints belonged to the old segmentation
            # A new segmentation: earlier review ticks and series results no longer apply.
            self.session.update(stage="segmented", settings=asdict(self.settings_form.values()),
                                recipe_sha256=file_sha256(RECIPE_FILES), overrides=[], identities=[],
                                series=None)
        self.segmented.emit()

    # -- 3 review ---------------------------------------------------------------------

    def _fill_review(self) -> None:
        s = self.settings_form.values()
        t = self.first_table.sort_values("cell").reset_index(drop=True)
        self.cell_table.blockSignals(True)  # filling is not a user tick (itemChanged)
        self.cell_table.setRowCount(len(t))
        known = set(t["cell"])
        self.identities = {c: v for c, v in self.identities.items() if c in known}
        rule_col = f"{s.exclude_channel}_mean_bgsub_au" if s.exclude_channel != "none" else None
        for i, row in t.iterrows():
            z = row["centroid_z_um"] / float(np.linalg.norm(np.asarray(self.frame["affine"])[:3, 0]))
            rgb = self.label_colors.get(int(row["label"]), (200, 200, 200))
            name = QTableWidgetItem(row["cell"])
            name.setForeground(theme_color(rgb))
            name.setData(Qt.UserRole, int(row["label"]))
            rule = ("excluded" if row["excluded_by_rule"] else "kept")
            if rule_col:
                rule += f": {GENE[s.exclude_channel]} {row[rule_col]:.1f} {'<' if row['excluded_by_rule'] else '≥'} {s.exclude_below:g}"
            include = QTableWidgetItem()
            include.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            include.setCheckState(Qt.Checked if row["included"] else Qt.Unchecked)
            cells = [name, QTableWidgetItem(""), QTableWidgetItem(f"{z:.0f}"),
                     QTableWidgetItem(f"{row['volume_um3']:.1f}"),
                     QTableWidgetItem(f"{row['ins6_mean_bgsub_au']:.1f}"), QTableWidgetItem(rule), include]
            for j, item in enumerate(cells):
                if j < INCLUDE_COL:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.cell_table.setItem(i, j, item)
            self.cell_table.setCellWidget(i, IDENTITY_COL, self._identity_combo(row["cell"]))
        self.cell_table.blockSignals(False)
        self.cell_table.resizeColumnsToContents()
        self.review_note.setText(f"Sample <b>{self.sample}</b>, first timepoint. Select the cells of interest: "
                                 f"by default, cells with {GENE.get(s.exclude_channel, s.exclude_channel)} mean − "
                                 f"background below {s.exclude_below:g} a.u. are left out. Tick or untick "
                                 "<b>Include</b> to change it; select a row to jump to the cell. Name each cell "
                                 "of interest in <b>Identity</b> (ASIL, ASIR, ASJL, ASJR). Spots are then "
                                 "detected only in a box around the included cells.")

    def _identity_combo(self, cell: str) -> QComboBox:
        box = QComboBox()
        box.addItem("—", "")
        for name in IDENTITIES:
            box.addItem(name, name)
        box.setCurrentIndex(max(0, box.findData(self.identities.get(cell, ""))))
        box.setProperty("cell", cell)
        box.currentIndexChanged.connect(lambda _i, b=box: self._on_identity(b))
        return box

    def _identity_boxes(self) -> dict:
        """{cell name: its Identity drop-down} in the Select cells table."""
        out = {}
        for i in range(self.cell_table.rowCount()):
            box = self.cell_table.cellWidget(i, IDENTITY_COL)
            if box is not None:
                out[self.cell_table.item(i, 0).text()] = box
        return out

    def _on_identity(self, box: QComboBox) -> None:
        cell, value = box.property("cell"), box.currentData()
        if value:  # each neuron is one cell: take the name from whichever cell had it
            for other, b in self._identity_boxes().items():
                if other != cell and b.currentData() == value:
                    b.blockSignals(True)
                    b.setCurrentIndex(0)
                    b.blockSignals(False)
        self.identities = {c: b.currentData() for c, b in self._identity_boxes().items() if b.currentData()}
        self._identities_changed()

    def set_identities(self, identities: dict) -> None:
        """Name the cells ({cell: 'ASIL', ...}), e.g. from a saved session."""
        self.identities = {c: v for c, v in identities.items() if v in IDENTITIES}
        for cell, box in self._identity_boxes().items():
            box.blockSignals(True)
            box.setCurrentIndex(max(0, box.findData(self.identities.get(cell, ""))))
            box.blockSignals(False)
        self._identities_changed()

    def _identities_changed(self) -> None:
        """Labels, results (and their CSV), plots and the saved state follow a renamed cell."""
        self._show_cell_labels()
        if self.sample in self.results:
            named = {(self.sample, c): v for c, v in self.identities.items()}
            self.results[self.sample] = apply_identities(self.results[self.sample], named)
            path = self.result_paths.get(self.sample)
            if path is not None and not self._restoring:
                try:
                    self.results[self.sample].to_csv(path, index=False)
                except OSError as exc:
                    self.window.append_log(f"Could not update {path.name}: {exc}")
            if self.plot_window.results:
                self.show_plot(open_window=False)
        if not self._restoring and self.session is not None:
            self.session.update(identities=sorted([c, v] for c, v in self.identities.items()))

    def _shown(self, cell: str) -> str:
        return shown_name(cell, self.identities.get(cell, ""))

    def overrides(self) -> dict:
        """{(sample, cell): include} where the Include box differs from the rule."""
        out = {}
        rule = dict(zip(self.first_table["cell"], self.first_table["excluded_by_rule"]))
        for i in range(self.cell_table.rowCount()):
            cell = self.cell_table.item(i, 0).text()
            include = self.cell_table.item(i, INCLUDE_COL).checkState() == Qt.Checked
            if include == bool(rule[cell]):  # differs from the rule's decision
                out[(self.sample, cell)] = include
        return out

    def set_include(self, cell: str, include: bool) -> None:
        for i in range(self.cell_table.rowCount()):
            if self.cell_table.item(i, 0).text() == cell:
                self.cell_table.item(i, INCLUDE_COL).setCheckState(Qt.Checked if include else Qt.Unchecked)

    def _jump_to_cell(self) -> None:
        rows = self.cell_table.selectionModel().selectedRows() or [
            self.cell_table.model().index(i.row(), 0) for i in self.cell_table.selectedIndexes()[:1]]
        if not rows or self.first is None:
            return
        label = self.cell_table.item(rows[0].row(), 0).data(Qt.UserRole)
        zs = np.nonzero((self.first[1]["cells"] == label).any(axis=(1, 2)))[0]
        if len(zs):
            self.viewer.slice_viewer.set_z(int(zs[len(zs) // 2]))

    def accept_review(self) -> None:
        if self.first_table is None:
            return
        self.first_table = apply_exclusion(self.first_table, self.settings_form.values(), self.overrides())
        self._show_cell_labels()
        self.interest = self.first_table[self.first_table["included"]].sort_values("cell").reset_index(drop=True)
        if self.roi_run is not None and not self._restoring:
            self.roi_run = None  # spots were for another selection
            self._show_spots()
            self._show_spots_3d()
        kept = list(self.interest["cell"])
        self.series_status.setText(f"Cells of interest: <b>{', '.join(kept) or 'none'}</b>. "
                                   f"{len(self.samples[self.sample])} timepoints available.")
        self.window.set_step_enabled(3, bool(kept))
        self.window.set_step_enabled(4, False)
        self.window.set_current_step(3)
        self.update_roi_preview()
        if not self._restoring and self.session is not None:
            self.session.update(stage="reviewed",
                                overrides=[[cell, include] for (_s, cell), include in self.overrides().items()])
        self.reviewed.emit()

    # -- 4 ROI and spots ----------------------------------------------------------------

    def update_roi_preview(self) -> None:
        """The ROI box around the cells of interest (first timepoint) on the slice view."""
        if self.first is None or self.interest is None or not len(self.interest):
            self.roi_overlay.set_box(None)
            return
        r = self.roi_form.values()
        spacing = [self._step_um(i) for i in range(3)]
        cells = self.first[1]["cells"]
        ids = self.interest["label"].to_numpy(int)
        box = roi_box(cells, ids, spacing, r.margin_xy, r.margin_z)
        ref = roi_box(cells, ids, spacing, max(r.ref_margin_xy, r.margin_xy), max(r.ref_margin_z, r.margin_z))
        self.roi_overlay.set_box(box, ref)

        def describe(b):
            size = [v[1] - v[0] for v in (b["z"], b["y"], b["x"])]
            um = [n * spacing[i] for i, n in enumerate(size)]
            frac = float(np.prod(size)) / float(np.prod(cells.shape))
            return f"{um[2]:.0f} × {um[1]:.0f} µm, {size[0]} slices — <b>{100 * frac:.1f} %</b> of the volume"

        self.roi_status.setText(
            f"<b>ROI</b> (amber) around <b>{', '.join(self.interest['cell'])}</b>: {describe(box)}.<br>"
            f"<b>Reference region</b> (grey): {describe(ref)}. Spots, their automatic threshold and the "
            "isolated single-spot reference (transcript estimate) are computed here; at every timepoint both "
            "boxes follow the cells.")
        if self.roi_run is not None and not self._restoring and self.roi_run[1].get("roi") != box:
            self.roi_status.setText(self.roi_status.text() + "<br><i>Margins changed: detect spots again.</i>")
        if not self._restoring:
            self._display_changed()

    def detect_spots_in_roi(self) -> None:
        if self.first is None or self.interest is None or not len(self.interest):
            return
        settings, roi = self.settings_form.values(), self.roi_form.values()
        path = self.samples[self.sample][0][1]
        targets = format_targets(self.interest[CENTROID].to_numpy(float))
        recipe, reg, store = self.recipe, self.reg, self.session

        def _job(params, ctx, progress, cancel, log):
            result = run_timepoint(path, params, targets=targets, roi=roi, recipe=recipe, reg=reg,
                                   progress=progress, cancel=cancel, log=log, span=(0.0, 0.97))
            if store is not None:
                progress(0.98, "saving state")
                store.save_roi(result[1])
            progress(1.0, "done")
            return result

        self._start(_job, settings, None, self._on_roi_done, "ROI")

    def _on_roi_done(self, result) -> None:
        self.roi_run = result
        values = result[1]
        self.roi_overlay.set_box(values["roi"], values.get("ref_roi"))
        self._show_spots()
        self._show_spots_3d()
        counts = ", ".join(f"{GENE[ch]} {len(values[f'spots_{ch}'])}" for ch in CHANNELS)
        refs = ", ".join(f"{GENE[ch]} {int(values['cell_table'][f'{ch}_n_ref_spots'].iloc[0]) if len(values['cell_table']) else 0}"
                         for ch in CHANNELS)
        self.update_roi_preview()
        self.roi_status.setText(self.roi_status.text() + f"<br>Spots in the ROI: {counts}.<br>Isolated reference "
                                f"spots (for the transcript estimate): {refs}.")
        self.window.set_step_enabled(4, True)
        self.window.set_current_step(4)
        if not self._restoring and self.session is not None:
            self.session.update(stage="roi_done", roi=asdict(self.roi_form.values()), series=None)
        self.roi_done.emit()

    # -- 4 series ---------------------------------------------------------------------

    def run_series(self) -> None:
        if self.roi_run is None or self.interest is None:
            return
        files = self.samples[self.sample]
        if not self.all_t.isChecked():
            files = [(t, p) for t, p in files if t <= self.last_t.value()]
        settings, overrides, roi = self.settings_form.values(), self.overrides(), self.roi_form.values()
        sid, first, recipe, reg, interest = self.sample, self.roi_run, self.recipe, self.reg, self.interest
        out = Path(self.out_edit.text()) / sid
        workers = self.workers_spin.value()
        named = {(sid, c): v for c, v in self.identities.items()}
        save_dir = self.session.dir if self.session is not None else None
        self._back_to_first_timepoint()
        self._set_tp_enabled(False)

        def _job(params, ctx, progress, cancel, log):
            if save_dir is not None:
                clear_timepoints(save_dir)  # this run's timepoints replace any earlier ones
            cells, _first, runs = track_series(files, interest, sample=sid, settings=params, roi=roi, recipe=recipe,
                                               reg=reg, first=first, progress=progress, cancel=cancel, log=log,
                                               workers=workers, save_dir=save_dir)
            cells = apply_identities(cells, named)
            path = export_cells(cells, out / "cells_per_timepoint.csv", runs, params, overrides, reg, roi)
            return cells, path

        self.series_status.setText(f"Running {len(files)} timepoints…")
        self._start(_job, settings, None, self._on_series, "Series")

    def _on_series(self, result) -> None:
        cells, path = result
        self.results[self.sample] = apply_identities(cells, {(self.sample, c): v for c, v in self.identities.items()})
        cells = self.results[self.sample]
        self.result_paths[self.sample] = Path(path)
        if not self._restoring and self.session is not None:
            self.session.update(stage="series_done", series={
                "results_csv": str(Path(path).resolve()), "all_timepoints": self.all_t.isChecked(),
                "last_t": self.last_t.value(), "workers": self.workers_spin.value()})
        self._load_other_samples(Path(self.out_edit.text()))
        n_t = cells["timepoint"].nunique()
        self.series_status.setText(f"<b>{self.sample}</b>: {n_t} timepoints, "
                                   f"{cells.loc[cells['included'], 'cell'].nunique()} cells included.<br>"
                                   f"Wrote {path.name} (+ provenance) in {path.parent}")
        self.window.set_step_enabled(5, True)
        self.window.set_current_step(5)
        self._set_tp_enabled(True)
        self.show_plot()
        self.series_done.emit()

    def _load_other_samples(self, out: Path) -> None:
        """Earlier runs of other samples (their CSVs in the output folder) join the batch-effect plot."""
        import pandas as pd

        for csv in sorted(out.glob("*/cells_per_timepoint.csv")):
            sid = csv.parent.name
            if sid not in self.results:
                try:
                    self.results[sid] = pd.read_csv(csv)
                    self.result_paths[sid] = csv
                except Exception as exc:  # noqa: BLE001
                    self.window.append_log(f"Skipped {csv}: {exc}")

    # -- saved state ------------------------------------------------------------------

    def _restore(self, bundle: dict) -> None:
        """Re-apply a saved state: settings, display, then each step reached --
        through the same handlers the steps use, without recomputing anything."""
        state = bundle["state"]
        sid = self.sample
        self._restoring = True
        try:
            known = {f.name for f in fields(Settings)}
            saved = {k: v for k, v in (state.get("settings") or {}).items() if k in known}
            if saved:
                self.settings_form.set_values(Settings(**saved))
            series = state.get("series") or {}
            if series:
                self.all_t.setChecked(bool(series.get("all_timepoints", True)))
                self.last_t.setValue(int(series.get("last_t", 0)))
                if series.get("workers"):
                    self.workers_spin.setValue(int(series["workers"]))
            display = state.get("display") or {}
            self._apply_display(display, view=False)
            done = ["loaded"]
            if bundle.get("segmentation") is not None:
                values, outlines, positions = bundle["segmentation"]
                path = self.samples[sid][0][1]
                first = (configured(self.cells_recipe, path, self.settings_form.values()), values)
                self._on_segmented((first, outlines, positions))
                done.append("segmented")
                if state.get("roi"):
                    known_roi = {f.name for f in fields(RoiSettings)}
                    self.roi_form.set_values(RoiSettings(**{k: v for k, v in state["roi"].items() if k in known_roi}))
                if stage_index(state.get("stage")) >= stage_index("reviewed"):
                    for cell, include in state.get("overrides") or []:
                        self.set_include(cell, bool(include))
                    self.set_identities(dict(state.get("identities") or []))
                    self.accept_review()
                    done.append("cells selected")
                if bundle.get("roi") is not None and self.interest is not None:
                    targets = format_targets(self.interest[CENTROID].to_numpy(float))
                    recipe = configured(self.recipe, path, self.settings_form.values(), targets=targets,
                                        roi=self.roi_form.values())
                    self._on_roi_done((recipe, bundle["roi"]))
                    done.append("ROI spots")
            if bundle.get("results") is not None:
                self._on_series((bundle["results"], Path(series["results_csv"])))
                done.append("series")
            self._apply_display(display, view=True)
        finally:
            self._restoring = False
        self.window.append_log(f"Restored {sid} ({' + '.join(done)}; saved {state.get('saved', '?')})")
        self.load_status.setText(self.load_status.text() + f"<br>Restored saved state: {' + '.join(done)}.")

    def start_fresh(self) -> None:
        """Discard the saved state of the selected (or loaded) sample and reload it."""
        sid = self.sample_combo.currentData() or self.sample
        if not sid:
            return
        if not self.synchronous:
            from PyQt5.QtWidgets import QMessageBox

            answer = QMessageBox.question(self.window, "Start fresh",
                                          f"Discard the saved segmentation, review and display state of {sid}? "
                                          "Exported results (CSV, plots) are kept.")
            if answer != QMessageBox.Yes:
                return
        SessionStore(self.out_edit.text(), sid).clear()
        self.session = None
        self.results.pop(sid, None)
        self.window.append_log(f"{sid}: saved state discarded")
        self.load(sid)

    def _display_state(self) -> dict:
        layers = [{"name": layer.name, "lut_name": layer.lut_name, "visible": layer.visible,
                   "opacity": layer.opacity, "gamma": layer.gamma,
                   "contrast_range": list(layer.contrast_range) if layer.contrast_range else None}
                  for layer in self.layers if isinstance(layer, ImageLayer)]
        return {"layers": layers, "outlines": self.show_outlines.isChecked(),
                "cell_labels": self.show_cell_labels.isChecked(), "spots": self.show_spots.isChecked(),
                "spots_3d": self.show_spots_3d.isChecked(), "spot_channel": self.spot_combo.currentText(),
                "spot_colors": {ch: (list(c) if isinstance(c, tuple) else c)
                                for ch, c in self.spot_color_choice.items()},
                "z": int(self.viewer.slice_viewer.current_z), "tab": self.viewer.as_widget().currentIndex(),
                "plot_window": self.plot_window.state(), "results_window": self.results_window.state(),
                "timepoint": self.tp_view["t"] if self.tp_view else None, "tp_auto": self.tp_auto.isChecked()}

    def _apply_display(self, display: dict, view: bool) -> None:
        """Channels, overlays and spot colours; with `view`, also z, tab and plot
        (last, once the restored steps have drawn their data)."""
        if not display:
            return
        if not view:
            saved = {d["name"]: d for d in display.get("layers", [])}
            spacing = tuple(float(s) for s in np.linalg.norm(np.asarray(self.frame["affine"])[:3, :3], axis=0))
            layers = []
            for name, key, lut, visible in LAYERS:
                d = saved.get(name, {})
                cr = d.get("contrast_range")
                layers.append(ImageLayer(name=name, data=self.channels[key], spacing_um_zyx=spacing,
                                         visible=d.get("visible", visible), lut_name=d.get("lut_name", lut),
                                         opacity=float(d.get("opacity", 1.0)), gamma=float(d.get("gamma", 1.0)),
                                         contrast_range=tuple(cr) if cr else self.contrast[key]))
            self.layer_panel.set_layers(layers)
            for box, key in ((self.show_outlines, "outlines"), (self.show_cell_labels, "cell_labels"),
                             (self.show_spots, "spots"), (self.show_spots_3d, "spots_3d")):
                if key in display:
                    box.setChecked(bool(display[key]))
            self.spot_color_choice = {ch: (("custom", tuple(c[1])) if isinstance(c, list) else c)
                                      for ch, c in (display.get("spot_colors") or {}).items()}
            if display.get("spot_channel"):
                self.spot_combo.blockSignals(True)
                self.spot_combo.setCurrentText(display["spot_channel"])
                self.spot_combo.blockSignals(False)
            self._on_spot_channel()
            return
        if display.get("plot_window"):
            self.plot_window.apply_state(display["plot_window"])
            if self.sample in self.results:
                self.show_plot(open_window=False)
                if display["plot_window"].get("open", True):
                    self.open_plot_window()
                results_state = display.get("results_window") or {}
                self.results_window.apply_state(results_state)
                if results_state.get("open", True):
                    self.open_results_window()
        if "tp_auto" in display:
            self.tp_auto.blockSignals(True)
            self.tp_auto.setChecked(bool(display["tp_auto"]))
            self.tp_auto.blockSignals(False)
        if display.get("timepoint") is not None and self.tp_spin.isEnabled():
            self.tp_spin.setValue(int(display["timepoint"]))  # loads it (debounced)
        tab = display.get("tab", 0)
        if tab != TAB_3D:  # the 3D view is created on demand; not opened just to restore
            self.viewer.as_widget().setCurrentIndex(tab)
        if "z" in display:
            self.viewer.slice_viewer.set_z(int(display["z"]))

    def _watch_display(self) -> None:
        """Any display change (re)starts the 2 s save timer."""
        changed = lambda *_: self._display_changed()  # noqa: E731
        self.layers.add_listener(changed)
        for box in (self.show_outlines, self.show_cell_labels, self.show_spots, self.show_spots_3d):
            box.toggled.connect(changed)
        self.spot_combo.currentIndexChanged.connect(changed)
        self.spot_color.colorChanged.connect(changed)
        self.compositor.z_listeners.append(changed)
        self.viewer.as_widget().currentChanged.connect(changed)
        self.plot_window.optionsChanged.connect(changed)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._flush_display_save)  # do not lose the last 2 s

    def _display_changed(self) -> None:
        if not self._restoring and self.session is not None and self.channels:
            self._save_timer.start()

    def _save_display(self) -> None:
        if self.session is not None and self.channels:
            self.session.update(display=self._display_state(), roi=asdict(self.roi_form.values()))

    def _flush_display_save(self) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_display()

    # -- 5 plots ----------------------------------------------------------------------

    def show_plot(self, open_window: bool = True) -> None:
        """Results to the plot window (opened unless `open_window` is False) and the table."""
        if self.sample not in self.results:
            return
        self.plot_window.set_results(self.results, self.sample, self.result_paths)
        self.results_window.set_results(self.results, self.result_paths, self.sample)
        if open_window:
            self.open_plot_window()
            self.open_results_window()

    def open_plot_window(self) -> None:
        if self.sample in self.results and not self.plot_window.results:
            self.plot_window.set_results(self.results, self.sample)
        self.plot_window.show()
        self.plot_window.raise_()
        self.plot_window.activateWindow()

    def open_results_window(self) -> None:
        if self.sample in self.results and not self.results_window.results:
            self.results_window.set_results(self.results, self.result_paths, self.sample)
        self.results_window.show()
        self.results_window.raise_()

    def save_plot(self, path: "str | Path | None" = None) -> "Path | None":
        """The plot window's figure in its chosen format (PNG by default), under <out>/."""
        target = Path(path) if path else Path(self.out_edit.text()) / f"{self.plot_window.default_name()}.png"
        written = self.plot_window.export(target)
        if written:
            self.window.append_log(f"Saved {written}")
        return written

    # -- display ----------------------------------------------------------------------

    def _spot_channel(self) -> str:
        return SPOT_CHANNELS[self.spot_combo.currentText()]

    def refresh_display(self) -> None:
        """Overlays and markers; the composite image itself follows z and the
        channel controls on its own (SliceCompositor)."""
        if self.sample is None:
            return
        self.compositor.refresh()
        self._show_overlay()
        self._show_spots()
        self._show_spots_3d()

    def _show_overlay(self) -> None:
        sv = self.viewer.slice_viewer
        if self.first is None or not self.show_outlines.isChecked():
            sv.set_overlay_labels(None, {}, {})
            return
        sv.set_overlay_labels(self.outlines, self.label_colors, {i: 255 for i in self.label_colors})
        sv.set_label_names(self._label_names())

    def _label_names(self) -> dict:
        """{label: cell name} of the timepoint shown (tracked cells only after the first)."""
        if self.tp_view is not None:
            return dict(self.tp_view["names"])
        if self.first_table is None:
            return {}
        return dict(zip(self.first_table["label"].astype(int), self.first_table["cell"]))

    def _spots(self):
        """The spot channel's spots of the timepoint shown, in whole-volume indices."""
        ch = self._spot_channel()
        if self.tp_view is not None:
            return self.tp_view["spots"].get(ch), ch
        if self.roi_run is None:
            return None, ch
        return spots_in_volume(self.roi_run[1], ch), ch

    def _show_spots(self) -> None:
        spots, ch = self._spots()
        sv = self.viewer.slice_viewer
        if spots is None or not self.show_spots.isChecked():
            sv.set_skeleton_points(None)
            return
        points = spots[["z_index", "y_index", "x_index"]].to_numpy(float)
        sv.set_skeleton_points(points, z_tolerance=1.0, color=self.spot_color.color())

    # -- timepoint browser ----------------------------------------------------------

    def _timepoints(self) -> list:
        return [t for t, _p in self.samples.get(self.sample, [])]

    def _set_tp_enabled(self, enabled: bool) -> None:
        """The browser works once the sample has series results (and its first
        timepoint is segmented with its cells of interest)."""
        enabled = bool(enabled and self.sample in self.results and self.first is not None
                       and self.interest is not None)
        ts = self._timepoints() if enabled else [0]
        for w in (self.tp_slider, self.tp_spin):
            w.blockSignals(True)
            w.setRange(min(ts), max(ts))
            w.setValue(self.tp_view["t"] if self.tp_view else min(ts))
            w.setEnabled(enabled)
            w.blockSignals(False)
        self.tp_auto.setEnabled(enabled)
        if not enabled:
            self.tp_status.setText("Run the series to browse the timepoints.")
        elif self.tp_view is None:
            self.tp_status.setText(f"Showing T = {min(ts)} (the first timepoint). Pick another to see its "
                                   "segmentation.")

    def _on_tp_changed(self, value: int, source: str) -> None:
        other = self.tp_spin if source == "slider" else self.tp_slider
        other.blockSignals(True)
        other.setValue(value)
        other.blockSignals(False)
        self._tp_timer.start()  # load when the slider stops moving

    def _reload_timepoint(self) -> None:
        if self.tp_view is not None:
            self.show_timepoint(self.tp_view["t"], force=True)

    def _targets_for(self, t: int) -> str:
        """Where the cells of interest were just before `t` (the series' own
        tracking), so a recomputed timepoint follows the same cells."""
        cells = self.results[self.sample]
        names = list(self.interest["cell"])
        points = []
        for name, start in zip(names, self.interest[CENTROID].to_numpy(float)):
            before = cells[(cells["cell"] == name) & (cells["timepoint"] < t)].sort_values("timepoint")
            points.append(before[CENTROID].to_numpy(float)[-1] if len(before) else start)
        return format_targets(points)

    def show_timepoint(self, t: int, force: bool = False) -> None:
        """Show timepoint `t` with its segmentation: saved by Run series, or
        computed now (then saved) for a series run before timepoints were saved."""
        if self.sample is None or not self.tp_spin.isEnabled():
            return
        files = dict(self.samples[self.sample])
        if t not in files:
            return
        if t == self._timepoints()[0]:
            self._back_to_first_timepoint()
            return
        if self.tp_view is not None and self.tp_view["t"] == t and not force:
            return
        if self._tp_loading:
            self._tp_pending = t
            return
        path, frame = files[t], self.first[1]["frame"]
        save_dir = self.session.dir if self.session is not None else None
        names = list(self.interest["cell"])
        settings, roi, recipe, reg = self.settings_form.values(), self.roi_form.values(), self.recipe, self.reg
        targets, auto = self._targets_for(t), self.tp_auto.isChecked()
        z_now = self.viewer.slice_viewer.current_z

        def _job(params, ctx, progress, cancel, log):
            import pandas as pd
            from scipy import ndimage

            from vanua.blocks._frame import as_frame, index_to_world

            # Only the slice being looked at is read now; the rest in the background.
            channels = open_lazy_channels(path, CHANNEL_INDEX, first_z=z_now)
            contrast = {}
            if auto:
                for name, ch in channels.items():
                    sample_for_contrast(ch, z_now)
                    contrast[name] = default_contrast(ch)
            progress(0.3, f"T={t}: slice {z_now} read")
            data = load_timepoint(save_dir, t) if save_dir is not None else None
            computed = data is None
            if computed:
                log(f"T={t}: no saved segmentation; computing it (about 30 s)")
                _r, values = run_timepoint(path, settings, targets=targets, roi=roi, recipe=recipe, reg=reg,
                                           cancel=cancel, progress=lambda f, m="": progress(0.6 + 0.3 * f, m))
                if save_dir is not None:
                    save_timepoint_view(save_dir, t, values, names, path)
                    data = load_timepoint(save_dir, t)
                if data is None:  # no session: use the run's values directly
                    table = values["cell_table"]
                    data = {"cells": values["cells"], "roi": values["roi"], "ref_roi": values.get("ref_roi"),
                            "names": {int(l): names[int(k)] for l, k in zip(table["label"], table["target"])},
                            "spots": {ch: spots_in_volume(values, ch) for ch in CHANNELS}}
            cells = data["cells"]
            # Outlines and badge positions slice by slice, as they are viewed.
            data["outlines"] = LazyOutlines(cells)
            data["positions"] = LazyPositions(cells)
            labels = sorted(data["names"])
            centres = ndimage.center_of_mass(np.ones_like(cells, dtype=np.uint8), cells, labels) if labels else []
            world = index_to_world(as_frame(frame), np.asarray(centres, float).reshape(-1, 3)) if labels else np.zeros((0, 3))
            data["badges"] = pd.DataFrame({"label": labels, "cell": [data["names"][l] for l in labels],
                                           "centroid_z_um": world[:, 0], "centroid_y_um": world[:, 1],
                                           "centroid_x_um": world[:, 2]})
            data.update(t=t, channels=channels, contrast=contrast, computed=computed)
            progress(1.0, f"T={t} ready")
            return data

        self._tp_loading = True
        self.tp_status.setText(f"Loading T = {t}…")

        def _done(data) -> None:
            self._tp_loading = False
            self._on_timepoint(data)
            pending, self._tp_pending = self._tp_pending, None
            if pending is not None and pending != t:
                self.show_timepoint(pending)

        self._start(_job, None, None, _done, "Timepoint")
        if not self.synchronous:
            # A failed or cancelled load must not leave the browser locked.
            thread = self._threads[-1] if self._threads else None
            if thread is not None:
                thread.failed.connect(lambda _m: setattr(self, "_tp_loading", False))
                thread.cancelled.connect(lambda: setattr(self, "_tp_loading", False))

    def _on_timepoint(self, data: dict) -> None:
        if self.tp_view is None:  # leaving the first timepoint: keep its display data
            self._t0_state = {"channels": self.channels, "contrast": self.contrast, "outlines": self.outlines,
                              "positions": self.label_positions, "colors": self.label_colors}
        t0_colors = self._t0_state["colors"]
        name_to_t0_label = dict(zip(self.first_table["cell"], self.first_table["label"].astype(int)))
        grey = (150, 150, 150)
        colors = {int(l): grey for l in np.unique(data["cells"]) if l}
        for label, name in data["names"].items():
            colors[int(label)] = t0_colors.get(name_to_t0_label.get(name, -1), grey)
        self.tp_view = data
        self._quick_contrast = dict(data["contrast"])
        self._apply_channels(data["channels"], data["contrast"] or self._t0_state["contrast"])
        self.outlines, self.label_positions, self.label_colors = data["outlines"], data["positions"], colors
        self._refresh_view(data["roi"], data.get("ref_roi"))
        found = sorted(data["names"].values())
        missing = [n for n in self.interest["cell"] if n not in found]
        note = f"Showing T = {data['t']}: {', '.join(found) or 'no cell of interest'} found"
        if missing:
            note += f" · <b>not found: {', '.join(missing)}</b>"
        if data.get("computed"):
            note += " · computed now (saved for next time)"
        self.tp_status.setText(note + ". Untracked cells are grey.")
        self.window.append_log(f"Display: T = {data['t']}")
        self._start_preload(data["channels"], refine_contrast=self.tp_auto.isChecked())
        self._display_changed()

    def _back_to_first_timepoint(self) -> None:
        if self.tp_view is None or self._t0_state is None:
            return
        s = self._t0_state
        self.tp_view = None
        self._apply_channels(s["channels"], s["contrast"])
        self._start_preload(s["channels"], refine_contrast=False)
        self.outlines, self.label_positions, self.label_colors = s["outlines"], s["positions"], s["colors"]
        roi = self.roi_run[1] if self.roi_run is not None else {}
        self._refresh_view(roi.get("roi"), roi.get("ref_roi"))
        t0 = self._timepoints()[0]
        for w in (self.tp_slider, self.tp_spin):
            w.blockSignals(True)
            w.setValue(t0)
            w.blockSignals(False)
        self.tp_status.setText(f"Showing T = {t0} (the first timepoint).")
        self._display_changed()

    def _start_preload(self, channels: dict, refine_contrast: bool) -> None:
        """Read every other slice of the shown channels in the background,
        nearest to the current slice first; then refine the first min / max
        (unless you changed it) and the histograms from the whole stack."""
        import threading

        if self._preload_cancel is not None:
            self._preload_cancel.set()  # the previous view's loading is no longer needed
        cancel = threading.Event()
        self._preload_cancel = cancel
        z = self.viewer.slice_viewer.current_z
        lazy = {k: ch for k, ch in channels.items() if hasattr(ch, "preload")}
        if not lazy:
            return

        def _job(params, ctx, progress, _cancel, log):
            for ch in lazy.values():
                ch.preload(z, cancel=cancel)
            return None

        def _done(_result=None) -> None:
            if cancel.is_set() or self.channels is not channels:
                return  # another sample or timepoint is shown now
            if refine_contrast:
                from dataclasses import replace as dc_replace

                for name, key, _lut, _visible in LAYERS:
                    layer = self.layers.by_name(name)
                    quick = self._quick_contrast.get(key)
                    if quick is not None and layer.contrast_range is not None \
                            and tuple(layer.contrast_range) == tuple(quick):
                        full = default_contrast(channels[key])
                        self.contrast[key] = full
                        self.layers.replace_at(self.layers.index_of(name), dc_replace(layer, contrast_range=full))
            self.layer_panel.refresh_histograms()
            self.compositor.refresh()
            self.window.append_log("All slices loaded")

        if self.synchronous:
            _job(None, None, None, None, None)
            _done()
            return
        thread = OpWorkerThread(_job, None, None)
        thread.succeeded.connect(lambda _r: _done())
        thread.finished.connect(lambda: self._preload_threads.remove(thread) if thread in self._preload_threads else None)
        self._preload_threads.append(thread)  # keep a reference (building.md §5)
        thread.start()

    def _apply_channels(self, channels: dict, contrast: dict) -> None:
        """New image data in every channel layer, keeping each layer's LUT,
        opacity, gamma and visibility; min / max from `contrast`, or kept."""
        from dataclasses import replace as dc_replace

        z = self.viewer.slice_viewer.current_z
        self.channels = channels
        layers = []
        for name, key, _lut, _visible in LAYERS:
            layer = self.layers.by_name(name)
            # Auto contrast off: your min / max carry over (bleaching shows as dimming);
            # on: each timepoint's own default min / max.
            cr = contrast.get(key) if self.tp_auto.isChecked() else layer.contrast_range
            layers.append(dc_replace(layer, data=channels[key], contrast_range=cr or layer.contrast_range))
        show_image(self.viewer.slice_viewer, channels["daf28"], self.frame)  # the z-navigation carrier
        self.layer_panel.set_layers(layers)
        self.viewer.slice_viewer.set_z(min(z, channels["daf28"].shape[0] - 1))
        self.compositor.refresh()

    def _refresh_view(self, roi, ref_roi) -> None:
        self._show_overlay()
        self._show_cell_labels()
        self._show_spots()
        self.roi_overlay.set_box(roi, ref_roi)
        if self.viewer.viewer_3d is not None:
            self.build_surfaces()
        else:
            self._show_spots_3d()

    # -- 3D ---------------------------------------------------------------------------

    def build_surfaces(self) -> None:
        if self.viewer.viewer_3d is None or self.first is None:
            return  # offscreen / no VTK: the 3D tab is a placeholder
        labels = self.tp_view["cells"] if self.tp_view is not None else self.first[1]["cells"]
        frame = self.first[1]["frame"]

        def _job(params, ctx, progress, cancel, log):
            mesh, _table = label_surfaces(labels, frame, step_size=1, cancel=cancel, progress=progress)
            return mesh

        self._start(_job, None, None, self._on_surfaces, "Surfaces")

    def _clear_3d(self) -> None:
        if self.viewer.viewer_3d is not None:
            clear_label_surfaces(self.viewer.viewer_3d, self.surface_actors + self.spot_actors + self.label_actors)
        self.surface_actors, self.spot_actors, self.label_actors = [], [], []

    def _on_surfaces(self, mesh) -> None:
        self._clear_3d()
        self.surface_actors = add_label_surfaces(self.viewer.viewer_3d, mesh, colors=self.label_colors,
                                                 opacity=0.45)
        self._show_spots_3d()
        self._show_cell_labels_3d()
        self.window.append_log(f"{len(self.surface_actors)} cell surfaces in the 3D tab")

    # -- cell labels ------------------------------------------------------------------

    def _excluded_cells(self) -> set:
        """Cells excluded right now: the review table's Include ticks once it
        is filled (so unticking a cell greys its label at once), else the rule."""
        if self.first_table is None or self.tp_view is not None:
            return set()  # after the first timepoint only the tracked cells are named
        if self.cell_table.rowCount():
            return {self.cell_table.item(i, 0).text() for i in range(self.cell_table.rowCount())
                    if self.cell_table.item(i, INCLUDE_COL) is not None
                    and self.cell_table.item(i, INCLUDE_COL).checkState() != Qt.Checked}
        return set(self.first_table.loc[~self.first_table["included"], "cell"])

    def _show_cell_labels(self) -> None:
        if self.first is None or self.first_table is None:
            self.cell_labels.clear()
            return
        names = {label: self._shown(name) for label, name in self._label_names().items()}
        self.cell_labels.visible = self.show_cell_labels.isChecked()
        self.cell_labels.set_cells(self.label_positions, names, self.label_colors,
                                   {self._shown(c) for c in self._excluded_cells()})
        self._show_cell_labels_3d()

    def _show_cell_labels_3d(self) -> None:
        v3 = self.viewer.viewer_3d
        if v3 is None:
            return
        for actor in self.label_actors:
            v3.renderer.RemoveActor(actor)
        self.label_actors = []
        if self.first_table is not None and self.show_cell_labels.isChecked():
            excluded = self._excluded_cells()
            t = self.tp_view["badges"] if self.tp_view is not None else self.first_table
            world = t[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
            for (_, row), point in zip(t.iterrows(), world_to_display(world, self.frame)):
                gone = row["cell"] in excluded
                rgb = self.label_colors.get(int(row["label"]), (255, 255, 255))
                text = self._shown(row["cell"])
                actor = _text_actor(f"{text} · excl." if gone else text, point, rgb, excluded=gone)
                v3.renderer.AddActor(actor)
                self.label_actors.append(actor)
        v3.render()

    def _show_spots_3d(self) -> None:
        v3 = self.viewer.viewer_3d
        if v3 is None:
            return
        for actor in self.spot_actors:
            v3.renderer.RemoveActor(actor)
        self.spot_actors = []
        spots, ch = self._spots()
        if spots is not None and self.show_spots_3d.isChecked() and len(spots):
            world = spots[[c for c in spots.columns if c.endswith("_um") and len(c) == 4]].to_numpy(float)
            actor = _points_actor(world_to_display(world, self.frame), self.spot_color.color())
            v3.renderer.AddActor(actor)
            self.spot_actors = [actor]
        v3.render()


def theme_color(rgb):
    from PyQt5.QtGui import QBrush, QColor

    return QBrush(QColor(*rgb))


def _text_actor(text: str, point_xyz, rgb, excluded: bool = False):
    """A badge that always faces the camera, at `point_xyz` (display coordinates):
    the 3D counterpart of the slice view's CellBadge -- the cell's colour as a
    translucent background, a lighter frame, black or white text for contrast."""
    from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D

    from .display import badge_text_color

    actor = vtkBillboardTextActor3D()
    actor.SetInput(f" {text} ")  # a little padding inside the background
    actor.SetPosition(*(float(v) for v in point_xyz))
    prop = actor.GetTextProperty()
    prop.SetFontFamilyToArial()
    prop.SetFontSize(15)
    prop.BoldOn()
    prop.SetJustificationToCentered()
    prop.SetVerticalJustificationToCentered()
    if excluded:
        prop.SetColor(0.67, 0.67, 0.71)
        prop.SetBackgroundColor(0.12, 0.12, 0.18)
        prop.SetFrameColor(0.55, 0.55, 0.6)
    else:
        prop.SetColor(*(c / 255.0 for c in badge_text_color(rgb)))
        prop.SetBackgroundColor(*(c / 255.0 for c in rgb))
        prop.SetFrameColor(*(min(1.0, c / 255.0 * 1.4 + 0.15) for c in rgb))
    prop.SetBackgroundOpacity(0.88)
    prop.FrameOn()
    prop.SetFrameWidth(1)
    return actor


def _points_actor(points_xyz: np.ndarray, rgb, radius: float = 0.15):
    """Small spheres at `points_xyz` (display coordinates)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkGlyph3D
    from vtkmodules.vtkFiltersSources import vtkSphereSource
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points_xyz, dtype=np.float64), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(pts)
    sphere = vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(8)
    sphere.SetPhiResolution(8)
    glyph = vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(poly)
    glyph.ScalingOff()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*(c / 255.0 for c in rgb))
    return actor


def build_app(synchronous: bool = False) -> HcrCellsApp:
    return HcrCellsApp(synchronous=synchronous)


TAB_3D = 3  # MultiViewHost tabs: Slice, MIP, Ortho, 3D


def show(app_obj: HcrCellsApp, init_3d: bool = False) -> None:
    """Show on a real display. The 3D tab is NOT created here: creating the
    VTK render window blocks the main thread right after show(), which is
    what left the window white at start. It is created the first time the
    3D tab is opened (:func:`initialize_3d`), or now with `init_3d=True`."""
    app_obj.window.resize(1500, 950)
    app_obj.window.show()
    theme.apply_dark_titlebar(app_obj.window)
    tabs = app_obj.viewer.as_widget()
    tabs.currentChanged.connect(lambda i: initialize_3d(app_obj) if i == TAB_3D else None)
    if init_3d:
        initialize_3d(app_obj)


def initialize_3d(app_obj: HcrCellsApp) -> None:
    """Create the VTK render window (main thread, window shown); once."""
    if app_obj.viewer.viewer_3d is not None:
        return
    import time

    started = time.perf_counter()
    app_obj.window.show_progress(0.0, "Starting the 3D view…")
    QApplication.processEvents()
    tabs = app_obj.viewer.as_widget()
    tabs.blockSignals(True)  # initialize() rebuilds the 3D tab; do not re-enter on its index change
    try:
        app_obj.viewer.initialize()
        tabs.setCurrentIndex(TAB_3D)
    finally:
        tabs.blockSignals(False)
        app_obj.window.clear_progress()
    if app_obj.viewer.viewer_3d is not None:
        app_obj.viewer.viewer_3d.set_orientation_labels("X+", "X-", "Y+", "Y-", "Z+", "Z-")
        app_obj.window.append_log(f"3D view ready ({time.perf_counter() - started:.1f} s)")
        if app_obj.first is not None:
            app_obj.build_surfaces()  # segmented before the 3D tab existed


def main(argv: "list[str] | None" = None) -> int:
    """Blocking start without a splash (tests, debugging); `pixi run gui` uses
    :mod:`hcrcells.launch`, which shows a splash while the imports load."""
    argv = list(sys.argv if argv is None else argv)
    theme.enable_high_dpi()
    qapp = QApplication(argv)
    qapp.setOrganizationName("hcrcells")  # QSettings home of the demo-mode toggle
    qapp.setApplicationName(APP_NAME)
    theme.install(qapp, app_id=APP_ID, app_name=APP_NAME)
    hcr = build_app()
    show(hcr)
    hcr.window.append_log(f"{APP_NAME} started.")
    folder = Path(argv[1]) if len(argv) > 1 else SAMPLE_DIR
    hcr.folder_edit.setText(str(folder))
    hcr.scan(folder)
    return qapp.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
