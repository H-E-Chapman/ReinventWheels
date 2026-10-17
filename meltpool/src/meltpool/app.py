"""MeltPool GUI on the vanua shell: Load -> Calibrate (flip, stabilise) -> Crop ->
Transient -> Liquid-gas boundary -> Measure -> Results -> Liquid-solid, with a
Tutorial / Demo mode (vanua.ui.guide.GuidedTour; View menu).

    python -m meltpool.app [MOVIE]

The steps call the recipe's own ops (``recipes/meltpool.toml``) on the movie
already open, so the GUI, the batch runner and the tests compute the same
numbers; the export records the recipe configured with exactly the settings used.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# vanua.ui before anything that imports VTK (building.md §5).
from vanua.ui import theme  # noqa: I001
from vanua.ui.blocks import PlotPanel, TableView, show_image
from vanua.ui.forms import ParamsForm
from vanua.ui.guide import GuidedTour, GuideStep
from vanua.ui.shell import MainWindow, WorkflowSpec, WorkflowStep
from vanua.ui.viewers.multi_view import MultiViewHost
from vanua.ui.workers import OpWorkerThread

import numpy as np
from PyQt5.QtCore import QObject, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainterPath, QPen
from PyQt5.QtWidgets import (QApplication, QCheckBox, QFileDialog, QGraphicsPathItem, QGraphicsRectItem, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSlider, QVBoxLayout, QWidget)

from . import movie as mv
from .contrast import ContrastPanel
from .opacity_panel import OpacityPanel
from .rect_tool import RectDrawer
from .volume_views import MovieVolumeViews
from . import boundary
from .ops import (BackgroundParams, CropParams, DynamicsParams, FlagParams, MedianParams, OPS, OverlayParams,
                  SolidBoundaryParams, StabiliseParams, TraceParams, crop_frame, dynamics, frames_table,
                  profile_trace, summary_table, flag_table, trace_one, trace_prepare, trace_time_points,
                  stabilise_region)
from .pipeline import (Calibration, Settings, configured, configured_solid, export_movie, load_recipe, registry,
                       series_figure, spectrum_figure)

APP_NAME = "MeltPool"
APP_ID = "org.vanua.apps.meltpool"
APP_DIR = Path(__file__).resolve().parents[2]
SYNTHETIC = APP_DIR / "data" / "synthetic_movie.tif"
#: The synthetic sample's calibration (the lab's weld videos: 4.44 um/px, 100 000 fps, build direction up).
SAMPLE_CALIBRATION = {"pixel_um": 4.44, "fps": 100000.0, "flip_vertical": False}
DEFAULT_OUT = Path.cwd() / "out"
DEMO_KEY = "meltpool/demoMode"
OPS_BY_NAME = {op.name: op for op in OPS}


def sample_path() -> Path:
    """The synthetic movie (written on first use), or MELTPOOL_SAMPLE if set."""
    env = os.environ.get("MELTPOOL_SAMPLE")
    return Path(env) if env else SYNTHETIC


def _column(*widgets) -> "tuple[QWidget, QVBoxLayout]":
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setAlignment(Qt.AlignTop)
    for w in widgets:
        layout.addWidget(w)
    return page, layout


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def _browse_row(edit: QLineEdit, pick) -> QWidget:
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, 1)
    button = QPushButton("Browse…")
    button.clicked.connect(pick)
    h.addWidget(button)
    return row


def _run_op(name: str, params, ctx, progress, cancel, log):
    return OPS_BY_NAME[name].run(params, ctx, progress, cancel, log)


class MeltPoolApp(QObject):
    """Builds the window and owns the workflow state. `synchronous=True` runs
    the workers' functions inline (offscreen tests, screenshots)."""

    loaded = pyqtSignal()
    cropped = pyqtSignal()
    transient_applied = pyqtSignal()
    stabilised = pyqtSignal()
    median_ready = pyqtSignal()
    solid_measured = pyqtSignal()
    previewed = pyqtSignal()
    measured = pyqtSignal()
    exported = pyqtSignal()

    def __init__(self, synchronous: bool = False) -> None:
        super().__init__()
        self.synchronous = synchronous
        self.recipe = load_recipe()
        self.reg = registry()
        self.source: Path | None = None
        self.movie_blob: dict | None = None
        self.frame: dict | None = None
        self.roi_blob: dict | None = None    # the cropped movie the tracing sees
        self.roi_frame: dict | None = None
        self.frames = None
        self.table = None
        self.spectrum = self.peaks = None
        self.summary = None
        self._threads: list = []

        self.viewer = MultiViewHost()
        steps = (
            WorkflowStep("1  Load", self._build_load()),
            WorkflowStep("2  Calibrate", self._build_calibrate()),
            WorkflowStep("3  Crop", self._build_crop()),
            WorkflowStep("4  Transient", self._build_transient()),
            WorkflowStep("5  Liquid-gas boundary", self._build_boundary()),
            WorkflowStep("6  Measure", self._build_measure()),
            WorkflowStep("7  Results", self._build_results()),
            WorkflowStep("8  Liquid-solid", self._build_solid()),
        )
        self.contrast = ContrastPanel()
        self.contrast.attach(self.viewer.slice_viewer)
        self.window = MainWindow(WorkflowSpec(title=APP_NAME, steps=steps, viewer=self.viewer,
                                              rendering_panel=self._rendering_dock()))
        for i in (1, 2, 3, 4, 5, 6, 7):
            self.window.set_step_enabled(i, False)
        self.window.cancel_button.clicked.connect(self._cancel)
        self.volume_views = MovieVolumeViews(self.viewer, self._volume_start, self.window.append_log,
                                             opacity=self.opacity_panel)
        self.tour = GuidedTour(self.window, [
            GuideStep(self.sample_button, "Click <b>Load sample</b> to open a high-speed X-ray movie of a weld "
                      "pool (synthetic, build direction up).", page=0, done=self.loaded),
            GuideStep(self.stabilise_button, "Click <b>Stabilise now</b>: every frame is registered to 1/100 px on "
                      "the substrate below the pool, so camera shake is not taken for pool motion.", page=1,
                      done=self.stabilised),
            GuideStep(self.crop_button, "Click <b>Apply crop</b>: only the region around the pool is analysed "
                      "(faster, and air and powder away from it are ignored). <b>Show on full frame</b> outlines "
                      "it first.", page=2, done=self.cropped),
            GuideStep(self.preview_button, "Drag the time slider, then click <b>Preview</b>: the liquid-gas "
                      "boundary -- the rim of the vapour depression -- as three lines: inner edge (cyan), band "
                      "centre (orange), outer edge (magenta).", page=4, done=self.previewed),
            GuideStep(self.run_button, "Click <b>Trace all frames</b>: depth, width and area of each line over "
                      "time, the start-up flagged, and the oscillation frequency.", page=5, done=self.measured),
            GuideStep(self.export_button, "Click <b>Export</b>: the per-frame CSV, the summary, the traced lines "
                      "and the spectrum (each with a provenance file), the plots and the overlay video.", page=6,
                      done=self.exported),
        ], settings_key=DEMO_KEY)
        self.tour.mode_changed.connect(self._demo_presets)
        self._demo_presets(self.tour.enabled)

    # -- pages ------------------------------------------------------------------------

    def _build_load(self) -> QWidget:
        self.sample_button = QPushButton("Load sample")
        theme.style_button_role(self.sample_button, "primary")
        self.sample_button.setToolTip("One click: a synthetic weld video (oscillating vapour depression)")
        self.sample_button.clicked.connect(self.load_sample)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("TIFF, HDF5, .npy or video (AVI, MP4, ...)")
        load_button = QPushButton("Load movie")
        theme.style_button_role(load_button, "primary")
        load_button.clicked.connect(lambda: self.load(self.path_edit.text()))
        self.load_status = _label("No movie loaded. Calibration (pixel size, frame rate, flip) is on the next "
                                  "page and can be changed after loading.")
        page, _ = _column(self.sample_button, _label("…or your own movie"), _browse_row(self.path_edit, self._pick),
                          load_button, self.load_status)
        return page

    def _build_calibrate(self) -> QWidget:
        self.calibration_form = ParamsForm(Calibration())
        self.apply_calibration_button = QPushButton("Apply calibration")
        self.apply_calibration_button.setToolTip("Use the pixel size, frame rate and flips below")
        self.apply_calibration_button.clicked.connect(self.apply_calibration_now)
        self.calibrate_status = _label("Pixel size and frame rate set the units; flip so the build direction is up.")
        self.stabilise_button = QPushButton("Stabilise now (sub-pixel)")
        theme.style_button_role(self.stabilise_button, "primary")
        self.stabilise_button.setToolTip("Register every frame in x and y on the substrate texture and shift it "
                                         "(Fourier, to 1/100 px). Mode and region below.")
        self.stabilise_button.clicked.connect(self.stabilise)
        unstabilise = QPushButton("Remove stabilisation")
        unstabilise.clicked.connect(self.remove_stabilisation)
        self.stabilise_form = ParamsForm(StabiliseParams(enabled=True))
        self.stabilise_status = _label("Not stabilised. <b>fixed</b> for a static sample (removes all motion), "
                                       "<b>jitter</b> for a moving one (removes shake, keeps the drift).")
        self.stabilise_status.setTextFormat(Qt.RichText)
        self.shifts: "np.ndarray | None" = None
        self._shifts_flip: "tuple | None" = None
        self.flip_v_button = QPushButton("Flip vertically")
        self.flip_v_button.setToolTip("Turn the movie upside down (applied at once; the build direction should be up)")
        self.flip_v_button.clicked.connect(lambda: self.toggle_flip("vertical"))
        self.flip_h_button = QPushButton("Flip horizontally")
        self.flip_h_button.setToolTip("Mirror the movie left-right (applied at once)")
        self.flip_h_button.clicked.connect(lambda: self.toggle_flip("horizontal"))
        flips = QWidget()
        row = QHBoxLayout(flips)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.flip_v_button)
        row.addWidget(self.flip_h_button)
        self.stab_draw_button = QPushButton("Draw stabilisation region")
        self.stab_draw_button.setCheckable(True)
        self.stab_draw_button.setToolTip("Drag a box (green, dashed) on solid material that does NOT contain the "
                                         "moving boundary: the frames are registered on it")
        self.stab_draw_button.toggled.connect(self._set_stab_draw_mode)
        stab_show = QPushButton("Show stabilisation region")
        stab_show.setToolTip("Outline the region that will be used -- the drawn one, or the automatic one "
                             "(the substrate below the pool)")
        stab_show.clicked.connect(self.show_stabilisation_region)
        stab_auto = QPushButton("Automatic region")
        stab_auto.clicked.connect(self.reset_stabilisation_region)
        self.stab_drawer = RectDrawer(self.viewer.slice_viewer, "#22c55e", "Draw region", self._on_stab_region)
        page, _ = _column(self.calibrate_status, flips, self.calibration_form, self.apply_calibration_button,
                          _label("<b>Stabilisation</b>"), self.stabilise_button, unstabilise,
                          self.stab_draw_button, stab_show, stab_auto, self.stabilise_status, self.stabilise_form)
        return page

    def _build_crop(self) -> QWidget:
        self.draw_button = QPushButton("Draw ROI on image")
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip("Shows the full frame; drag a rectangle with the left mouse button")
        self.draw_button.toggled.connect(self._set_draw_mode)
        self.crop_button = QPushButton("Apply crop")
        theme.style_button_role(self.crop_button, "primary")
        self.crop_button.clicked.connect(self.apply_crop)
        outline = QPushButton("Show on full frame")
        outline.clicked.connect(self.show_crop_outline)
        full = QPushButton("Use full frame")
        full.clicked.connect(self.reset_crop)
        self.frame_size_label = _label("")
        self.crop_form = ParamsForm(CropParams())
        self.crop_status = _label("Click <b>Draw ROI on image</b> and drag a box, or type the edges below "
                                  "(pixel columns/rows of the frame as shown). Keep some unmelted surface on each "
                                  "side of the depression: the original surface is found there.")
        self.crop_status.setTextFormat(Qt.RichText)
        self._roi_item: "QGraphicsRectItem | None" = None
        self._drag_start: "tuple[float, float] | None" = None
        page, _ = _column(self.draw_button, self.crop_button, outline, full, self.frame_size_label,
                          self.crop_status, self.crop_form)
        return page

    def _build_solid(self) -> QWidget:
        self.median_button = QPushButton("Compute stabilised median")
        theme.style_button_role(self.median_button, "primary")
        self.median_button.setToolTip("Per-pixel median over the frames (of the cropped, stabilised movie): the "
                                      "fluctuating liquid-gas boundary averages out, a stationary one stays")
        self.median_button.clicked.connect(self.compute_median)
        self.detail_check = QCheckBox("Show fine detail (band-pass)")
        self.detail_check.setToolTip("Beam profile and detector row banding removed, fine lines kept")
        self.detail_check.toggled.connect(lambda _on: self._show_median())
        self.trace_button = QPushButton("Trace line (click points)")
        self.trace_button.setCheckable(True)
        self.trace_button.setToolTip("Click along the liquid-solid line, from one end at the surface to the other")
        self.trace_button.toggled.connect(self._set_trace_mode)
        undo = QPushButton("Undo point")
        undo.clicked.connect(self.undo_point)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_points)
        self.measure_line_button = QPushButton("Measure traced line")
        theme.style_button_role(self.measure_line_button, "primary")
        self.measure_line_button.clicked.connect(lambda: self.measure_solid(manual=True))
        self.auto_button = QPushButton("Automatic search")
        self.auto_button.setToolTip("Reports a line only if it stands out against the same search where there is "
                                    "no pool (detection threshold below)")
        self.auto_button.clicked.connect(lambda: self.measure_solid(manual=False))
        self.export_solid_button = QPushButton("Export liquid-solid")
        self.export_solid_button.clicked.connect(lambda: self.export_solid(self.out_edit.text()))
        self.solid_status = _label("Compute the median first. Stabilise on the Calibrate page for the sharpest "
                                   "median (sub-pixel).")
        self.solid_status.setTextFormat(Qt.RichText)
        self.median_form = ParamsForm(MedianParams())
        self.solid_form = ParamsForm(SolidBoundaryParams())
        self.solid_table = TableView()
        self.solid_table.setMinimumHeight(80)
        self.median: "np.ndarray | None" = None
        self.solid_points: list = []
        self.solid_result = None
        self._solid_params: "SolidBoundaryParams | None" = None
        self._line_item: "QGraphicsPathItem | None" = None
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(undo)
        row.addWidget(clear)
        page, _ = _column(self.median_button, self.detail_check, self.trace_button, buttons,
                          self.measure_line_button, self.auto_button, self.export_solid_button, self.solid_status,
                          self.solid_table, self.median_form, self.solid_form)
        return page

    def _build_transient(self) -> QWidget:
        self.transient_preview_button = QPushButton("Preview transient frame")
        self.transient_preview_button.setToolTip("The frame at the slider position minus its running median")
        self.transient_preview_button.clicked.connect(self.preview_transient)
        self.transient_apply_button = QPushButton("Apply")
        theme.style_button_role(self.transient_apply_button, "primary")
        self.transient_apply_button.clicked.connect(self.apply_transient)
        self.transient_status = _label("Optional. Subtracting a running median removes what stays put and leaves "
                                       "the <b>transient</b> features. Tick <i>Subtract running median</i>, "
                                       "preview, then Apply. By default it is for display only (views, overlay "
                                       "video); a stationary rim cancels in these frames, so tracing it there may "
                                       "work less well.")
        self.transient_status.setTextFormat(Qt.RichText)
        self.background_form = ParamsForm(BackgroundParams())
        self._background_applied = BackgroundParams()
        page, _ = _column(self.transient_preview_button, self.transient_apply_button, self.transient_status,
                          self.background_form)
        return page

    def _background_blobs(self, params: BackgroundParams) -> "tuple[dict, dict]":
        out = _run_op("meltpool.subtract_background", params, {"movie": self.roi_blob}, lambda *a: None, None,
                      lambda m: None)
        return out["movie"], out["display_movie"]

    def _update_background(self) -> None:
        """The analysis and display movies from the ROI movie and the applied background setting."""
        if self.roi_blob is None:
            self.analysis_blob = self.display_blob = None
            return
        self.analysis_blob, self.display_blob = self._background_blobs(self._background_applied)

    def preview_transient(self) -> None:
        if self.roi_blob is None:
            return
        errors = self.background_form.errors()
        if errors:
            self.transient_status.setText("\n".join(errors))
            return
        params = self.background_form.values()
        params.enabled = True
        _analysis, display = self._background_blobs(params)
        i = self.slider.value()
        self._display(mv.Movie(display).mean_of(i, 1), self.roi_frame, kind="transient")
        self.transient_status.setText(f"Frame {i} minus its running median ({params.mode}, {params.window} frames, "
                                      f"{params.operation}). Click <b>Apply</b> to use it.")

    def apply_transient(self) -> None:
        if self.roi_blob is None:
            return
        errors = self.background_form.errors()
        if errors:
            self.transient_status.setText("\n".join(errors))
            return
        self._background_applied = self.background_form.values()
        self._update_background()
        self._refresh_volume()
        p = self._background_applied
        if p.enabled:
            self.transient_status.setText(f"Applied: running median ({p.mode}, {p.window} frames, {p.operation}) "
                                          f"for <b>{p.apply_to}</b>.")
        else:
            self.transient_status.setText("Off: the frames are shown and traced as they are.")
        self.window.append_log("Transient: " + (f"{p.operation} running median, {p.window} frames, {p.apply_to}"
                                                if p.enabled else "off"))
        self._show_current()
        self.window.append_log("Next: 5 Liquid-gas boundary (click it when ready)")
        self.transient_applied.emit()

    def _build_boundary(self) -> QWidget:
        self.preview_button = QPushButton("Preview")
        theme.style_button_role(self.preview_button, "primary")
        self.preview_button.clicked.connect(self.preview)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self._show_current)
        self.frame_label = _label("")
        self.boundary_status = _label("Traces the dark rim of the vapour depression (the liquid-gas boundary) "
                                     "as three lines: inner edge (gas side), band centre, outer edge (liquid side).")
        self.boundary_status.setTextFormat(Qt.RichText)
        self.trace_form = ParamsForm(TraceParams())
        self._line_items: dict = {}
        page, _ = _column(self.slider, self.frame_label, self.preview_button, self.boundary_status,
                          self.trace_form)
        return page

    def _build_measure(self) -> QWidget:
        self.run_button = QPushButton("Trace all frames")
        theme.style_button_role(self.run_button, "primary")
        self.run_button.clicked.connect(self.measure)
        self.flag_form = ParamsForm(FlagParams())
        self.dynamics_form = ParamsForm(DynamicsParams())
        self.measure_status = _label("Traces every Nth frame (settings on the Liquid-gas boundary page) -- "
                                     "minutes for a long movie; Cancel stops it. Then the start-up is flagged and "
                                     "the spectra of depth, width and area are computed.")
        page, _ = _column(self.run_button, self.measure_status, _label("<b>Start-up flags</b>"), self.flag_form,
                          _label("<b>Dynamics</b>"), self.dynamics_form)
        return page

    def _build_results(self) -> QWidget:
        self.summary_label = _label("")
        self.summary_label.setTextFormat(Qt.RichText)
        self.out_edit = QLineEdit(str(DEFAULT_OUT))
        self.export_button = QPushButton("Export CSV + plot")
        theme.style_button_role(self.export_button, "primary")
        self.export_button.clicked.connect(lambda: self.export(self.out_edit.text()))
        self.export_status = _label("")
        self.overlay_button = QPushButton("Export overlay video")
        self.overlay_button.setToolTip("The movie with the traced lines and a time stamp drawn (MP4 or GIF); "
                                       "brightness as on screen unless Display min/max are set")
        self.overlay_button.clicked.connect(lambda: self.export_overlay(self.out_edit.text()))
        self.overlay_form = ParamsForm(OverlayParams())
        self.plot = PlotPanel()
        self.plot.setMinimumHeight(420)
        self.spectrum_plot = PlotPanel()
        self.spectrum_plot.setMinimumHeight(240)
        self.table_view = TableView()
        self.table_view.setMinimumHeight(240)
        page, _ = _column(self.summary_label, _browse_row(self.out_edit, self._pick_out), self.export_button,
                          self.export_status, self.overlay_button, self.overlay_form, self.plot,
                          self.spectrum_plot, self.table_view)
        return page

    def _pick(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in mv.MOVIE_SUFFIXES)
        path, _ = QFileDialog.getOpenFileName(self.window, "Open movie", "",
                                              f"Movies ({patterns});;All files (*)")
        if path:
            self.path_edit.setText(path)

    def _pick_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self.window, "Output folder", self.out_edit.text())
        if path:
            self.out_edit.setText(path)

    # -- threads ------------------------------------------------------------------------

    def _start(self, fn, params, ctx, on_success, label: str) -> None:
        if self.synchronous:
            on_success(fn(params, ctx, lambda *a: None, threading.Event(), self.window.append_log))
            return
        thread = OpWorkerThread(fn, params, ctx)
        thread.progress.connect(lambda e: self.window.show_progress(e.fraction, f"{label}: {e.message}"))
        thread.log.connect(self.window.append_log)
        thread.succeeded.connect(lambda result: (self.window.clear_progress(), on_success(result)))
        thread.failed.connect(lambda msg: (self.window.clear_progress("Failed"),
                                           self.window.append_log(f"{label} failed: {msg}")))
        thread.cancelled.connect(lambda: self.window.clear_progress("Cancelled"))
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)
        self._threads.append(thread)
        self.window.show_progress(0.0, f"{label}…")
        thread.start()

    def _rendering_dock(self) -> QWidget:
        """The right-hand dock: display contrast (2D views), then which values are transparent in 3D."""
        self.opacity_panel = OpacityPanel()
        dock = QWidget()
        layout = QVBoxLayout(dock)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.contrast)
        layout.addWidget(self.opacity_panel)
        layout.addStretch(1)
        return dock

    def _volume_start(self, job, label, done, failed) -> None:
        """MovieVolumeViews runs its stack building through the app's worker helper."""
        def _run(p, c, progress, cancel, log):
            try:
                return ("ok", job(progress, cancel))
            except Exception as exc:  # noqa: BLE001 -- reported in the log, the Slice view keeps working
                return ("error", str(exc))

        self._start(_run, None, None, lambda r: done(r[1]) if r[0] == "ok" else failed(r[1]), label)

    def _refresh_volume(self) -> None:
        blob = getattr(self, "display_blob", None) or self.roi_blob
        if getattr(self, "volume_views", None) is not None and blob is not None:
            lit = ~self.frames["blank"].to_numpy(bool) if self.frames is not None else None
            self.volume_views.set_movie(blob, self.roi_frame, lit)

    def _cancel(self) -> None:
        for thread in list(self._threads):
            thread.cancel()

    def _demo_presets(self, on: bool) -> None:
        """Demo mode: a coarser time step so 'Measure all frames' takes a minute or two on a 12 GB movie."""
        values = self.trace_form.values()
        values.stride = 4 if on else TraceParams.stride
        self.trace_form.set_values(values)

    # -- actions --------------------------------------------------------------------------

    def load_sample(self) -> None:
        path = sample_path()
        if path.resolve() == SYNTHETIC.resolve() and not path.is_file():
            sys.path.insert(0, str(APP_DIR / "scripts"))
            from make_synthetic import make

            make(path)
        if path.resolve() == SYNTHETIC.resolve():
            cal = self.calibration_form.values()
            for key, value in SAMPLE_CALIBRATION.items():
                setattr(cal, key, value)
            self.calibration_form.set_values(cal)
        self.path_edit.setText(str(path))
        self.load(path)

    def load(self, path: "str | Path") -> None:
        path = Path(path)
        cal = self.calibration_form.values()
        params = OPS_BY_NAME["meltpool.open_movie"].params(path=str(path), dataset=cal.dataset, pixel_um=cal.pixel_um,
                                                           pixel_from_file=cal.pixel_from_file, fps=cal.fps,
                                                           flip_vertical=cal.flip_vertical,
                                                           flip_horizontal=cal.flip_horizontal)
        self.load_status.setText(f"Reading {path.name} (per-frame intensities)…")

        def _job(p, ctx, progress, cancel, log):
            return _run_op("meltpool.open_movie", p, ctx, progress, cancel, log)

        self._start(_job, params, {}, lambda result: self._on_loaded(path, result), "Load")

    def _on_loaded(self, path: Path, result: dict) -> None:
        self.source = path
        self.movie_blob, self.frame, self.frames = result["movie"], result["frame"], result["frames"]
        self.table, self.summary, self.profiles, self.spectrum, self.peaks = None, None, None, None, None
        self._crop_quietly()
        n, (h, w) = self.movie_blob["n"], self.movie_blob["shape"]
        lit = np.flatnonzero(~self.frames["blank"].to_numpy(bool))
        trace = self.trace_form.values()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n - trace.n_avg))
        self.slider.setValue(int(lit[0] + (lit[-1] - lit[0]) * 0.6) if len(lit) else 0)
        self.slider.blockSignals(False)
        meta = self.movie_blob.get("file_meta", {})
        notes = [meta.get("format", "")]
        if meta.get("dataset"):
            notes.append(f"dataset /{meta['dataset']}")
        if np.dtype(self.movie_blob["dtype"]).itemsize == 1:
            notes.append("8-bit")
        if meta.get("pixel_um") and self.calibration_form.values().pixel_from_file:
            notes.append(f"pixel size {meta['pixel_um']:.4g} µm from the file")
        if meta.get("fps_in_file"):
            notes.append(f"file says {meta['fps_in_file']:g} fps (playback; not used -- set the recording rate "
                         "on the Calibrate page)")
        self.load_status.setText(f"{path.name}  ({'; '.join(filter(None, notes))})\n{n} frames of {h} x {w}; "
                                 f"{int(self.frames.blank.sum())} blank (shutter closed); first lit frame "
                                 f"{int(lit[0]) if len(lit) else '-'}")
        self.window.append_log(f"Loaded {path.name}")
        self._show_current()
        self.viewer.slice_viewer.fit_to_view()
        for i in (1, 2, 3, 4, 5, 7):
            self.window.set_step_enabled(i, True)
        self.window.append_log("Next: 2 Calibrate (click it when ready)")
        self.loaded.emit()

    def apply_calibration_now(self) -> None:
        """The button: calibration (and flips) as typed, applied to the open movie."""
        if self.source is None:
            return
        self.apply_calibration()
        self._show_current()
        cal = self.calibration_form.values()
        self.calibrate_status.setText(f"Calibration: {self.movie_blob['pixel_um']:.4g} µm/px, {cal.fps:g} fps.")
        self.window.append_log(f"Calibration: {self.movie_blob['pixel_um']:.4g} um/px, {cal.fps:g} fps")

    def apply_calibration(self) -> None:
        """Calibration changes after loading: re-describe the movie (cheap), re-time the frames."""
        if self.source is None:
            return
        cal = self.calibration_form.values()
        blob = mv.describe_movie(self.source, cal.flip_vertical, self.movie_blob.get("dataset", cal.dataset),
                                 log=lambda m: None)
        blob["flip_horizontal"] = bool(cal.flip_horizontal)
        blob["fps"] = cal.fps
        file_pixel = blob.get("file_meta", {}).get("pixel_um")
        blob["pixel_um"] = float(file_pixel) if (cal.pixel_from_file and file_pixel) else cal.pixel_um
        if self.shifts is not None:
            if self._shifts_flip == (cal.flip_vertical, cal.flip_horizontal):
                blob["shifts"] = self.shifts.tolist()
            else:  # shifts were measured on the other orientation
                self.shifts = None
                self.stabilise_status.setText("Stabilisation removed: the flip changed. Stabilise again.")
        self.movie_blob = blob
        from vanua.core.datastore import WorldFrame

        pixel = blob["pixel_um"]
        self.frame = WorldFrame.from_spacing((1.0, pixel, pixel), unit="um").as_document()
        self.frames = frames_table(self.frames["mean_intensity"].to_numpy(), cal.fps,
                                   OPS_BY_NAME["meltpool.open_movie"].params().blank_fraction)
        self._crop_quietly()

    # -- crop -------------------------------------------------------------------------------

    def _crop_bounds(self) -> "tuple[int, int, int, int]":
        """(y0, y1, x0, x1) from the form: 0 for Right/Bottom = the frame edge,
        swapped if typed the wrong way round, clamped to the frame."""
        h, w = (int(v) for v in self.movie_blob["shape"])
        c = self.crop_form.values()
        x0, x1 = c.x0, (c.x1 or w)
        y0, y1 = c.y0, (c.y1 or h)
        x0, x1 = sorted((min(x0, w), min(x1, w)))
        y0, y1 = sorted((min(y0, h), min(y1, h)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(f"the ROI is {x1 - x0} x {y1 - y0} px; it needs at least 8 x 8 "
                             f"(frame {w} x {h} px)")
        return y0, y1, x0, x1

    def _crop_blob(self) -> "tuple[dict, dict]":
        y0, y1, x0, x1 = self._crop_bounds()
        blob = mv.crop_blob(self.movie_blob, y0, y1, x0, x1)
        return blob, crop_frame(self.frame, y0, x0)

    def _crop_quietly(self) -> None:
        try:
            self.roi_blob, self.roi_frame = self._crop_blob()
        except ValueError:
            self.roi_blob, self.roi_frame = self.movie_blob, self.frame
        self._update_background()
        self._refresh_volume()
        if self.movie_blob is not None:
            h, w = self.movie_blob["shape"]
            self.frame_size_label.setText(f"Frame: {w} x {h} px (x 0–{w}, y 0–{h}; y = 0 at the top).")

    def _show_full_frame(self) -> None:
        full = mv.Movie(self.movie_blob)
        self.viewer.slice_viewer.set_overlay_labels(None, {}, {})
        self._display(full.mean_of(self.slider.value(), 3), self.frame)

    def _draw_roi_box(self, y0: float, y1: float, x0: float, x1: float) -> None:
        """The ROI as a rectangle on the image layer (pixel coordinates; follows zoom and pan)."""
        rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if self._roi_item is None:
            self._roi_item = QGraphicsRectItem(rect, self.viewer.slice_viewer.pixmap_item)
            pen = QPen(QColor("#f59e0b"))
            pen.setWidth(2)
            pen.setCosmetic(True)  # 2 screen pixels at any zoom
            self._roi_item.setPen(pen)
        else:
            self._roi_item.setRect(rect)
        self._roi_item.show()

    def _hide_roi_box(self) -> None:
        if self._roi_item is not None:
            self._roi_item.hide()

    def show_crop_outline(self) -> None:
        """The full frame with the ROI drawn on it."""
        if self.movie_blob is None:
            return
        try:
            y0, y1, x0, x1 = self._crop_bounds()
        except ValueError as exc:
            self.crop_status.setText(str(exc))
            return
        self._show_full_frame()
        self._draw_roi_box(y0, y1, x0, x1)
        self.viewer.slice_viewer.fit_to_view()

    # drag selection: vanua's "paint" left tool reports press/drag/release in image pixels
    def _set_draw_mode(self, on: bool) -> None:
        view = self.viewer.slice_viewer
        if on and self.movie_blob is None:
            self.draw_button.setChecked(False)
            return
        if on and self.stab_draw_button.isChecked():
            self.stab_draw_button.setChecked(False)
        if on:
            self.show_crop_outline() if self._roi_valid() else self._show_full_frame()
            view.fit_to_view()
            view.set_paint_radius(None)
            view.set_paint_hint_labels("Draw ROI", None)
            view.set_left_tool("paint")
            view.on_painted(self._on_drag)
            self.crop_status.setText("Drag a rectangle on the image with the left mouse button.")
        else:
            view.on_painted(None)
            view.set_paint_hint_labels(None, None)
            view.set_left_tool("crosshair")

    def _roi_valid(self) -> bool:
        try:
            self._crop_bounds()
            return True
        except ValueError:
            return False

    def _on_drag(self, x: float, y: float, phase: str, button: str) -> None:
        if button != "left":
            return
        h, w = self.movie_blob["shape"]
        x, y = min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h))
        if phase == "start" or self._drag_start is None:
            self._drag_start = (x, y)
        sx, sy = self._drag_start
        self._draw_roi_box(sy, y, sx, x)
        if phase != "end":
            return
        self._drag_start = None
        x0, x1 = int(np.floor(min(sx, x))), int(np.ceil(max(sx, x)))
        y0, y1 = int(np.floor(min(sy, y))), int(np.ceil(max(sy, y)))
        self.set_roi(x0, y0, x1, y1)
        self.draw_button.setChecked(False)

    def set_roi(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Put a ROI (pixel columns x0:x1, rows y0:y1 of the shown full frame) in
        the form and on screen; Apply crop commits it."""
        h, w = self.movie_blob["shape"]
        self.crop_form.set_values(CropParams(x0=x0, x1=x1, y0=y0, y1=y1))
        try:
            y0, y1, x0, x1 = self._crop_bounds()
        except ValueError as exc:
            self.crop_status.setText(str(exc))
            return
        self._draw_roi_box(y0, y1, x0, x1)
        self.crop_status.setText(f"ROI: x {x0}–{x1}, y {y0}–{y1} ({x1 - x0} x {y1 - y0} px of {w} x {h}). "
                                 "Click <b>Apply crop</b> to use it.")

    def apply_crop(self) -> None:
        if self.movie_blob is None:
            return
        if self.draw_button.isChecked():
            self.draw_button.setChecked(False)
        try:
            self.roi_blob, self.roi_frame = self._crop_blob()
        except ValueError as exc:
            self.crop_status.setText(str(exc))
            return
        self._update_background()
        self._refresh_volume()
        y0, y1, x0, x1 = self.roi_blob["crop"]
        self.crop_status.setText(f"Analysing x {x0}–{x1}, y {y0}–{y1} ({x1 - x0} x {y1 - y0} px).")
        self.window.append_log(f"Crop: x {x0}:{x1}, y {y0}:{y1}")
        self._show_current()
        self.viewer.slice_viewer.fit_to_view()
        self.window.append_log("Next: 4 Transient (optional) or 5 Liquid-gas boundary (click it when ready)")
        self.cropped.emit()

    def reset_crop(self) -> None:
        self.crop_form.set_values(CropParams())
        self.apply_crop()

    # -- stabilisation --------------------------------------------------------------------

    def stabilise(self) -> None:
        if self.source is None:
            return
        self.shifts = None
        self.apply_calibration()  # the movie as read, without old shifts
        params = self.stabilise_form.values()
        params.enabled = True
        cal = self.calibration_form.values()
        flip = (cal.flip_vertical, cal.flip_horizontal)
        ctx = {"movie": self.movie_blob, "frames": self.frames}
        self.stabilise_status.setText("Registering every frame…")

        def _job(p, c, progress, cancel, log):
            return _run_op("meltpool.stabilise", p, c, progress, cancel, log)

        self._start(_job, params, ctx, lambda result: self._on_stabilised(result, params, flip), "Stabilise")

    def _on_stabilised(self, result: dict, params, flip: tuple) -> None:
        self.shifts = np.asarray(result["movie"]["shifts"], dtype=float)
        self._shifts_flip = flip
        self.movie_blob = result["movie"]
        self._crop_quietly()
        lit = ~self.frames["blank"].to_numpy(bool)
        dy, dx = self.shifts[lit, 0], self.shifts[lit, 1]
        self.stabilise_status.setText(
            f"Stabilised (<b>{params.mode}</b>, 1/{params.upsample} px): shake dy sd {dy.std():.3f} px "
            f"(range {dy.min():+.2f}..{dy.max():+.2f}), dx sd {dx.std():.3f} px (range {dx.min():+.2f}..{dx.max():+.2f}).")
        self.window.append_log(f"Stabilised: dy sd {dy.std():.3f} px, dx sd {dx.std():.3f} px")
        self.window.append_log("Next: 3 Crop (click it when ready)")
        self._show_current()
        self.stabilised.emit()

    def toggle_flip(self, which: str) -> None:
        """Flip the movie now (vertical: build direction up; horizontal: mirror)."""
        cal = self.calibration_form.values()
        if which == "vertical":
            cal.flip_vertical = not cal.flip_vertical
        else:
            cal.flip_horizontal = not cal.flip_horizontal
        self.calibration_form.set_values(cal)
        if self.source is None:
            return
        self.apply_calibration()
        self._show_current()
        self.viewer.slice_viewer.fit_to_view()
        state = ", ".join(n for n, on in (("flipped vertically", cal.flip_vertical),
                                          ("flipped horizontally", cal.flip_horizontal)) if on) or "as recorded"
        self.calibrate_status.setText(f"Movie shown {state}.")
        self.window.append_log(f"Orientation: {state}")

    def _full_frame_view(self) -> None:
        """The whole frame (no ROI), as read -- the stabilisation region is in these pixels."""
        blob = dict(self.movie_blob)
        blob.pop("shifts", None)
        self._display(mv.Movie(blob).mean_of(self.slider.value(), 3), self.frame)
        self.viewer.slice_viewer.fit_to_view()

    def _set_stab_draw_mode(self, on: bool) -> None:
        if on and self.movie_blob is None:
            self.stab_draw_button.setChecked(False)
            return
        if on:
            if self.draw_button.isChecked():
                self.draw_button.setChecked(False)
            self._full_frame_view()
            self.stab_drawer.start(self.movie_blob["shape"])
            self.stabilise_status.setText("Drag a box on solid material away from the moving boundary.")
        else:
            self.stab_drawer.stop()

    def _on_stab_region(self, x0: int, y0: int, x1: int, y1: int) -> None:
        params = self.stabilise_form.values()
        params.x0, params.y0, params.x1, params.y1 = x0, y0, x1, y1
        self.stabilise_form.set_values(params)
        self.stab_draw_button.setChecked(False)
        self.stab_drawer.show(x0, y0, x1, y1)
        self.stabilise_status.setText(f"Stabilisation region: x {x0}–{x1}, y {y0}–{y1} ({x1 - x0} x {y1 - y0} px). "
                                      "Click <b>Stabilise now</b>.")

    def show_stabilisation_region(self) -> None:
        if self.movie_blob is None:
            return
        blob = dict(self.movie_blob)
        blob.pop("shifts", None)
        params = self.stabilise_form.values()
        try:
            rows, cols = stabilise_region(blob, self.frames, params)
        except ValueError as exc:
            self.stabilise_status.setText(str(exc))
            return
        self._full_frame_view()
        self.stab_drawer.show(cols.start, rows.start, cols.stop, rows.stop)
        how = "drawn" if (params.x0 or params.x1 or params.y0 or params.y1) else "automatic (substrate below the pool)"
        self.stabilise_status.setText(f"Stabilisation region ({how}): x {cols.start}–{cols.stop}, "
                                      f"y {rows.start}–{rows.stop}. It must not contain the moving boundary.")

    def reset_stabilisation_region(self) -> None:
        params = self.stabilise_form.values()
        params.x0 = params.y0 = params.x1 = params.y1 = 0
        self.stabilise_form.set_values(params)
        self.show_stabilisation_region()

    def remove_stabilisation(self) -> None:
        self.shifts = None
        if self.source is not None:
            self.apply_calibration()
            self._show_current()
        self.stabilise_status.setText("Not stabilised.")

    # -- liquid-solid boundary on the stabilised median -------------------------------------

    def compute_median(self) -> None:
        if self.roi_blob is None:
            return
        if self.draw_button.isChecked():
            self.draw_button.setChecked(False)
        params = self.median_form.values()
        ctx = {"movie": self.roi_blob, "frames": self.frames}
        self.solid_status.setText("Computing the median…")

        def _job(p, c, progress, cancel, log):
            return _run_op("meltpool.median_image", p, c, progress, cancel, log)

        self._start(_job, params, ctx, self._on_median, "Median")

    def _on_median(self, result: dict) -> None:
        self.median = np.asarray(result["image"])[0]
        self._median_blob, self._median_frame = self.roi_blob, self.roi_frame
        note = "stabilised" if self.shifts is not None else "<b style='color:#ef4444'>not stabilised</b>"
        self.solid_status.setText(f"Median ready ({note}). Trace the liquid-solid line (click points) or try the "
                                  "automatic search.")
        self._show_median()
        self.viewer.slice_viewer.fit_to_view()
        self.median_ready.emit()

    def _surface_estimate(self) -> float:
        """Surface row of the median, in its own pixels (for the band-pass view)."""
        surface = mv.original_surface(self.median)
        return surface if np.isfinite(surface) else 0.0

    def _show_median(self) -> None:
        if self.median is None:
            return
        plane = self.median
        if self.detail_check.isChecked():
            plane = mv.band_pass(self.median, self._surface_estimate()).astype(np.float32)
        self._display(plane, self._median_frame, keep_line=True,
                      kind="detail" if self.detail_check.isChecked() else "median")
        self._draw_line()

    def _set_trace_mode(self, on: bool) -> None:
        view = self.viewer.slice_viewer
        if on and self.median is None:
            self.trace_button.setChecked(False)
            self.solid_status.setText("Compute the median first.")
            return
        if on:
            view.set_paint_radius(None)
            view.set_paint_hint_labels("Add point", None)
            view.set_left_tool("paint")
            view.on_painted(self._on_trace_click)
            self.solid_status.setText("Click along the liquid-solid line, one end at the surface to the other.")
        else:
            view.on_painted(None)
            view.set_paint_hint_labels(None, None)
            view.set_left_tool("crosshair")

    def _on_trace_click(self, x: float, y: float, phase: str, button: str) -> None:
        if phase != "start" or button != "left":
            return
        oy, ox = mv.crop_offset(self._median_blob)
        self.solid_points.append((float(x) + ox, float(y) + oy))
        self._draw_line()
        self.solid_status.setText(f"{len(self.solid_points)} points. Click <b>Measure traced line</b> when done.")

    def undo_point(self) -> None:
        if self.solid_points:
            self.solid_points.pop()
            self._draw_line()

    def clear_points(self) -> None:
        self.solid_points = []
        self._draw_line()

    def _draw_line(self, xs=None, ys=None) -> None:
        """The traced points (or a found line) on the median, in its own pixel coordinates."""
        if self.median is None:
            return
        oy, ox = mv.crop_offset(self._median_blob)
        if xs is None:
            pts = sorted(self.solid_points)
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        path = QPainterPath()
        for k, (x, y) in enumerate(zip(xs, ys)):
            (path.lineTo if k else path.moveTo)(x - ox, y - oy)
        if len(xs) == 1:
            path.addEllipse(xs[0] - ox - 1.5, ys[0] - oy - 1.5, 3.0, 3.0)
        if self._line_item is None:
            self._line_item = QGraphicsPathItem(self.viewer.slice_viewer.pixmap_item)
            pen = QPen(QColor("#22d3ee"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            self._line_item.setPen(pen)
        self._line_item.setPath(path)
        self._line_item.show()

    def measure_solid(self, manual: bool) -> None:
        if self.median is None:
            self.solid_status.setText("Compute the median first.")
            return
        if self.trace_button.isChecked():
            self.trace_button.setChecked(False)
        params = self.solid_form.values()
        if manual:
            if len(self.solid_points) < 2:
                self.solid_status.setText("Trace at least two points first.")
                return
            params.points = "; ".join(f"{x:.2f},{y:.2f}" for x, y in sorted(self.solid_points))
        else:
            params.points = ""
        ctx = {"image": self.median[np.newaxis], "frame": self._median_frame, "movie": self._median_blob}

        def _job(p, c, progress, cancel, log):
            return _run_op("meltpool.solid_boundary", p, c, progress, cancel, log)

        self._start(_job, params, ctx, lambda result: self._on_solid(result, params), "Liquid-solid")

    def _on_solid(self, result: dict, params) -> None:
        self.solid_result, self._solid_params = result, params
        row = result["table"].iloc[0]
        self.solid_table.set_frame(result["table"])
        if row["detected"]:
            if row["method"] == "auto":
                self._draw_line(result["line"]["x"], result["line"]["y"])
            depth = [c for c in result["table"].columns if c.startswith("depth_")][0]
            width = [c for c in result["table"].columns if c.startswith("width_")][0]
            how = "traced by hand" if row["method"] == "manual" else f"found automatically (z = {row['z']:.1f})"
            self.solid_status.setText(f"Liquid-solid line {how}: depth <b>{row[depth]:.0f}</b>, width "
                                      f"<b>{row[width]:.0f}</b> ({depth.rsplit('_', 1)[1]}).")
        else:
            self.solid_status.setText(f"<b>Not detectable</b> automatically (z = {row['z']:.1f}, threshold "
                                      f"{params.z_min}): the line does not stand out from the substrate. "
                                      "Trace it by hand.")
        self.solid_measured.emit()

    def export_solid(self, out_dir: "str | Path") -> dict:
        if self.solid_result is None or self.source is None:
            return {}
        from vanua.blocks.plot import save_figure
        from vanua.blocks.provenance import export_table

        out_dir = Path(out_dir)
        stem = self.source.stem
        recipe = configured_solid(self.source, self.settings(), self.median_form.values(), self._solid_params)
        table = self.solid_result["table"].copy()
        table.insert(0, "run", stem)
        written = {"solid": export_table(table, out_dir / f"{stem}_solid.csv", recipe, "solid", self.reg,
                                         extra={"line": self.solid_result["line"]})}
        written["figure"] = save_figure(self._solid_figure(), out_dir / f"{stem}_solid.png")
        self.solid_status.setText("Wrote:\n" + "\n".join(q.name for q in written.values()))
        self.window.append_log(f"Exported liquid-solid line to {out_dir}")
        return written

    def _solid_figure(self):
        from matplotlib.figure import Figure

        oy, ox = mv.crop_offset(self._median_blob)
        fig = Figure(figsize=(7, 4), dpi=110, layout="constrained")
        ax = fig.add_subplot()
        lo, hi = self.contrast.window() if self.contrast._plane is not None else np.percentile(self.median, [1, 99])
        h, w = self.median.shape
        ax.imshow(self.median, cmap="gray", vmin=lo, vmax=hi, extent=(ox, ox + w, oy + h, oy))
        line = self.solid_result["line"]
        if line["detected"]:
            ax.plot(line["x"], line["y"], color="#22d3ee", lw=1.2, label="liquid-solid line")
            ax.legend(loc="lower right", fontsize=7)
        ax.set_xlabel("x (px, full frame)")
        ax.set_ylabel("y (px)")
        ax.set_title("Stabilised median" if self.shifts is not None else "Median (not stabilised)", fontsize=9)
        return fig

    def _display(self, plane: np.ndarray, frame: dict, keep_line: bool = False, kind: str = "frame") -> None:
        """Every frame shown goes through here: the view, then the contrast panel
        (its histogram, and the kept window or an auto stretch)."""
        self._hide_roi_box()
        if getattr(self, "stab_drawer", None) is not None and not self.stab_drawer.active:
            self.stab_drawer.hide()
        if not keep_line and getattr(self, "_line_item", None) is not None:
            self._line_item.hide()
        if getattr(self, "_line_items", None):
            self._draw_boundary(None)
        show_image(self.viewer.slice_viewer, plane[np.newaxis], frame)
        self.contrast.show_plane(plane, kind)

    def _on_slider(self, value: int) -> None:
        t = self.frames["time_ms"].iloc[value] if self.frames is not None else 0.0
        self.frame_label.setText(f"Frame {value}  ({t:.2f} ms after beam-on)")

    def _show_current(self) -> None:
        if self.roi_blob is None:
            return
        display = getattr(self, "display_blob", None) or self.roi_blob
        movie = mv.Movie(display)
        i = self.slider.value()
        self._on_slider(i)
        self.viewer.slice_viewer.set_overlay_labels(None, {}, {})
        self._display(movie.mean_of(i, 3), self.roi_frame,
                      kind="transient" if display.get("background") else "frame")
        prof = getattr(self, "profiles", None)
        if prof is not None and len(prof["frames"]):
            k = int(np.argmin(np.abs(np.asarray(prof["frames"]) - i)))
            if abs(int(prof["frames"][k]) - i) <= max(1, self.trace_form.values().stride):
                self._draw_boundary(profile_trace(prof, k))
    def preview(self) -> None:
        if self.roi_blob is None:
            return
        errors = self.trace_form.errors()
        if errors:
            self.boundary_status.setText("\n".join(errors))
            return
        params = self.trace_form.values()
        analysis = getattr(self, "analysis_blob", None) or self.roi_blob
        movie, frame, i, frames = mv.Movie(analysis), self.roi_frame, self.slider.value(), self.frames

        def _job(p, c, progress, cancel, log):
            points = trace_time_points(movie.n, frames, params)
            surface, cx = trace_prepare(movie, points if len(points) else np.array([i]), params)
            return trace_one(movie, i, surface, cx, params, frame)

        self._start(_job, None, None, self._on_preview, "Preview")

    def _on_preview(self, out: dict) -> None:
        analysis = getattr(self, "analysis_blob", None) or self.roi_blob
        # a transient frame (values around 0) must not get a movie frame's display window: it would be black
        self._display(out["image"], self.roi_frame, kind="transient" if analysis.get("background") else "frame")
        self._draw_boundary(out["trace"])
        v = out["values"]
        self.boundary_status.setText(
            "Depth (inner / centre / outer): " + " / ".join(f"<b>{v['depth_' + L]:.0f}</b>" for L in boundary.LINES)
            + " µm; width " + " / ".join(f"{v['width_' + L]:.0f}" for L in boundary.LINES) + " µm.")
        self.previewed.emit()

    def _draw_boundary(self, trace: "dict | None") -> None:
        """The three traced lines on the image layer (inner cyan, centre orange, outer magenta)."""
        for line in boundary.LINES:
            item = self._line_items.get(line)
            if trace is None:
                if item is not None:
                    item.hide()
                continue
            x, y = boundary.line_xy(trace, line)
            path = QPainterPath()
            for k, (xi, yi) in enumerate(zip(x, y)):
                (path.lineTo if k else path.moveTo)(float(xi), float(yi))
            if item is None:
                item = QGraphicsPathItem(self.viewer.slice_viewer.pixmap_item)
                pen = QPen(QColor(*boundary.COLOURS[line]))
                pen.setWidth(2)
                pen.setCosmetic(True)
                item.setPen(pen)
                self._line_items[line] = item
            item.setPath(path)
            item.show()

    def measure(self) -> None:
        if self.roi_blob is None:
            return
        errors = self.trace_form.errors() + self.flag_form.errors()
        if errors:
            self.measure_status.setText("\n".join(errors))
            return
        trace, flags, dyn = self.trace_form.values(), self.flag_form.values(), self.dynamics_form.values()
        ctx = {"movie": getattr(self, "analysis_blob", None) or self.roi_blob, "frame": self.roi_frame,
               "frames": self.frames}

        def _job(p, c, progress, cancel, log):
            series = _run_op("meltpool.trace_boundary", trace, c, progress, cancel, log)
            table = flag_table(series["table"], flags)
            try:
                spectrum, peaks = dynamics(table, dyn)
            except ValueError as exc:  # too few steady time points: sizes still reported
                log(f"no spectrum: {exc}")
                spectrum = peaks = None
            return {"table": table, "summary": summary_table(table, peaks), "profiles": series["profiles"],
                    "spectrum": spectrum, "peaks": peaks}

        self.measure_status.setText("Tracing…")
        self._start(_job, None, ctx, self._on_measured, "Trace")

    def _on_measured(self, result: dict) -> None:
        self.table, self.summary, self.profiles = result["table"], result["summary"], result["profiles"]
        self.spectrum, self.peaks = result["spectrum"], result["peaks"]
        s = self.summary.iloc[0]
        self.measure_status.setText(f"{int(s.n_time_points)} time points traced.")
        if s.n_steady:
            lines = "<br>".join(
                f"{L}: depth <b>{s[f'depth_{L}_median_um']:.0f} µm</b> (IQR {s[f'depth_{L}_q25_um']:.0f}–"
                f"{s[f'depth_{L}_q75_um']:.0f}), width {s[f'width_{L}_median_um']:.0f} µm, area "
                f"{s[f'area_{L}_median_um2']:.0f} µm²" for L in boundary.LINES)
            text = (f"Steady from <b>{s.steady_from_ms:.2f} ms</b> ({int(s.n_transient)} start-up and "
                    f"{int(s.n_blank)} blank time points flagged, left out). Steady-state medians:<br>{lines}")
            hz = s.get("depth_centre_dominant_hz", float("nan"))
            if np.isfinite(hz):
                text += f"<br>Dominant depth oscillation (band centre): <b>{hz:.0f} Hz</b>"
        else:
            text = "<b style='color:#ef4444'>Never steady</b> within the tolerance: every time point is flagged."
        self.summary_label.setText(text)
        self.plot.show_figure(series_figure(self.table))
        if self.spectrum is not None:
            self.spectrum_plot.show_figure(spectrum_figure(self.spectrum, self.peaks))
        self.table_view.set_frame(self.table)
        self.window.set_step_enabled(6, True)
        self.window.append_log("Next: 7 Results (click it when ready)")
        self.measured.emit()

    def settings(self) -> Settings:
        """Everything as run, so the exported recipe reproduces these numbers headless."""
        stabilise = self.stabilise_form.values()
        stabilise.enabled = self.shifts is not None
        return Settings(calibration=self.calibration_form.values(), stabilise=stabilise,
                        crop=self.crop_form.values(), background=self._background_applied,
                        trace=self.trace_form.values(), flags=self.flag_form.values(),
                        dynamics=self.dynamics_form.values(),
                        overlay=getattr(self, "_last_overlay", None) or self.overlay_form.values())

    def _overlay_params(self, out_dir: Path, stem: str) -> OverlayParams:
        """The form's settings; brightness as on screen for movie frames unless set in the form."""
        p = self.overlay_form.values()
        suffix = Path(p.out_path).suffix or ".mp4"
        p.out_path = str(Path(out_dir) / f"{stem}_overlay{suffix}")
        if not p.high > p.low:
            kind = "transient" if (getattr(self, "display_blob", None) or {}).get("background") else "frame"
            kept = self.contrast._windows.get(kind)
            if self.contrast._kind == kind and self.contrast._plane is not None:
                (p.low, p.high), p.gamma = self.contrast.window(), self.contrast.gamma()
            elif kept:
                (p.low, p.high), p.gamma = kept[0], kept[1]
        return p

    def _write_overlay(self, out_dir, stem: str) -> "str | None":
        if self.profiles is None:
            return None
        params = self._overlay_params(Path(out_dir), stem)
        display = getattr(self, "display_blob", None) or self.roi_blob
        result = _run_op("meltpool.overlay_movie", params, {"movie": display, "profiles": self.profiles},
                         lambda *a: None, None, self.window.append_log)
        self._last_overlay = params
        return result["movie_file"]

    def export_overlay(self, out_dir) -> "Path | None":
        """Only the overlay video (the Export button writes it too)."""
        if self.profiles is None:
            self.export_status.setText("Trace all frames first.")
            return None
        stem = self.source.stem
        self.export_status.setText("Writing the overlay video…")
        path = Path(self._write_overlay(out_dir, stem))
        self.export_status.setText(f"Wrote {path.name}")
        return path

    def export(self, out_dir: "str | Path") -> dict:
        if self.table is None:
            return {}
        out_dir = Path(out_dir)
        stem = self.source.stem
        self.export_status.setText("Writing (the overlay video and the movie's SHA-256 take a moment)…")
        overlay = self._write_overlay(out_dir, stem)  # first: its settings go into the recorded recipe
        recipe = configured(self.recipe, self.source, self.settings(), out_dir / f"{stem}_depth.png")
        values = {"table": self.table, "summary": self.summary, "profiles": self.profiles, "overlay": overlay,
                  "spectrum": self.spectrum, "peaks": self.peaks}
        written = export_movie(recipe, values, out_dir, stem, self.reg)
        self.export_status.setText("Wrote:\n" + "\n".join(p.name for p in written.values()))
        self.window.append_log(f"Exported to {out_dir}")
        self.exported.emit()
        return written


def build_app(synchronous: bool = False) -> MeltPoolApp:
    return MeltPoolApp(synchronous=synchronous)


def show(app_obj: MeltPoolApp) -> None:
    app_obj.window.resize(1500, 900)
    app_obj.window.show()
    app_obj.viewer.initialize()
    theme.apply_dark_titlebar(app_obj.window)


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    theme.enable_high_dpi()
    qapp = QApplication(argv)
    qapp.setOrganizationName("meltpool")
    qapp.setApplicationName(APP_NAME)
    theme.install(qapp, app_id=APP_ID, app_name=APP_NAME)
    app_obj = build_app()
    show(app_obj)
    app_obj.window.append_log(f"{APP_NAME} started.")
    if len(argv) > 1:
        app_obj.path_edit.setText(argv[1])
        app_obj.load(argv[1])
    return qapp.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
