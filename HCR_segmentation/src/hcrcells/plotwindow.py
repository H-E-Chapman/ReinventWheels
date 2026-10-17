"""The plot window and the results table of the HCR Cells GUI.

* :class:`PlotWindow` -- the bleaching plots in their own window: plot kind and
  sample, the matplotlib toolbar (zoom, pan), export in a chosen format
  (PNG, TIFF, PDF, SVG, EPS, JPEG; DPI for raster), and a collapsible
  "Advanced" section with every :class:`~hcrcells.plots.PlotOptions` choice.
* :class:`ResultsTable` -- the per-cell table, sortable by any column (click
  a header), exported as CSV or Excel in the sorted order, with a provenance
  sidecar pointing at the series results it came from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout,
                             QWidget)

from vanua.ui.blocks import DataFrameModel, TableView

from .pipeline import CHANNELS, GENE
from .plots import (BANDS, COUNTS, EXPORT_FORMATS, FITS, INTENSITIES, KINDS, LABEL_ROLES, RASTER, Y_MODES,
                    PlotOptions, export_figure, fits_table, make_figure)

NORMALIZATION = {"normalized": "Normalized to first T", "absolute": "Absolute"}

#: Right-hand column: every control at least this tall (logical px), rows this far apart.
ROW_HEIGHT = 30
ROW_SPACING = 10

__all__ = ["PlotWindow", "ResultsTable", "ResultsWindow", "SortableFrameModel"]


def _combo(items: dict, current) -> QComboBox:
    box = QComboBox()
    for key, text in items.items():
        box.addItem(text, key)
    box.setCurrentIndex(max(0, box.findData(current)))
    return box


class _Collapsible(QWidget):
    """A '▸ title' toggle that shows / hides `content` (collapsed by default)."""

    def __init__(self, title: str, content: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.content = content
        content.setVisible(False)
        self.toggle.toggled.connect(self._set_open)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle)
        layout.addWidget(content)

    def _set_open(self, open_: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        self.content.setVisible(open_)

    def set_open(self, open_: bool) -> None:
        self.toggle.setChecked(open_)


class PlotWindow(QWidget):
    """Top-level window: the figure (with the matplotlib toolbar) on the left,
    every control in a scrollable column on the right -- plot and sample,
    normalized / absolute values, the fit, export, and the collapsible
    Advanced and Labels sections. Double-click an axis label or title on the
    figure to edit it. Feed it with :meth:`set_results`; `optionsChanged`
    fires after any change, so the app can save the choices with the session."""

    optionsChanged = pyqtSignal()
    PANEL_WIDTH = 360

    def __init__(self, parent=None, out_dir=None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("HCR Cells — Plots")
        self.resize(1400, 860)
        self.results: dict = {}
        self.paths: dict = {}
        self.out_dir = out_dir or (lambda: Path("out"))
        self.figure = None
        self._labels: dict = {}          # role -> {"text", "auto"}: the user's label texts
        self._label_rows: list = []
        self._building = False
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self.redraw)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        import matplotlib.figure

        self._canvas_cls, self._toolbar_cls = FigureCanvasQTAgg, NavigationToolbar2QT
        self.canvas = FigureCanvasQTAgg(matplotlib.figure.Figure())
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # -- the figure, left ---------------------------------------------------------------
        self.note = QLabel("")
        self.note.setWordWrap(True)
        left = QWidget()
        self._left = QVBoxLayout(left)
        self._left.setContentsMargins(0, 0, 0, 0)
        self._left.addWidget(self.toolbar)
        self._left.addWidget(self.canvas, 1)
        self._left.addWidget(self.note)

        # -- every control, right ----------------------------------------------------------
        panel = QWidget()
        col = QVBoxLayout(panel)
        # Never squeeze the rows to fit the window: the column scrolls instead.
        col.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
        col.setSpacing(ROW_SPACING)
        col.setAlignment(Qt.AlignTop)

        self.kind_combo = _combo(KINDS, "cells")
        self.sample_combo = QComboBox()
        plot_form = QFormLayout()
        plot_form.setVerticalSpacing(ROW_SPACING)
        plot_form.addRow("Plot", self.kind_combo)
        plot_form.addRow("Sample", self.sample_combo)
        col.addWidget(_group("Plot", plot_form))

        # Normalized (to each cell's first timepoint) or absolute, for intensity and spots separately.
        self.int_norm = _combo(NORMALIZATION, "normalized")
        self.count_norm = _combo(NORMALIZATION, "normalized")
        self.int_norm.setToolTip("Intensity: each cell's value divided by its value at the first plotted "
                                 "timepoint, or the absolute value (a.u.)")
        self.count_norm.setToolTip("Spots (the measure chosen under Advanced): normalized to the first plotted "
                                   "timepoint, or absolute per cell")
        values_form = QFormLayout()
        values_form.setVerticalSpacing(ROW_SPACING)
        values_form.addRow("Intensity", self.int_norm)
        values_form.addRow("Spots", self.count_norm)
        col.addWidget(_group("Values", values_form))

        self.fit_combo = _combo(FITS, "off")
        self.fit_combo.setToolTip("Fit each mean curve (Mean of cells, Batch effect). Auto: linear or "
                                  "exponential decay to a plateau, whichever has the lower AICc -- the "
                                  "exponential's extra parameter is penalised, so it wins only for real curvature.")
        self.fit_summary = QLabel("")
        self.fit_summary.setWordWrap(True)
        self.fit_summary.setTextFormat(Qt.RichText)
        self.fit_export = QPushButton("Export fits (CSV)…")
        self.fit_export.clicked.connect(lambda: self.export_fits())
        fit_box = QVBoxLayout()
        fit_box.addWidget(self.fit_combo)
        fit_box.addWidget(self.fit_summary)
        fit_box.addWidget(self.fit_export)
        col.addWidget(_group("Fit to mean curve", fit_box))

        self.format_combo = QComboBox()
        self.format_combo.addItems(list(EXPORT_FORMATS))
        self.format_combo.setToolTip("PNG / TIFF / JPEG: images at the chosen DPI. PDF / SVG / EPS: vector, for "
                                     "editing in Illustrator or Inkscape (EPS cannot store transparency: the "
                                     "error bands become opaque).")
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" dpi")
        self.export_button = QPushButton("Export plot…")
        self.export_button.clicked.connect(lambda: self.export())
        self.format_combo.currentTextChanged.connect(
            lambda f: self.dpi_spin.setEnabled(EXPORT_FORMATS[f] in RASTER))
        export_form = QFormLayout()
        export_form.setVerticalSpacing(ROW_SPACING)
        export_form.addRow("Format", self.format_combo)
        export_form.addRow("Resolution", self.dpi_spin)
        export_form.addRow("", self.export_button)
        col.addWidget(_group("Export", export_form))

        self.advanced = _Collapsible("Advanced plot settings", self._build_advanced())
        col.addWidget(self.advanced)
        self.labels_page = QWidget()
        self._labels_form = QFormLayout(self.labels_page)
        self._labels_form.setContentsMargins(12, 0, 0, 0)
        self._labels_form.setVerticalSpacing(ROW_SPACING)
        self.labels_section = _Collapsible("Labels", self.labels_page)
        col.addWidget(self.labels_section)
        hint = QLabel("Tip: double-click an axis label or title on the plot to edit it.")
        hint.setWordWrap(True)
        col.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(self.PANEL_WIDTH)
        layout = QHBoxLayout(self)
        layout.addWidget(left, 1)
        layout.addWidget(scroll)

        for box in (self.kind_combo, self.sample_combo, self.int_norm, self.count_norm, self.fit_combo):
            box.currentIndexChanged.connect(self._changed)

    # -- advanced -----------------------------------------------------------------------

    def _build_advanced(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(12, 0, 0, 0)
        form.setVerticalSpacing(ROW_SPACING)
        # One form row per channel: a nested layout in a single row was squeezed
        # to one row's height, the checkboxes drawn on top of each other.
        self.channel_checks = {}
        for i, ch in enumerate(CHANNELS):
            box = QCheckBox(GENE[ch])
            box.setChecked(True)
            box.toggled.connect(self._changed)
            self.channel_checks[ch] = box
            form.addRow("Channels" if i == 0 else "", box)
        self.t_from = QSpinBox()
        self.t_to = QSpinBox()
        for spin, prefix in ((self.t_from, "from T = "), (self.t_to, "to T = ")):
            spin.setRange(0, 9999)
            spin.setPrefix(prefix)
            spin.valueChanged.connect(self._changed)
        self.t_all = QCheckBox("all timepoints")
        self.t_all.setChecked(True)
        self.t_all.toggled.connect(lambda on: (self.t_from.setEnabled(not on), self.t_to.setEnabled(not on),
                                               self._changed()))
        self.t_from.setEnabled(False)
        self.t_to.setEnabled(False)
        form.addRow("Timepoints", self.t_all)
        form.addRow("", self.t_from)
        form.addRow("", self.t_to)
        self.count_combo = _combo(COUNTS, "transcripts_est")
        form.addRow("Spots measure", self.count_combo)
        self.int_combo = _combo(INTENSITIES, "mean_bgsub_au")
        form.addRow("Intensity measure", self.int_combo)
        self.band_combo = _combo(BANDS, "sd")
        form.addRow("Error band", self.band_combo)
        self.show_cells = QCheckBox("individual cells behind means")
        form.addRow("", self.show_cells)
        self.y_mode = _combo(Y_MODES, "from_zero")
        self.y_min = QDoubleSpinBox()
        self.y_max = QDoubleSpinBox()
        for spin, prefix, value in ((self.y_min, "min ", 0.0), (self.y_max, "max ", 1.5)):
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(3)
            spin.setPrefix(prefix)
            spin.setValue(value)
            spin.valueChanged.connect(self._changed)
        form.addRow("Y axis", self.y_mode)
        form.addRow("", self.y_min)
        form.addRow("", self.y_max)
        self.share_y = QCheckBox("same y range across channels")
        self.share_y.setChecked(True)
        self.log_y = QCheckBox("log scale")
        self.markers = QCheckBox("point markers")
        self.markers.setChecked(True)
        for w in (self.share_y, self.log_y, self.markers):
            form.addRow("", w)
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(lambda: self.set_options(PlotOptions(kind=self.options().kind, labels=self._labels)))
        form.addRow("", reset)
        for w in (self.count_combo, self.int_combo, self.band_combo, self.y_mode):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.show_cells, self.share_y, self.log_y, self.markers):
            w.toggled.connect(self._changed)
        self.y_mode.currentIndexChanged.connect(self._sync_enabled)
        self._sync_enabled()
        return page

    # -- control heights ----------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 -- Qt override
        super().showEvent(event)
        self.fit_control_heights()

    def changeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        from PyQt5.QtCore import QEvent

        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self.fit_control_heights()

    def fit_control_heights(self, root=None) -> None:
        """Every input control at least as tall as its text needs: the line
        height of the font it is drawn with, plus frame and padding. On a real
        200 % display the theme's font was drawn taller than the controls had
        been sized for, so their text was clipped (the squeezed Advanced
        settings); sizing from the drawn font's own metrics avoids that."""
        from PyQt5.QtWidgets import QAbstractSpinBox

        root = root or self
        for w in root.findChildren((QComboBox, QAbstractSpinBox, QLineEdit, QPushButton, QCheckBox)):
            # A fixed generous floor (the controls still looked crushed on a real
            # 200 % display), raised further if the drawn font needs more.
            need = max(ROW_HEIGHT, w.fontMetrics().height() + (12 if isinstance(w, QPushButton) else 10))
            if w.minimumHeight() < need:
                w.setMinimumHeight(need)

    def _sync_enabled(self) -> None:
        manual = self.y_mode.currentData() == "manual"
        self.y_min.setEnabled(manual)
        self.y_max.setEnabled(manual)

    # -- labels -------------------------------------------------------------------------

    def set_label(self, role: str, text: "str | None") -> None:
        """The user's text for a label role; empty or None: back to the automatic text."""
        auto = (getattr(self.figure, "_hcr_auto", {}) or {}).get(role, "")
        text = (text or "").strip()
        if not text or text == auto:
            self._labels.pop(role, None)
        else:
            self._labels[role] = {"text": text, "auto": auto}
        self._changed()

    def _rebuild_label_rows(self) -> None:
        """One row per label role in the current figure: the text (automatic one
        as the placeholder), a reset, and a warning when the automatic text has
        changed since the label was set (e.g. switched to absolute values)."""
        while self._labels_form.rowCount():
            self._labels_form.removeRow(0)
        autos = getattr(self.figure, "_hcr_auto", {}) or {}
        for role, auto in autos.items():
            edit = QLineEdit(self._labels.get(role, {}).get("text", ""))
            edit.setPlaceholderText(auto or "(none)")
            edit.setToolTip(f"Automatic: {auto or '(none)'}. Leave empty for the automatic text.")
            edit.editingFinished.connect(lambda r=role, e=edit: self._label_edited(r, e))
            reset = QToolButton()
            reset.setText("↺")
            reset.setToolTip("Back to the automatic text")
            reset.clicked.connect(lambda _c=False, r=role: self.set_label(r, None))
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(edit, 1)
            h.addWidget(reset)
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(row)
            stored = self._labels.get(role)
            if stored and stored.get("auto") and stored["auto"] != auto:
                warn = QLabel(f"⚠ automatic text is now: <i>{auto}</i>")
                warn.setWordWrap(True)
                warn.setStyleSheet("color: #f59e0b;")
                v.addWidget(warn)
            self._labels_form.addRow(LABEL_ROLES.get(role, role), box)
        reset_all = QPushButton("Reset all labels")
        reset_all.clicked.connect(self._reset_labels)
        self._labels_form.addRow("", reset_all)
        self.fit_control_heights(self.labels_page)

    def _label_edited(self, role: str, edit: QLineEdit) -> None:
        current = self._labels.get(role, {}).get("text", "")
        if edit.text().strip() != current:
            self.set_label(role, edit.text())

    def _reset_labels(self) -> None:
        self._labels = {}
        self._changed()

    def _on_click(self, event) -> None:
        """Double-click an axis label or title: edit its text in place."""
        if not event.dblclick or self.figure is None:
            return
        candidates = []
        if getattr(self.figure, "_suptitle", None) is not None:
            candidates.append(self.figure._suptitle)
        for ax in self.figure.axes:
            candidates += [ax.title, ax.xaxis.label, ax.yaxis.label]
        for text in candidates:
            role = getattr(text, "_hcr_role", None)
            if role and text.get_visible() and text.get_text() and text.contains(event)[0]:
                from PyQt5.QtWidgets import QInputDialog

                new, ok = QInputDialog.getText(self, "Edit label", f"{LABEL_ROLES.get(role, role)} "
                                               "(empty: automatic text):", QLineEdit.Normal, text.get_text())
                if ok:
                    self.set_label(role, new)
                return

    # -- options <-> controls ---------------------------------------------------------------

    def options(self) -> PlotOptions:
        return PlotOptions(
            kind=self.kind_combo.currentData(),
            channels=tuple(ch for ch, box in self.channel_checks.items() if box.isChecked()) or CHANNELS,
            t_min=None if self.t_all.isChecked() else self.t_from.value(),
            t_max=None if self.t_all.isChecked() else self.t_to.value(),
            count=self.count_combo.currentData(), intensity=self.int_combo.currentData(),
            count_relative=self.count_norm.currentData() == "normalized",
            intensity_relative=self.int_norm.currentData() == "normalized",
            band=self.band_combo.currentData(), show_cells=self.show_cells.isChecked(),
            y_mode=self.y_mode.currentData(), y_min=self.y_min.value(), y_max=self.y_max.value(),
            share_y=self.share_y.isChecked(), log_y=self.log_y.isChecked(), markers=self.markers.isChecked(),
            fit=self.fit_combo.currentData(), labels={k: dict(v) for k, v in self._labels.items()})

    def set_options(self, opts: PlotOptions) -> None:
        self._building = True
        try:
            self.kind_combo.setCurrentIndex(max(0, self.kind_combo.findData(opts.kind)))
            for ch, box in self.channel_checks.items():
                box.setChecked(ch in opts.channels)
            self.t_all.setChecked(opts.t_min is None and opts.t_max is None)
            if opts.t_min is not None:
                self.t_from.setValue(opts.t_min)
            if opts.t_max is not None:
                self.t_to.setValue(opts.t_max)
            self.count_combo.setCurrentIndex(max(0, self.count_combo.findData(opts.count)))
            self.count_norm.setCurrentIndex(self.count_norm.findData("normalized" if opts.count_relative else "absolute"))
            self.int_combo.setCurrentIndex(max(0, self.int_combo.findData(opts.intensity)))
            self.int_norm.setCurrentIndex(self.int_norm.findData("normalized" if opts.intensity_is_relative() else "absolute"))
            self.band_combo.setCurrentIndex(max(0, self.band_combo.findData(opts.band)))
            self.show_cells.setChecked(opts.show_cells)
            self.y_mode.setCurrentIndex(max(0, self.y_mode.findData(opts.y_mode)))
            self.y_min.setValue(opts.y_min)
            self.y_max.setValue(opts.y_max)
            self.share_y.setChecked(opts.share_y)
            self.log_y.setChecked(opts.log_y)
            self.markers.setChecked(opts.markers)
            self.fit_combo.setCurrentIndex(max(0, self.fit_combo.findData(opts.fit)))
            self._labels = {k: dict(v) for k, v in (opts.labels or {}).items()}
        finally:
            self._building = False
        self._sync_enabled()
        self._changed()

    def state(self) -> dict:
        """Everything to save with the session."""
        return {"options": asdict(self.options()), "sample": self.sample_combo.currentText(),
                "advanced_open": self.advanced.toggle.isChecked(),
                "labels_open": self.labels_section.toggle.isChecked(), "format": self.format_combo.currentText(),
                "dpi": self.dpi_spin.value(), "open": self.isVisible()}

    def apply_state(self, state: dict) -> None:
        if not state:
            return
        if state.get("format"):
            self.format_combo.setCurrentText(state["format"])
        if state.get("dpi"):
            self.dpi_spin.setValue(int(state["dpi"]))
        self.advanced.set_open(bool(state.get("advanced_open")))
        self.labels_section.set_open(bool(state.get("labels_open")))
        if state.get("sample"):
            self.sample_combo.setCurrentText(state["sample"])
        self.set_options(PlotOptions.from_dict(state.get("options")))

    # -- data and drawing -----------------------------------------------------------------------

    def set_results(self, results: dict, current: "str | None" = None, paths: "dict | None" = None) -> None:
        """`results`: {sample id: per-cell table over T}; `current`: the sample to
        show; `paths`: {sample id: its CSV} (recorded with exported fits)."""
        self.results = results
        if paths is not None:
            self.paths = dict(paths)
        self._building = True
        try:
            keep = current or self.sample_combo.currentText()
            self.sample_combo.clear()
            self.sample_combo.addItems(sorted(results))
            if keep in results:
                self.sample_combo.setCurrentText(keep)
            ts = [int(t) for df in results.values() for t in df["timepoint"].unique()]
            if ts:
                for spin in (self.t_from, self.t_to):
                    spin.setRange(min(ts), max(ts))
                if self.t_all.isChecked():
                    self.t_from.setValue(min(ts))
                    self.t_to.setValue(max(ts))
        finally:
            self._building = False
        self.redraw()

    def _changed(self, *_) -> None:
        if self._building:
            return
        self.sample_combo.setEnabled(self.kind_combo.currentData() != "batch")
        self._redraw_timer.start()
        self.optionsChanged.emit()

    def redraw(self) -> None:
        import pandas as pd

        opts = self.options()
        sample = self.sample_combo.currentText()
        if not self.results:
            return
        if opts.kind == "batch":
            cells = pd.concat(self.results.values(), ignore_index=True)
            title = "Batch-effect check: samples shown separately"
            n = cells["sample"].nunique()
            self.note.setText(f"{n} sample(s): {', '.join(sorted(self.results))}."
                              + (" Run another sample to compare." if n < 2 else ""))
        else:
            if sample not in self.results:
                return
            cells, title = self.results[sample], sample
            self.note.setText({"cells": "Per cell. Top: intensity; bottom: spots. Y axes include every point.",
                               "mean": "Mean of the included cells with the chosen error band.",
                               "counting": "Two ways of counting transcripts per cell. Top: the mean of "
                                           "each over the cells (Spots: normalized or absolute). Bottom: "
                                           "every cell at every timepoint, peaks against Big-FISH."}[opts.kind])
        self.figure = make_figure(cells, opts, title)
        # A new canvas per figure (matplotlib cannot move a figure between canvases).
        old_canvas, old_toolbar = self.canvas, self.toolbar
        self.canvas = self._canvas_cls(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.toolbar = self._toolbar_cls(self.canvas, self)
        self._left.replaceWidget(old_toolbar, self.toolbar)
        self._left.replaceWidget(old_canvas, self.canvas)
        for w in (old_toolbar, old_canvas):
            w.hide()
            w.deleteLater()
        self.canvas.draw_idle()
        self._rebuild_label_rows()
        self._show_fit_summary(opts)

    def _show_fit_summary(self, opts: PlotOptions) -> None:
        fits = fits_table(self.figure) if self.figure is not None else None
        enabled = opts.kind in ("mean", "batch", "counting")
        self.fit_combo.setEnabled(enabled)
        if not enabled:
            self.fit_summary.setText("<i>Fits apply to the Mean of cells, Batch effect and Counting methods "
                                     "plots.</i>")
        elif opts.fit == "off":
            self.fit_summary.setText("")
        elif fits is None or fits.empty:
            self.fit_summary.setText("<i>Too few timepoints to fit (at least 3; 4 for an exponential).</i>")
        else:
            rows = "".join(
                f"<tr><td>{r.channel}</td><td>{r.sample if opts.kind == 'batch' else ''}{r.measure}</td>"
                f"<td><b>{r.model}</b></td><td>R² {r.r2:.3f}</td><td>"
                + (f"t½ {r.half_life_T:.3g} T" if r.model == "exponential" else f"{r.slope_per_T:.3g}/T")
                + f"</td><td>ΔAICc {abs(r.aicc_linear - r.aicc_exponential):.1f}</td></tr>"
                for r in fits.itertuples())
            self.fit_summary.setText(
                "<table cellspacing='4'>" + rows + "</table><small>ΔAICc: how much better the chosen model "
                "is (&gt; 2: clear; &lt; 2: the two fit about equally).</small>")
        self.fit_export.setEnabled(enabled and opts.fit != "off" and fits is not None and not fits.empty)

    def default_name(self) -> str:
        opts = self.options()
        stem = {"cells": "bleaching", "mean": "bleaching_mean", "batch": "batch_effect",
                "counting": "counting_methods"}[opts.kind]
        return stem if opts.kind == "batch" else f"{self.sample_combo.currentText()}_{stem}"

    def export(self, path: "str | Path | None" = None) -> "Path | None":
        """Save the figure in the chosen format; asks for a file name unless `path` is given."""
        if self.figure is None:
            return None
        ext = EXPORT_FORMATS[self.format_combo.currentText()]
        if path is None:
            start = Path(self.out_dir()) / f"{self.default_name()}.{ext}"
            name, _ = QFileDialog.getSaveFileName(self, "Export plot", str(start),
                                                  f"{self.format_combo.currentText()} (*.{ext})")
            if not name:
                return None
            path = name
        path = Path(path)
        if path.suffix.lower().lstrip(".") != ext:
            path = path.with_suffix(f".{ext}")
        return export_figure(self.figure, path, dpi=self.dpi_spin.value())

    def export_fits(self, path: "str | Path | None" = None) -> "Path | None":
        """The fits as CSV, with a provenance sidecar: the options and the
        per-cell CSVs (with their SHA-256) the mean curves were computed from."""
        import hashlib

        if self.figure is None:
            return None
        fits = fits_table(self.figure)
        if fits.empty:
            return None
        if path is None:
            start = Path(self.out_dir()) / f"{self.default_name()}_fits.csv"
            name, _ = QFileDialog.getSaveFileName(self, "Export fits", str(start), "CSV (*.csv)")
            if not name:
                return None
            path = name
        path = Path(path).with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.to_csv(path, index=False)
        opts = self.options()
        samples = sorted(self.results) if opts.kind == "batch" else [self.sample_combo.currentText()]
        sources = []
        for s in samples:
            src = self.paths.get(s)
            if src and Path(src).is_file():
                sources.append({"sample": s, "path": str(Path(src).resolve()),
                                "sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest()})
        doc = {"what": "fits of the mean curve(s) shown in the plot window", "options": asdict(opts),
               "models": {"linear": "y = a + b*t", "exponential": "y = c + (y0 - c)*exp(-(t - t0)/tau)",
                          "choice": "lower AICc (auto), or the model chosen"},
               "sources": sources, "rows": int(len(fits))}
        path.with_name(path.name + ".provenance.json").write_text(json.dumps(doc, indent=2, default=str),
                                                                   encoding="utf-8")
        return path


def _group(title: str, inner) -> QWidget:
    """A titled box for the right-hand control column."""
    from PyQt5.QtWidgets import QGroupBox

    box = QGroupBox(title)
    box.setLayout(inner)
    return box


class SortableFrameModel(DataFrameModel):
    """vanua's DataFrameModel plus sorting on the underlying values (numbers as
    numbers, not as their display text)."""

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:  # noqa: A003 -- Qt override
        frame = self.frame()
        if frame is None or column < 0 or column >= len(frame.columns):
            return
        self.layoutAboutToBeChanged.emit()
        self._frame = frame.sort_values(frame.columns[column], ascending=order == Qt.AscendingOrder,
                                        kind="stable", na_position="last").reset_index(drop=True)
        self.layoutChanged.emit()


class ResultsTable(QWidget):
    """The per-cell results: click a column header to sort; export the table
    (in its current sort order) as CSV or Excel. `source_csv` (the series
    results with their provenance) is recorded in the export's sidecar."""

    FORMATS = {"CSV (.csv)": "csv", "Excel (.xlsx)": "xlsx"}

    def __init__(self, parent=None, out_dir=None) -> None:
        super().__init__(parent)
        self.out_dir = out_dir or (lambda: Path("out"))
        self.source_csv: "Path | None" = None
        self.name = "results"
        self.view = TableView()
        self.model = SortableFrameModel(parent=self.view)
        self.view.setModel(self.model)
        self.view.model_ = self.model
        # vanua's TableView sizes every column to its contents on every change,
        # which re-measures all ~16,000 cells of a 40-timepoint table each time
        # (tens of seconds per load); here widths are set once (_fit_columns)
        # and stay draggable.
        from PyQt5.QtWidgets import QHeaderView

        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.setSortingEnabled(True)
        self.view.horizontalHeader().setSortIndicatorShown(True)
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.format_combo = QComboBox()
        self.format_combo.addItems(list(self.FORMATS))
        self.export_button = QPushButton("Export table…")
        self.export_button.clicked.connect(lambda: self.export())
        row = QHBoxLayout()
        hint = QLabel("Click a column header to sort.")
        row.addWidget(hint, 1)
        row.addWidget(self.format_combo)
        row.addWidget(self.export_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.view, 1)

    def set_frame(self, frame, name: str = "results", source_csv=None) -> None:
        self.name, self.source_csv = name, (Path(source_csv) if source_csv else None)
        self.model.set_frame(frame.reset_index(drop=True) if frame is not None else None)
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self._fit_columns()

    def _fit_columns(self, sample_rows: int = 30) -> None:
        """Column widths from the header and the first rows' text.
        resizeColumnsToContents() rendered every cell (160 rows x ~100 columns
        = ~370,000 calls, ~4.5 s of every load); a sample gives the same widths."""
        frame = self.model.frame()
        if frame is None:
            return
        metrics = self.view.fontMetrics()
        header = self.view.horizontalHeader()
        rows = min(sample_rows, len(frame))
        for col in range(len(frame.columns)):
            texts = [str(self.model.headerData(col, Qt.Horizontal) or "")]
            texts += [str(self.model.data(self.model.index(r, col)) or "") for r in range(rows)]
            header.resizeSection(col, max(metrics.horizontalAdvance(s) for s in texts) + 24)

    def frame(self):
        return self.model.frame()

    def export(self, path: "str | Path | None" = None) -> "Path | None":
        frame = self.frame()
        if frame is None:
            return None
        ext = self.FORMATS[self.format_combo.currentText()]
        if path is None:
            start = Path(self.out_dir()) / f"{self.name}.{ext}"
            name, _ = QFileDialog.getSaveFileName(self, "Export table", str(start),
                                                  f"{self.format_combo.currentText()} (*.{ext})")
            if not name:
                return None
            path = name
        path = Path(path)
        if path.suffix.lower().lstrip(".") != ext:
            path = path.with_suffix(f".{ext}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if ext == "xlsx":
            frame.to_excel(path, index=False, sheet_name=str(self.name)[:31] or "results")
        else:
            frame.to_csv(path, index=False)
        self._write_provenance(path, frame)
        return path

    def _write_provenance(self, path: Path, frame) -> None:
        """The numbers trace to the series run: copy its provenance and say
        what this file is (a sorted view of it)."""
        doc = {}
        if self.source_csv is not None:
            sidecar = self.source_csv.with_name(self.source_csv.name + ".provenance.json")
            if sidecar.is_file():
                doc = json.loads(sidecar.read_text(encoding="utf-8"))
        doc["derived_export"] = {"path": str(path.resolve()), "format": path.suffix.lstrip("."),
                                 "rows": int(len(frame)), "columns": list(map(str, frame.columns)),
                                 "from": str(self.source_csv) if self.source_csv else None,
                                 "note": "the same rows and values, in the order shown (sorted in the GUI)"}
        path.with_name(path.name + ".provenance.json").write_text(json.dumps(doc, indent=2, default=str),
                                                                   encoding="utf-8")


class ResultsWindow(QWidget):
    """The results table in its own window, like the plots: a sample picker,
    the sortable table, and CSV / Excel export (ResultsTable)."""

    def __init__(self, parent=None, out_dir=None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("HCR Cells — Results")
        self.resize(1200, 600)
        self.results: dict = {}
        self.paths: dict = {}
        self.sample_combo = QComboBox()
        self.sample_combo.currentTextChanged.connect(lambda _s: self._show())
        self.table = ResultsTable(out_dir=out_dir)
        self.note = QLabel("")
        top = QHBoxLayout()
        top.addWidget(QLabel("Sample"))
        top.addWidget(self.sample_combo)
        top.addWidget(self.note, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)

    def set_results(self, results: dict, paths: dict, current: "str | None" = None) -> None:
        self.results, self.paths = results, paths
        keep = current or self.sample_combo.currentText()
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        self.sample_combo.addItems(sorted(results))
        if keep in results:
            self.sample_combo.setCurrentText(keep)
        self.sample_combo.blockSignals(False)
        self._show()

    def _show(self) -> None:
        sample = self.sample_combo.currentText()
        if sample not in self.results:
            return
        frame = self.results[sample]
        name = sample.replace(" ", "_").replace("(", "").replace(")", "")
        self.table.set_frame(frame, name=name if " " in sample else f"{sample}_cells",
                             source_csv=self.paths.get(sample))
        if {"cell", "timepoint"} <= set(frame.columns):
            self.note.setText(f"{len(frame)} rows: {frame['cell'].nunique()} cells × "
                              f"{frame['timepoint'].nunique()} timepoints")
        else:
            self.note.setText(f"{len(frame)} rows")

    def state(self) -> dict:
        return {"open": self.isVisible(), "sample": self.sample_combo.currentText(),
                "format": self.table.format_combo.currentText()}

    def apply_state(self, state: dict) -> None:
        if state.get("format"):
            self.table.format_combo.setCurrentText(state["format"])
        if state.get("sample") in self.results:
            self.sample_combo.setCurrentText(state["sample"])
