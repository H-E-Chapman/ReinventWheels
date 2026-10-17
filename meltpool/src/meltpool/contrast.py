"""Display contrast for the slice view: a histogram of the shown frame with
draggable min/max handles and a gamma point (vanua's HistogramWidget), exact
min/max/gamma boxes, Auto and Full-range buttons, and "keep for every frame".

Display only: the analysis always works on the raw intensities.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                             QWidget)
from vanua.ui.widgets.histogram import HistogramWidget

AUTO_PERCENTILE = (0.5, 99.5)


class ContrastPanel(QWidget):
    #: (lo, hi, gamma) whenever the window applied to the view changes.
    changed = pyqtSignal(float, float, float)

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self._viewer = None
        self._plane: "np.ndarray | None" = None
        self._kind: "str | None" = None
        self._windows: dict = {}  # kind -> ((lo, hi), gamma, keep)
        self._suspend = False

        self.histogram = HistogramWidget(show_gamma_curve=True)
        self.histogram.setMinimumHeight(140)
        self.histogram.rangeChanged.connect(self._on_handles)
        self.histogram.gammaChanged.connect(self._on_gamma_drag)

        self.min_box, self.max_box = QDoubleSpinBox(), QDoubleSpinBox()
        for box in (self.min_box, self.max_box):
            box.setDecimals(1)
            box.setRange(-1e9, 1e9)
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self._on_boxes)
        self.gamma_box = QDoubleSpinBox()
        self.gamma_box.setRange(0.1, 5.0)
        self.gamma_box.setSingleStep(0.05)
        self.gamma_box.setValue(1.0)
        self.gamma_box.setKeyboardTracking(False)
        self.gamma_box.valueChanged.connect(self._on_gamma_box)

        self.auto_button = QPushButton("Auto")
        self.auto_button.setToolTip(f"Stretch the {AUTO_PERCENTILE[0]}–{AUTO_PERCENTILE[1]} percentile of this frame")
        self.auto_button.clicked.connect(self.auto)
        self.full_button = QPushButton("Full range")
        self.full_button.setToolTip("The frame's own minimum to maximum")
        self.full_button.clicked.connect(self.full_range)
        self.keep = QCheckBox("Keep for every frame")
        self.keep.setToolTip("On: scrubbing, cropping and previews keep this window. Off: every frame is "
                             "auto-stretched. Dragging a handle turns it on.")
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        note = QLabel("Display only: the analysis always uses the raw intensities.")
        note.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.auto_button)
        buttons.addWidget(self.full_button)
        form = QFormLayout()
        form.addRow("Min", self.min_box)
        form.addRow("Max", self.max_box)
        form.addRow("Gamma", self.gamma_box)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Contrast</b>"))
        layout.addWidget(self.histogram)
        layout.addLayout(buttons)
        layout.addLayout(form)
        layout.addWidget(self.keep)
        layout.addWidget(self.stats)
        layout.addWidget(note)
        layout.addStretch(1)

    # -- wiring -------------------------------------------------------------------------

    def attach(self, slice_viewer) -> None:
        self._viewer = slice_viewer

    def show_plane(self, plane: np.ndarray, kind: str = "frame") -> None:
        """Call after a new image is put on the viewer: refresh the histogram, then
        apply the kept window or an auto stretch. A window is kept per `kind` of image
        ("frame", "median", "detail", ...): a 20-60 grey window kept for movie frames
        would blacken a band-passed image whose values are about +-0.1."""
        plane = np.asarray(plane, dtype=np.float64)
        if self._plane is not None and getattr(self, "_kind", None) is not None:
            self._windows[self._kind] = (self.window(), self.gamma_box.value(), self.keep.isChecked())
        self._kind = kind
        self._plane = plane
        self._set_precision(plane)
        kept = self._windows.get(kind)
        self._suspend = True
        self.keep.setChecked(bool(kept and kept[2]))
        self._suspend = False
        if kept and kept[2]:
            (lo, hi), gamma = kept[0], kept[1]
        else:
            lo, hi = self._auto_range(plane)
            gamma = kept[1] if kept else self.gamma_box.value()
        self._set(lo, hi, gamma)
        digits = max(0, self.min_box.decimals() - 1)
        self.stats.setText(f"Image min {plane.min():.{digits}f}, max {plane.max():.{digits}f}, "
                           f"mean {plane.mean():.{digits + 1}f}")

    def _set_precision(self, plane: np.ndarray) -> None:
        """Enough decimals for the image's own range (the boxes round what they show,
        and the view uses exactly what they show)."""
        span = float(np.percentile(plane, 99.5) - np.percentile(plane, 0.5)) or float(plane.max() - plane.min()) or 1.0
        decimals = int(np.clip(np.ceil(-np.log10(span)) + 3, 1, 8))
        self._suspend = True
        for box in (self.min_box, self.max_box):
            box.setDecimals(decimals)
            box.setSingleStep(span / 100.0)
        self._suspend = False

    def window(self) -> "tuple[float, float]":
        return float(self.min_box.value()), float(self.max_box.value())

    def gamma(self) -> float:
        return float(self.gamma_box.value())

    def auto(self) -> None:
        if self._plane is not None:
            self._set(*self._auto_range(self._plane), self.gamma_box.value())

    def full_range(self) -> None:
        if self._plane is not None:
            self._set(float(self._plane.min()), float(self._plane.max()), self.gamma_box.value())

    # -- internals ----------------------------------------------------------------------

    @staticmethod
    def _auto_range(plane: np.ndarray) -> "tuple[float, float]":
        lo, hi = np.percentile(plane, AUTO_PERCENTILE)
        if hi <= lo:
            lo, hi = float(plane.min()), float(plane.max()) + 1.0
        return float(lo), float(hi)

    def _set(self, lo: float, hi: float, gamma: float) -> None:
        if hi <= lo:
            hi = lo + 1e-6
        self._suspend = True
        try:
            if self._plane is not None:
                # Histogram over the frame's own range, widened to include the window.
                span = (min(lo, float(self._plane.min())), max(hi, float(self._plane.max())))
                self.histogram.set_data(np.clip(self._plane, *span), (lo, hi))
            else:
                self.histogram.set_contrast_range(lo, hi)
            self.histogram.set_gamma(gamma)
            self.min_box.setValue(lo)
            self.max_box.setValue(hi)
            self.gamma_box.setValue(gamma)
        finally:
            self._suspend = False
        self._apply(*self.window(), self.gamma())  # exactly what the boxes show

    def _apply(self, lo: float, hi: float, gamma: float) -> None:
        if self._viewer is not None and self._plane is not None:
            self._viewer.set_level_range(lo, hi)
            self._viewer.set_gamma(gamma)
        self.changed.emit(lo, hi, gamma)

    def _on_handles(self, lo: float, hi: float) -> None:
        if self._suspend:
            return
        self.keep.setChecked(True)
        self._suspend = True
        self.min_box.setValue(lo)
        self.max_box.setValue(hi)
        self._suspend = False
        self._apply(lo, hi, self.gamma_box.value())

    def _on_gamma_drag(self, gamma: float) -> None:
        if self._suspend:
            return
        self._suspend = True
        self.gamma_box.setValue(gamma)
        self._suspend = False
        self._apply(*self.window(), gamma)

    def _on_boxes(self, _value: float) -> None:
        if self._suspend:
            return
        self.keep.setChecked(True)
        self._set(*self.window(), self.gamma_box.value())

    def _on_gamma_box(self, gamma: float) -> None:
        if not self._suspend:
            self._set(*self.window(), gamma)
