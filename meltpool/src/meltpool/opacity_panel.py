"""Which values are transparent in the 3D view: vanua's OpacityRampWidget (draggable
points over the stack's histogram) plus exact numbers and presets.

The opacity is a trapezoid by default -- transparent below `a`, ramping to `max
opacity` at `b`, opaque to `c`, ramping back to transparent at `d` and above -- so
any band of values can be shown and everything else made transparent. Dragging
points on the curve gives any other shape.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)
from vanua.ui.widgets.opacity_ramp import OpacityPoint, OpacityRampWidget


class OpacityPanel(QWidget):
    #: The opacity changed (on release, or when numbers / a preset are applied).
    changed = pyqtSignal()

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.ramp = OpacityRampWidget()
        self.ramp.setMinimumHeight(120)
        self.ramp.pointsCommitted.connect(self._on_ramp)
        self._range = (0.0, 1.0)
        self._sample: "np.ndarray | None" = None

        self.boxes = {}
        form = QFormLayout()
        for key, label in (("a", "Transparent below"), ("b", "Opaque from"), ("c", "Opaque to"),
                           ("d", "Transparent above")):
            box = QDoubleSpinBox()
            box.setRange(-1e9, 1e9)
            box.setDecimals(1)
            box.setKeyboardTracking(False)
            self.boxes[key] = box
            form.addRow(label, box)
        self.max_opacity = QDoubleSpinBox()
        self.max_opacity.setRange(0.0, 1.0)
        self.max_opacity.setSingleStep(0.05)
        self.max_opacity.setValue(0.6)
        form.addRow("Max opacity", self.max_opacity)
        apply = QPushButton("Apply numbers")
        apply.clicked.connect(self.apply_numbers)
        bright = QPushButton("Bright visible")
        bright.setToolTip("Dark values transparent: e.g. the gas of a vapour depression inside darker metal")
        bright.clicked.connect(lambda: self.preset("bright"))
        dark = QPushButton("Dark visible")
        dark.setToolTip("Bright values transparent: e.g. hide the air, show the metal")
        dark.clicked.connect(lambda: self.preset("dark"))
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.preset("default"))
        presets = QHBoxLayout()
        for b in (bright, dark, reset):
            presets.addWidget(b)
        self.note = QLabel("Values where the curve is at 0 are transparent. Drag points; double-click to add or "
                           "remove one. Applies to the 3D tab.")
        self.note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(QLabel("<b>3D transparency</b>"))
        layout.addWidget(self.ramp)
        layout.addLayout(presets)
        layout.addLayout(form)
        layout.addWidget(apply)
        layout.addWidget(self.note)

    # -- data ---------------------------------------------------------------------------------

    def set_data(self, stack: np.ndarray) -> None:
        """The volume about to be rendered: its histogram behind the curve; the curve kept
        if it still fits the new range, else the 'bright visible' preset."""
        sample = np.asarray(stack)[:: max(1, len(stack) // 40)].ravel()
        if sample.size > 2_000_000:
            sample = sample[:: sample.size // 2_000_000]
        self._sample = sample.astype(np.float64)
        lo, hi = float(self._sample.min()), float(self._sample.max())
        old = self._range
        self._range = (lo, hi if hi > lo else lo + 1.0)
        span = self._range[1] - self._range[0]
        decimals = int(np.clip(np.ceil(-np.log10(span)) + 3, 1, 8))
        for box in self.boxes.values():
            box.setDecimals(decimals)
            box.setSingleStep(span / 100.0)
        self.ramp.set_data(self._sample)
        points = self.ramp.points()
        if old == (0.0, 1.0) or not points or points[0].value < lo - span or points[-1].value > self._range[1] + span:
            self.preset("bright", emit=False)

    def function(self):
        return self.ramp.to_piecewise_function()

    def points(self) -> "list[OpacityPoint]":
        return self.ramp.points()

    # -- editing --------------------------------------------------------------------------------

    def set_trapezoid(self, a: float, b: float, c: float, d: float, opacity: float, emit: bool = True) -> None:
        lo, hi = self._range
        a, b, c, d = sorted((a, b, c, d))
        # a band that reaches the end of the data starts / ends opaque there (no zero at the edge)
        pts = []
        if a > lo:
            pts += [OpacityPoint(lo, 0.0), OpacityPoint(a, 0.0)]
        pts += [OpacityPoint(max(b, lo), opacity), OpacityPoint(min(c, hi) if c > b else max(b, lo), opacity)]
        if d < hi:
            pts += [OpacityPoint(d, 0.0), OpacityPoint(hi, 0.0)]
        # identical values would make an ill-formed function: nudge them apart
        eps = (hi - lo) * 1e-6 or 1e-6
        out = []
        for p in pts:
            if out and p.value <= out[-1].value:
                p = OpacityPoint(out[-1].value + eps, p.opacity)
            out.append(p)
        self.ramp.set_points(out)
        for key, v in zip("abcd", (a, b, c, d)):
            self.boxes[key].blockSignals(True)
            self.boxes[key].setValue(v)
            self.boxes[key].blockSignals(False)
        self.max_opacity.setValue(opacity)
        if emit:
            self.changed.emit()

    def apply_numbers(self) -> None:
        self.set_trapezoid(*(self.boxes[k].value() for k in "abcd"), self.max_opacity.value())

    def preset(self, name: str, emit: bool = True) -> None:
        if self._sample is None:
            return
        lo, hi = self._range
        p = np.percentile(self._sample, (1, 50, 99))
        op = self.max_opacity.value() or 0.6
        if name == "bright":
            self.set_trapezoid(p[1], p[2], hi, hi, op, emit)
        elif name == "dark":
            self.set_trapezoid(lo, lo, p[0], p[1], op, emit)
        else:
            self.set_trapezoid(p[0], p[2], hi, hi, 0.8, emit)

    def _on_ramp(self) -> None:
        self.changed.emit()
