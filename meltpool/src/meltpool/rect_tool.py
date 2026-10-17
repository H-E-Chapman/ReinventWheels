"""Drag a rectangle on the slice view (vanua's "paint" left tool reports press /
drag / release in image pixels); shown as an outline on the image layer, so it
follows zoom and pan. Used for the stabilisation region."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsRectItem


class RectDrawer:
    """`start()` puts the view in drawing mode; one drag later `on_done(x0, y0, x1, y1)`
    gets the box in image pixels (sorted, clamped to `shape`) and the view is back to
    its normal tool. `show(...)` / `hide()` draw a box without the mouse."""

    def __init__(self, slice_viewer, colour: str, hint: str, on_done: Callable[[int, int, int, int], None]) -> None:
        self.view = slice_viewer
        self.colour = colour
        self.hint = hint
        self.on_done = on_done
        self.shape: "tuple[int, int] | None" = None
        self.active = False
        self._item: "QGraphicsRectItem | None" = None
        self._start: "tuple[float, float] | None" = None

    def start(self, shape: "tuple[int, int]") -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.active = True
        self._start = None
        self.view.set_paint_radius(None)
        self.view.set_paint_hint_labels(self.hint, None)
        self.view.set_left_tool("paint")
        self.view.on_painted(self._on_paint)

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.view.on_painted(None)
        self.view.set_paint_hint_labels(None, None)
        self.view.set_left_tool("crosshair")

    def show(self, x0: float, y0: float, x1: float, y1: float) -> None:
        rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if self._item is None:
            self._item = QGraphicsRectItem(rect, self.view.pixmap_item)
            pen = QPen(QColor(self.colour))
            pen.setWidth(2)
            pen.setCosmetic(True)
            pen.setStyle(2)  # dashed: tells it apart from the ROI
            self._item.setPen(pen)
        else:
            self._item.setRect(rect)
        self._item.show()

    def hide(self) -> None:
        if self._item is not None:
            self._item.hide()

    def _on_paint(self, x: float, y: float, phase: str, button: str) -> None:
        if button != "left" or self.shape is None:
            return
        h, w = self.shape
        x, y = min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h))
        if phase == "start" or self._start is None:
            self._start = (x, y)
        sx, sy = self._start
        self.show(sx, sy, x, y)
        if phase != "end":
            return
        self._start = None
        box = (int(np.floor(min(sx, x))), int(np.floor(min(sy, y))), int(np.ceil(max(sx, x))), int(np.ceil(max(sy, y))))
        self.stop()
        self.on_done(*box)
