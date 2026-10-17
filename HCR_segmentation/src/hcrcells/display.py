"""Display widgets for the HCR Cells composite image.

* :class:`SliceCompositor` -- paints the composite of the visible channel
  layers at the slice viewer's current z.
* :class:`CombinedHistogram` -- ONE chart with every channel's histogram in
  its LUT colour and that channel's min/max handles (drag them).
* :class:`ChannelPanel` -- the combined histogram above one compact row of
  controls per channel: visibility, LUT, gamma, opacity, exact min/max.
* :func:`label_slice_slider` -- an axis label and live position readout on a
  slice viewer's slice scroll bar (Z, or Y / X for the ortho panes).

All edits go through the :class:`~vanua.core.layers.LayerStack` (layers are
immutable; ``replace_at`` notifies listeners), exactly as vanua's own
RenderingPanel does -- this replaces that panel only because it draws one
histogram per channel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import QEvent, QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (QCheckBox, QDoubleSpinBox, QFrame, QGraphicsDropShadowEffect, QGraphicsItem,
                             QGraphicsObject, QGridLayout, QHBoxLayout, QLabel, QSizePolicy,
                             QSlider, QSpinBox, QVBoxLayout, QWidget)

from vanua.core.layers import ImageLayer, LayerStack
from vanua.core.lut import LUT_REGISTRY, composite_channels
from vanua.ui.viewers.slice2d import rgb_to_qimage
from vanua.ui.widgets.lut_picker import LutPicker

__all__ = ["SliceCompositor", "CombinedHistogram", "ChannelPanel", "label_slice_slider", "lut_color",
           "install_checkmark_style", "CellBadge", "badge_text_color"]


_CHECK_SVG = Path(__file__).resolve().parent / "assets" / "check.svg"


def install_checkmark_style(app) -> None:
    """Checked boxes show a tick on the normal box, not a filled box.

    vanua's theme (not to be edited, I3) draws a checked QCheckBox as a box
    filled with the accent colour. This appends app-level rules on top:
    same box and size, the theme's background, and a tick (assets/check.svg,
    in the theme's accent colour). Also used for the review table's Include
    ticks. Idempotent."""
    if app is None or "hcrcells-checkmark" in app.styleSheet():
        return
    from vanua.ui import theme

    t = theme.DEFAULT_THEME
    url = _CHECK_SVG.as_posix()
    app.setStyleSheet(app.styleSheet() + f"""
        /* hcrcells-checkmark */
        QCheckBox::indicator:checked, QTableView::indicator:checked {{
            background-color: {t.bg_alt}; border: 1px solid {t.accent}; image: url("{url}");
        }}
        QTableView::indicator {{
            width: 13px; height: 13px; background-color: {t.bg_alt};
            border: 1px solid {theme.BORDER}; border-radius: 2px;
        }}
    """)


# -- spot colour --------------------------------------------------------------------------------

#: Marker colours offered for spots: all bright enough to read on the dark
#: viewer background (no black, no dark blue).
SPOT_PALETTE = {
    "Red": (255, 60, 60), "Orange": (255, 150, 0), "Yellow": (255, 235, 0), "Green": (0, 230, 0),
    "Cyan": (0, 230, 230), "Sky blue": (90, 160, 255), "Magenta": (255, 0, 255), "White": (255, 255, 255),
}


def _lab(rgb) -> np.ndarray:
    """sRGB (0-255) -> CIE L*a*b* (D65), for perceptual colour distance."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ c
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def contrast_color(rgb) -> "tuple[int, int, int]":
    """The palette colour that differs most from `rgb` (CIE76 ΔE): the
    default spot colour, so markers stand out on their own channel."""
    target = _lab(rgb)
    return max(SPOT_PALETTE.values(), key=lambda c: float(np.linalg.norm(_lab(c) - target)))


class SpotColorPicker(QWidget):
    """Spot marker colour: 'Auto (highest contrast)' -- follows the spot
    channel's LUT colour via :func:`contrast_color` -- a named colour, or a
    custom one (colour dialog). `colorChanged` fires with the colour to use."""

    colorChanged = pyqtSignal(tuple)
    AUTO, CUSTOM = "Auto (highest contrast)", "Custom…"

    def __init__(self, parent: "QWidget | None" = None) -> None:
        from PyQt5.QtWidgets import QComboBox

        super().__init__(parent)
        self._auto = (255, 255, 255)
        self._custom: "tuple[int, int, int] | None" = None
        self.combo = QComboBox()
        self.combo.addItem(self.AUTO)
        for name, rgb in SPOT_PALETTE.items():
            self.combo.addItem(_swatch(rgb), name)
        self.combo.addItem(self.CUSTOM)
        # Sized to its longest item, the combo was wider than the dock
        # (horizontal scroll); let it shrink to the dock instead.
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(8)
        self.combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo)
        self._last_index = 0
        self.combo.activated.connect(self._on_activated)
        self._refresh_auto_swatch()

    def set_auto_color(self, channel_rgb) -> None:
        """The spot channel's colour changed (or another channel was chosen)."""
        self._auto = contrast_color(channel_rgb)
        self._refresh_auto_swatch()

    def _refresh_auto_swatch(self) -> None:
        self.combo.setItemIcon(0, _swatch(self._auto))

    def choice(self):
        """None (auto), a palette name, or ('custom', rgb) -- to remember per channel."""
        text = self.combo.currentText()
        if text == self.AUTO:
            return None
        if text == self.CUSTOM:
            return ("custom", self._custom)
        return text

    def set_choice(self, choice) -> None:
        if choice is None:
            self.combo.setCurrentIndex(0)
        elif isinstance(choice, tuple):
            self._custom = tuple(choice[1])
            self.combo.setItemIcon(self.combo.count() - 1, _swatch(self._custom))
            self.combo.setCurrentIndex(self.combo.count() - 1)
        else:
            self.combo.setCurrentText(choice)
        self._last_index = self.combo.currentIndex()

    def color(self) -> "tuple[int, int, int]":
        text = self.combo.currentText()
        if text == self.AUTO:
            return self._auto
        if text == self.CUSTOM and self._custom is not None:
            return self._custom
        return SPOT_PALETTE.get(text, self._auto)

    def _on_activated(self, index: int) -> None:
        if self.combo.itemText(index) == self.CUSTOM:
            from PyQt5.QtWidgets import QColorDialog

            start = QColor(*(self._custom or self.color()))
            picked = QColorDialog.getColor(start, self, "Spot colour")
            if not picked.isValid():
                self.combo.setCurrentIndex(self._last_index)  # cancelled: keep the previous choice
                return
            self._custom = (picked.red(), picked.green(), picked.blue())
            self.combo.setItemIcon(index, _swatch(self._custom))
        self._last_index = self.combo.currentIndex()
        self.colorChanged.emit(self.color())


def _swatch(rgb, size: int = 14):
    from PyQt5.QtGui import QIcon

    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


def lut_color(lut_name: str) -> "tuple[int, int, int]":
    """The colour a LUT ramps to (its brightest entry) -- the channel's colour in the chart."""
    lut = LUT_REGISTRY.get(lut_name, LUT_REGISTRY["Grays"])
    return tuple(int(v) for v in lut.table[-1])


# -- composite ---------------------------------------------------------------------------


class SliceCompositor:
    """Paints the additive composite of the visible layers of `stack`
    (per-channel LUT, min/max, gamma, opacity -- vanua.core.lut.composite_channels)
    for the slice viewer's *current* z.

    vanua's CompositeDisplay always composites the middle plane and so
    fights z-scrolling (building.md §5). Here the slice viewer keeps one
    channel loaded only as the carrier of z navigation, outlines and spot
    markers (separate scene items, untouched), and its plane pixmap is
    repainted with the composite after every z change and every layer edit.
    """

    def __init__(self, stack: LayerStack, slice_viewer) -> None:
        self.stack = stack
        self.viewer = slice_viewer
        #: Called with the current z after every repaint (the slice viewer has
        #: one z callback, taken here; other per-slice overlays hook in here).
        self.z_listeners: list = []
        stack.add_listener(self.refresh)
        slice_viewer.on_z_changed(lambda _z: self.refresh())
        # The window/level mouse tool redraws the carrier plane; paint over it again.
        slice_viewer.on_window_level_changed(lambda _lo, _hi: self.refresh())

    def refresh(self) -> None:
        layers = [layer for layer in self.stack if isinstance(layer, ImageLayer)]
        if not layers or self.viewer.z_count == 0:
            return
        z = self.viewer.current_z
        planes = [np.asarray(layer.data[min(z, layer.data.shape[0] - 1)]) for layer in layers]
        levels = [layer.contrast_range or (float(p.min()), float(p.max()) or 1.0)
                  for layer, p in zip(layers, planes)]
        rgb = composite_channels(planes, [LUT_REGISTRY.get(layer.lut_name, LUT_REGISTRY["Grays"]) for layer in layers],
                                 levels, opacities=[layer.opacity for layer in layers],
                                 gammas=[layer.gamma for layer in layers],
                                 visible=[layer.visible for layer in layers])
        self.viewer.pixmap_item.setPixmap(QPixmap.fromImage(rgb_to_qimage(rgb)))
        self.viewer.as_widget().update()
        for listener in self.z_listeners:
            listener(z)


# -- cell labels ------------------------------------------------------------------------------


def label_positions(labels: np.ndarray) -> "dict[int, list[tuple[int, float, float]]]":
    """{z: [(label, y, x), ...]}: each label's centre on every plane it occupies."""
    from scipy import ndimage

    out: dict = {}
    for z in range(labels.shape[0]):
        plane = labels[z]
        ids = np.unique(plane)
        ids = ids[ids > 0]
        if len(ids):
            centres = ndimage.center_of_mass(np.ones_like(plane, dtype=np.uint8), plane, ids)
            out[z] = [(int(i), float(cy), float(cx)) for i, (cy, cx) in zip(ids, centres)]
    return out


class CellLabelOverlay:
    """Cell letters on the slice view, at each cell's centre on the current
    plane, in the cell's outline colour (grey + '(excl.)' when excluded).

    The text items are children of the viewer's public `pixmap_item`, so they
    sit in its scene without touching the viewer's internals, survive the
    composite repaints (setPixmap keeps children), and ignore the view's zoom
    so the letters stay readable at any magnification."""

    def __init__(self, slice_viewer) -> None:
        self.viewer = slice_viewer
        self.items: list = []
        self.positions: dict = {}
        self.names: dict = {}
        self.colors: dict = {}
        self.excluded: set = set()
        self.visible = True

    def set_cells(self, positions: dict, names: dict, colors: dict, excluded=()) -> None:
        self.positions, self.names, self.colors = positions, dict(names), dict(colors)
        self.excluded = set(excluded)
        self.refresh(self.viewer.current_z)

    def clear(self) -> None:
        self.positions, self.names = {}, {}
        self._remove()

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self.refresh(self.viewer.current_z)

    def _remove(self) -> None:
        for item in self.items:
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)
        self.items = []

    def refresh(self, z: int) -> None:
        self._remove()
        if not self.visible:
            return
        for label, y, x in self.positions.get(z, ()):
            name = self.names.get(label)
            if name is None:
                continue
            badge = CellBadge(name, self.colors.get(label, (255, 255, 255)), excluded=name in self.excluded,
                              parent=self.viewer.pixmap_item)
            badge.setPos(x + 0.5, y + 0.5)  # the badge draws itself centred on its position
            self.items.append(badge)


def badge_text_color(rgb) -> "tuple[int, int, int]":
    """Near-black on a light fill, white on a dark one (WCAG relative luminance)."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    luminance = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return (24, 24, 32) if luminance > 0.35 else (255, 255, 255)


class CellBadge(QGraphicsObject):
    """A rounded 'pill' with the cell's letter: filled with the cell's
    colour (slightly translucent), a thin lighter rim, a soft shadow, and
    black or white text for contrast. Excluded cells: a dark pill with a
    dashed grey rim and muted 'E · excl.'. Ignores the view's zoom, so it
    stays the same size on screen, and is centred on its position."""

    PAD_X, PAD_Y, RADIUS = 7.0, 2.5, 8.0

    def __init__(self, text: str, rgb, excluded: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.excluded = excluded
        self.text = f"{text} · excl." if excluded else text
        self.rgb = tuple(int(v) for v in rgb)
        self.font = QFont()
        self.font.setPointSizeF(9.5)
        self.font.setWeight(QFont.DemiBold)
        from PyQt5.QtGui import QFontMetricsF

        metrics = QFontMetricsF(self.font)
        w = max(metrics.horizontalAdvance(self.text) + 2 * self.PAD_X, metrics.height() + 2 * self.PAD_Y)
        h = metrics.height() + 2 * self.PAD_Y
        self._rect = QRectF(-w / 2, -h / 2, w, h)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(10)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setOffset(0, 1.2)
        shadow.setColor(QColor(0, 0, 0, 150))
        self.setGraphicsEffect(shadow)

    def boundingRect(self) -> QRectF:  # noqa: N802 -- Qt override
        return self._rect.adjusted(-1, -1, 1, 1)

    def paint(self, painter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        radius = min(self.RADIUS, self._rect.height() / 2)
        if self.excluded:
            fill, rim, ink = QColor(30, 30, 46, 210), QPen(QColor(150, 150, 160), 1.0, Qt.DashLine), QColor(170, 170, 180)
        else:
            fill = QColor(*self.rgb, 225)
            rim = QPen(QColor(*self.rgb).lighter(150), 1.0)
            ink = QColor(*badge_text_color(self.rgb))
        painter.setPen(rim)
        painter.setBrush(fill)
        painter.drawRoundedRect(self._rect, radius, radius)
        painter.setFont(self.font)
        painter.setPen(ink)
        painter.drawText(self._rect, Qt.AlignCenter, self.text)


# -- ROI box ---------------------------------------------------------------------------------------

ROI_COLOR = (255, 196, 0)  # amber: distinct from every channel LUT and cell colour


REF_COLOR = (170, 170, 185)  # the reference region: quieter than the ROI


class _BoxItem:
    """One dashed rectangle with a small tag, on the slices the box spans."""

    def __init__(self, parent, color, text: str, width: float) -> None:
        from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsSimpleTextItem

        self.box: "dict | None" = None
        self.rect = QGraphicsRectItem(parent)
        pen = QPen(QColor(*color), width, Qt.DashLine)
        pen.setCosmetic(True)  # screen pixels at any zoom
        self.rect.setPen(pen)
        self.rect.setZValue(9)
        self.tag = QGraphicsSimpleTextItem(text, parent)
        font = QFont()
        font.setPointSizeF(8.5)
        font.setWeight(QFont.DemiBold)
        self.tag.setFont(font)
        self.tag.setBrush(QColor(*color))
        self.tag.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.tag.setZValue(9)

    def set_box(self, box: "dict | None") -> None:
        self.box = box
        if box is not None:
            (y0, y1), (x0, x1) = box["y"], box["x"]
            self.rect.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            self.tag.setPos(x0 + 2, y0 + 1)

    def refresh(self, z: int) -> None:
        shown = self.box is not None and self.box["z"][0] <= z < self.box["z"][1]
        self.rect.setVisible(shown)
        self.tag.setVisible(shown)


class RoiOverlay:
    """The ROI (dashed amber, 'ROI') and the larger reference region (dashed
    grey, 'reference') on the slice view, each on the slices it spans.
    Children of the viewer's public `pixmap_item`, like the cell badges."""

    def __init__(self, slice_viewer) -> None:
        self.viewer = slice_viewer
        self.ref = _BoxItem(slice_viewer.pixmap_item, REF_COLOR, "reference", 1.2)
        self.roi = _BoxItem(slice_viewer.pixmap_item, ROI_COLOR, "ROI", 2.0)
        self.set_box(None)

    @property
    def box(self) -> "dict | None":
        return self.roi.box

    @property
    def ref_box(self) -> "dict | None":
        return self.ref.box

    def set_box(self, box: "dict | None", ref_box: "dict | None" = None) -> None:
        self.roi.set_box(box)
        self.ref.set_box(ref_box if box is not None else None)
        self.refresh(self.viewer.current_z)

    def refresh(self, z: int) -> None:
        self.roi.refresh(z)
        self.ref.refresh(z)


# -- combined histogram ----------------------------------------------------------------------


def volume_histogram(volume, bins: int = 256) -> "tuple[np.ndarray, np.ndarray]":
    """Counts and bin edges over the whole volume (every 4th pixel in y and x
    of every plane: the shape of the distribution, at 1/16 of the cost)."""
    if hasattr(volume, "loaded_sample"):  # a lazy channel: the slices read so far (refreshed when all are in)
        sample = np.asarray(volume.loaded_sample())
    else:
        sample = np.asarray(volume[:, ::4, ::4])
    if np.issubdtype(sample.dtype, np.integer) and sample.dtype.itemsize == 1:
        counts = np.bincount(sample.ravel(), minlength=256).astype(float)
        return counts, np.arange(257, dtype=float)
    lo, hi = float(sample.min()), float(sample.max())
    counts, edges = np.histogram(sample, bins=bins, range=(lo, hi if hi > lo else lo + 1))
    return counts.astype(float), edges.astype(float)


class CombinedHistogram(QWidget):
    """Every channel's histogram in one chart, each in its LUT colour (log
    counts, each scaled to its own peak so a sparse channel stays visible),
    with that channel's min and max as draggable handles in the same colour.

    Press near a handle and drag; the nearest handle of a visible channel is
    taken. `rangeChanged(name, lo, hi)` fires while dragging,
    `rangeCommitted(name, lo, hi)` on release.
    """

    rangeChanged = pyqtSignal(str, float, float)
    rangeCommitted = pyqtSignal(str, float, float)

    _PAD_L, _PAD_R, _PAD_T, _PAD_B = 8, 8, 18, 22
    _GRAB_PX = 8

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self._channels: dict = {}   # name -> {counts, edges, color, lo, hi, visible}
        self._order: list = []
        self._drag: "tuple[str, int] | None" = None   # (channel, 0 = min / 1 = max)
        self._x_range = (0.0, 1.0)

    # model ---------------------------------------------------------------------------

    def set_channel(self, name: str, counts, edges, color, lo: float, hi: float, visible: bool) -> None:
        self._channels[name] = {"counts": np.asarray(counts, float), "edges": np.asarray(edges, float),
                                "color": tuple(color), "lo": float(lo), "hi": float(hi), "visible": bool(visible)}
        if name not in self._order:
            self._order.append(name)
        edges_all = [c["edges"] for c in self._channels.values()]
        self._x_range = (min(e[0] for e in edges_all), max(e[-1] for e in edges_all))
        self.update()

    def update_channel(self, name: str, **values) -> None:
        if name in self._channels:
            self._channels[name].update(values)
            self.update()

    def clear(self) -> None:
        self._channels, self._order = {}, []
        self.update()

    def channel_range(self, name: str) -> "tuple[float, float]":
        c = self._channels[name]
        return c["lo"], c["hi"]

    # geometry --------------------------------------------------------------------------

    def _plot_rect(self) -> QRectF:
        return QRectF(self._PAD_L, self._PAD_T, self.width() - self._PAD_L - self._PAD_R,
                      self.height() - self._PAD_T - self._PAD_B)

    def _x_of(self, value: float) -> float:
        r = self._plot_rect()
        lo, hi = self._x_range
        return r.left() + (value - lo) / ((hi - lo) or 1.0) * r.width()

    def _value_of(self, x: float) -> float:
        r = self._plot_rect()
        lo, hi = self._x_range
        return lo + (x - r.left()) / (r.width() or 1.0) * (hi - lo)

    # painting --------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 -- Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._plot_rect()
        p.fillRect(self.rect(), self.palette().base())
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawRect(r)
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.5))
        p.setFont(small)
        # x ticks
        lo, hi = self._x_range
        for v in np.linspace(lo, hi, 5):
            x = self._x_of(v)
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 3))
            p.drawText(QRectF(x - 30, r.bottom() + 4, 60, 14), Qt.AlignHCenter, f"{v:g}")
        visible = [n for n in self._order if self._channels[n]["visible"]]
        # curves
        for name in visible:
            c = self._channels[name]
            counts = np.log1p(c["counts"])
            peak = counts.max() or 1.0
            centres = 0.5 * (c["edges"][:-1] + c["edges"][1:])
            path = QPainterPath(QPointF(self._x_of(centres[0]), r.bottom()))
            for xv, yv in zip(centres, counts / peak):
                path.lineTo(QPointF(self._x_of(xv), r.bottom() - yv * r.height()))
            path.lineTo(QPointF(self._x_of(centres[-1]), r.bottom()))
            color = QColor(*c["color"])
            fill = QColor(color)
            fill.setAlpha(45)
            p.fillPath(path, fill)
            p.setPen(QPen(color, 1.4))
            p.drawPath(path)
        # handles: min = triangle at the bottom pointing right, max = pointing left
        for name in visible:
            c = self._channels[name]
            color = QColor(*c["color"])
            for k, value in enumerate((c["lo"], c["hi"])):
                x = self._x_of(value)
                p.setPen(QPen(color, 2 if self._drag == (name, k) else 1.3, Qt.DashLine))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                d = 6 if k == 0 else -6
                tri = QPolygonF([QPointF(x, r.bottom()), QPointF(x + d, r.bottom() - 5), QPointF(x, r.bottom() - 10)])
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawPolygon(tri)
        # legend
        x = r.left() + 4
        for name in self._order:
            c = self._channels[name]
            color = QColor(*c["color"]) if c["visible"] else QColor(110, 110, 110)
            p.setPen(color)
            text = name if c["visible"] else f"{name} (off)"
            p.drawText(QPointF(x, self._PAD_T - 5), text)
            x += p.fontMetrics().horizontalAdvance(text) + 12
        p.end()

    # interaction -----------------------------------------------------------------------

    def _handle_at(self, x: float) -> "tuple[str, int] | None":
        best, best_d = None, self._GRAB_PX + 1
        for name in self._order:
            c = self._channels[name]
            if not c["visible"]:
                continue
            for k, value in enumerate((c["lo"], c["hi"])):
                d = abs(self._x_of(value) - x)
                if d < best_d:
                    best, best_d = (name, k), d
        return best

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._drag = self._handle_at(event.pos().x())
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            self.setCursor(Qt.SizeHorCursor if self._handle_at(event.pos().x()) else Qt.ArrowCursor)
            return
        name, k = self._drag
        c = self._channels[name]
        value = min(max(self._value_of(event.pos().x()), self._x_range[0]), self._x_range[1])
        if k == 0:
            c["lo"] = min(value, c["hi"] - 1e-6)
        else:
            c["hi"] = max(value, c["lo"] + 1e-6)
        self.update()
        self.rangeChanged.emit(name, c["lo"], c["hi"])

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            name, _ = self._drag
            self._drag = None
            self.update()
            self.rangeCommitted.emit(name, *self.channel_range(name))


# -- channel panel -----------------------------------------------------------------------------


class _ChannelRow:
    """One channel's controls, addressing the stack by name (never a captured index)."""

    def __init__(self, panel: "ChannelPanel", name: str) -> None:
        self.panel, self.name = panel, name
        self.suspend = False
        self.widget = QFrame()
        self.widget.setFrameShape(QFrame.StyledPanel)
        grid = QGridLayout(self.widget)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        self.visible_check = QCheckBox(name)
        self.lut_picker = LutPicker(preview_size=(36, 14))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setPrefix("γ ")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(0, 100)
        self.opacity_spin.setSuffix("%")
        self.min_spin = QDoubleSpinBox()
        self.max_spin = QDoubleSpinBox()
        for spin, prefix in ((self.min_spin, "min "), (self.max_spin, "max ")):
            # 8/16-bit intensities. A ±1e9 range made Qt size each box for
            # "-1000000000.0" (330 px), pushing the dock into a horizontal scroll.
            spin.setRange(0, 65535)
            spin.setDecimals(1)
            spin.setPrefix(prefix)
        # Three short lines so a row fits the dock's width (no horizontal scroll):
        #   [✓ name]        [LUT      ]
        #   Opacity [slider      ] [ %]
        #   [min ] [max ] [γ ]
        for spin in (self.min_spin, self.max_spin, self.gamma_spin, self.opacity_spin):
            spin.setMinimumWidth(0)
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.lut_picker.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        grid.addWidget(self.visible_check, 0, 0, 1, 1)
        grid.addWidget(self.lut_picker, 0, 1, 1, 2)
        grid.addWidget(QLabel("Opacity"), 1, 0)
        grid.addWidget(self.opacity_slider, 1, 1)
        grid.addWidget(self.opacity_spin, 1, 2)
        grid.addWidget(self.min_spin, 2, 0)
        grid.addWidget(self.max_spin, 2, 1)
        grid.addWidget(self.gamma_spin, 2, 2)
        for col in range(3):
            grid.setColumnStretch(col, 1)

        self.visible_check.toggled.connect(lambda v: self._replace(self._layer().with_visible(v)))
        self.lut_picker.lutChanged.connect(lambda n: self._replace(self._layer().with_lut_name(n)))
        self.gamma_spin.valueChanged.connect(lambda v: self._replace(self._layer().with_gamma(v)))
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        self.opacity_spin.valueChanged.connect(self._on_opacity)
        self.min_spin.editingFinished.connect(self._on_minmax)
        self.max_spin.editingFinished.connect(self._on_minmax)

    def _layer(self) -> ImageLayer:
        return self.panel.stack.by_name(self.name)

    def _replace(self, layer: ImageLayer) -> None:
        if not self.suspend:
            self.panel.stack.replace_at(self.panel.stack.index_of(self.name), layer)

    def _on_opacity(self, value: int) -> None:
        if self.suspend:
            return
        self.suspend = True
        self.opacity_slider.setValue(value)
        self.opacity_spin.setValue(value)
        self.suspend = False
        self._replace(self._layer().with_opacity(value / 100.0))

    def _on_minmax(self) -> None:
        lo, hi = sorted((self.min_spin.value(), self.max_spin.value()))
        self._replace(self._layer().with_contrast_range((lo, hi)))

    def show_range(self, lo: float, hi: float) -> None:
        """Live feedback while a handle is dragged (no stack edit yet)."""
        self.suspend = True
        self.min_spin.setValue(lo)
        self.max_spin.setValue(hi)
        self.suspend = False

    def sync(self, layer: ImageLayer) -> None:
        self.suspend = True
        try:
            self.visible_check.setChecked(layer.visible)
            self.lut_picker.set_lut_name(layer.lut_name)
            self.gamma_spin.setValue(layer.gamma)
            self.opacity_slider.setValue(round(layer.opacity * 100))
            self.opacity_spin.setValue(round(layer.opacity * 100))
            lo, hi = layer.contrast_range or (0.0, 1.0)
            self.min_spin.setValue(lo)
            self.max_spin.setValue(hi)
            r, g, b = lut_color(layer.lut_name)
            self.visible_check.setStyleSheet(f"QCheckBox {{ color: rgb({r},{g},{b}); font-weight: bold; }}")
        finally:
            self.suspend = False


class ChannelPanel(QWidget):
    """The combined histogram above one compact row per channel. Construct
    with the LayerStack the compositor draws; edits flow through the stack.
    Rows are rebuilt only when the set of layers changes, never mid-drag."""

    def __init__(self, stack: LayerStack, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.stack = stack
        self.histogram = CombinedHistogram()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setAlignment(Qt.AlignTop)
        hint = QLabel("Drag a ▶ / ◀ handle to set that channel's min / max.")
        hint.setWordWrap(True)
        self._layout.addWidget(self.histogram)
        self._layout.addWidget(hint)
        self.rows: dict[str, _ChannelRow] = {}
        self._hist_cache: dict = {}   # id(layer.data) -> (counts, edges)
        self.histogram.rangeChanged.connect(lambda n, lo, hi: self.rows[n].show_range(lo, hi))
        self.histogram.rangeCommitted.connect(self._commit_range)
        self._batching = False
        stack.add_listener(self.refresh)
        self.refresh()

    def set_layers(self, layers: "list[ImageLayer]") -> None:
        """Replace every layer at once: one rebuild, not one per layer."""
        self._batching = True
        try:
            self.stack.clear()
            for layer in layers:
                self.stack.append(layer)
        finally:
            self._batching = False
        self.refresh()

    def refresh_histograms(self) -> None:
        """Recompute every histogram (e.g. once a lazy channel's slices are all in)."""
        self._hist_cache = {}
        self.refresh()

    def _commit_range(self, name: str, lo: float, hi: float) -> None:
        self.stack.replace_at(self.stack.index_of(name), self.stack.by_name(name).with_contrast_range((lo, hi)))

    def _histogram_of(self, layer: ImageLayer):
        key = id(layer.data)
        if key not in self._hist_cache:
            self._hist_cache[key] = volume_histogram(layer.data)
        return self._hist_cache[key]

    def refresh(self) -> None:
        if self._batching:
            return
        layers = [layer for layer in self.stack if isinstance(layer, ImageLayer)]
        names = [layer.name for layer in layers]
        if names != list(self.rows):
            # Delete replaced rows; never orphan them with setParent(None).
            # Adding a widget to a visible layout queues a show for the next
            # event-loop turn; a row orphaned before that turn (load appends
            # four layers, each a rebuild) was then shown as a top-level
            # window -- the stray per-channel pop-ups.
            for row in self.rows.values():
                self._layout.removeWidget(row.widget)
                row.widget.hide()
                row.widget.deleteLater()
            self.rows, self._hist_cache = {}, {}
            self.histogram.clear()
            for layer in layers:
                row = _ChannelRow(self, layer.name)
                self._layout.addWidget(row.widget)
                self.rows[layer.name] = row
        for layer in layers:
            self.rows[layer.name].sync(layer)
            counts, edges = self._histogram_of(layer)
            lo, hi = layer.contrast_range or (edges[0], edges[-1])
            self.histogram.set_channel(layer.name, counts, edges, lut_color(layer.lut_name), lo, hi, layer.visible)


# -- slider labels ---------------------------------------------------------------------------------


def label_slice_slider(slice_viewer, axis: str, spacing_of=None) -> QLabel:
    """Put an axis label ('Z', 'Y', 'X') above the slice viewer's slice scroll
    bar and a live 'index / last' readout (plus µm when `spacing_of()` gives
    the step along that axis) below it. Works through the viewer's public
    `z_scrollbar` only: the bar is moved into a small labelled column in the
    same place of its container's layout."""
    bar = slice_viewer.z_scrollbar
    container = bar.parentWidget()
    layout = container.layout()
    index = layout.indexOf(bar)
    column = QWidget(container)
    v = QVBoxLayout(column)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(1)
    name = QLabel(axis)
    name.setAlignment(Qt.AlignHCenter)
    font = QFont(name.font())
    font.setBold(True)
    name.setFont(font)
    readout = QLabel("")
    readout.setAlignment(Qt.AlignHCenter)
    small = QFont(readout.font())
    small.setPointSizeF(max(6.5, small.pointSizeF() - 2))
    readout.setFont(small)
    layout.removeWidget(bar)
    v.addWidget(name)
    v.addWidget(bar, 1)
    v.addWidget(readout)
    column.setToolTip(f"{axis} slice")
    layout.insertWidget(index, column)

    # The viewer moves the bar with its signals blocked (wheel, set_z), so
    # valueChanged misses those; the bar always repaints when it moves, so
    # the readout follows its paint events instead.
    watcher = _BarWatcher(bar, readout, spacing_of, parent=column)
    bar.installEventFilter(watcher)
    watcher.update_text()
    return readout


class _BarWatcher(QObject):
    def __init__(self, bar, readout: QLabel, spacing_of, parent=None) -> None:
        super().__init__(parent)
        self.bar, self.readout, self.spacing_of = bar, readout, spacing_of
        self._shown = None

    def update_text(self) -> None:
        value, last = self.bar.value(), self.bar.maximum()
        if (value, last) == self._shown:
            return
        self._shown = (value, last)
        step = self.spacing_of() if self.spacing_of else None
        text = f"{value}/{last}"
        if step:
            text += f"\n{value * step:.1f} µm"
        self.readout.setText(text)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 -- Qt override
        if event.type() == QEvent.Paint:
            self.update_text()
        return False
