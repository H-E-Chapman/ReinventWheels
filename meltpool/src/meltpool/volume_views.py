"""The movie as a (time, y, x) volume for the viewer's MIP, Ortho and 3D tabs.

The Slice tab shows one (averaged) frame. The other tabs need a volume; for a
movie the natural one is the frames stacked along time:

- MIP: the maximum over time of every pixel (for a vapour depression -- bright gas
  in darker metal -- its largest extent during the movie);
- Ortho: the stack cut along y and along x -- x-t and y-t kymographs, one row or
  column of pixels followed through time;
- 3D: a volume rendering with time as depth.

The stack is the movie as the analysis sees it (ROI, stabilisation) and is capped
at a memory budget (256 MB) by taking every Nth frame. In MIP / Ortho one frame step is
drawn as one pixel's length (times N); in 3D the time axis is shortened, if need be, to
the frame's larger side, so a long movie is not a needle seen end-on. Built lazily, the
first time one of those tabs is shown after the movie (or its ROI / stabilisation)
changed; while it builds, the log says so and the tabs keep the previous volume.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import movie as mv

MAX_BYTES = 256 * 2**20  # the stack's memory budget
TAB_SLICE, TAB_MIP, TAB_ORTHO, TAB_3D = 0, 1, 2, 3


def movie_stack(blob: dict, lit: "np.ndarray | None" = None, max_bytes: int = MAX_BYTES,
                progress=None, cancel=None) -> "tuple[np.ndarray, int, np.ndarray]":
    """(stack (t, y, x) in the movie's own dtype -- stabilised frames rounded back to it --,
    frame step, frame indices), within `max_bytes` by taking every Nth frame."""
    from vanua.core.cancellation import check_cancel

    movie = mv.Movie(blob)
    frames = np.flatnonzero(lit) if lit is not None and lit.any() else np.arange(movie.n)
    h, w = movie.shape
    dtype = np.dtype(np.float32) if blob.get("background") else np.dtype(blob["dtype"])
    step = max(1, int(np.ceil(len(frames) * h * w * dtype.itemsize / max_bytes)))
    picks = frames[::step]
    stack = np.empty((len(picks), h, w), dtype)
    info = np.iinfo(dtype) if dtype.kind in "ui" else None
    for k, i in enumerate(picks):
        if k % 50 == 0:
            check_cancel(cancel)
            if progress:
                progress(k / len(picks), f"stacking frame {i}")
        a = movie.raw(int(i))
        if info is not None and a.dtype != dtype:  # stabilised (Fourier-shifted) frames are float
            a = np.clip(np.rint(a), info.min, info.max)
        stack[k] = a
    movie.close()
    return stack, step, picks


class MovieVolumeViews:
    """Keeps the MIP / Ortho / 3D tabs of a MultiViewHost showing the current movie.

    `start(fn, label, on_success)` runs `fn(progress, cancel)` off the GUI thread (or
    inline) -- the app's own worker helper -- so building a large stack never freezes
    the window."""

    def __init__(self, viewer, start: Callable, log: Callable[[str], None] = lambda m: None, opacity=None) -> None:
        self.viewer = viewer
        self.opacity = opacity  # an OpacityPanel: which values are transparent in 3D
        self.volume_actor = None
        if opacity is not None:
            opacity.changed.connect(self._update_opacity)
        self._start = start
        self._log = log
        self._blob = self._frame = self._lit = None
        self._built_for = None
        self._building = False
        self.stack = None
        self.step = 1
        viewer.as_widget().currentChanged.connect(self._on_tab)

    def set_movie(self, blob: "dict | None", frame: "dict | None", lit: "np.ndarray | None" = None) -> None:
        """The movie (ROI and stabilisation included) the volume tabs should show."""
        self._blob, self._frame, self._lit = blob, frame, lit
        if self.viewer.as_widget().currentIndex() != TAB_SLICE:
            self.build()

    def _key(self):
        b = self._blob
        if b is None:
            return None
        shifts = b.get("shifts")
        return (b.get("path"), b.get("n"), tuple(b.get("crop") or ()), b.get("flip_vertical"), b.get("flip_horizontal"),
                str(sorted((b.get("background") or {}).items())),
                None if shifts is None else float(np.asarray(shifts).sum()))

    def _on_tab(self, index: int) -> None:
        if index != TAB_SLICE:
            self.build()
            self._fit_later(index)

    def _fit_later(self, index: int) -> None:
        """Fit the shown tab's views once Qt has laid it out: a view fitted while its tab is
        hidden has no size, and fitting that zooms in absurdly (a black view)."""
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(0, lambda: self._fit(index))

    def _fit(self, index: int) -> None:
        if self.stack is None:
            return
        v = self.viewer
        if index == TAB_MIP:
            v.mip_viewer.fit_to_view()
        elif index == TAB_ORTHO:
            for pane in (v.ortho_viewer.xy, v.ortho_viewer.xz, v.ortho_viewer.yz):
                pane.fit_to_view()

    def build(self) -> None:
        if self._blob is None or self._building or self._key() == self._built_for:
            return
        blob, lit, key = self._blob, self._lit, self._key()
        self._building = True

        def _job(progress, cancel):
            return movie_stack(blob, lit, progress=progress, cancel=cancel)

        def _done(result):
            self._building = False
            self.stack, self.step, picks = result
            self._built_for = key
            self._show()
            self._log(f"MIP / Ortho / 3D: {len(picks)} frames"
                      + (f" (every {self.step}th, to fit memory)" if self.step > 1 else ""))
            if self._key() != key and self.viewer.as_widget().currentIndex() != TAB_SLICE:
                self.build()  # the movie changed while this one was being built

        self._log("MIP / Ortho / 3D: building the volume"
                  + (" (running median on every frame: this can take a few minutes)" if blob.get("background") else "")
                  + "...")
        self._start(_job, "Volume", _done, self._failed)

    def _failed(self, message: str) -> None:
        self._building = False
        self._log(f"MIP / Ortho / 3D not built: {message}")

    def _spacing(self) -> "tuple[float, float, float]":
        from vanua.blocks._frame import as_frame

        _z, sy, sx = (float(v) for v in as_frame(self._frame).spacing()[:3])
        return (sx * self.step, sy, sx)  # one frame step drawn as one pixel's length (times the frame step)

    def _show(self) -> None:
        spacing = self._spacing()
        v = self.viewer
        v.mip_viewer.set_volume(self.stack, axis=0, spacing_um_zyx=spacing, percentile=(0.5, 99.5))
        v.ortho_viewer.set_volume(self.stack, spacing_um_zyx=spacing, percentile=(0.5, 99.5))
        v.ortho_viewer.set_axis_labels(xy=None, xz=(None, None, "time", None), yz=(None, None, "time", None))
        self._fit_later(v.as_widget().currentIndex())  # only the tab on screen has a size to fit to
        if self.opacity is not None:
            self.opacity.set_data(self.stack)
        if v.viewer_3d is not None:
            self._show_3d(spacing)

    def _spacing_3d(self, spacing) -> "tuple[float, float, float]":
        """Time no longer than the frame's larger side: 15000 frames of 900 pixels would
        otherwise be a needle 17 times longer than it is wide."""
        _n, h, w = self.stack.shape
        st, sy, sx = spacing
        return (min(st, max(h * sy, w * sx) / max(1, len(self.stack))), sy, sx)

    def _show_3d(self, spacing) -> None:
        from vanua.ui.viewers.vtk3d import build_volume_actor

        v3 = self.viewer.viewer_3d
        # rows run down in an image and y runs up in VTK: flip, so the frame is upright
        upright = self.stack[:, ::-1, :]
        spacing = self._spacing_3d(spacing)
        if self.opacity is not None:
            actor = build_volume_actor(upright, spacing, opacity_function=self.opacity.function())
        else:
            lo, hi = np.percentile(self.stack[:: max(1, len(self.stack) // 20)], (1.0, 99.5))
            actor = build_volume_actor(upright, spacing, opacity_range=(float(lo), float(hi)))
        prop = actor.GetProperty()
        prop.ShadeOff()  # shading speckles noisy X-ray frames and darkens them
        self._set_colour(prop)
        new_size = actor.GetBounds()
        fresh = v3.bounds is None or tuple(np.round(v3.bounds, 3)) != tuple(np.round(new_size, 3))
        if fresh:
            v3.bounds = None  # another stack size (ROI, flip): re-centre the camera on it
        v3.set_volume(actor)
        self.volume_actor = actor
        if fresh:
            self._oblique_camera()

    def _oblique_camera(self) -> None:
        """Frame upright, seen a little from the side and above, so time has depth."""
        v3 = self.viewer.viewer_3d
        cam = v3.renderer.GetActiveCamera()
        x0, x1, y0, y1, z0, z1 = v3.bounds
        centre = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        cam.SetFocalPoint(*centre)
        cam.SetPosition(centre[0], centre[1], centre[2] + 1.0)
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.Azimuth(-35.0)
        cam.Elevation(20.0)
        cam.OrthogonalizeViewUp()
        v3.renderer.ResetCamera()
        v3.render()

    def _set_colour(self, prop) -> None:
        """Grey from black to white across the values that are visible -- from where the
        opacity curve leaves 0 to where it first reaches its top -- not across the whole
        data range, where the values that are shown would all be dark grey."""
        from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

        lo = hi = None
        if self.opacity is not None:
            pts = [(p.value, p.opacity) for p in self.opacity.points()]
            top = max((o for _v, o in pts), default=0.0)
            if top > 0:
                first = next(k for k, (_v, o) in enumerate(pts) if o > 0)
                lo = pts[first - 1][0] if first > 0 else pts[first][0]
                hi = next(v for v, o in pts if o >= top)
        if lo is None or not hi > lo:
            lo, hi = (float(v) for v in np.percentile(self.stack[:: max(1, len(self.stack) // 20)], (1.0, 99.5)))
            hi = hi if hi > lo else lo + 1.0
        colour = vtkColorTransferFunction()
        colour.AddRGBPoint(lo, 0.0, 0.0, 0.0)
        colour.AddRGBPoint(hi, 1.0, 1.0, 1.0)
        prop.SetColor(colour)

    def _update_opacity(self) -> None:
        """Only the transfer function changes: no rebuild of the volume."""
        v3 = self.viewer.viewer_3d
        if v3 is None or self.volume_actor is None or self.opacity is None:
            return
        prop = self.volume_actor.GetProperty()
        prop.SetScalarOpacity(self.opacity.function())
        self._set_colour(prop)
        v3.render()
