"""Lazy display data: show the slice being looked at now, load the rest in the background.

The ImageJ TIFFs store every z-slice of every channel as its own page, so
one slice of one channel can be read on its own in milliseconds.

* :class:`TiffStack` -- one timepoint file, opened once, shared by its channels.
* :class:`LazyChannel` -- one channel as a ZYX array (vanua's
  :class:`~vanua.core.readers._lazy.LazyPlaneVolume`) whose slices are read on
  first use and cached; :meth:`LazyChannel.preload` fills in every other slice
  (nearest to the current one first) -- on a background thread in the GUI.
* :class:`LazyOutlines`, :class:`LazyPositions` -- a timepoint's cell outlines
  and badge positions, computed slice by slice as they are viewed.

The analysis (segmentation, series) still reads whole volumes in its workers;
only the Display is lazy. Qt-free.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from vanua.core.readers._lazy import LazyPlaneVolume

__all__ = ["TiffStack", "LazyChannel", "LazyOutlines", "LazyPositions", "open_lazy_channels",
           "sample_for_contrast"]


class TiffStack:
    """One ZCYX (or CZYX) ImageJ TIFF; pages are read under a lock, so the
    display thread and a background preload can share the file."""

    def __init__(self, path) -> None:
        import tifffile

        self.path = Path(path)
        self._tif = tifffile.TiffFile(self.path)
        series = self._tif.series[0]
        axes = series.axes.upper()
        if "T" in axes:
            raise ValueError(f"{self.path.name}: one timepoint per file expected (axes {axes})")
        grid_axes = [a for a in axes if a not in "YX"]
        self.grid_shape = tuple(series.shape[axes.index(a)] for a in grid_axes)
        self.grid_axes = "".join(grid_axes)       # e.g. "ZC": the order of the pages
        self.n_z = series.shape[axes.index("Z")] if "Z" in axes else 1
        self.n_c = series.shape[axes.index("C")] if "C" in axes else 1
        self.shape_yx = (series.shape[axes.index("Y")], series.shape[axes.index("X")])
        self.dtype = series.dtype
        self._pages = series.pages
        self._lock = threading.Lock()

    def page_index(self, z: int, c: int) -> int:
        coords = {"Z": z, "C": c}
        return int(np.ravel_multi_index(tuple(coords[a] for a in self.grid_axes), self.grid_shape))

    def read(self, z: int, c: int) -> np.ndarray:
        with self._lock:
            return self._pages[self.page_index(z, c)].asarray()

    def close(self) -> None:
        with self._lock:
            self._tif.close()


class LazyChannel(LazyPlaneVolume):
    """One channel: a ZYX array read slice by slice, every slice cached once read."""

    def __init__(self, stack: TiffStack, channel: int) -> None:
        self.stack, self.channel = stack, channel
        self._cache: dict = {}
        self._cache_lock = threading.Lock()
        super().__init__((stack.n_z, *stack.shape_yx), stack.dtype, self._plane)

    def _plane(self, z: int) -> np.ndarray:
        with self._cache_lock:
            plane = self._cache.get(z)
        if plane is None:
            plane = self.stack.read(z, self.channel)
            with self._cache_lock:
                self._cache[z] = plane
        return plane

    @property
    def loaded(self) -> int:
        return len(self._cache)

    @property
    def complete(self) -> bool:
        return len(self._cache) == self.shape[0]

    def preload(self, start: int = 0, cancel=None) -> None:
        """Read every slice not read yet, nearest to `start` first."""
        n = self.shape[0]
        for z in sorted(range(n), key=lambda k: (abs(k - start), k)):
            if cancel is not None and cancel.is_set():
                return
            self._plane(z)

    def projection(self) -> np.ndarray:
        """Maximum along z over the slices read so far (all of them once
        :attr:`complete`): the MIP, without forcing the rest to be read now."""
        with self._cache_lock:
            planes = list(self._cache.values())
        if not planes:
            return np.zeros(self.shape[1:], self.dtype)
        out = planes[0].copy()
        for plane in planes[1:]:
            np.maximum(out, plane, out=out)
        return out

    def loaded_sample(self, step: int = 4) -> np.ndarray:
        """The slices read so far, every `step`-th pixel: for histograms and the
        display min / max before the whole stack is in."""
        with self._cache_lock:
            planes = [p[::step, ::step] for _z, p in sorted(self._cache.items())]
        return np.stack(planes) if planes else np.zeros((0, 1, 1), self.dtype)

    def __array__(self, dtype=None, copy=None):  # numpy conversion: the whole stack
        out = np.stack([self._plane(z) for z in range(self.shape[0])])
        return out.astype(dtype) if dtype is not None else out


def tiff_frame(path):
    """The file's WorldFrame from its ImageJ tags alone (no pixels read) --
    the same rules as vanua's TIFF reader (vanua.blocks.io._read_tiff): z from
    the ImageJ 'spacing', y / x from the resolution tags, the unit normalised to µm."""
    import tifffile

    from vanua.core.datastore import WorldFrame

    with tifffile.TiffFile(path) as tif:
        meta = tif.imagej_metadata or {}
        page = tif.series[0].pages[0]
        res = page.tags.get("XResolution"), page.tags.get("YResolution")

    def _pixel(tag) -> float:
        if tag is None:
            return 1.0
        num, den = tag.value
        return den / num if num else 1.0

    sz = float(meta.get("spacing", 1.0))
    sy, sx = _pixel(res[1]), _pixel(res[0])
    unit = str(meta.get("unit", "um")).replace("\\u00B5", "u").replace("µ", "u").replace("micron", "um")
    return WorldFrame.from_spacing((sz, sy, sx), unit=unit or "um")


def open_lazy_channels(path, channel_index: dict, first_z: "int | None" = None) -> dict:
    """{name: LazyChannel} for one timepoint file; reads slice `first_z` of every
    channel now (the one shown first), nothing else."""
    stack = TiffStack(path)
    channels = {name: LazyChannel(stack, c) for name, c in channel_index.items()}
    z = stack.n_z // 2 if first_z is None else min(max(int(first_z), 0), stack.n_z - 1)
    for ch in channels.values():
        ch[z]
    return channels


def sample_for_contrast(channel: LazyChannel, z: int, around: int = 3) -> np.ndarray:
    """The current slice and `around` on each side (read now if needed): a quick
    stand-in for the whole stack when choosing the first display min / max."""
    n = channel.shape[0]
    for k in range(max(z - around, 0), min(z + around + 1, n)):
        channel[k]
    return channel.loaded_sample()


class LazyOutlines:
    """A label image's outlines, one slice at a time (find_boundaries on the
    slice asked for), cached: ZYX-indexable, so the slice viewer can show it."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels
        self.shape, self.dtype, self.ndim = labels.shape, labels.dtype, 3
        self._cache: dict = {}

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            z = int(key)
            if z not in self._cache:
                from skimage.segmentation import find_boundaries

                plane = self.labels[z]
                out = np.zeros_like(plane)
                edge = find_boundaries(plane, mode="inner")
                out[edge] = plane[edge]
                self._cache[z] = out
            return self._cache[z]
        return np.stack([self[z] for z in range(self.shape[0])])[key[1:] if isinstance(key, tuple) else slice(None)]


    def __array__(self, dtype=None, copy=None):  # numpy conversion: every slice
        out = np.stack([self[z] for z in range(self.shape[0])])
        return out.astype(dtype) if dtype is not None else out


class LazyPositions:
    """{z: [(label, y, x), ...]} computed per slice on first use (the cell badges)."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels
        self._cache: dict = {}

    def get(self, z, default=()):
        z = int(z)
        if not 0 <= z < self.labels.shape[0]:
            return default
        if z not in self._cache:
            from scipy import ndimage

            plane = self.labels[z]
            ids = np.unique(plane)
            ids = ids[ids > 0]
            centres = ndimage.center_of_mass(np.ones_like(plane, dtype=np.uint8), plane, ids) if len(ids) else []
            self._cache[z] = [(int(i), float(cy), float(cx)) for i, (cy, cx) in zip(ids, centres)]
        return self._cache[z] or default

    def __getitem__(self, z):
        return self.get(z, [])

    def items(self):
        return ((z, self.get(z, [])) for z in range(self.labels.shape[0]))
