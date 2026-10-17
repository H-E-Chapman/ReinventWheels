"""Time-lapse access and the image algorithms behind MeltPool, Qt-free.

A movie is described by a plain dict (the ``movie`` blob) so it can travel
between recipe steps. :func:`describe_movie` picks the reader from the
file's *content* (magic bytes), not its extension -- an ImageJ TIFF saved as
``.avi`` is still a TIFF:

- TIFF, one page per frame, memory-mapped when the pages are uncompressed and
  evenly spaced (a 12 GB movie never has to fit in memory);
- HDF5 / NeXus: one 3D dataset, axis 0 = time (h5py, read lazily);
- NumPy ``.npy``: a 3D array, memory-mapped;
- video (AVI, MP4, MOV, MKV ...): decoded with PyAV, colour -> grey, exact
  frame seeking by timestamp and a cache of decoded frames;
- an in-memory ``(T, Y, X)`` array (synthetic movies, tests).

``flip_vertical`` is applied at read time, so
every later step works with the build direction up: air above, material
below, row index increasing downwards into the substrate.

The algorithms on the frames: sub-pixel stabilisation (:func:`estimate_shifts`,
Fourier shifts), the running-median (transient) frames, the original surface
and the depression centre, the dark-rim trace of the vapour depression
(:func:`rim_trace`; the three-line tracer is in ``boundary.py``), the temporal
median and the liquid-solid search on it. :func:`estimate_velocity` (the sample
drift) checks that "jitter" stabilisation keeps a steady drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["Movie", "describe_movie", "sniff", "MOVIE_SUFFIXES", "describe_tiff", "describe_hdf5", "describe_npy",
           "describe_video", "describe_array", "crop_blob", "crop_offset", "xshift", "frame_means", "estimate_velocity",
           "estimate_shifts", "fourier_shift_frame", "original_surface", "depression_centre", "rim_trace",
           "region_below", "median_image", "band_pass", "solid_line_auto"]


# -- reading ---------------------------------------------------------------------

#: Extensions offered in file dialogs and picked up by the batch runner. The
#: reader itself is chosen from the content (:func:`sniff`), not from these.
MOVIE_SUFFIXES = (".tif", ".tiff", ".h5", ".hdf5", ".hdf", ".he5", ".nxs", ".npy",
                  ".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".webm")

_MAGIC = ((b"II*\x00", "tiff"), (b"MM\x00*", "tiff"), (b"II+\x00", "tiff"), (b"MM\x00+", "tiff"),
          (b"\x89HDF\r\n\x1a\n", "hdf5"), (b"\x93NUMPY", "npy"))


def sniff(path: "str | Path") -> str:
    """'tiff' | 'hdf5' | 'npy' | 'video', from the file's first bytes (HDF5
    may sit behind a user block at 512, 1024, ... bytes)."""
    path = Path(path)
    with path.open("rb") as handle:
        head = handle.read(8)
        for magic, kind in _MAGIC:
            if head.startswith(magic):
                return kind
        offset = 512
        size = path.stat().st_size
        while offset < min(size, 1 << 20):
            handle.seek(offset)
            if handle.read(8) == b"\x89HDF\r\n\x1a\n":
                return "hdf5"
            offset *= 2
    return "video"


def describe_movie(path: "str | Path", flip_vertical: bool = True, dataset: str = "", log=print) -> dict:
    """A movie blob for any supported file. ``blob["file_meta"]`` holds what the
    file says about calibration (``pixel_um``; ``fps_in_file`` -- often a
    playback rate, so it is reported, never used)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    kind = sniff(path)
    expected = {".tif": "tiff", ".tiff": "tiff", ".h5": "hdf5", ".hdf5": "hdf5", ".hdf": "hdf5", ".he5": "hdf5",
                ".nxs": "hdf5", ".npy": "npy"}.get(path.suffix.lower(), "video")
    if kind != expected:
        log(f"{path.name}: the content is {kind.upper()}, not what the extension suggests; reading it as {kind}")
    if kind == "tiff":
        return describe_tiff(path, flip_vertical)
    if kind == "hdf5":
        return describe_hdf5(path, flip_vertical, dataset, log)
    if kind == "npy":
        return describe_npy(path, flip_vertical)
    return describe_video(path, flip_vertical, log)


def _imagej_pixel_um(tif) -> "float | None":
    meta = tif.imagej_metadata or {}
    unit = str(meta.get("unit", "")).replace("\\u00B5", "u").replace("µ", "u").lower()
    tag = tif.pages[0].tags.get("XResolution")
    if tag is None or unit not in ("um", "micron", "microns"):
        return None
    num, den = tag.value
    return float(den / num) if num else None


def describe_tiff(path: "str | Path", flip_vertical: bool = True) -> dict:
    """A movie blob for a TIFF with one 2D page per frame."""
    import tifffile

    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        n = len(pages)
        first = pages[0]
        if first.ndim != 2:
            raise ValueError(f"{path.name}: pages are {first.shape}; expected one 2D frame per page")
        shape = tuple(int(s) for s in first.shape)
        dtype = np.dtype(first.dtype).str
        memmap = None
        if n > 1 and first.is_contiguous and pages[1].is_contiguous:
            off0 = int(first.dataoffsets[0])
            step = int(pages[1].dataoffsets[0]) - off0
            last = pages[n - 1]
            if (last.shape == first.shape and last.is_contiguous
                    and int(last.dataoffsets[0]) == off0 + (n - 1) * step):
                memmap = {"offset": off0, "step": step}
        meta = {"format": "TIFF" + (" (ImageJ)" if tif.is_imagej else "")}
        pixel = _imagej_pixel_um(tif)
        if pixel:
            meta["pixel_um"] = pixel
        fps = (tif.imagej_metadata or {}).get("fps")
        if fps:
            meta["fps_in_file"] = float(fps)
    return {"kind": "tiff", "path": str(path), "n": n, "shape": shape, "dtype": dtype,
            "memmap": memmap, "flip_vertical": bool(flip_vertical), "file_meta": meta}


def _h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError("reading HDF5 needs h5py: pip install h5py") from None
    return h5py


def describe_hdf5(path: "str | Path", flip_vertical: bool = True, dataset: str = "", log=print) -> dict:
    """One 3D dataset, axis 0 = time. `dataset` empty: the largest 3D dataset
    in the file (named in the log, so a wrong pick is visible)."""
    h5py = _h5py()
    path = Path(path)
    with h5py.File(path, "r") as f:
        if not dataset:
            found: list = []
            f.visititems(lambda name, obj: found.append((obj.size, name))
                         if isinstance(obj, h5py.Dataset) and obj.ndim == 3 else None)
            if not found:
                raise ValueError(f"{path.name}: no 3D dataset (time, y, x) in the file; name one with 'dataset'")
            dataset = max(found)[1]
            if len(found) > 1:
                log(f"{path.name}: {len(found)} 3D datasets; using the largest, /{dataset}")
        if dataset not in f or not isinstance(f[dataset], h5py.Dataset):
            raise ValueError(f"{path.name}: no dataset {dataset!r}")
        d = f[dataset]
        if d.ndim != 3:
            raise ValueError(f"{path.name}: /{dataset} is {d.shape}; expected (time, y, x)")
        meta = {"format": "HDF5", "dataset": dataset, "attrs": sorted(d.attrs.keys())[:20]}
        n, shape, dtype = int(d.shape[0]), (int(d.shape[1]), int(d.shape[2])), d.dtype.str
    return {"kind": "hdf5", "path": str(path), "dataset": dataset, "n": n, "shape": shape, "dtype": dtype,
            "flip_vertical": bool(flip_vertical), "file_meta": meta}


def describe_npy(path: "str | Path", flip_vertical: bool = True) -> dict:
    path = Path(path)
    a = np.load(path, mmap_mode="r")
    if a.ndim != 3:
        raise ValueError(f"{path.name}: array is {a.shape}; expected (time, y, x)")
    return {"kind": "npy", "path": str(path), "n": int(a.shape[0]), "shape": (int(a.shape[1]), int(a.shape[2])),
            "dtype": a.dtype.str, "flip_vertical": bool(flip_vertical), "file_meta": {"format": "NumPy"}}


def _av():
    try:
        import av
    except ImportError:
        raise ImportError("reading video needs PyAV: pip install av") from None
    return av


#: Pixel formats with more than 8 bits of grey: decoded as gray16 instead of gray.
_DEEP = ("gray10", "gray12", "gray14", "gray16", "yuv420p10", "yuv422p10", "yuv444p10", "yuv420p12", "rgb48", "bgr48")


def describe_video(path: "str | Path", flip_vertical: bool = True, log=print) -> dict:
    """Frame count and every frame's timestamp, from the packets (no decoding),
    so any frame can be sought exactly later."""
    av = _av()
    path = Path(path)
    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001 -- an unreadable file is a clear error, not a traceback
        raise ValueError(f"{path.name}: not a readable movie ({exc})") from None
    with container:
        if not container.streams.video:
            raise ValueError(f"{path.name}: no video stream")
        stream = container.streams.video[0]
        pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None and p.size)
        if not pts:
            raise ValueError(f"{path.name}: no frames")
        cc = stream.codec_context
        pix = (cc.pix_fmt or "") if hasattr(cc, "pix_fmt") else ""
        deep = any(pix.startswith(p) for p in _DEEP)
        rate = float(stream.average_rate) if stream.average_rate else None
        meta = {"format": f"video ({cc.name}, {pix or '?'})", "fps_in_file": rate}
        if not deep:
            log(f"{path.name}: {cc.name} {pix}: 8-bit grey levels (a 16-bit original has been reduced)")
    lut = None
    if pix.startswith("pal8"):
        lut, grey = _palette_lut(path)
        meta["palette"] = "grey ramp" if grey else "colour palette -> luma"
        if not grey:
            log(f"{path.name}: palette video with a colour palette; converted to grey (luma)")
    elif not pix.startswith("gray") and pix:
        log(f"{path.name}: colour video ({pix}); converted to grey (luma)")
    return {"kind": "video", "path": str(path), "n": len(pts), "shape": (int(cc.height), int(cc.width)),
            "dtype": np.dtype(np.uint16 if deep else np.uint8).str, "pts": pts, "gray16": deep,
            "palette_lut": lut, "flip_vertical": bool(flip_vertical), "file_meta": meta}


def _palette_lut(path: Path) -> "tuple[list[int], bool]":
    """Grey value of each palette entry of a palette ('pal8') video.

    The palette travels with the first packet only, so it is read here, from a
    straight decode of the first frame; a decoder that has sought elsewhere
    has an all-zero palette (every pixel black). PyAV gives the palette as
    (A, R, G, B) rows."""
    av = _av()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        _indices, palette = frame.to_ndarray()
    palette = np.asarray(palette, dtype=np.float64)
    alpha_first = bool((palette[:, 0] == 255).all()) and not bool((palette[:, 3] == 255).all())
    r, g, b = (palette[:, 1], palette[:, 2], palette[:, 3]) if alpha_first else (palette[:, 2], palette[:, 1], palette[:, 0])
    grey = np.clip(np.rint(0.299 * r + 0.587 * g + 0.114 * b), 0, 255).astype(np.uint8)
    if not palette[:, 1:].any() and not palette[:, :3].any():
        raise ValueError(f"{path.name}: the palette of the first frame is empty")
    return grey.tolist(), bool(np.array_equal(r, g) and np.array_equal(g, b))


class _VideoReader:
    """Random access into a video: sequential decoding when frames are asked
    for in order (the sliding window), a seek to the nearest key frame before
    a jump, and an LRU cache of decoded frames."""

    def __init__(self, blob: Mapping[str, Any], cache: int = 256) -> None:
        from collections import OrderedDict

        av = _av()
        self.container = av.open(blob["path"])
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.pts = list(blob["pts"])
        self.index_of = {p: i for i, p in enumerate(self.pts)}
        self.fmt = "gray16le" if blob.get("gray16") else "gray"
        lut = blob.get("palette_lut")
        self.lut = np.asarray(lut, dtype=np.uint8) if lut else None
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.size = cache
        self._frames = None
        self._next = None  # index the decoder will produce next

    def _put(self, i: int, a: np.ndarray) -> None:
        self.cache[i] = a
        self.cache.move_to_end(i)
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)

    def get(self, i: int) -> np.ndarray:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        if self._frames is None or self._next is None or not (self._next <= i <= self._next + 64):
            self.container.seek(int(self.pts[i]), stream=self.stream, backward=True, any_frame=False)
            self._frames = self.container.decode(self.stream)
            self._next = None
        for frame in self._frames:
            k = self.index_of.get(frame.pts)
            if k is None:
                continue
            if self.lut is not None:
                indices, _palette = frame.to_ndarray()  # the frame's own palette is zero after a seek
                a = self.lut[indices]
            else:
                a = frame.to_ndarray(format=self.fmt)
            self._put(k, a)
            self._next = k + 1
            if k >= i:
                break
        else:
            self._frames, self._next = None, None  # decoder exhausted: seek next time
        if i not in self.cache:
            raise IndexError(f"frame {i} could not be decoded")
        return self.cache[i]

    def close(self) -> None:
        self.container.close()


def describe_array(array: np.ndarray, flip_vertical: bool = False) -> dict:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) array, got {array.shape}")
    return {"kind": "array", "array": array, "n": int(array.shape[0]), "shape": tuple(array.shape[1:]),
            "dtype": array.dtype.str, "flip_vertical": bool(flip_vertical)}


def crop_offset(blob: Mapping[str, Any]) -> "tuple[int, int]":
    """(row, column) of the movie's first pixel in the full (flipped) frame."""
    crop = blob.get("crop")
    return (int(crop[0]), int(crop[2])) if crop else (0, 0)


def crop_blob(blob: Mapping[str, Any], y0: int, y1: int, x0: int, x1: int) -> dict:
    """The movie restricted to rows y0:y1, columns x0:x1 of the full frame as
    displayed (after the flip). 0 for y1/x1 means "to the end". Not composable
    with an earlier crop: bounds are always in full-frame pixels."""
    h, w = (int(v) for v in blob["shape"])
    y1, x1 = (y1 or h), (x1 or w)
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError(f"crop rows {y0}:{y1}, columns {x0}:{x1} outside the {h} x {w} frame or empty")
    out = dict(blob)
    out["crop"] = [int(y0), int(y1), int(x0), int(x1)]
    return out


class Movie:
    """Random access to frames of a movie blob, as float32, flipped (and
    cropped) if the blob says so. `shape` is the shape returned."""

    def __init__(self, blob: Mapping[str, Any]) -> None:
        self.blob = dict(blob)
        self.n = int(blob["n"])
        self.full_shape = tuple(blob["shape"])
        crop = blob.get("crop")
        self._crop = (slice(crop[0], crop[1]), slice(crop[2], crop[3])) if crop else None
        self.shape = (crop[1] - crop[0], crop[3] - crop[2]) if crop else self.full_shape
        self.flip = bool(blob.get("flip_vertical", False))
        self.flip_h = bool(blob.get("flip_horizontal", False))
        shifts = blob.get("shifts")
        self._shifts = np.asarray(shifts, dtype=np.float64) if shifts is not None else None
        self._bg = dict(blob["background"]) if blob.get("background") else None
        self._bg_cache: "dict[int, np.ndarray]" = {}
        # with a running median, consecutive medians share most of their frames: keep the
        # last ones read (decoded, shifted, cropped) instead of reading them again
        self._base_cache: "dict[int, np.ndarray]" = {}
        self._base_keep = 0
        if self._bg is not None:
            frame_bytes = max(1, int(np.prod(self.shape)) * 4)
            want = int(self._bg.get("window", 50)) + int(self._bg.get("every", 5)) + 2
            self._base_keep = max(2, min(want, (128 * 2**20) // frame_bytes))
        self._raw = None
        self._tif = None
        self._array = None
        self._h5 = None
        self._video = None
        kind = blob["kind"]
        if kind == "array":
            self._array = np.asarray(blob["array"])
        elif kind == "npy":
            self._array = np.load(blob["path"], mmap_mode="r")
        elif kind == "hdf5":
            self._h5 = _h5py().File(blob["path"], "r")
            self._array = self._h5[blob["dataset"]]  # h5py slices lazily, one frame at a time
        elif kind == "video":
            self._video = _VideoReader(blob)
        elif kind == "tiff":
            if blob.get("memmap"):
                self._raw = np.memmap(blob["path"], dtype=np.uint8, mode="r")
                self._dtype = np.dtype(blob["dtype"])
                self._nbytes = int(np.prod(self.full_shape)) * self._dtype.itemsize
            else:
                import tifffile

                self._tif = tifffile.TiffFile(blob["path"])
        else:
            raise ValueError(f"unknown movie kind {blob['kind']!r}")

    def raw(self, i: int) -> np.ndarray:
        """Frame `i` as the analysis sees it: flipped, stabilised, cropped and -- if the
        blob says so -- with the running median subtracted (the transient part)."""
        if self._bg is not None:
            return self.transient(i)
        return self._base(i)

    def background(self, i: int) -> np.ndarray:
        """The running median for frame `i` (recomputed every `every` frames, from at most
        `max_frames` frames spread over the window)."""
        bg = self._bg
        window, every = max(1, int(bg.get("window", 50))), max(1, int(bg.get("every", 5)))
        anchor = (i // every) * every
        if anchor not in self._bg_cache:
            if bg.get("mode", "centred") == "trailing":
                lo, hi = anchor - window, anchor
            else:
                lo, hi = anchor - window // 2, anchor + window // 2 + 1
            lo, hi = max(0, lo), min(self.n, hi)
            if hi <= lo:  # the first frames of a trailing window: use what follows
                lo, hi = anchor + 1, min(self.n, anchor + 1 + window)
            ks = np.arange(lo, hi)
            ks = ks[np.linspace(0, len(ks) - 1, min(len(ks), int(bg.get("max_frames", 25)))).astype(int)]
            self._bg_cache[anchor] = np.median(np.stack([self._base(int(k)).astype(np.float32) for k in ks]), axis=0)
            while len(self._bg_cache) > 6:
                self._bg_cache.pop(next(iter(self._bg_cache)))
        return self._bg_cache[anchor]

    def transient(self, i: int) -> np.ndarray:
        """Frame minus its running median ('subtract', grey levels) or frame / median - 1
        ('divide', relative change)."""
        base = self._base(i).astype(np.float32)
        med = self.background(i)
        if self._bg.get("operation", "subtract") == "divide":
            return base / np.maximum(med, 1e-6) - 1.0
        return base - med

    def _base(self, i: int) -> np.ndarray:
        if not 0 <= i < self.n:
            raise IndexError(f"frame {i} outside 0..{self.n - 1}")
        if not self._base_keep:
            return self._read_base(i)
        a = self._base_cache.pop(i, None)
        if a is None:
            a = self._read_base(i)
            a.flags.writeable = False  # shared between calls
        self._base_cache[i] = a  # most recently used last
        while len(self._base_cache) > self._base_keep:
            self._base_cache.pop(next(iter(self._base_cache)))
        return a

    def _read_base(self, i: int) -> np.ndarray:
        if self._array is not None:
            a = np.asarray(self._array[i])
        elif self._video is not None:
            a = self._video.get(i)
        elif self._raw is not None:
            o = self.blob["memmap"]["offset"] + i * self.blob["memmap"]["step"]
            a = np.frombuffer(self._raw[o:o + self._nbytes], dtype=self._dtype).reshape(self.full_shape)
        else:
            a = self._tif.pages[i].asarray()
        if self.flip:
            a = a[::-1]
        if self.flip_h:
            a = a[:, ::-1]
        if self._shifts is not None:
            dy, dx = self._shifts[i]
            if dy or dx:
                a = fourier_shift_frame(a, dy, dx)
        return a[self._crop] if self._crop else a

    def frame(self, i: int) -> np.ndarray:
        return self.raw(i).astype(np.float32)

    def mean_of(self, start: int, count: int) -> np.ndarray:
        stop = min(self.n, start + count)
        acc = np.zeros(self.shape, np.float64)
        for k in range(start, stop):
            acc += self.raw(k)
        return (acc / max(1, stop - start)).astype(np.float32)

    def close(self) -> None:
        if self._tif is not None:
            self._tif.close()
        if self._h5 is not None:
            self._h5.close()
        if self._video is not None:
            self._video.close()


# -- sub-pixel stabilisation ------------------------------------------------------------

_PAD = 8


def fourier_shift_frame(a: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift a frame by (dy, dx) px with a Fourier phase ramp: exact for sub-pixel
    shifts, no interpolation blur. Mirrored padding keeps the edges from wrapping."""
    from scipy import ndimage as ndi

    p = np.pad(np.asarray(a, dtype=np.float32), _PAD, mode="reflect")
    out = np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(p), (dy, dx))).real
    return out[_PAD:-_PAD, _PAD:-_PAD].astype(np.float32)


def _register_view(a: np.ndarray, region, sigmas) -> np.ndarray:
    """The registration signal: texture of the region with slow shading (beam
    profile) removed, Hann-windowed so the region's borders do not register."""
    from scipy import ndimage as ndi
    from skimage.filters import window

    sub = np.asarray(a, dtype=np.float32)[region]
    hp = ndi.gaussian_filter(sub, sigmas[0]) - ndi.gaussian_filter(sub, sigmas[1])
    return hp * window("hann", hp.shape)


def estimate_shifts(movie: Movie, lit: np.ndarray, region: "tuple[slice, slice]", *, mode: str = "jitter",
                    n_ref: int = 50, upsample: int = 100, sigmas: "tuple[float, float]" = (1.0, 12.0),
                    detrend: int = 51, progress=None, cancel=None, log=print) -> np.ndarray:
    """Per-frame (dy, dx) corrections that register every frame, to `upsample`-th
    of a pixel, on the texture inside `region` (rows, cols of the displayed frame).

    mode "fixed":  each frame against the median of `n_ref` frames from the middle
                   of the lit run -- removes all motion (a static sample).
    mode "jitter": consecutive frames, summed, minus a running mean over `detrend`
                   frames -- removes shake, keeps a steady drift (a moving sample).
    Blank frames get (0, 0)."""
    from scipy import ndimage as ndi
    from skimage.registration import phase_cross_correlation
    from vanua.core.cancellation import check_cancel

    idx = np.flatnonzero(lit)
    out = np.zeros((movie.n, 2), np.float64)
    if len(idx) < 3:
        return out

    def register(ref_view, i):
        s, _err, _ = phase_cross_correlation(ref_view, _register_view(movie.raw(i), region, sigmas),
                                             upsample_factor=upsample, normalization=None)
        return s

    if mode == "fixed":
        mid = idx[len(idx) // 2]
        picks = idx[max(0, np.searchsorted(idx, mid) - n_ref // 2):][:n_ref]
        ref = np.median(np.stack([movie.frame(k) for k in picks]), axis=0)
        ref_view = _register_view(ref, region, sigmas)
        for j, i in enumerate(idx):
            if j % 100 == 0:
                check_cancel(cancel)
                if progress:
                    progress(j / len(idx), f"registering frame {i}")
            out[i] = register(ref_view, i)
    elif mode == "jitter":
        # Each frame against the mean of its neighbours, shifted to cancel the steady
        # drift: the neighbours' shake averages out, so nothing accumulates. The
        # camera-fixed pattern (detector, flat field) is divided out first -- it does
        # not move with the sample and would pull every shift towards zero.
        from skimage.filters import window

        picks = idx[np.linspace(0, len(idx) - 1, min(len(idx), 100)).astype(int)]
        static = np.median(np.stack([movie.frame(int(k)) for k in picks]), axis=0)
        np.maximum(static, 1.0, out=static)
        drift = estimate_velocity(movie, lit, lag=min(40, max(2, len(idx) // 4)), samples=6, log=lambda m: None)
        u = float(drift["px_per_frame"])
        if abs(u) * len(idx) < 20.0:
            # The sample hardly travels: the temporal median *is* the scene, so dividing
            # by it would leave only noise. Register the frames as they are.
            static = np.ones_like(static)
        hann = window("hann", (region[0].stop - region[0].start, region[1].stop - region[1].start))
        half = max(2, detrend // 6)

        def texture(i):
            a = movie.frame(int(i)) / static
            sub = a[region]
            return ndi.gaussian_filter(sub, sigmas[0]) - ndi.gaussian_filter(sub, sigmas[1])

        cache: dict = {}
        for j, i in enumerate(idx):
            if j % 100 == 0:
                check_cancel(cancel)
                if progress:
                    progress(j / len(idx), f"registering frame {i}")
            nb = [k for k in idx[max(0, j - half): j + half + 1] if k != i]
            for k in list(cache):
                if k < nb[0] if nb else True:
                    del cache[k]
            for k in nb + [i]:
                if k not in cache:
                    cache[k] = texture(k)
            ref = np.mean([xshift(cache[k], u * (i - k)) for k in nb], axis=0)
            out[i], _err, _ = phase_cross_correlation(ref * hann, cache[i] * hann, upsample_factor=upsample,
                                                      normalization=None)
        out[idx] -= np.median(out[idx], axis=0)    # shake only, centred; the steady drift u is kept
        log(f"stabilisation: steady drift {u:+.4f} px/frame kept")
    else:
        raise ValueError(f"unknown stabilisation mode {mode!r} (fixed | jitter)")
    log(f"stabilisation ({mode}): dy sd {out[idx, 0].std():.3f} px (range {out[idx, 0].min():+.2f}..{out[idx, 0].max():+.2f}), "
        f"dx sd {out[idx, 1].std():.3f} px (range {out[idx, 1].min():+.2f}..{out[idx, 1].max():+.2f})")
    return out


def frame_means(movie: Movie, progress=None, cancel=None, subsample: int = 8) -> np.ndarray:
    from vanua.core.cancellation import check_cancel

    means = np.empty(movie.n, np.float64)
    for i in range(movie.n):
        if i % 256 == 0:
            check_cancel(cancel)
            if progress:
                progress(i / movie.n, f"frame {i}/{movie.n}")
        means[i] = movie.raw(i)[::subsample, ::subsample].mean()
    return means


# -- drift (a check on jitter stabilisation) ------------------------------------------------------------------

def estimate_velocity(movie: Movie, lit: np.ndarray, *, lag: int = 40, samples: int = 8,
                      context: int = 600, progress=None, cancel=None, log=print) -> dict:
    """Sample drift in px/frame along x: phase correlation of frames `lag`
    apart, after removing what is static in the camera frame (a long temporal
    median: detector pattern, a steady pool) -- what is left is the texture
    that moves with the sample."""
    from scipy import ndimage as ndi
    from skimage.registration import phase_cross_correlation
    from vanua.core.cancellation import check_cancel

    lit_idx = np.flatnonzero(lit)
    if len(lit_idx) < lag + 20:
        raise ValueError("too few lit frames to estimate the drift")
    lo, hi = int(lit_idx[0]), int(lit_idx[-1])
    centres = np.linspace(lo + (hi - lo) * 0.2, hi - lag - 10, samples).astype(int)
    estimates = []
    for j, i in enumerate(centres):
        check_cancel(cancel)
        if progress:
            progress(j / len(centres), f"drift at frame {i}")
        a0, a1 = max(lo, i - context // 3), min(hi, i + 2 * context // 3)
        static = np.median(np.stack([movie.frame(k) for k in range(a0, a1, max(1, context // 60))]), axis=0)
        np.maximum(static, 1.0, out=static)

        def high_pass(k: int) -> np.ndarray:
            # A ratio, not a difference: the detector pattern is multiplicative (flat field).
            return ndi.gaussian_filter(movie.mean_of(k, 3) / static - 1.0, 1.0)

        # Plain (unwhitened) cross-correlation: whitening lets per-frame noise
        # and the static residue win, giving a spurious zero shift.
        shift, _err, _ = phase_cross_correlation(high_pass(i), high_pass(i + lag), upsample_factor=20,
                                                 normalization=None)
        estimates.append(float(-shift[1] / lag))  # content moved by -shift over `lag` frames
    estimates_arr = np.asarray(estimates)
    value = float(np.median(estimates_arr))
    spread = float(np.median(np.abs(estimates_arr - value)))
    log(f"drift {value:+.4f} px/frame (MAD {spread:.4f}; {len(estimates)} samples)")
    return {"px_per_frame": value, "method": "phase correlation", "estimates": estimates,
            "mad": spread, "lag": lag}


# -- surface, depression, rim ---------------------------------------------------------------------------------

def xshift(a: np.ndarray, dx: float) -> np.ndarray:
    """Shift content by `dx` px along x (positive: to larger x), linear
    interpolation, edges extended."""
    if dx == 0:
        return a
    w = a.shape[-1]
    src = np.arange(w, dtype=np.float64) - dx
    np.clip(src, 0, w - 1, out=src)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, w - 1)
    f = (src - i0).astype(np.float32)
    return a[..., i0] * (1 - f) + a[..., i1] * f


def material_mask(image: np.ndarray, sigma: float = 3.0, open_radius: int = 3) -> np.ndarray:
    """Material (substrate, deposit) = darker than air, Otsu on the smoothed frame."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk, opening

    smooth = ndi.gaussian_filter(image, sigma)
    mask = smooth < threshold_otsu(smooth)
    if open_radius:
        mask = opening(mask, disk(open_radius))
    return ndi.binary_fill_holes(mask)


def free_surface(image: np.ndarray) -> np.ndarray:
    """Per column, the row where the column goes from air (above) to material:
    the first crossing of the mid level between the air and material medians."""
    from scipy import ndimage as ndi

    sm = ndi.gaussian_filter(image, 1.0)
    mat = material_mask(image)
    if mat.all() or not mat.any():
        return np.full(image.shape[1], np.nan)
    mid = 0.5 * (np.median(sm[~mat]) + np.median(sm[mat]))
    dark = sm < mid if np.median(sm[mat]) < np.median(sm[~mat]) else sm > mid
    rows = np.argmax(dark, axis=0).astype(float)
    rows[~dark.any(axis=0)] = np.nan
    return ndi.median_filter(rows, 5, mode="nearest")



def original_surface(image: np.ndarray) -> float:
    """Row of the original (unmelted) surface: the lower envelope of the per-column
    free surface -- a bead or spatter stands above it and must not pull it up."""
    top = free_surface(image)
    return float(np.nanpercentile(top, 80)) if np.isfinite(top).any() else float("nan")


def depression_centre(image: np.ndarray, surface: float, depth: int = 60, smooth: float = 25.0) -> float:
    """Column of the vapour-depression centre in a (median) image: the dark-ridge
    strength just below the surface, summed per column and smoothed over `smooth` px,
    so the two walls of a bowl merge into one central peak and small pores do not count."""
    from scipy import ndimage as ndi

    ridge = dark_ridges(image)
    top = int(np.ceil(surface)) + 10  # the surface line's own response reaches ~3 sigma (9 px) below it
    band = ridge[top: top + depth]
    if not band.size or band.max() <= 0:
        return image.shape[1] / 2
    strong = band >= 0.3 * np.percentile(band, 99.9)  # a robust "strong": not one extreme pixel
    profile = ndi.gaussian_filter1d(np.where(strong, band, 0.0).sum(axis=0), smooth)
    return float(np.argmax(profile))

def _dp_path(score: np.ndarray, step: int) -> np.ndarray:
    """Column index per row of `score` (rows = angles) maximising the summed score,
    moving at most `step` columns between rows (dynamic programming)."""
    n, m = score.shape
    acc = score.copy()
    back = np.zeros(score.shape, np.int64)
    cols = np.arange(m)
    for t in range(1, n):
        prev = acc[t - 1]
        best = np.full(m, -np.inf)
        arg = np.zeros(m, np.int64)
        for d in range(-step, step + 1):
            shifted = np.full(m, -np.inf)
            if d >= 0:
                shifted[d:] = prev[:m - d]
            else:
                shifted[:d] = prev[-d:]
            better = shifted > best
            best[better] = shifted[better]
            arg[better] = (cols - d)[better]
        acc[t] = score[t] + best
        back[t] = arg
    path = np.zeros(n, np.int64)
    path[-1] = int(np.argmax(acc[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def polar_trace(score_img: np.ndarray, cx: float, cy: float, r_lo, r_hi, *, n_theta: int = 181,
                dr: float = 0.5, max_step_px: float = 1.0) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The best smooth path r(theta) through `score_img` around (cx, cy), theta from
    ~0 (right, along the surface) to ~pi (left), r in [r_lo, r_hi] (scalars or
    per-angle arrays). Returns x, y of the path and the score along it."""
    from scipy import ndimage as ndi

    h, w = score_img.shape
    thetas = np.linspace(0.05, np.pi - 0.05, n_theta)
    rs = np.arange(0.0, float(np.max(r_hi)) + dr, dr)
    yy = cy + np.outer(np.sin(thetas), rs)
    xx = cx + np.outer(np.cos(thetas), rs)
    sc = ndi.map_coordinates(score_img, [yy, xx], order=1, mode="constant", cval=0.0)
    lo = np.broadcast_to(np.asarray(r_lo, float), thetas.shape)[:, None]
    hi = np.broadcast_to(np.asarray(r_hi, float), thetas.shape)[:, None]
    allowed = (rs[None, :] >= lo) & (rs[None, :] <= hi) & (xx >= 0) & (xx <= w - 1) & (yy <= h - 1)
    sc = np.where(allowed, sc, -1e9)
    path = _dp_path(sc, max(1, int(round(max_step_px / dr))))
    r = rs[path]
    return cx + np.cos(thetas) * r, cy + np.sin(thetas) * r, sc[np.arange(n_theta), path]


def dark_ridges(image: np.ndarray, sigmas=(1.0, 2.0, 3.0)) -> np.ndarray:
    """Dark-line strength (Sato), computed on a mirror-padded image: an unpadded
    filter answers strongly along the image border, which is not a line."""
    from skimage.filters import sato

    pad = int(np.ceil(4 * max(sigmas)))
    padded = np.pad(np.asarray(image, dtype=np.float64), pad, mode="reflect")
    return sato(padded, sigmas=tuple(sigmas), black_ridges=True)[pad:-pad, pad:-pad]


def rim_trace(image: np.ndarray, surface: float, cx: float, *, r_min: float = 15.0, r_max: float = 150.0,
              sigmas=(1.0, 2.0, 3.0)) -> "tuple[np.ndarray, np.ndarray]":
    """The vapour-depression rim: the dark phase-contrast line below the surface,
    traced as a smooth path around (cx, surface). The surface band itself is ignored."""
    ridge = dark_ridges(image, sigmas)
    blind = max(0, int(np.ceil(surface)) + 4)
    ridge[:blind] = 0.0
    x, y, _ = polar_trace(ridge, cx, surface, r_min, r_max)
    seen = y >= blind  # inside the masked surface band the path has nothing to follow: drop it
    if seen.sum() >= 3:
        x, y = x[seen], y[seen]
    return x, y


def region_below(shape, x: np.ndarray, y: np.ndarray, surface: float) -> np.ndarray:
    """Pixels between the surface line and a path that starts and ends near it:
    the path is closed along the surface, so nothing between it and the surface is lost."""
    from skimage.draw import polygon

    xs = np.r_[x[0], x, x[-1]]
    ys = np.r_[surface, y, surface]
    mask = np.zeros(shape, bool)
    rr, cc = polygon(ys, xs, shape)
    mask[rr, cc] = True
    mask[: max(0, int(np.ceil(surface)))] = False
    return mask


# -- stabilised median and the liquid-solid line --------------------------------------------

def median_image(movie: Movie, frames: np.ndarray, progress=None, cancel=None) -> np.ndarray:
    """Per-pixel temporal median of the given frames (the fluctuating liquid-gas
    boundary averages out; a stationary boundary stays). Chunked by rows to bound memory."""
    from vanua.core.cancellation import check_cancel

    frames = np.asarray(frames, int)
    if len(frames) > 2000:
        frames = frames[np.linspace(0, len(frames) - 1, 2000).astype(int)]
    h, w = movie.shape
    rows_per_chunk = max(1, int(2e8 // (len(frames) * w * 4)))
    if rows_per_chunk >= h:
        stack = np.stack([movie.frame(int(i)) for i in frames])
        return np.median(stack, axis=0).astype(np.float32)
    out = np.empty((h, w), np.float32)
    for r0 in range(0, h, rows_per_chunk):
        check_cancel(cancel)
        if progress:
            progress(r0 / h, f"median rows {r0}-{min(h, r0 + rows_per_chunk)}")
        stack = np.stack([movie.frame(int(i))[r0:r0 + rows_per_chunk] for i in frames])
        out[r0:r0 + rows_per_chunk] = np.median(stack, axis=0)
    return out


def band_pass(image: np.ndarray, surface: float, s1: float = 2.0, s2: float = 8.0, edge: int = 70,
              exclude: "tuple[float, float] | None" = None) -> np.ndarray:
    """Fine detail of a median image: the beam profile divided out, a difference of
    Gaussians, and the detector row banding removed. The banding is each row's median
    over the columns outside `exclude` (the pool zone) -- so every column a control
    search uses is corrected exactly like the pool's -- or, without `exclude`, the
    `edge` columns at each side interpolated across."""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float64).copy()
    top = max(0, int(np.ceil(surface)) + 1)
    if top < img.shape[0]:
        img[:top] = np.median(img[top:top + 40])
    rel = img / np.maximum(ndi.gaussian_filter(img, 30.0), 1e-6)
    d = ndi.gaussian_filter(rel, s1) - ndi.gaussian_filter(rel, s2)
    w = d.shape[1]
    if exclude is not None:
        cols = np.ones(w, bool)
        cols[max(0, int(exclude[0])):min(w, int(np.ceil(exclude[1])))] = False
        if cols.sum() >= 8:
            return d - np.median(d[:, cols], axis=1, keepdims=True)
    e = max(4, min(edge, w // 4))
    left, right = np.median(d[:, :e], axis=1), np.median(d[:, w - e:], axis=1)
    x = np.arange(w)[None, :]
    return d - (left[:, None] + (right - left)[:, None] * (x - e / 2) / max(1.0, w - e))


def solid_line_auto(median: np.ndarray, surface: float, rim_x: np.ndarray, rim_y: np.ndarray, cx: float, *,
                    gap: float = 15.0, reach: float = 70.0, z_min: float = 3.0, n_controls: int = 4) -> dict:
    """Look for a faint dark line outside the vapour-depression rim (the liquid-solid
    boundary) in a stabilised median image, and report it only if it stands out:
    its ridge response must beat the same search at `n_controls` places without a
    pool by `z_min` standard deviations. Returns x, y, z and `detected`."""
    half_zone = float(np.max(np.abs(rim_x - cx)) + reach)
    d = band_pass(median, surface, exclude=(cx - half_zone, cx + half_zone))
    ridge = np.clip(-d, 0.0, None)
    ridge[: max(0, int(np.ceil(surface)) + 4)] = 0.0
    # rim radius on the search's angle grid (the rim may have lost its ends at the surface)
    ang = np.arctan2(rim_y - surface, rim_x - cx)
    order = np.argsort(ang)
    thetas = np.linspace(0.05, np.pi - 0.05, 181)
    r_rim = np.interp(thetas, ang[order], np.hypot(rim_x - cx, rim_y - surface)[order])
    x, y, sc = polar_trace(ridge, cx, surface, r_rim + gap, r_rim + reach)
    ok = sc > -1e8
    score = float(sc[ok].mean()) if ok.any() else 0.0
    w = median.shape[1]
    half = float(np.max(np.abs(rim_x - cx)) + reach)
    candidates = [c for c in np.linspace(0, w - 1, 4 * n_controls + 1) if abs(c - cx) > 1.5 * half]
    picks = [candidates[int(k)] for k in np.linspace(0, len(candidates) - 1, min(n_controls, len(candidates)))] \
        if candidates else []
    controls = []
    for c in picks:
        _, _, csc = polar_trace(ridge, float(c), surface, r_rim + gap, r_rim + reach)
        if (csc > -1e8).sum() > len(csc) // 3:
            controls.append(float(csc[csc > -1e8].mean()))
    controls_a = np.asarray(controls)
    z = float((score - controls_a.mean()) / controls_a.std()) if len(controls_a) >= 2 and controls_a.std() > 0 \
        else float("nan")
    return {"x": x, "y": y, "score": score, "controls": controls, "z": z,
            "detected": bool(np.isfinite(z) and z >= z_min)}
