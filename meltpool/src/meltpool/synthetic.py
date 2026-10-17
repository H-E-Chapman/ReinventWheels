"""A synthetic radiography movie with a known pool, for tests and the demo
fallback. Qt-free.

Built in the display orientation (air on top, material below, surface at
row ``surface``) and then, like the real data, stored upside down unless
``upside_down=False``. Ingredients:

- a static detector pattern (vertical stripes, multiplicative) -- fixed in the camera;
- a textured solid (pores) translating at ``velocity`` px/frame -- moves with the sample;
- the pool: an ellipse (semi-axes ``a`` x ``b`` px) centred on the surface at
  ``centre_x``, material-dark, whose texture is redrawn every frame (liquid);
  it grows linearly from nothing during ``ramp`` frames after the blank ones;
- or (``depression=True``) a vapour depression: a bright half-ellipse inside a dark rim
  ``rim_width`` px wide, its size oscillating (``osc_amp``, ``osc_period`` frames) and,
  with ``ramp``, growing from half to full size during the start-up;
- fast powder dots in the air;
- ``n_blank`` dark frames first (shutter closed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyntheticSpec:
    n: int = 360
    height: int = 120
    width: int = 240
    surface: int = 60
    velocity: float = 0.35
    a: float = 40.0
    b: float = 15.0
    centre_x: float = 120.0
    n_blank: int = 20
    ramp: int = 0
    air: float = 1800.0
    material: float = 1100.0
    liquid_contrast: float = 0.08  # interior activity ~0.06, as measured inside the real pools (478/479)
    seed: int = 0
    texture: float = 0.08          # pore texture of the solid (sd); the real weld substrate is ~0.015
    pattern: float = 0.05          # camera-fixed vertical stripes (amplitude)
    # sub-pixel shake of the whole scene (sd, px); the detector pattern stays put, as in real data
    jitter_px: float = 0.0
    # a vapour depression instead of the liquid pool: a brighter (gas) bowl below the surface with
    # a dark phase-contrast rim; its size fluctuates by `rim_wobble` from frame to frame
    depression: bool = False
    rim_dark: float = 0.55
    rim_wobble: float = 0.02
    rim_width: float = 2.5         # thickness of the dark rim band at the bottom (px)
    osc_amp: float = 0.0           # sinusoidal size oscillation (fraction) ...
    osc_period: float = 0.0        # ... with this period (frames); 0 = none
    # a faint static dark line `solid_gap` px outside the depression (the liquid-solid boundary)
    solid_line_contrast: float = 0.0
    solid_gap: float = 18.0

    def jitter(self) -> np.ndarray:
        """The (dy, dx) shake applied to each frame -- the truth for stabilisation tests."""
        if self.jitter_px <= 0:
            return np.zeros((self.n, 2))
        return np.random.default_rng(self.seed + 99).normal(0.0, self.jitter_px, (self.n, 2))

    def depression_mask(self) -> np.ndarray:
        """Truth: the bowl below the surface at its mean size."""
        m = self.truth_mask()
        m[: self.surface] = False
        return m

    def solid_line(self) -> "tuple[float, float]":
        """Semi-axes (x, y) of the liquid-solid line."""
        return self.a + self.solid_gap, self.b + self.solid_gap

    def truth_mask(self, scale: float = 1.0) -> np.ndarray:
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        a, b = self.a * scale, self.b * scale
        if a <= 0 or b <= 0:
            return np.zeros((self.height, self.width), bool)
        return ((xx - self.centre_x) / a) ** 2 + ((yy - self.surface) / b) ** 2 <= 1.0

    def scale_at(self, t: int) -> float:
        if t < self.n_blank:
            return 0.0
        if self.ramp <= 0:
            return 1.0
        return float(min(1.0, (t - self.n_blank + 1) / self.ramp))


def make_movie(spec: SyntheticSpec, upside_down: bool = True) -> np.ndarray:
    from scipy import ndimage as ndi

    rng = np.random.default_rng(spec.seed)
    h, w = spec.height, spec.width
    pattern = 1.0 + spec.pattern * np.sin(np.arange(w) / 3.0)[None, :] * np.ones((h, 1))
    # The solid texture lives on a long strip; frame t sees it shifted by velocity * t.
    margin = int(abs(spec.velocity) * spec.n) + 8
    strip_w = w + 2 * margin
    pores = ndi.gaussian_filter(rng.normal(0, 1, (h, strip_w)), 2.0)
    pores = 1.0 - spec.texture * pores / pores.std()
    yy = np.arange(h)[:, None]
    yg, xg = np.mgrid[0:h, 0:w]
    below = yg >= spec.surface
    jitter = spec.jitter()
    movie = np.empty((spec.n, h, w), np.uint16)
    for t in range(spec.n):
        if t < spec.n_blank:
            movie[t] = rng.integers(30, 60, (h, w))
            continue
        x = np.arange(w) + margin - spec.velocity * t        # content moves by +velocity px/frame
        x0 = np.floor(x).astype(int)
        f = x - x0
        solid = pores[:, x0] * (1 - f) + pores[:, x0 + 1] * f
        img = np.where(yy >= spec.surface, spec.material * solid, spec.air)
        pool = spec.truth_mask(spec.scale_at(t)) if not spec.depression else np.zeros((h, w), bool)
        if spec.solid_line_contrast:
            sa, sb = spec.solid_line()
            rho = np.hypot((xg - spec.centre_x) / sa, (yg - spec.surface) / sb)
            img = np.where(below & (np.abs(rho - 1.0) < 1.0 / sb), img * (1.0 - spec.solid_line_contrast), img)
        if spec.depression:
            k = 1.0 + spec.rim_wobble * rng.uniform(-1, 1)
            if spec.osc_period > 0:
                k += spec.osc_amp * np.sin(2 * np.pi * t / spec.osc_period)
            if spec.ramp > 0:  # start-up: the depression grows from half to full size
                k *= 0.5 + 0.5 * spec.scale_at(t)
            a_t, b_t = spec.a * k, spec.b * k
            rho = np.hypot((xg - spec.centre_x) / a_t, (yg - spec.surface) / b_t)
            img = np.where(below & (rho < 1.0), spec.material * 1.12, img)
            img = np.where(below & (rho >= 1.0) & (rho < 1.0 + spec.rim_width / b_t), img * spec.rim_dark, img)
        if pool.any():
            liquid = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), 2.0)
            liquid = 1.0 + spec.liquid_contrast * liquid / liquid.std()
            img = np.where(pool, spec.material * liquid, img)
        for _ in range(6):  # powder in flight
            cy, cx = rng.integers(0, max(1, spec.surface - 10)), rng.integers(0, w)
            img[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] *= 0.6
        if jitter[t].any():
            from .movie import fourier_shift_frame

            img = fourier_shift_frame(img, *jitter[t])
        img = img * pattern + rng.normal(0, 8, (h, w))
        movie[t] = np.clip(img, 0, 65535).astype(np.uint16)
    return movie[:, ::-1] if upside_down else movie


def write_movie(path: "str | Path", movie: np.ndarray) -> Path:
    """One page per frame, uncompressed -- the layout of the real movies."""
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path) as tif:
        for frame in movie:
            tif.write(frame, contiguous=False)
    return path


def write_hdf5(path: "str | Path", movie: np.ndarray, dataset: str = "entry/data/data") -> Path:
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=movie, chunks=(1, *movie.shape[1:]))
        f.create_dataset("entry/instrument/dark", data=movie[:2])  # a smaller 3D decoy
    return path


def write_video(path: "str | Path", movie: np.ndarray, codec: str = "rawvideo", rate: int = 25) -> Path:
    """Lossless video: `rawvideo` (8-bit grey, AVI) or `ffv1` (16-bit grey, MKV)."""
    import av

    path = Path(path)
    deep = movie.dtype.itemsize > 1
    fmt = "gray16le" if deep else "gray"
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.width, stream.height = movie.shape[2], movie.shape[1]
        stream.pix_fmt = fmt
        for frame in movie:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format=fmt)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def to_uint8(movie: np.ndarray, lo: float = 0.0, hi: float = 2400.0) -> np.ndarray:
    """What an 8-bit export does: a linear map of [lo, hi] onto 0..255."""
    return np.clip((movie.astype(np.float32) - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)
