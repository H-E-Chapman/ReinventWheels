"""Image quality checks for a group experiment: drift across the z-stack and
duplicate images of one worm. Qt-free.

* :func:`estimate_z_drift` -- how far each z-slice is shifted sideways (y, x)
  from the rest of the stack: a worm that was not held still, or a bump to the
  optical table during the stack. Neighbouring slices are matched by phase
  correlation; the steps add up to each slice's offset.
* :func:`apply_z_drift` -- moves every slice back (registration).
* :func:`drift_decision` -- small drift: register; big drift: leave the image
  out of the analysis.
* :func:`fingerprint` / :func:`similarity` -- a small DAPI projection per image
  and how alike two of them are once aligned; the same worm saved twice (under
  two numbers, say) is near 1, two different worms far lower.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DriftSettings", "estimate_z_drift", "apply_z_drift", "drift_decision", "fingerprint", "similarity"]


@dataclass
class DriftSettings:
    """When drift is corrected and when the image is left out (µm)."""

    # On the bleaching stacks (one worm, 40 repeats) the slices line up to
    # within 0.75-1.1 µm, the same in every repeat to ±0.1 µm, in steps of at
    # most 0.1 µm: the head's own slight tilt (or a repeatable stage motion),
    # not something to correct. So:
    register_above: float = 1.5   # total drift above this is corrected ...
    jump_above: float = 0.5       # ... and so is a sudden move between two neighbouring slices (a bump)
    exclude_above: float = 3.0    # total drift above this: left out of the analysis
    channel: int = 3              # the channel the drift is measured on (DAPI)


def _bandpass(plane: np.ndarray, band) -> np.ndarray:
    """Difference of Gaussians (sigmas in pixels): keeps the nuclei-sized
    texture, removes pixel noise and the slow shapes (the body outline, a
    background gradient) that change with z and pulled every match towards
    no shift at all."""
    from scipy import ndimage

    return ndimage.gaussian_filter(plane, band[0]) - ndimage.gaussian_filter(plane, band[1])


def _planes(volume, downsample: int, band):
    return [_bandpass(np.asarray(volume[z], np.float32)[::downsample, ::downsample], band)
            for z in range(volume.shape[0])]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _overlap(a: np.ndarray, b: np.ndarray, shift) -> "tuple[np.ndarray, np.ndarray]":
    """The parts of `a` and `b` that overlap once `b` is moved by the integer `shift` (dy, dx)."""
    dy, dx = (int(round(s)) for s in shift)
    h, w = a.shape
    ya, yb = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xa, xb = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    return a[ya, xa], b[yb, xb]


def estimate_z_drift(volume, spacing, *, downsample: int = 2, band=(1.0, 3.0), min_contrast: float = 0.5):
    """Each slice's sideways offset from the stack, as a table (one row per z):
    `dy_px`, `dx_px` (full-resolution pixels), `dy_um`, `dx_um`, `step_um`
    (the move from the slice before) and `match` (how alike the two
    neighbouring slices are once aligned, 0-1).

    Offsets are relative to the median position, so a stack that only moved at
    its ends is corrected at its ends. Slices with too little structure to
    match (the empty ends of a stack: contrast below `min_contrast` x the
    median) take no step.

    Checked by moving the slices of real stacks by a known drift (1.8 µm
    across the stack, dim and cropped stacks included): 92-97 % of it is
    measured; with plain smoothing instead of the band-pass, 50-70 %."""
    from skimage.registration import phase_cross_correlation

    import pandas as pd

    n = volume.shape[0]
    planes = _planes(volume, downsample, band)
    contrast = np.array([float(p.std()) for p in planes])
    usable = contrast >= min_contrast * float(np.median(contrast))
    steps = np.zeros((n, 2))
    match = np.full(n, np.nan)
    for z in range(1, n):
        if not (usable[z] and usable[z - 1]):
            continue
        shift, _err, _phase = phase_cross_correlation(planes[z - 1], planes[z], upsample_factor=10,
                                                      normalization=None)
        # `shift` moves slice z onto slice z-1: the content of slice z sits at -shift.
        steps[z] = -np.asarray(shift, float) * downsample
        match[z] = _ncc(*_overlap(planes[z - 1], planes[z], shift))
    offsets = np.cumsum(steps, axis=0)
    offsets -= np.median(offsets[usable] if usable.any() else offsets, axis=0)
    sy, sx = float(spacing[1]), float(spacing[2])
    step_um = np.hypot(steps[:, 0] * sy, steps[:, 1] * sx)
    return pd.DataFrame({"z": np.arange(n), "dy_px": offsets[:, 0], "dx_px": offsets[:, 1],
                         "dy_um": offsets[:, 0] * sy, "dx_um": offsets[:, 1] * sx, "step_um": step_um,
                         "match": match, "usable": usable})


def drift_summary(drift) -> dict:
    """`drift_um`: the largest distance between the positions of any two
    slices; `max_step_um`: the largest move between neighbouring slices;
    `min_match`: the poorest neighbour match."""
    pts = drift[["dy_um", "dx_um"]].to_numpy(float)
    spread = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2))) if len(pts) else 0.0
    match = drift["match"].dropna()
    return {"drift_um": spread, "max_step_um": float(drift["step_um"].max()) if len(drift) else 0.0,
            "min_match": float(match.min()) if len(match) else float("nan")}


def drift_decision(summary: dict, settings: "DriftSettings | None" = None) -> str:
    """'ok' (no correction), 'registered' (small drift, corrected) or
    'excluded' (big drift: left out of the analysis)."""
    s = settings or DriftSettings()
    if summary["drift_um"] > s.exclude_above:
        return "excluded"
    moved = summary["drift_um"] > s.register_above or summary["max_step_um"] > s.jump_above
    return "registered" if moved else "ok"


def apply_z_drift(volume, drift) -> np.ndarray:
    """Every slice moved back by its offset (linear interpolation; the edge
    pixels repeat into the gap). Same dtype as the input."""
    from scipy import ndimage

    volume = np.asarray(volume)
    out = np.empty_like(volume)
    offsets = drift[["dy_px", "dx_px"]].to_numpy(float)
    for z in range(volume.shape[0]):
        dy, dx = offsets[z]
        if abs(dy) < 0.05 and abs(dx) < 0.05:
            out[z] = volume[z]
            continue
        moved = ndimage.shift(volume[z].astype(np.float32), (-dy, -dx), order=1, mode="nearest")
        if np.issubdtype(volume.dtype, np.integer):
            info = np.iinfo(volume.dtype)
            moved = np.clip(np.rint(moved), info.min, info.max)
        out[z] = moved.astype(volume.dtype)
    return out


# -- duplicates -----------------------------------------------------------------------------


def fingerprint(volume, downsample: int = 4, band=(1.0, 4.0)) -> np.ndarray:
    """A small maximum projection of the stack, band-passed to its nuclei
    pattern (DAPI: particular to each worm and where it lies on the slide;
    the body outline, which two worms lying alike share, is removed), normalized."""
    mip = np.asarray(volume).max(axis=0).astype(np.float32)
    h, w = (mip.shape[0] // downsample) * downsample, (mip.shape[1] // downsample) * downsample
    small = mip[:h, :w].reshape(h // downsample, downsample, w // downsample, downsample).mean(axis=(1, 3))
    small = _bandpass(small, band)
    small -= small.mean()
    sd = small.std()
    return small / sd if sd > 0 else small


def similarity(a: np.ndarray, b: np.ndarray, min_overlap: float = 0.5) -> float:
    """How alike two fingerprints are once aligned (shift by phase
    correlation, then correlation of the overlap): ~1 for the same worm, low
    for two different worms. NaN when they are of different sizes or overlap
    too little."""
    from skimage.registration import phase_cross_correlation

    if a.shape != b.shape:
        return float("nan")
    shift, _e, _p = phase_cross_correlation(a, b, normalization=None)
    oa, ob = _overlap(a, b, shift)
    if oa.size < min_overlap * a.size:
        return float("nan")
    return _ncc(oa, ob)
