"""HCR Cells app ops -- only what no vanua block covers.

The cell masks come from blocks (smooth, threshold, clean, label, filter); the
app adds three things:

* ``hcrcells.subtract_background``: removes the slowly varying body
  autofluorescence (strong in the 488 / ins-6 channel) so a global threshold
  picks out the HCR-positive somata instead of the whole worm.
* ``hcrcells.detect_spots``: HCR puncta as local maxima of a
  difference-of-Gaussians, thresholded at k x a robust (MAD) noise estimate of
  that same image -- so the threshold follows each timepoint's own noise.
* ``hcrcells.channel_stats``: per-cell intensity (raw, local background,
  background-subtracted), saturation and spot count for one HCR channel.

Physical parameters are in the frame's unit (µm for these TIFFs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from vanua.core.cancellation import check_cancel
from vanua.core.ops import OpSpec
from vanua.blocks._frame import as_frame, index_to_world, length_column, spacing_zyx, world_axis_names

__all__ = ["OPS", "subtract_background", "detect_spots", "channel_stats"]


def _sigma_voxels(frame, sigma_zyx: "tuple[float, float, float]") -> np.ndarray:
    return np.asarray(sigma_zyx, dtype=float) / np.asarray(spacing_zyx(frame))


# -- background --------------------------------------------------------------------------


@dataclass
class BackgroundParams:
    radius: float = field(default=4.0, metadata={
        "label": "Background radius", "range": (0.1, 100.0), "step": 0.5,
        "help": "Gaussian sigma (frame unit) of the background estimate; larger than a soma"})


def subtract_background(image: np.ndarray, frame, radius: float) -> np.ndarray:
    from scipy import ndimage

    image = np.asarray(image, dtype=np.float32)
    sigma = _sigma_voxels(as_frame(frame), (radius,) * 3)
    return np.clip(image - ndimage.gaussian_filter(image, sigma), 0, None)


def _subtract_background(params: BackgroundParams, ctx, progress, cancel, log):
    out = subtract_background(ctx["image"], ctx["frame"], params.radius)
    progress(1.0, "background subtracted")
    return {"image": out}


# -- spots -------------------------------------------------------------------------------


@dataclass
class SpotParams:
    sigma_xy: float = field(default=0.12, metadata={"label": "Spot sigma xy", "range": (0.01, 5.0), "step": 0.01,
                                                    "help": "Gaussian sigma of one punctum in x/y (frame unit)"})
    sigma_z: float = field(default=0.35, metadata={"label": "Spot sigma z", "range": (0.01, 10.0), "step": 0.05,
                                                   "help": "Gaussian sigma of one punctum in z (frame unit)"})
    threshold_mode: Literal["elbow", "noise"] = field(default="elbow", metadata={
        "label": "Threshold", "help": "elbow: knee of the spot-count vs threshold curve (Big-FISH's automated "
                                      "setting); noise: median + k x MAD noise of the filtered image"})
    threshold_k: float = field(default=5.0, metadata={"label": "Threshold (x noise)", "range": (0.5, 50.0),
                                                      "help": "Used when threshold is 'noise'"})
    min_separation: float = field(default=0.2, metadata={"label": "Min separation xy", "range": (0.05, 5.0),
                                                         "help": "Minimum distance between two spots in x/y (frame unit)"})


def elbow_threshold(values: np.ndarray, n: int = 200) -> float:
    """Knee of log(count above t) vs t: the point farthest from the chord
    joining the curve's ends (both axes scaled to 0..1). Noise maxima make the
    steep part, real puncta the long flat tail, as in Big-FISH."""
    values = np.asarray(values, dtype=float).ravel()
    if len(values) < 3:
        return float(values.min()) if len(values) else 0.0
    ordered = np.sort(values)
    ts = np.linspace(np.percentile(ordered, 1), np.percentile(ordered, 99.99), n)
    # Count above each t from one sort (searchsorted), not one pass per t:
    # identical counts, ~50x faster on a 2.6-million-voxel display sample.
    above = len(ordered) - np.searchsorted(ordered, ts, side="right")
    counts = np.log(np.maximum(above, 1))
    x = (ts - ts[0]) / ((ts[-1] - ts[0]) or 1.0)
    y = (counts - counts.min()) / (np.ptp(counts) or 1.0)
    distance = np.abs((x[0] - x[-1]) * (y - y[0]) - (x - x[0]) * (y[0] - y[-1]))
    return float(ts[int(np.argmax(distance))])


def detect_spots(image: np.ndarray, frame, *, sigma_xy: float = 0.12, sigma_z: float = 0.35,
                 threshold_mode: str = "elbow", threshold_k: float = 5.0, min_separation: float = 0.2, cancel=None):
    """Spots as DoG local maxima. Returns (table, threshold, noise)."""
    import pandas as pd
    from scipy import ndimage
    from skimage.feature import peak_local_max

    frame = as_frame(frame)
    image = np.asarray(image, dtype=np.float32)
    s1 = _sigma_voxels(frame, (sigma_z, sigma_xy, sigma_xy))
    dog = ndimage.gaussian_filter(image, s1)
    check_cancel(cancel)
    dog -= ndimage.gaussian_filter(image, 3.0 * s1)
    check_cancel(cancel)
    median = float(np.median(dog))
    noise = 1.4826 * float(np.median(np.abs(dog - median)))
    # uint8 data with a black surround can give MAD 0; a floor keeps the
    # threshold above quantisation noise instead of accepting every ripple.
    noise = max(noise, 1e-3)
    noise_threshold = median + threshold_k * noise
    radius_xy = max(1, int(round(min_separation / spacing_zyx(frame)[1])))
    radius_z = max(1, int(round(min_separation / spacing_zyx(frame)[0])))
    footprint = np.ones((2 * radius_z + 1, 2 * radius_xy + 1, 2 * radius_xy + 1), bool)
    if threshold_mode == "noise":
        threshold = noise_threshold
        peaks = peak_local_max(dog, footprint=footprint, threshold_abs=threshold, exclude_border=False)
    else:
        # Candidates at a low noise threshold, then the elbow over their DoG values.
        candidates = peak_local_max(dog, footprint=footprint, threshold_abs=median + 3.0 * noise, exclude_border=False)
        threshold = max(elbow_threshold(dog[tuple(candidates.T)]), noise_threshold) if len(candidates) else noise_threshold
        peaks = candidates[dog[tuple(candidates.T)] > threshold] if len(candidates) else candidates
    world = index_to_world(frame, peaks) if len(peaks) else np.zeros((0, 3))
    table = {"z_index": peaks[:, 0], "y_index": peaks[:, 1], "x_index": peaks[:, 2]}
    for axis, column in zip(world_axis_names(frame), world.T):
        table[length_column(axis, frame)] = column
    table["dog"] = dog[tuple(peaks.T)] if len(peaks) else np.zeros(0)
    table["intensity"] = image[tuple(peaks.T)] if len(peaks) else np.zeros(0)
    return pd.DataFrame(table), threshold, noise


def _detect_spots(params: SpotParams, ctx, progress, cancel, log):
    table, threshold, noise = detect_spots(ctx["image"], ctx["frame"], sigma_xy=params.sigma_xy,
                                           sigma_z=params.sigma_z, threshold_mode=params.threshold_mode,
                                           threshold_k=params.threshold_k,
                                           min_separation=params.min_separation, cancel=cancel)
    log(f"{len(table)} spots ({params.threshold_mode} threshold {threshold:.3f}; noise {noise:.3f})")
    progress(1.0, f"{len(table)} spots")
    return {"table": table}


# -- per-cell channel statistics ---------------------------------------------------------


@dataclass
class ChannelStatsParams:
    channel: str = field(default="hcr", metadata={"help": "Column prefix, e.g. ins6"})
    shell_inner: float = field(default=0.5, metadata={"range": (0.0, 20.0),
                                                      "help": "Background shell starts this far outside the cell (frame unit)"})
    shell_outer: float = field(default=2.0, metadata={"range": (0.1, 50.0),
                                                      "help": "Background shell ends this far outside the cell (frame unit)"})
    saturation: float = field(default=255.0, metadata={"help": "Detector maximum (255 for 8-bit)"})
    isolation: float = field(default=3.0, metadata={"range": (0.0, 100.0),
                                                    "help": "Reference spots lie at least this far from any cell (frame unit)"})
    window_xy: float = field(default=0.2, metadata={"range": (0.05, 5.0),
                                                    "help": "Half-width in x/y of the box summed around a reference spot"})
    window_z: float = field(default=0.5, metadata={"range": (0.0, 10.0),
                                                   "help": "Half-depth in z of the box summed around a reference spot"})


def reference_spot(image: np.ndarray, spots, labels: np.ndarray, frame, *, isolation: float = 3.0,
                   window_xy: float = 0.2, window_z: float = 0.5, background_sigma: float = 2.0):
    """Median integrated, background-subtracted intensity of one isolated
    spot, and how many spots it is the median of. Big-FISH's dense-region
    decomposition uses the same idea: a cell's signal divided by one
    transcript's signal is an estimate of how many transcripts it holds."""
    from scipy import ndimage

    frame = as_frame(frame)
    if len(spots) == 0:
        return float("nan"), 0
    spacing = np.asarray(spacing_zyx(frame))
    image = np.asarray(image, dtype=np.float32)
    z, y, x = (spots[c].to_numpy(int) for c in ("z_index", "y_index", "x_index"))
    distance = ndimage.distance_transform_edt(labels == 0, sampling=spacing)
    isolated = np.nonzero(distance[z, y, x] > isolation)[0]
    if len(isolated) == 0:
        return float("nan"), 0
    background = ndimage.gaussian_filter(image, background_sigma / spacing)
    rz = int(round(window_z / spacing[0]))
    ry = int(round(window_xy / spacing[1]))
    rx = int(round(window_xy / spacing[2]))
    sums = []
    for i in isolated:
        box = image[max(z[i] - rz, 0):z[i] + rz + 1, max(y[i] - ry, 0):y[i] + ry + 1, max(x[i] - rx, 0):x[i] + rx + 1]
        sums.append(float(box.sum() - background[z[i], y[i], x[i]] * box.size))
    return float(np.median(sums)), int(len(isolated))


def shell_measures(labels: np.ndarray, image: np.ndarray, spacing, ids, *, shell_inner: float = 0.5,
                   shell_outer: float = 2.0, cancel=None):
    """(voxel count, summed intensity, background) per label in `ids`.

    Background is the median intensity in a shell `shell_inner`..`shell_outer`
    (frame unit) outside the cell, where each shell voxel belongs to its
    nearest cell -- neighbouring cells never share background. Computed on a
    crop around all cells plus the shell: the same numbers as on the whole
    volume (no shell voxel lies outside it), at a fraction of the cost."""
    from scipy import ndimage

    if shell_outer <= shell_inner:
        raise ValueError("shell_outer must be larger than shell_inner")
    spacing = np.asarray(spacing, float)
    ids = np.asarray(ids, int)
    occupied = np.nonzero(labels)
    if len(ids) == 0 or len(occupied[0]) == 0:
        return np.zeros(len(ids)), np.zeros(len(ids)), np.full(len(ids), np.nan)
    pad = np.ceil(shell_outer / spacing).astype(int) + 1
    box = tuple(slice(max(int(a.min()) - int(q), 0), min(int(a.max()) + int(q) + 1, n))
                for a, q, n in zip(occupied, pad, labels.shape))
    lab = np.asarray(labels[box])
    img = np.asarray(image[box], dtype=np.float64)
    distance, nearest = ndimage.distance_transform_edt(lab == 0, sampling=spacing, return_indices=True)
    check_cancel(cancel)
    owner = lab[tuple(nearest)]
    shell_labels = np.where((distance > shell_inner) & (distance <= shell_outer), owner, 0)
    count = ndimage.sum_labels(np.ones_like(lab, np.int64), lab, ids)
    total = ndimage.sum_labels(img, lab, ids)
    background = np.array([float(np.median(img[shell_labels == i])) if np.any(shell_labels == i) else np.nan
                           for i in ids])
    return count, total, background


def channel_stats(table, labels: np.ndarray, image: np.ndarray, spots, frame, *, channel: str,
                  shell_inner: float = 0.5, shell_outer: float = 2.0, saturation: float = 255.0,
                  isolation: float = 3.0, window_xy: float = 0.2, window_z: float = 0.5, cancel=None):
    """Add `<channel>_*` columns to the per-cell `table` (one row per label)."""
    from scipy import ndimage

    frame = as_frame(frame)
    labels = np.asarray(labels)
    image = np.asarray(image, dtype=np.float64)
    table = table.copy()
    ids = table["label"].to_numpy(int)
    voxel_count, total, background = shell_measures(labels, image, spacing_zyx(frame), ids,
                                                    shell_inner=shell_inner, shell_outer=shell_outer, cancel=cancel)
    mean = total / np.maximum(voxel_count, 1)
    saturated = ndimage.sum_labels((image >= saturation).astype(np.int64), labels, ids) / np.maximum(voxel_count, 1)
    spot_label = labels[spots["z_index"].to_numpy(int), spots["y_index"].to_numpy(int),
                        spots["x_index"].to_numpy(int)] if len(spots) else np.zeros(0, int)
    table[f"{channel}_mean_au"] = mean
    table[f"{channel}_integrated_au"] = total
    table[f"{channel}_background_au"] = background
    table[f"{channel}_mean_bgsub_au"] = mean - background
    table[f"{channel}_integrated_bgsub_au"] = total - background * voxel_count
    table[f"{channel}_saturated_fraction"] = saturated
    table[f"{channel}_spots"] = [int(np.sum(spot_label == i)) for i in ids]
    check_cancel(cancel)
    reference, n_reference = reference_spot(image, spots, labels, frame, isolation=isolation,
                                            window_xy=window_xy, window_z=window_z)
    table[f"{channel}_ref_spot_au"] = reference
    table[f"{channel}_n_ref_spots"] = n_reference
    table[f"{channel}_transcripts_est"] = (table[f"{channel}_integrated_bgsub_au"] / reference
                                           if reference > 0 else np.nan)
    return table


def _channel_stats(params: ChannelStatsParams, ctx, progress, cancel, log):
    table = channel_stats(ctx["table"], ctx["labels"], ctx["image"], ctx["spots"], ctx["frame"],
                          channel=params.channel, shell_inner=params.shell_inner,
                          shell_outer=params.shell_outer, saturation=params.saturation,
                          isolation=params.isolation, window_xy=params.window_xy, window_z=params.window_z,
                          cancel=cancel)
    log(f"{params.channel}: reference spot {table[f'{params.channel}_ref_spot_au'].iloc[0] if len(table) else float('nan'):.1f} a.u. "
        f"from {table[f'{params.channel}_n_ref_spots'].iloc[0] if len(table) else 0} isolated spots")
    progress(1.0, f"{params.channel}: {len(table)} cells")
    return {"table": table}


# -- cell intensity (no spots): what the cell-selection rule needs -----------------------


@dataclass
class CellIntensityParams:
    channel: str = field(default="hcr", metadata={"help": "Column prefix, e.g. ins6"})
    shell_inner: float = field(default=0.5, metadata={"range": (0.0, 20.0)})
    shell_outer: float = field(default=2.0, metadata={"range": (0.1, 50.0)})


def _cell_intensity(params: CellIntensityParams, ctx, progress, cancel, log):
    frame = as_frame(ctx["frame"])
    table = ctx["table"].copy()
    ids = table["label"].to_numpy(int)
    count, total, background = shell_measures(np.asarray(ctx["labels"]), np.asarray(ctx["image"]),
                                              spacing_zyx(frame), ids, shell_inner=params.shell_inner,
                                              shell_outer=params.shell_outer, cancel=cancel)
    mean = total / np.maximum(count, 1)
    ch = params.channel
    table[f"{ch}_mean_au"] = mean
    table[f"{ch}_background_au"] = background
    table[f"{ch}_mean_bgsub_au"] = mean - background
    progress(1.0, f"{ch}: {len(table)} cells")
    return {"table": table}


# -- selecting cells and the ROI around them ---------------------------------------------------


@dataclass
class SelectParams:
    targets: str = field(default="", metadata={
        "help": "World centroids (z,y,x in frame unit) of the cells of interest, ';'-separated; "
                "empty: every cell"})
    max_distance: float = field(default=3.0, metadata={"range": (0.1, 100.0),
                                                       "help": "A cell matches a target within this distance"})
    margin_xy: float = field(default=8.0, metadata={"range": (0.0, 500.0),
                                                    "help": "ROI margin around the cells in x/y (frame unit)"})
    margin_z: float = field(default=3.0, metadata={"range": (0.0, 500.0),
                                                   "help": "ROI margin around the cells in z (frame unit)"})
    ref_margin_xy: float = field(default=25.0, metadata={
        "range": (0.0, 1000.0), "help": "Reference-region margin in x/y (frame unit): spot detection, its "
                                        "threshold and the isolated single-spot reference use this larger box"})
    ref_margin_z: float = field(default=10.0, metadata={"range": (0.0, 1000.0),
                                                        "help": "Reference-region margin in z (frame unit)"})


def parse_targets(text: str) -> np.ndarray:
    rows = [[float(v) for v in item.split(",")] for item in text.split(";") if item.strip()]
    return np.asarray(rows, float).reshape(-1, 3)


def format_targets(points) -> str:
    return ";".join(",".join(f"{v:.4f}" for v in p) for p in np.asarray(points, float).reshape(-1, 3))


def roi_box(labels: np.ndarray, ids, spacing, margin_xy: float, margin_z: float) -> dict:
    """Index box {"z": [start, stop], "y": ..., "x": ...} around the voxels of
    `ids`, widened by the margins (frame unit), clipped to the volume."""
    mask = np.isin(labels, np.asarray(ids, int))
    if not mask.any():
        return {a: [0, int(n)] for a, n in zip("zyx", labels.shape)}
    margins = (margin_z / spacing[0], margin_xy / spacing[1], margin_xy / spacing[2])
    box = {}
    for axis, a, m, n in zip("zyx", np.nonzero(mask), margins, labels.shape):
        box[axis] = [max(int(a.min()) - int(np.ceil(m)), 0), min(int(a.max()) + int(np.ceil(m)) + 1, int(n))]
    return box


def match_targets(table, frame, targets: str, max_distance: float = 3.0):
    """The rows of `table` matched to the targets (Hungarian assignment on
    centroid distance, within `max_distance`), with their `target` index,
    in target order. Empty `targets`: every row. Used by the select step and
    by the parallel series runner's tracking, so both match identically."""
    from scipy.optimize import linear_sum_assignment

    frame = as_frame(frame)
    cols = [length_column(f"centroid_{a}", frame) for a in world_axis_names(frame)]
    points = parse_targets(targets)
    if len(points) == 0:
        selected = table.copy()
        selected["target"] = np.arange(len(selected))
        return selected
    if len(table) == 0:
        return table.assign(target=[])
    cost = np.linalg.norm(table[cols].to_numpy(float)[:, None, :] - points[None, :, :], axis=2)
    rows, target_idx = linear_sum_assignment(cost)
    keep = [(r, k) for r, k in zip(rows, target_idx) if cost[r, k] <= max_distance]
    selected = table.iloc[[r for r, _ in keep]].copy()
    selected["target"] = [k for _, k in keep]
    return selected.sort_values("target").reset_index(drop=True)


def select_cells(table, labels: np.ndarray, frame, *, targets: str = "", max_distance: float = 3.0,
                 margin_xy: float = 8.0, margin_z: float = 3.0):
    """(selected rows with their `target` index, ROI box). Cells are matched
    to the targets (Hungarian assignment on centroid distance, within
    `max_distance`), so the cells of interest are followed as the sample drifts."""
    frame = as_frame(frame)
    selected = match_targets(table, frame, targets, max_distance)
    box = roi_box(np.asarray(labels), selected["label"].to_numpy(int), spacing_zyx(frame), margin_xy, margin_z)
    return selected, box


def _select(params: SelectParams, ctx, progress, cancel, log):
    selected, box = select_cells(ctx["table"], ctx["labels"], ctx["frame"], targets=params.targets,
                                 max_distance=params.max_distance, margin_xy=params.margin_xy,
                                 margin_z=params.margin_z)
    labels = np.asarray(ctx["labels"])
    # The reference region: the same cells, wider margins -- never smaller than the ROI.
    ref = roi_box(labels, selected["label"].to_numpy(int), spacing_zyx(as_frame(ctx["frame"])),
                  max(params.ref_margin_xy, params.margin_xy), max(params.ref_margin_z, params.margin_z))
    n_targets = len(parse_targets(params.targets))
    if n_targets and len(selected) < n_targets:
        log(f"{n_targets - len(selected)} cell(s) of interest not found within {params.max_distance} "
            "of their last position")
    shape = labels.shape
    for name, b in (("ROI", box), ("reference region", ref)):
        size = np.prod([v[1] - v[0] for v in b.values()]) / max(float(np.prod(shape)), 1.0)
        log(f"{name} z{b['z']} y{b['y']} x{b['x']}: {100 * size:.1f} % of the volume")
    progress(1.0, f"{len(selected)} cells of interest")
    return {"table": selected, "roi": box, "ref_roi": ref}


@dataclass
class CropParams:
    pass


def crop_frame(frame, box: dict) -> dict:
    """The crop's frame: the same affine with the origin moved to the crop's
    first voxel, so every world coordinate is unchanged."""
    from vanua.core.datastore import WorldFrame

    frame = as_frame(frame)
    m = frame.matrix.copy()
    offset = np.array([box["z"][0], box["y"][0], box["x"][0]], float)
    m[:3, 3] = m[:3, 3] + m[:3, :3] @ offset
    doc = frame.as_document()
    doc["affine"] = [list(map(float, row)) for row in m]
    return WorldFrame.from_dict(doc).as_document()


def box_slices(box: dict):
    return tuple(slice(*box[a]) for a in "zyx")


def _crop_image(params: CropParams, ctx, progress, cancel, log):
    box = ctx["roi"]
    image = np.ascontiguousarray(np.asarray(ctx["image"])[box_slices(box)])
    progress(1.0, f"cropped to {image.shape}")
    return {"image": image, "frame": crop_frame(ctx["frame"], box)}


def _crop_labels(params: CropParams, ctx, progress, cancel, log):
    labels = np.ascontiguousarray(np.asarray(ctx["labels"])[box_slices(ctx["roi"])])
    progress(1.0, f"cropped to {labels.shape}")
    return {"labels": labels}


# -- drift across the z-stack (group experiments) ------------------------------------------------


@dataclass
class DriftParams:
    register_above: float = field(default=1.5, metadata={
        "label": "Register drift above", "unit": "µm", "range": (0.0, 50.0), "step": 0.1,
        "help": "Total sideways drift across the stack above which the slices are moved back into line"})
    jump_above: float = field(default=0.5, metadata={
        "label": "Register a jump above", "unit": "µm", "range": (0.0, 50.0), "step": 0.1,
        "help": "A sudden move between two neighbouring slices (a bump) above this is corrected too"})
    exclude_above: float = field(default=3.0, metadata={
        "label": "Exclude drift above", "unit": "µm", "range": (0.0, 100.0), "step": 0.5,
        "help": "Images that drifted more than this are left out of the analysis"})


def _estimate_drift(params: DriftParams, ctx, progress, cancel, log):
    from .qc import DriftSettings, drift_decision, drift_summary, estimate_z_drift

    drift = estimate_z_drift(ctx["image"], spacing_zyx(as_frame(ctx["frame"])))
    summary = drift_summary(drift)
    summary["decision"] = drift_decision(summary, DriftSettings(params.register_above, params.jump_above,
                                                                params.exclude_above))
    log(f"drift {summary['drift_um']:.2f} µm, largest jump {summary['max_step_um']:.2f} µm: {summary['decision']}")
    progress(1.0, summary["decision"])
    return {"table": drift, "summary": summary}


@dataclass
class ApplyDriftParams:
    apply: Literal["auto", "always", "never"] = field(default="auto", metadata={
        "label": "Register", "help": "auto: when the drift check says 'registered' (or 'excluded', if the "
                                     "image is analysed anyway); always / never"})


def _apply_drift(params: ApplyDriftParams, ctx, progress, cancel, log):
    from .qc import apply_z_drift

    decision = (ctx["summary"] or {}).get("decision", "ok")
    if params.apply == "never" or (params.apply == "auto" and decision == "ok"):
        progress(1.0, "not moved")
        return {"image": ctx["image"]}
    image = apply_z_drift(ctx["image"], ctx["drift"])
    progress(1.0, "slices moved back into line")
    return {"image": image}


# -- cache readers (the parallel series runner's second stage) -----------------------------------


@dataclass
class CachePathParams:
    path: str = field(default="", metadata={"label": "File", "help": "Cache file written by the first stage"})


def _read_label_cache(params: CachePathParams, ctx, progress, cancel, log):
    """Cell masks saved by the first stage (.npy). The images and their frame
    are read from the source again, so the frame is never taken from a cache."""
    labels = np.load(params.path)
    progress(1.0, f"cells {labels.shape}")
    return {"labels": labels.astype(np.int32, copy=False)}


def _read_table_cache(params: CachePathParams, ctx, progress, cancel, log):
    import pandas as pd

    table = pd.read_csv(params.path)
    progress(1.0, f"{len(table)} rows")
    return {"table": table}


OPS = (
    OpSpec(name="hcrcells.subtract_background", summary="Subtract a broad Gaussian background (autofluorescence)",
           inputs={"image": "image", "frame": "blob"}, outputs={"image": "image"},
           params=BackgroundParams, run=_subtract_background, version="1"),
    OpSpec(name="hcrcells.detect_spots", summary="HCR puncta: DoG local maxima above an elbow (or k x MAD noise) threshold",
           inputs={"image": "image", "frame": "blob"}, outputs={"table": "table"},
           params=SpotParams, run=_detect_spots, version="2"),
    OpSpec(name="hcrcells.channel_stats", summary="Per-cell intensity, background, saturation and spot count for one channel",
           inputs={"table": "table", "labels": "labels", "image": "image", "spots": "table", "frame": "blob"},
           outputs={"table": "table"}, params=ChannelStatsParams, run=_channel_stats, version="3"),
    OpSpec(name="hcrcells.cell_intensity", summary="Per-cell mean, local background and mean - background of one channel",
           inputs={"table": "table", "labels": "labels", "image": "image", "frame": "blob"},
           outputs={"table": "table"}, params=CellIntensityParams, run=_cell_intensity, version="1"),
    OpSpec(name="hcrcells.select_cells", summary="Keep the cells of interest (matched to targets) and box them in an ROI",
           inputs={"table": "table", "labels": "labels", "frame": "blob"},
           outputs={"table": "table", "roi": "blob", "ref_roi": "blob"}, params=SelectParams, run=_select,
           version="2"),
    OpSpec(name="hcrcells.crop_image", summary="Crop an image to the ROI; the frame keeps world coordinates",
           inputs={"image": "image", "frame": "blob", "roi": "blob"},
           outputs={"image": "image", "frame": "blob"}, params=CropParams, run=_crop_image, version="1"),
    OpSpec(name="hcrcells.read_label_cache", summary="Cell masks cached by the series runner's first stage",
           inputs={}, outputs={"labels": "labels"}, params=CachePathParams, run=_read_label_cache, version="1"),
    OpSpec(name="hcrcells.read_table_cache", summary="A per-cell table cached by the series runner's first stage",
           inputs={}, outputs={"table": "table"}, params=CachePathParams, run=_read_table_cache, version="1"),
    OpSpec(name="hcrcells.estimate_drift", summary="Sideways drift of each z-slice (phase correlation of neighbours)",
           inputs={"image": "image", "frame": "blob"}, outputs={"table": "table", "summary": "blob"},
           params=DriftParams, run=_estimate_drift, version="1"),
    OpSpec(name="hcrcells.apply_drift", summary="Move each z-slice back by its drift (registration)",
           inputs={"image": "image", "drift": "table", "summary": "blob"}, outputs={"image": "image"},
           params=ApplyDriftParams, run=_apply_drift, version="1"),
    OpSpec(name="hcrcells.crop_labels", summary="Crop a label image to the ROI",
           inputs={"labels": "labels", "roi": "blob"}, outputs={"labels": "labels"},
           params=CropParams, run=_crop_labels, version="1"),
)
