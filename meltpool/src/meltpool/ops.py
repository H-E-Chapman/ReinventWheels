"""MeltPool ops: only what no vanua block does -- opening a movie, stabilisation, crop,
running-median (transient) frames, the liquid-gas boundary (vapour-depression rim)
traced per time point, start-up flags, the steady summary, the dynamics (spectra),
the overlay video, and the liquid-solid line on the stabilised median.
Qt-free; registered next to the blocks with ``build_registry(*OPS)``."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from vanua.blocks._frame import as_frame, unit_suffix
from vanua.core.cancellation import check_cancel
from vanua.core.datastore import WorldFrame
from vanua.core.ops import OpSpec

from . import boundary
from . import movie as mv

__all__ = ["OPS", "CropParams", "TraceParams", "crop_frame", "pixel_geometry", "trace_one"]

MEASURED = ("area", "width", "depth")


def pixel_geometry(frame) -> "tuple[float, float, float]":
    """(y size, x size, pixel area) in the frame's unit -- exact for an oblique
    frame: the area is |col_y x col_x| of the affine, not a product of spacings."""
    m = as_frame(frame).matrix[:3, :3]
    col_y, col_x = m[:, 1], m[:, 2]
    return float(np.linalg.norm(col_y)), float(np.linalg.norm(col_x)), float(np.linalg.norm(np.cross(col_y, col_x)))


# -- open --------------------------------------------------------------------------

@dataclass
class OpenParams:
    path: str = field(default="", metadata={
        "label": "Movie", "help": "TIFF (one frame per page), HDF5/NeXus, .npy, or video (AVI, MP4, MOV, MKV ...)"})
    dataset: str = field(default="", metadata={
        "label": "HDF5 dataset", "help": "Path of the (time, y, x) dataset; empty = the largest 3D dataset"})
    pixel_um: float = field(default=4.44, metadata={"label": "Pixel size", "unit": "µm", "range": (1e-3, 1e4),
                                                    "step": 0.01, "decimals": 3})
    pixel_from_file: bool = field(default=True, metadata={
        "label": "Pixel size from file", "help": "Use the file's own pixel size (ImageJ TIFF) when it has one"})
    fps: float = field(default=20000.0, metadata={
        "label": "Frame rate", "unit": "fps", "range": (1e-3, 1e8),
        "help": "The recording rate. A rate stored in a file is usually a playback rate and is not used; "
                "for an export that kept every Nth frame enter recording rate / N"})
    flip_vertical: bool = field(default=True, metadata={"label": "Flip vertically",
                                                        "help": "Recorded upside down: show the build direction up"})
    flip_horizontal: bool = field(default=False, metadata={"label": "Flip horizontally",
                                                           "help": "Mirror left-right (e.g. to make the travel direction the same in every run)"})
    blank_fraction: float = field(default=0.5, metadata={
        "label": "Blank below", "range": (0.0, 1.0), "step": 0.05,
        "help": "A frame whose mean is below this fraction of the run's median is blank (shutter closed)"})


def frames_table(means: np.ndarray, fps: float, blank_fraction: float):
    import pandas as pd

    blank = means < blank_fraction * np.median(means)
    first = int(np.argmax(~blank)) if (~blank).any() else 0
    index = np.arange(len(means))
    return pd.DataFrame({"frame": index, "time_ms": (index - first) / fps * 1000.0,
                         "mean_intensity": means, "blank": blank})


def _open_movie(params: OpenParams, ctx, progress, cancel, log):
    blob = mv.describe_movie(params.path, params.flip_vertical, params.dataset, log)
    blob["flip_horizontal"] = bool(params.flip_horizontal)
    blob["fps"] = float(params.fps)
    meta = blob.get("file_meta", {})
    pixel = params.pixel_um
    if params.pixel_from_file and meta.get("pixel_um"):
        pixel = float(meta["pixel_um"])
        if abs(pixel - params.pixel_um) > 1e-6:
            log(f"pixel size {pixel:.4g} um from the file (entered: {params.pixel_um:.4g} um)")
    blob["pixel_um"] = pixel
    if meta.get("fps_in_file"):
        log(f"the file says {meta['fps_in_file']:g} fps (a playback rate, not used); using {params.fps:g} fps")
    if blob["kind"] == "tiff" and not blob["memmap"]:
        log("pages are not evenly spaced/uncompressed: reading page by page (slower)")
    movie = mv.Movie(blob)
    try:
        means = mv.frame_means(movie, progress, cancel)
    finally:
        movie.close()
    table = frames_table(means, params.fps, params.blank_fraction)
    frame = WorldFrame.from_spacing((1.0, pixel, pixel), unit="um")
    log(f"{meta.get('format', blob['kind'])}: {blob['n']} frames of {blob['shape'][0]} x {blob['shape'][1]} "
        f"({np.dtype(blob['dtype']).name}); {int(table.blank.sum())} blank")
    progress(1.0, f"{blob['n']} frames")
    return {"movie": blob, "frame": frame.as_document(), "frames": table}


# -- crop --------------------------------------------------------------------------------

@dataclass
class CropParams:
    x0: int = field(default=0, metadata={"label": "Left edge (x from)", "unit": "px", "range": (0, 100000),
                                         "help": "First pixel column kept, counted from the left of the frame as shown"})
    x1: int = field(default=0, metadata={"label": "Right edge (x to)", "unit": "px", "range": (0, 100000),
                                         "help": "Column where the ROI ends, counted from the LEFT of the frame "
                                                 "(not a margin); 0 = the right edge of the frame"})
    y0: int = field(default=0, metadata={"label": "Top edge (y from)", "unit": "px", "range": (0, 100000),
                                         "help": "First pixel row kept, counted from the top of the frame as shown"})
    y1: int = field(default=0, metadata={"label": "Bottom edge (y to)", "unit": "px", "range": (0, 100000),
                                         "help": "Row where the ROI ends, counted from the TOP; 0 = the bottom edge"})


def crop_frame(frame, y0: int, x0: int) -> dict:
    """The frame of a crop starting at full-frame pixel (y0, x0): the same
    affine with its origin moved to that pixel -- spacing and direction kept."""
    f = as_frame(frame)
    m = f.matrix.copy()
    m[:3, 3] = (m @ np.array([0.0, y0, x0, 1.0]))[:3]
    return WorldFrame(unit=f.unit, affine=tuple(tuple(float(v) for v in row) for row in m),
                      axes=f.axes, space=f.space).as_document()


def _crop(params: CropParams, ctx, progress, cancel, log):
    blob = mv.crop_blob(ctx["movie"], params.y0, params.y1, params.x0, params.x1)
    y0, y1, x0, x1 = blob["crop"]
    full = "full frame" if (y0, x0, y1, x1) == (0, 0, *blob["shape"]) else "ROI"
    log(f"{full}: rows {y0}:{y1}, columns {x0}:{x1} ({y1 - y0} x {x1 - x0} px)")
    progress(1.0, f"{y1 - y0} x {x1 - x0} px")
    return {"movie": blob, "frame": crop_frame(ctx["frame"], y0, x0)}


# -- flags ---------------------------------------------------------------------------------

@dataclass
class FlagParams:
    column: str = field(default="area_centre_um2", metadata={"label": "Steady-state column"})
    tolerance: float = field(default=0.10, metadata={
        "label": "Tolerance", "range": (0.0, 10.0), "step": 0.01,
        "help": "Steady when the rolling median is within this fraction of the late-run median"})
    rolling: int = field(default=21, metadata={"label": "Rolling window", "unit": "time points", "range": (1, 10001),
                                               "help": "Also: how many consecutive steady points start the steady state"})
    late_fraction: float = field(default=0.5, metadata={"label": "Late part of run", "range": (0.05, 1.0),
                                                        "step": 0.05})


def flag_table(table, p: FlagParams):
    if p.column not in table.columns:
        raise ValueError(f"no column {p.column!r}; columns: {', '.join(map(str, table.columns))}")
    table = table.copy()
    blank = table["blank"].to_numpy(bool)
    values = table[p.column].astype(float).where(~table["blank"])
    rolled = values.rolling(p.rolling, center=True, min_periods=1).median().to_numpy()
    live = np.flatnonzero(~blank)
    transient = ~blank.copy()
    late_median = float("nan")
    if len(live):
        late = live[int(len(live) * (1 - p.late_fraction)):]
        late_median = float(np.nanmedian(values.to_numpy()[late]))
        within = np.abs(rolled - late_median) <= p.tolerance * abs(late_median)
        within[blank] = False
        run = min(p.rolling, len(live))
        steady_from = next((k for k in live if within[k:k + run].all() and len(within[k:k + run]) == run), None)
        if steady_from is not None:
            transient = (~blank) & (np.arange(len(table)) < steady_from)
    table["transient"] = transient
    table["flagged"] = blank | transient
    table["flag_reason"] = np.where(blank, "blank", np.where(transient, "transient", ""))
    table.attrs["late_median"] = late_median
    return table


def _flag(params: FlagParams, ctx, progress, cancel, log):
    table = flag_table(ctx["table"], params)
    n_tr, n_bl = int(table.transient.sum()), int(table.blank.sum())
    steady = table.loc[~table.flagged, "time_ms"]
    log(f"{n_bl} blank, {n_tr} transient; steady from {steady.min():.2f} ms" if len(steady)
        else f"{n_bl} blank, {n_tr} transient; never steady")
    progress(1.0, "flagged")
    return {"table": table}


# -- summary --------------------------------------------------------------------------------

@dataclass
class SummaryParams:
    pass


def summary_table(table, peaks=None):
    import pandas as pd

    ok = table.loc[~table["flagged"].astype(bool)]
    row = {"n_time_points": int(len(table)), "n_blank": int(table["blank"].sum()),
           "n_transient": int(table["transient"].sum()), "n_steady": int(len(ok)),
           "steady_from_ms": float(ok["time_ms"].min()) if len(ok) else float("nan"),
           "steady_to_ms": float(ok["time_ms"].max()) if len(ok) else float("nan")}
    for column in table.columns:
        if any(str(column).startswith(m + "_") for m in MEASURED):
            v = ok[column].astype(float)
            base, _, suffix = str(column).rpartition("_")
            for stat, value in (("median", v.median()), ("q25", v.quantile(0.25)), ("q75", v.quantile(0.75))):
                row[f"{base}_{stat}_{suffix}"] = float(value) if len(v) else float("nan")
    if peaks is not None:
        for line in boundary.LINES:
            top = peaks[peaks["series"].str.startswith(f"depth_{line}_") & (peaks["rank"] == 1)] if len(peaks) else []
            row[f"depth_{line}_dominant_hz"] = float(top["freq_hz"].iloc[0]) if len(top) else float("nan")
    return pd.DataFrame([row])


def _summary(params: SummaryParams, ctx, progress, cancel, log):
    table = summary_table(ctx["table"], ctx.get("peaks"))
    progress(1.0, "summarised")
    return {"table": table}



# -- dynamics -----------------------------------------------------------------------------------

@dataclass
class DynamicsParams:
    detrend: bool = field(default=True, metadata={"label": "Remove linear trend"})
    n_peaks: int = field(default=3, metadata={"label": "Peaks reported", "range": (1, 20)})
    steady_only: bool = field(default=True, metadata={
        "label": "Steady part only", "help": "Leave the flagged (blank / start-up) time points out of the spectra"})


def dynamics(table, params: DynamicsParams):
    """(spectrum table, peaks table) of every depth, width and area column."""
    import pandas as pd

    if params.steady_only and "flagged" in table:
        ok = ~table["flagged"].astype(bool).to_numpy()
        # the longest run of steady points: a spectrum needs evenly spaced samples
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        runs = list(zip(edges[::2], edges[1::2]))
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            table = table.iloc[a:b]
    t = table["time_ms"].to_numpy(float)
    series = [c for c in table.columns if str(c).startswith(("depth_", "width_", "area_"))]
    spec_cols, peaks, freq = {}, [], None
    for c in series:
        f, amp = boundary.spectrum(t, table[c].to_numpy(float), params.detrend)
        if not len(f):
            continue
        freq = f
        spec_cols[f"amp_{c}"] = amp
        for rank, (fk, ak) in enumerate(boundary.dominant(f, amp, params.n_peaks), 1):
            peaks.append({"series": c, "rank": rank, "freq_hz": fk, "amplitude": ak})
    if freq is None:
        raise ValueError("too few steady time points for a spectrum (need 8)")
    return (pd.DataFrame({"freq_hz": freq, **spec_cols}),
            pd.DataFrame(peaks, columns=["series", "rank", "freq_hz", "amplitude"]))


def _dynamics(params: DynamicsParams, ctx, progress, cancel, log):
    spectrum, peaks = dynamics(ctx["table"], params)
    top = peaks[peaks["series"].str.startswith("depth_centre") & (peaks["rank"] == 1)]
    if len(top):
        log(f"dominant depth oscillation (band centre): {top['freq_hz'].iloc[0]:.0f} Hz")
    progress(1.0, "spectrum")
    return {"spectrum": spectrum, "peaks": peaks}


# -- liquid-gas boundary tracing (the rim of the vapour depression) ----------------------------

@dataclass
class TraceParams:
    n_avg: int = field(default=3, metadata={"label": "Frames averaged", "unit": "frames", "range": (1, 500),
                                            "help": "Each time point is the mean of this many frames (3 = 30 us at 100 kfps)"})
    stride: int = field(default=1, metadata={"label": "Every Nth frame", "range": (1, 10000)})
    start: int = field(default=0, metadata={"label": "First frame", "range": (0, 10**9), "help": "0: first lit frame"})
    stop: int = field(default=0, metadata={"label": "Last frame", "range": (0, 10**9), "help": "0: the end"})
    r_min: float = field(default=15.0, metadata={"label": "Min radius", "unit": "px", "range": (2.0, 5000.0)})
    r_max: float = field(default=150.0, metadata={"label": "Max radius", "unit": "px", "range": (10.0, 5000.0),
                                                  "help": "How far below/around the centre the rim may be"})
    center_x: int = field(default=0, metadata={"label": "Centre column", "unit": "px", "range": (0, 100000),
                                               "help": "Full-frame column of the depression centre; 0 = found and tracked"})
    surface_y: int = field(default=0, metadata={"label": "Surface row", "unit": "px", "range": (0, 100000),
                                                "help": "Full-frame row of the original surface; 0 = found"})
    edge_fraction: float = field(default=0.5, metadata={
        "label": "Edge level", "range": (0.05, 0.95), "step": 0.05,
        "help": "Band edges where the intensity is this fraction of the way from the band minimum to each side"})


def trace_settings(p: TraceParams) -> boundary.TraceSettings:
    return boundary.TraceSettings(r_min=p.r_min, r_max=p.r_max, edge_fraction=p.edge_fraction)


def trace_time_points(n: int, frames, p: TraceParams) -> np.ndarray:
    lit = np.flatnonzero(~frames["blank"].to_numpy(bool))
    if len(lit) == 0:
        return np.zeros(0, int)
    start = max(p.start, int(lit[0]))
    stop = min(p.stop or n, n - p.n_avg + 1)
    return np.arange(start, stop, p.stride)


def trace_prepare(movie: mv.Movie, points: np.ndarray, p: TraceParams) -> "tuple[float, float]":
    """Surface row and initial centre column (both in the movie's own pixels). Found on the
    plain frames: in transient (running-median) frames the steady air/material contrast is gone."""
    oy, ox = mv.crop_offset(movie.blob)
    plain = movie if not movie.blob.get("background") else mv.Movie(
        {k: v for k, v in movie.blob.items() if k != "background"})
    early = plain.mean_of(int(points[0]), min(20, max(1, len(points))))
    surface = float(p.surface_y - oy) if p.surface_y else mv.original_surface(early)
    if not np.isfinite(surface):
        raise ValueError("no air/material surface found; set the surface row by hand")
    s = trace_settings(p)
    cx = float(p.center_x - ox) if p.center_x else boundary.recentre(early, surface, mv.depression_centre(early, surface), s)
    return surface, cx


def trace_one(movie: mv.Movie, i: int, surface: float, cx: float, p: TraceParams, frame, track: bool = True) -> dict:
    """One time point: the averaged image, the three lines and their measurements."""
    s = trace_settings(p)
    image = movie.mean_of(int(i), p.n_avg)
    if track and not p.center_x:
        cx = boundary.recentre(image, surface, cx, s, iterations=1)
    trace = boundary.trace_frame(image, surface, cx, s)
    py, px, area = pixel_geometry(frame)
    return {"image": image, "trace": trace, "values": boundary.measure(trace, image.shape, (py, px), area)}


def _trace_boundary(params: TraceParams, ctx, progress, cancel, log):
    import pandas as pd

    movie, frame, frames = mv.Movie(ctx["movie"]), as_frame(ctx["frame"]), ctx["frames"]
    unit = unit_suffix(frame)
    py, px, _ = pixel_geometry(frame)
    oy, ox = mv.crop_offset(movie.blob)
    points = trace_time_points(movie.n, frames, params)
    if len(points) == 0:
        raise ValueError("no time points (all frames blank, or start/stop leave none)")
    surface, cx = trace_prepare(movie, points, params)
    log(f"surface row {surface + oy:.1f}, depression centre column {cx + ox:.1f} (full frame)")
    rows, radii, centres = [], {line: [] for line in boundary.LINES}, []
    time_ms = frames["time_ms"].to_numpy(float)
    blank = frames["blank"].to_numpy(bool)
    for j, i in enumerate(points):
        check_cancel(cancel)
        if j % 10 == 0:
            progress(j / len(points), f"frame {i} ({j + 1}/{len(points)})")
        out = trace_one(movie, int(i), surface, cx, params, frame)
        cx = out["trace"]["cx"]
        centres.append(cx)
        for line in boundary.LINES:
            radii[line].append(out["trace"]["r"][line])
        v = out["values"]
        row = {"frame": int(i), "time_ms": float(time_ms[i]), "blank": bool(blank[i:i + params.n_avg].any()),
               f"centre_x_{unit}": (cx + ox) * px}
        for line in boundary.LINES:
            row[f"depth_{line}_{unit}"] = v[f"depth_{line}"]
            row[f"width_{line}_{unit}"] = v[f"width_{line}"]
            row[f"area_{line}_{unit}2"] = v[f"area_{line}"]
        rows.append(row)
    table = pd.DataFrame(rows)
    profiles = {"theta": boundary.thetas(), "frames": points.astype(int), "time_ms": time_ms[points],
                "cx": np.asarray(centres), "surface": float(surface), "offset": (int(oy), int(ox)),
                "pixel": (py, px), "unit": unit, "r": {line: np.asarray(radii[line]) for line in boundary.LINES}}
    log(f"{len(table)} time points traced (n_avg {params.n_avg}, stride {params.stride})")
    progress(1.0, f"{len(table)} time points")
    return {"table": table, "profiles": profiles}


def profile_trace(profiles: dict, k: int) -> dict:
    return {"cx": float(profiles["cx"][k]), "surface": float(profiles["surface"]),
            "r": {line: np.asarray(profiles["r"][line][k]) for line in boundary.LINES}}



# -- overlay video of the traced lines -----------------------------------------------------------

@dataclass
class OverlayParams:
    out_path: str = field(default="out/overlay.mp4", metadata={
        "label": "Save to", "help": ".mp4 (H.264 / MPEG-4, needs PyAV) or .gif"})
    fps_out: float = field(default=25.0, metadata={"label": "Playback rate", "unit": "fps", "range": (1.0, 120.0)})
    every: int = field(default=1, metadata={"label": "Every Nth time point", "range": (1, 10000)})
    lines: str = field(default="inner,centre,outer", metadata={
        "label": "Lines", "help": "Any of inner, centre, outer (comma-separated)"})
    timestamp: bool = field(default=True, metadata={"label": "Time stamp", "help": "Frame number and time on each frame"})
    low: float = field(default=0.0, metadata={"label": "Display min", "help": "Min = max: 0.5-99.5 percentile"})
    high: float = field(default=0.0, metadata={"label": "Display max"})
    gamma: float = field(default=1.0, metadata={"label": "Display gamma", "range": (0.1, 5.0), "step": 0.05})


def _overlay(params: OverlayParams, ctx, progress, cancel, log):
    movie, profiles = mv.Movie(ctx["movie"]), ctx["profiles"]
    lines = tuple(x.strip() for x in params.lines.split(",") if x.strip() in boundary.LINES) or boundary.LINES
    picks = range(0, len(profiles["frames"]), params.every)
    first = movie.frame(int(profiles["frames"][0]))
    lo, hi = (params.low, params.high) if params.high > params.low else tuple(np.percentile(first, (0.5, 99.5)))
    times = profiles.get("time_ms")

    def frames():
        for j, k in enumerate(picks):
            check_cancel(cancel)
            if j % 20 == 0:
                progress(j / max(1, len(picks)), f"overlay frame {j + 1}/{len(picks)}")
            i = int(profiles["frames"][k])
            label = ""
            if params.timestamp:
                label = f"frame {i}" + (f"   t = {float(times[k]):.3f} ms" if times is not None else "")
            yield boundary.overlay_frame(movie.frame(i), profile_trace(profiles, k), (lo, hi), lines=lines,
                                         label=label, gamma=params.gamma)

    path = boundary.write_overlay(frames(), params.out_path, params.fps_out)
    movie.close()
    log(f"overlay video: {path} ({len(picks)} frames, lines {', '.join(lines)})")
    progress(1.0, "overlay written")
    return {"movie_file": path}



# -- transient features: subtract a running median ------------------------------------------------

@dataclass
class BackgroundParams:
    enabled: bool = field(default=False, metadata={
        "label": "Subtract running median", "help": "Show what changes: each frame minus the median of its neighbours"})
    window: int = field(default=50, metadata={"label": "Median window", "unit": "frames", "range": (3, 100000)})
    mode: Literal["centred", "trailing"] = field(default="centred", metadata={
        "label": "Window", "help": "centred: frames before and after; trailing: only earlier frames"})
    operation: Literal["subtract", "divide"] = field(default="subtract", metadata={
        "label": "Operation", "help": "subtract: frame - median (grey levels); divide: frame / median - 1 (relative)"})
    every: int = field(default=5, metadata={"label": "Recompute every", "unit": "frames", "range": (1, 10000),
                                            "help": "The median is recomputed every N frames (speed on long movies)"})
    apply_to: Literal["display", "display and analysis"] = field(default="display", metadata={
        "label": "Use for",
        "help": "display: only what you see (views, overlay video); display and analysis: tracing uses the "
                "transient frames too -- a stationary rim cancels in them"})


def background_spec(p: BackgroundParams) -> dict:
    return {"window": int(p.window), "mode": p.mode, "operation": p.operation, "every": int(p.every)}


def _background(params: BackgroundParams, ctx, progress, cancel, log):
    blob = dict(ctx["movie"])
    blob.pop("background", None)
    display, analysis = dict(blob), dict(blob)
    if params.enabled:
        display["background"] = background_spec(params)
        if params.apply_to == "display and analysis":
            analysis["background"] = background_spec(params)
        log(f"running median ({params.mode}, {params.window} frames, {params.operation}) for "
            f"{params.apply_to}")
    else:
        log("no background subtraction")
    progress(1.0, "background" if params.enabled else "off")
    return {"movie": analysis, "display_movie": display}


# -- stabilise --------------------------------------------------------------------------------

@dataclass
class StabiliseParams:
    enabled: bool = field(default=False, metadata={
        "label": "Stabilise (sub-pixel)", "help": "Register every frame to 1/upsample px and shift it (Fourier)"})
    mode: Literal["jitter", "fixed"] = field(default="jitter", metadata={
        "label": "Mode",
        "help": "jitter: remove shake, keep a steady sample drift (moving sample). "
                "fixed: register to one reference, remove all motion (static sample)"})
    x0: int = field(default=0, metadata={"label": "Region left", "unit": "px", "range": (0, 100000),
                                         "help": "Region registered on (full frame as shown). All 0 = the "
                                                 "material from 10 px below the surface down, full width"})
    x1: int = field(default=0, metadata={"label": "Region right", "unit": "px", "range": (0, 100000)})
    y0: int = field(default=0, metadata={"label": "Region top", "unit": "px", "range": (0, 100000)})
    y1: int = field(default=0, metadata={"label": "Region bottom", "unit": "px", "range": (0, 100000)})
    upsample: int = field(default=100, metadata={"label": "Precision", "unit": "1/px", "range": (1, 1000),
                                                 "help": "Shifts are found to 1/upsample of a pixel"})
    n_ref: int = field(default=50, metadata={"label": "Reference frames", "range": (1, 5000),
                                             "help": "fixed: median of this many frames is the reference"})
    detrend: int = field(default=51, metadata={"label": "Drift window", "unit": "frames", "range": (3, 100001),
                                               "help": "jitter: motion slower than this is kept"})
    texture_sigma: float = field(default=12.0, metadata={
        "label": "Shading removal", "unit": "px", "range": (2.0, 200.0),
        "help": "Shading (beam profile) broader than this is removed before registering"})


def stabilise_region(blob, frames, p: StabiliseParams) -> "tuple[slice, slice]":
    h, w = (int(v) for v in blob["shape"])
    if p.x0 or p.x1 or p.y0 or p.y1:
        x0, x1 = sorted((min(p.x0, w), min(p.x1 or w, w)))
        y0, y1 = sorted((min(p.y0, h), min(p.y1 or h, h)))
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise ValueError(f"stabilisation region {x1 - x0} x {y1 - y0} px is too small (min 16 x 16)")
        return slice(y0, y1), slice(x0, x1)
    # Automatic: the solid substrate only. The pool must stay out of the region -- a moving
    # boundary inside it would be "stabilised" away (its motion taken for camera shake).
    movie = mv.Movie(blob)
    lit = np.flatnonzero(~frames["blank"].to_numpy(bool))
    image = movie.mean_of(int(lit[len(lit) // 2]) if len(lit) else 0, 5)
    surface = mv.original_surface(image)
    if not np.isfinite(surface) or surface + 40 >= h:
        return slice(0, h), slice(0, w)
    top = int(surface) + 10
    try:
        cx = mv.depression_centre(image, surface)
        _rx, ry = mv.rim_trace(image, surface, cx)
        top = max(top, int(np.ceil(ry.max())) + 20)  # below the deepest point of the pool
    except Exception:  # noqa: BLE001 -- no recognisable depression: keep the band below the surface
        pass
    if h - top < 32:
        top = max(0, h - 32)
    return slice(top, h), slice(0, w)


def _stabilise(params: StabiliseParams, ctx, progress, cancel, log):
    import pandas as pd

    blob = dict(ctx["movie"])
    frames = ctx["frames"]
    blob.pop("shifts", None)
    if not params.enabled:
        log("stabilisation off")
        shifts = np.zeros((blob["n"], 2))
    else:
        region = stabilise_region(blob, frames, params)
        log(f"stabilising on rows {region[0].start}:{region[0].stop}, columns {region[1].start}:{region[1].stop}")
        movie = mv.Movie(blob)
        try:
            shifts = mv.estimate_shifts(movie, ~frames["blank"].to_numpy(bool), region, mode=params.mode,
                                        n_ref=params.n_ref, upsample=params.upsample,
                                        sigmas=(1.0, params.texture_sigma), detrend=params.detrend,
                                        progress=progress, cancel=cancel, log=log)
        finally:
            movie.close()
        blob["shifts"] = shifts.tolist()
    progress(1.0, "stabilised" if params.enabled else "off")
    table = pd.DataFrame({"frame": np.arange(blob["n"]), "dy_px": shifts[:, 0], "dx_px": shifts[:, 1]})
    return {"movie": blob, "shifts": table}


# -- median image -------------------------------------------------------------------------------

@dataclass
class MedianParams:
    start: int = field(default=0, metadata={"label": "First frame", "range": (0, 10**9),
                                            "help": "0: from the first lit frame"})
    stop: int = field(default=0, metadata={"label": "Last frame", "range": (0, 10**9), "help": "0: to the end"})


def _median(params: MedianParams, ctx, progress, cancel, log):
    movie = mv.Movie(ctx["movie"])
    lit = np.flatnonzero(~ctx["frames"]["blank"].to_numpy(bool))
    lit = lit[(lit >= params.start) & ((lit < params.stop) if params.stop else True)]
    if len(lit) == 0:
        raise ValueError("no lit frames in the chosen range")
    try:
        image = mv.median_image(movie, lit, progress, cancel)
    finally:
        movie.close()
    log(f"median of {len(lit)} frames ({int(lit[0])}..{int(lit[-1])})"
        + (" -- stabilised" if ctx["movie"].get("shifts") is not None else " -- NOT stabilised"))
    progress(1.0, "median")
    return {"image": image[np.newaxis]}


# -- liquid-solid boundary on the median ----------------------------------------------------------

@dataclass
class SolidBoundaryParams:
    points: str = field(default="", metadata={
        "label": "Traced points", "help": "x,y; x,y; ... in full-frame pixels, along the liquid-solid line "
                                          "from one end at the surface to the other (empty: automatic search)"})
    surface_y: int = field(default=0, metadata={"label": "Surface row", "unit": "px", "range": (0, 100000),
                                                "help": "Full-frame row of the surface; 0 = found in the median"})
    center_x: int = field(default=0, metadata={"label": "Depression centre", "unit": "px", "range": (0, 100000),
                                               "help": "Automatic search: centre column; 0 = from the rim"})
    z_min: float = field(default=3.0, metadata={
        "label": "Detection threshold", "unit": "z", "range": (0.0, 50.0),
        "help": "Automatic search reports a line only if it beats searches without a pool by this many SD"})


def parse_points(text: str) -> np.ndarray:
    pts = []
    for part in filter(None, (p.strip() for p in text.replace("\n", ";").split(";"))):
        x, _, y = part.partition(",")
        pts.append((float(x), float(y)))
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def line_row(xs, ys, surface, pixel, pixel_area, shape, offset) -> dict:
    """Depth, width and enclosed area of a line below the surface (frame units)."""
    py, px = pixel
    mask = mv.region_below(shape, np.asarray(xs) - offset[1], np.asarray(ys) - offset[0], surface - offset[0])
    return {"depth": float((np.max(ys) - surface) * py), "width": float((np.max(xs) - np.min(xs)) * px),
            "area": float(mask.sum() * pixel_area), "x_min": float(np.min(xs) * px), "x_max": float(np.max(xs) * px),
            "deepest_x": float(xs[int(np.argmax(ys))] * px)}


def _solid_boundary(params: SolidBoundaryParams, ctx, progress, cancel, log):
    import pandas as pd

    image = np.asarray(ctx["image"])[0]
    blob = ctx["movie"]
    oy, ox = mv.crop_offset(blob)
    py, px, area = pixel_geometry(ctx["frame"])
    unit = unit_suffix(as_frame(ctx["frame"]))
    if params.surface_y:
        surface = float(params.surface_y)
    else:
        surface = mv.original_surface(image) + oy
    row = {"method": "", "detected": False, "z": float("nan"), "n_points": 0, "points": params.points.strip()}
    pts = parse_points(params.points)
    if len(pts) >= 2:
        pts = pts[np.argsort(pts[:, 0])]
        xs, ys = pts[:, 0], pts[:, 1]
        row.update(method="manual", detected=True, n_points=len(pts))
        log(f"liquid-solid line traced by hand: {len(pts)} points")
    else:
        local_surface = surface - oy
        if params.center_x:
            cx = float(params.center_x - ox)
        else:
            cx = mv.depression_centre(image, local_surface)
        rx, ry = mv.rim_trace(image, local_surface, cx)
        if not params.center_x:
            for _ in range(2):  # re-centre on the traced rim: the midpoint of its two ends
                cx = float((rx.min() + rx.max()) / 2)
                rx, ry = mv.rim_trace(image, local_surface, cx)
        found = mv.solid_line_auto(image, local_surface, rx, ry, cx, z_min=params.z_min)
        xs, ys = found["x"] + ox, found["y"] + oy
        row.update(method="auto", detected=found["detected"], z=found["z"], n_points=len(xs))
        log(f"automatic liquid-solid search: z = {found['z']:.1f} against {len(found['controls'])} places "
            f"without a pool ({'detected' if found['detected'] else 'NOT detectable'}; threshold {params.z_min})")
    if row["detected"]:
        r = line_row(xs, ys, surface, (py, px), area, image.shape, (oy, ox))
    else:
        r = {k: float("nan") for k in ("depth", "width", "area", "x_min", "x_max", "deepest_x")}
    table = pd.DataFrame([{**row, f"surface_y_{unit}": surface * py, f"depth_{unit}": r["depth"],
                           f"width_{unit}": r["width"], f"area_{unit}2": r["area"], f"x_min_{unit}": r["x_min"],
                           f"x_max_{unit}": r["x_max"], f"deepest_x_{unit}": r["deepest_x"]}])
    progress(1.0, row["method"] or "none")
    return {"table": table, "line": {"x": [float(v) for v in xs], "y": [float(v) for v in ys],
                                     "detected": row["detected"]}}

OPS = (
    OpSpec(name="meltpool.open_movie",
           summary="Open a movie (TIFF, HDF5, .npy, video -- chosen by content); frame, per-frame means, blank flags",
           inputs={}, outputs={"movie": "blob", "frame": "blob", "frames": "table"}, params=OpenParams,
           run=_open_movie, version="2"),
    OpSpec(name="meltpool.trace_boundary",
           summary="Liquid-gas boundary per time point: the vapour-depression rim's inner edge, band centre and outer edge; depth, width, area; profiles",
           inputs={"movie": "blob", "frame": "blob", "frames": "table"},
           outputs={"table": "table", "profiles": "blob"}, params=TraceParams, run=_trace_boundary, version="1"),
    OpSpec(name="meltpool.overlay_movie", summary="The movie with the traced lines (and a time stamp) drawn, as MP4 or GIF",
           inputs={"movie": "blob", "profiles": "blob"}, outputs={"movie_file": "file"}, params=OverlayParams,
           run=_overlay, version="1"),
    OpSpec(name="meltpool.subtract_background",
           summary="Transient features: each frame minus (or divided by) a running median; for display only, or for the analysis too",
           inputs={"movie": "blob"}, outputs={"movie": "blob", "display_movie": "blob"}, params=BackgroundParams,
           run=_background, version="1"),
    OpSpec(name="meltpool.stabilise",
           summary="Sub-pixel stabilisation: register every frame (x and y) on the texture, Fourier-shift it",
           inputs={"movie": "blob", "frames": "table"}, outputs={"movie": "blob", "shifts": "table"},
           params=StabiliseParams, run=_stabilise, version="1"),
    OpSpec(name="meltpool.median_image", summary="Per-pixel temporal median of the (stabilised) movie",
           inputs={"movie": "blob", "frames": "table"}, outputs={"image": "image"}, params=MedianParams,
           run=_median, version="1"),
    OpSpec(name="meltpool.solid_boundary",
           summary="Liquid-solid line on the median: traced by hand, or an automatic search reported only if it beats no-pool controls",
           inputs={"image": "image", "frame": "blob", "movie": "blob"}, outputs={"table": "table", "line": "blob"},
           params=SolidBoundaryParams, run=_solid_boundary, version="1"),
    OpSpec(name="meltpool.crop", summary="Restrict the movie to a region of interest (frame origin moved with it)",
           inputs={"movie": "blob", "frame": "blob"}, outputs={"movie": "blob", "frame": "blob"},
           params=CropParams, run=_crop, version="1"),
    OpSpec(name="meltpool.flag_transient", summary="Flag blank and start-up time points (auto steady state)",
           inputs={"table": "table"}, outputs={"table": "table"}, params=FlagParams, run=_flag, version="1"),
    OpSpec(name="meltpool.dynamics",
           summary="Amplitude spectra of depth, width and area (steady part); dominant frequencies",
           inputs={"table": "table"}, outputs={"spectrum": "table", "peaks": "table"}, params=DynamicsParams,
           run=_dynamics, version="1"),
    OpSpec(name="meltpool.steady_summary",
           summary="Steady-state median and IQR of every size/shape column; the dominant depth frequencies",
           inputs={"table": "table", "peaks": "table"}, outputs={"table": "table"}, params=SummaryParams,
           run=_summary, version="2"),
)
