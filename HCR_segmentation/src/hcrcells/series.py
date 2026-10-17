"""Headless: photobleaching series, cells of interest tracked across T, spots in an ROI.

    python -m hcrcells.series originals out [--samples S1,S2] [--last 5] [--override S1:E=exclude]

Every '<series> - T=<n>.tif' file in the folder is one timepoint of one
sample (the series name is the CZI it was exported from; 'repeatall.2.czi'
-> sample S2). Per sample:

1. the first timepoint's cells are segmented (whole volume, no spots) and
   named A, B, ... by depth;
2. the cells of interest are chosen by the rule in
   :func:`hcrcells.pipeline.apply_exclusion` (no ins-6 signal: excluded),
   with per-cell overrides;
3. every timepoint is segmented on the whole volume (the worm may drift), its
   cells of interest are matched to their last known positions (Hungarian
   assignment on centroid distance, within `max_distance` µm), and spots and
   per-cell measurements are computed only in a loose ROI around them.

A cell of interest missing at some T keeps its last position for matching,
so one miss does not lose it; it has no row at that T (a gap in the curves,
not a wrong number).
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np

from .plots import PlotOptions, cells_figure, relative_to_first
from .plots import batch_figure as _batch_figure
from .plots import mean_figure as _mean_figure
from .ops import format_targets
from .pipeline import (CHANNELS, GENE, RoiSettings, Settings, apply_exclusion, default_workers, load_recipe,
                       registry, run_spots, run_timepoint, segment_cells, settings_document)

CENTROID = ["centroid_z_um", "centroid_y_um", "centroid_x_um"]

COUNT = "transcripts_est"     # the per-cell count plotted: spots + dense signal in transcript equivalents
INTENSITY = "mean_bgsub_au"   # background-subtracted mean intensity
_SERIES_NAME = re.compile(r"^(?P<series>.+?) - T=(?P<t>\d+)\.tiff?$", re.IGNORECASE)


def sample_series(folder: Path) -> "dict[str, list[tuple[int, Path]]]":
    """{series name: [(timepoint, path), ...]} from '<series> - T=<n>.tif' files."""
    found: dict = {}
    for path in Path(folder).iterdir():
        match = _SERIES_NAME.match(path.name)
        if match:
            found.setdefault(match["series"], []).append((int(match["t"]), path))
    return {name: sorted(files) for name, files in sorted(found.items())}


def sample_id(series: str) -> str:
    """Short id for a series name: 'N2_..._40xrepeatall.2.czi' -> 'S2'."""
    match = re.search(r"\.(\d+)\.czi$", series)
    return f"S{match.group(1)}" if match else series


def match_cells(reference, table, max_distance: float = 3.0) -> dict:
    """{label in table: label in reference}, by centroid distance (µm)."""
    from scipy.optimize import linear_sum_assignment

    cols = ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
    if len(reference) == 0 or len(table) == 0:
        return {}
    a, b = reference[cols].to_numpy(float), table[cols].to_numpy(float)
    cost = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    rows, ref_cols = linear_sum_assignment(cost)
    return {int(table["label"].iloc[i]): int(reference["label"].iloc[j])
            for i, j in zip(rows, ref_cols) if cost[i, j] <= max_distance}


def cell_names(reference) -> dict:
    """First-timepoint labels -> 'A', 'B', ... ordered by depth then y, so names are stable."""
    order = reference.sort_values(["centroid_z_um", "centroid_y_um"])["label"].tolist()
    return {int(label): string.ascii_uppercase[i] for i, label in enumerate(order)}


def name_first(table):
    names = cell_names(table)
    return table.assign(cell=[names[int(lbl)] for lbl in table["label"]])


def spots_in_volume(values: dict, channel: str):
    """The channel's spots with voxel indices in the whole volume (the run
    reports them in its crop -- the reference region: shifted by its corner)."""
    spots = values[f"spots_{channel}"].copy()
    box = values.get("ref_roi", values.get("roi"))
    if box is not None and len(spots):
        for axis in "zyx":
            spots[f"{axis}_index"] = spots[f"{axis}_index"] + int(box[axis][0])
    return spots


def _rows_from(values_table, names, t, path, sample, boxes):
    """One timepoint's rows: cell names from the targets, both boxes, identity columns."""
    table = values_table.copy()
    table["cell"] = [names[int(k)] for k in table["target"]]
    for prefix, box in boxes:
        if box is not None:
            for axis in "zyx":
                table[f"{prefix}_{axis}0"], table[f"{prefix}_{axis}1"] = int(box[axis][0]), int(box[axis][1])
    table["timepoint"] = t
    table["sample"] = sample
    table["source"] = path.name
    table["included"] = True
    return table


# -- parallel stages (module level: the worker processes import and call them) ----------------


def _stage1_segment(path: str, settings: dict, cache_dir: str, t: int) -> dict:
    """Worker: segment one timepoint's cells (whole volume), cache masks and table."""
    from vanua.blocks.provenance import provenance_document

    reg = registry()
    r, values = segment_cells(Path(path), Settings(**settings), reg=reg)
    cache = Path(cache_dir)
    labels = values["cells"]
    labels_path = cache / f"T{t:05d}_cells.npy"
    table_path = cache / f"T{t:05d}_cells.csv"
    np.save(labels_path, labels.astype(np.uint16 if labels.max(initial=0) < 65535 else np.int32))
    values["cell_table"].to_csv(table_path, index=False)
    return {"t": t, "labels": str(labels_path), "table": str(table_path), "frame": values["frame"],
            "provenance": provenance_document(r, "cell_table", reg)}


def save_timepoint_view(save_dir, t: int, values: dict, names: list, source) -> None:
    """Save one timepoint's segmentation for the Display's timepoint browser:
    all cell masks, the tracked cells' names, spots in whole-volume indices,
    both boxes (:func:`hcrcells.session.save_timepoint`). Skipped without masks."""
    if save_dir is None or values.get("cells") is None:
        return
    from .session import save_timepoint

    table = values["cell_table"]
    tracked = {int(label): names[int(k)] for label, k in zip(table["label"], table["target"])}
    spots = {ch: spots_in_volume(values, ch) for ch in CHANNELS if f"spots_{ch}" in values}
    save_timepoint(save_dir, t, values["cells"], spots, tracked, values["roi"], values.get("ref_roi"),
                   Path(source).name)


def _stage2_spots(path: str, labels_path: str, table_path: str, targets: str, settings: dict, roi: dict,
                  save_dir: "str | None" = None, t: int = 0, names: "list | None" = None) -> dict:
    """Worker: one timepoint's cells of interest, boxes, spots and measurements,
    from the cache; also saves its segmentation for the timepoint browser."""
    from vanua.blocks.provenance import provenance_document

    reg = registry()
    r, values = run_spots(Path(path), labels_path, table_path, Settings(**settings), targets=targets,
                          roi=RoiSettings(**roi), reg=reg)
    save_timepoint_view(save_dir, t, values, names or [], path)
    out = {"table": values["cell_table"], "roi": values["roi"], "ref_roi": values["ref_roi"],
           "provenance": provenance_document(r, "cell_table", reg)}
    for p in (labels_path, table_path):
        Path(p).unlink(missing_ok=True)  # each timepoint's cache goes as soon as it is used
    return out


def track_series_parallel(files, interest, *, sample: str = "", settings: "Settings | None" = None,
                          roi: "RoiSettings | None" = None, first=None, workers: "int | None" = None,
                          cache_dir=None, progress=None, cancel=None, log=print, save_dir=None):
    """:func:`track_series` with timepoints processed in parallel worker processes.

    Stage 1 (parallel): every timepoint's cells are segmented on the whole
    volume -- the costly part -- and cached. Tracking (in order, in this
    process): as soon as timepoints up to T are segmented, the cells of
    interest are matched to their last positions (the select step's own
    matching, :func:`hcrcells.ops.match_targets`). Stage 2 (parallel): T's
    selection, ROI and reference region, spots and measurements, from the
    cache (recipes/spots.toml). The first timepoint runs here as in
    :func:`track_series` (or is `first`). Same numbers as the sequential run."""
    import multiprocessing
    import shutil
    import tempfile
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from dataclasses import asdict

    import pandas as pd

    from vanua.blocks.provenance import provenance_document
    from vanua.core.cancellation import OpCancelled

    from .ops import match_targets

    settings = settings or Settings()
    roi = roi or RoiSettings()
    workers = workers or default_workers()
    reg = registry()
    names = list(interest["cell"])
    last = {name: row for name, row in zip(names, interest[CENTROID].to_numpy(float))}
    n = len(files)
    rows, runs = {}, {}

    # The first timepoint here: it fixes nothing new (targets are its own cells), and its
    # arrays are kept for the overlays, as in the sequential runner.
    import time

    t0, path0 = files[0]
    targets0 = format_targets([last[k] for k in names])
    started = time.perf_counter()
    if first is not None:
        r0, values0 = first
    else:
        r0, values0 = run_timepoint(path0, settings, targets=targets0, roi=roi, reg=reg, cancel=cancel)
    first_seconds = time.perf_counter() - started
    rows[0] = _rows_from(values0["cell_table"], names, t0, path0, sample,
                         (("roi", values0["roi"]), ("ref", values0.get("ref_roi"))))
    runs[0] = [provenance_document(r0, "cell_table", reg)]
    save_timepoint_view(save_dir, t0, values0, names, path0)
    for name, point in zip(rows[0]["cell"], rows[0][CENTROID].to_numpy(float)):
        last[name] = point
    if n == 1:
        cells = rows[0]
        return cells, values0, runs[0]

    cache = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="hcrcells-cache-"))
    cache.mkdir(parents=True, exist_ok=True)
    sdict, rdict = asdict(settings), asdict(roi)
    done_units, total_units = 1, 2 * (n - 1) + 1  # first timepoint + two stages per other timepoint

    # Progress: whole units (a timepoint's stage done), plus a heartbeat while
    # workers run -- the work in flight estimated from the throughput so far --
    # so the bar never stands still for the half minute a first segmentation takes.
    pool_start = time.perf_counter()
    expected_unit = max(10.0, 1.5 * (first_seconds if first is None else 30.0))  # until real completions arrive
    shown = {"fraction": 0.0, "last_event": pool_start, "beat": 0.0}

    def report(fraction: float, message: str) -> None:
        if progress:
            shown["fraction"] = max(shown["fraction"], min(fraction, 0.995 if fraction < 1 else 1.0))
            progress(shown["fraction"], message)

    def tick(message: str) -> None:
        shown["last_event"] = time.perf_counter()
        report(done_units / total_units, message)

    def heartbeat() -> None:
        now = time.perf_counter()
        if now - shown["beat"] < 1.0:
            return
        shown["beat"] = now
        units_done_in_pool = done_units - 1
        unit = ((now - pool_start) * workers / units_done_in_pool) if units_done_in_pool else expected_unit
        running = min(workers, total_units - done_units)
        partial = min(running * (now - shown["last_event"]) / unit, 0.9 * running)
        elapsed = int(now - pool_start)
        what = (f"segmenting {len(stage1)} timepoint(s)" if stage1 else "") + \
               (", " if stage1 and stage2 else "") + (f"measuring {len(stage2)}" if stage2 else "")
        report((done_units + partial) / total_units,
               f"{sample}: {what} in {workers} workers — {len(rows)}/{n} done, {elapsed // 60}:{elapsed % 60:02d} elapsed")

    if log:
        log(f"{sample}: {n - 1} timepoints in {workers} parallel workers")
    tick(f"{sample} T={t0} (1/{n})")
    stage1, stage2, segmented = {}, {}, {}
    next_i = 1
    # 'spawn': the same start method on every platform, and nothing of the GUI is inherited.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        try:
            for i in range(1, n):
                t, path = files[i]
                stage1[pool.submit(_stage1_segment, str(path), sdict, str(cache), t)] = i
            while stage1 or stage2:
                if cancel is not None and cancel.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise OpCancelled("cancelled")
                finished, _ = wait(list(stage1) + list(stage2), timeout=0.25, return_when=FIRST_COMPLETED)
                if not finished:
                    heartbeat()
                for fut in finished:
                    if fut in stage1:
                        i = stage1.pop(fut)
                        segmented[i] = fut.result()
                        done_units += 1
                        tick(f"{sample} T={files[i][0]} segmented")
                    else:
                        i = stage2.pop(fut)
                        res = fut.result()
                        t, path = files[i]
                        rows[i] = _rows_from(res["table"], names, t, path, sample,
                                             (("roi", res["roi"]), ("ref", res["ref_roi"])))
                        runs[i] = [segmented[i]["provenance"], res["provenance"]]
                        done_units += 1
                        tick(f"{sample} T={t} measured ({len(rows)}/{n})")
                # Tracking, strictly in timepoint order: T's targets are the positions after T-1.
                while next_i in segmented and next_i not in {v for v in stage2.values()} and next_i not in rows:
                    seg = segmented[next_i]
                    targets = format_targets([last[k] for k in names])
                    table = pd.read_csv(seg["table"])
                    matched = match_targets(table, seg["frame"], targets, settings.max_distance)
                    for k, point in zip(matched["target"], matched[CENTROID].to_numpy(float)):
                        last[names[int(k)]] = point  # a cell not found keeps its last position
                    t, path = files[next_i]
                    stage2[pool.submit(_stage2_spots, str(path), seg["labels"], seg["table"], targets,
                                       sdict, rdict, str(save_dir) if save_dir else None, t, names)] = next_i
                    next_i += 1
        finally:
            if cache_dir is None:
                shutil.rmtree(cache, ignore_errors=True)
    tick(f"{sample}: {n} timepoints")
    cells = pd.concat([rows[i] for i in range(n)], ignore_index=True)
    front = ["sample", "timepoint", "cell", "included", "label"]
    flat_runs = [doc for i in range(n) for doc in runs[i]]
    return cells[front + [c for c in cells.columns if c not in front]], values0, flat_runs


def track_series(files, interest, *, sample: str = "", settings: "Settings | None" = None,
                 roi: "RoiSettings | None" = None, recipe=None, reg=None, first=None, progress=None,
                 cancel=None, log=print, workers: int = 1, save_dir=None):
    """One sample's timepoints -> (per-cell table over T, first timepoint's
    values, provenance runs), for the cells of interest only.

    `interest`: the first timepoint's cells of interest -- a table with
    `cell` (name) and their centroid columns (µm). Each timepoint is told
    where they were last (`select.targets`), so the recipe follows them and
    boxes them in the ROI. `first`: an already computed (recipe, values) of
    the first file with these targets (the GUI's ROI step). `save_dir`: a
    session folder -- each timepoint's segmentation is saved there for the
    Display's timepoint browser (~0.3 MB each).

    Only the first timepoint's arrays are kept: one timepoint's arrays are
    ~0.5 GB, so keeping 40 would not fit in memory."""
    if workers and workers > 1 and len(files) > 2:
        return track_series_parallel(files, interest, sample=sample, settings=settings, roi=roi, first=first,
                                     workers=workers, progress=progress, cancel=cancel, log=log,
                                     save_dir=save_dir)
    import pandas as pd

    from vanua.blocks.provenance import provenance_document
    from vanua.core.cancellation import check_cancel

    settings = settings or Settings()
    reg = reg or registry()
    recipe = recipe or load_recipe()
    names = list(interest["cell"])
    last = {name: row for name, row in zip(names, interest[CENTROID].to_numpy(float))}
    rows, runs = [], []
    first_values = None
    for i, (t, path) in enumerate(files):
        check_cancel(cancel)
        if progress:
            progress(i / len(files), f"{sample} T={t} ({i + 1}/{len(files)})")
        if log:
            log(f"{sample} T={t}: {path.name}")
        targets = format_targets([last[n] for n in names])
        if i == 0 and first is not None:
            r, values = first
        else:
            # This timepoint's recipe progress fills its own share of the series bar.
            r, values = run_timepoint(path, settings, targets=targets, roi=roi, recipe=recipe, reg=reg,
                                      cancel=cancel, progress=progress, span=(i / len(files), (i + 1) / len(files)),
                                      prefix=f"{sample} T={t} ({i + 1}/{len(files)}) ")
        table = values["cell_table"].copy()
        table["cell"] = [names[int(k)] for k in table["target"]]
        for name, point in zip(table["cell"], table[CENTROID].to_numpy(float)):
            last[name] = point  # a cell not found keeps its last position
        for prefix, key in (("roi", "roi"), ("ref", "ref_roi")):
            box = values.get(key)
            if box is not None:
                for axis in "zyx":
                    table[f"{prefix}_{axis}0"], table[f"{prefix}_{axis}1"] = int(box[axis][0]), int(box[axis][1])
        table["timepoint"] = t
        table["sample"] = sample
        table["source"] = path.name
        table["included"] = True
        rows.append(table)
        runs.append(provenance_document(r, "cell_table", reg))
        save_timepoint_view(save_dir, t, values, names, path)
        if i == 0:
            first_values = values
        del values
    if progress:
        progress(1.0, f"{sample}: {len(files)} timepoints")
    cells = pd.concat(rows, ignore_index=True)
    front = ["sample", "timepoint", "cell", "included", "label"]
    return cells[front + [c for c in cells.columns if c not in front]], first_values, runs


def cells_of_interest(table, settings: "Settings | None" = None, overrides=None):
    """The first timepoint's cells named, the rule applied, overrides honoured."""
    named = apply_exclusion(table, settings, overrides)
    return named, named[named["included"]].reset_index(drop=True)


def export_cells(cells, out: Path, runs, settings: "Settings | None" = None, overrides=None, reg=None,
                 roi: "RoiSettings | None" = None) -> Path:
    """CSV + provenance sidecar (every timepoint's run, the selection rule, matching, ROI)."""
    from vanua.blocks.provenance import export_table

    settings = settings or Settings()
    recipe = load_recipe()
    path = Path(out)
    return export_table(cells, path, recipe, "cell_table", reg or registry(), extra={
        "runs": runs, "settings": settings_document(settings),
        "exclusion": {"rule": f"{settings.exclude_channel}_mean_bgsub_au at the first timepoint < "
                              f"{settings.exclude_below} a.u.",
                      "overrides": {f"{k[0]}:{k[1]}": v for k, v in (overrides or {}).items()}},
        "cells_of_interest": sorted(cells["cell"].unique().tolist()),
        "roi": {"margins_um": settings_document(roi or RoiSettings()),
                "note": "roi_*: the box around the cells of interest; ref_*: the larger reference region "
                        "in which spots, their threshold, the single-spot reference and the per-cell "
                        "measurements were computed (index ranges per timepoint)"},
        "matching": {"method": "Hungarian on centroid distance, each timepoint to the last known positions",
                     "max_distance_um": settings.max_distance},
        "identities": identities_document(cells)})


def identities_document(cells) -> dict:
    """{cell: neuron} of the cells named (the `identity` column), e.g. {'A': 'ASIL'}."""
    if "identity" not in cells:
        return {}
    named = cells[cells["identity"].fillna("").astype(str) != ""]
    return {str(c): str(i) for c, i in zip(named["cell"], named["identity"])}


def update_identities(cells, path: "str | Path") -> Path:
    """Rewrite a results CSV after cells were (re)named, and bring its
    provenance sidecar up to date: the output's columns and rows, the
    identities, and when they were changed. The measurements, and every
    other record of how they were made, are unchanged."""
    from datetime import datetime, timezone

    from vanua.blocks.provenance import sidecar_path

    path = Path(path)
    cells.to_csv(path, index=False)
    sidecar = sidecar_path(path)
    if sidecar.exists():
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        output = document.setdefault("output", {"path": str(path.resolve())})
        output.update(rows=int(len(cells)), columns=list(map(str, cells.columns)))
        document["identities"] = identities_document(cells)
        document["identities_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sidecar.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return path


# -- figures ---------------------------------------------------------------------------


def head_crop(labels: np.ndarray, margin: int = 120):
    """(y, x) slices around every segmented cell, plus a margin, in pixels."""
    mip = np.asarray(labels).max(0) > 0
    if not mip.any():
        return slice(0, mip.shape[0]), slice(0, mip.shape[1])
    ys, xs = np.nonzero(mip)
    return (slice(max(ys.min() - margin, 0), min(ys.max() + margin, mip.shape[0])),
            slice(max(xs.min() - margin, 0), min(xs.max() + margin, mip.shape[1])))


def bleaching_figure(cells, title: str = ""):
    """Per cell, default options (see hcrcells.plots)."""
    return cells_figure(cells, PlotOptions(kind="cells"), title)


def mean_figure(cells, title: str = ""):
    """Mean of the included cells, default options (see hcrcells.plots)."""
    return _mean_figure(cells, PlotOptions(kind="mean"), title)


def batch_figure(cells):
    """Each sample's mean separately, default options (see hcrcells.plots)."""
    return _batch_figure(cells, PlotOptions(kind="batch"))


def overlay_figure(values, table, pad_um: float = 4.0):
    """One panel per cell: the slice through its centroid (daf-28 magenta,
    ins-4 green, ins-6 cyan), its outline, and spots within ±1 slice."""
    import matplotlib.figure
    from scipy import ndimage
    from skimage.segmentation import find_boundaries

    labels = values["cells"]
    imgs = [np.asarray(values[c], float) for c in CHANNELS]
    sy = float(np.linalg.norm(np.asarray(values["frame"]["affine"])[:3, 1]))
    pad = int(round(pad_um / sy))
    colors = {"daf28": (1, 0, 1), "ins4": (0, 1, 0), "ins6": (0, 1, 1)}
    n = len(table)
    fig = matplotlib.figure.Figure(figsize=(3.2 * max(n, 1), 3.6), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, max(n, 1)))
    for ax, (_, row) in zip(axes, table.sort_values("cell").iterrows()):
        lbl = int(row["label"])
        zz, yy, xx = ndimage.center_of_mass(labels == lbl)
        z, y, x = int(round(zz)), int(round(yy)), int(round(xx))
        sl = (slice(max(y - pad, 0), y + pad), slice(max(x - pad, 0), x + pad))
        rgb = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start, 3))
        for ch, img in zip(CHANNELS, imgs):
            hi = np.percentile(img[max(z - 3, 0):z + 4][(slice(None),) + sl], 99.8) or 1
            rgb += np.clip(img[z][sl] / hi, 0, 1)[..., None] * np.array(colors[ch])
        ax.imshow(np.clip(rgb, 0, 1), extent=(sl[1].start, sl[1].stop, sl[0].stop, sl[0].start))
        ey, ex = np.nonzero(find_boundaries(labels[z][sl] == lbl, mode="inner"))
        ax.scatter(ex + sl[1].start, ey + sl[0].start, s=0.4, c="yellow")
        for ch, mk in zip(CHANNELS, ("+", "x", "1")):
            s = spots_in_volume(values, ch)
            near = s[(abs(s["z_index"] - z) <= 1) & (s["y_index"].between(sl[0].start, sl[0].stop - 1))
                     & (s["x_index"].between(sl[1].start, sl[1].stop - 1))]
            near = near[labels[near["z_index"], near["y_index"], near["x_index"]] == lbl]
            ax.scatter(near["x_index"], near["y_index"], marker=mk, s=14, c=[colors[ch]], linewidths=0.7)
        state = "" if row.get("included", True) else "  (excluded)"
        ax.set_title(f"cell {row['cell']}  z={z}  {row['volume_um3']:.0f} µm³{state}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Slice through each cell centroid — outline yellow; daf-28 magenta (+), ins-4 green (x), ins-6 cyan (Y)",
                 fontsize=9)
    return fig


def overview_figure(values, table, crop=None):
    """MIP of the head: HCR composite with every cell's outline (MIP) and name."""
    import matplotlib.figure
    from scipy import ndimage
    from skimage.segmentation import find_boundaries

    labels = values["cells"]
    crop = crop or head_crop(labels)
    colors = {"daf28": (1, 0, 1), "ins4": (0, 1, 0), "ins6": (0, 1, 1)}
    rgb = 0
    for ch in CHANNELS:
        mip = np.asarray(values[ch], float).max(0)[crop]
        rgb = rgb + np.clip(mip / (np.percentile(mip, 99.9) or 1), 0, 1)[..., None] * np.array(colors[ch])
    fig = matplotlib.figure.Figure(figsize=(6.5, 6), layout="constrained")
    ax = fig.subplots()
    ax.imshow(np.clip(rgb, 0, 1), extent=(crop[1].start, crop[1].stop, crop[0].stop, crop[0].start))
    ey, ex = np.nonzero(find_boundaries(labels.max(0)[crop], mode="inner"))
    ax.scatter(ex + crop[1].start, ey + crop[0].start, s=0.3, c="yellow")
    for _, row in table.iterrows():
        zz, yy, xx = ndimage.center_of_mass(labels == int(row["label"]))
        state = "" if row.get("included", True) else " excl."
        ax.text(xx + 18, yy, f"{row['cell']} (z{zz:.0f}){state}", color="white", fontsize=10, weight="bold")
    ax.set_title("Head MIP: daf-28 magenta, ins-4 green, ins-6 cyan; cell outlines yellow", fontsize=9)
    return fig


def parse_overrides(text: str) -> dict:
    """'S1:E=exclude,S2:B=include' -> {('S1', 'E'): False, ('S2', 'B'): True}."""
    out = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        key, _, value = item.partition("=")
        sample, cell = key.split(":")
        out[(sample.strip(), cell.strip())] = value.strip().lower() != "exclude"
    return out


def parse_identities(text: str) -> dict:
    """'S1:A=ASIL,S1:B=ASJR' -> {('S1', 'A'): 'ASIL', ('S1', 'B'): 'ASJR'}."""
    from .pipeline import IDENTITIES

    out = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        key, _, value = item.partition("=")
        sample, cell = key.split(":")
        value = value.strip().upper().replace("-", "")
        if value not in IDENTITIES:
            raise ValueError(f"{item}: identity must be one of {', '.join(IDENTITIES)}")
        out[(sample.strip(), cell.strip())] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--samples", default="", help="Only these sample ids, e.g. 'S2'")
    parser.add_argument("--last", type=int, default=None, help="Only timepoints up to this T (quick checks)")
    parser.add_argument("--compare-counting", action="store_true",
                        help="Also save counting_methods.png: peaks vs the Big-FISH estimate, per channel")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker processes (0: automatic, from CPUs and free memory; 1: sequential)")
    parser.add_argument("--override", default="", help="Per-cell overrides of the exclusion rule, "
                                                       "e.g. 'S1:E=exclude,S2:B=include'")
    parser.add_argument("--identity", default="", help="Which neuron each cell is, e.g. "
                                                       "'S1:A=ASIL,S1:B=ASIR,S1:C=ASJL,S1:D=ASJR'")
    args = parser.parse_args(argv)
    import pandas as pd

    from .pipeline import apply_identities

    from vanua.blocks.plot import save_figure

    settings, roi = Settings(), RoiSettings()
    overrides = parse_overrides(args.override)
    identities = parse_identities(args.identity)
    wanted = {s.strip() for s in args.samples.split(",") if s.strip()}
    reg, recipe = registry(), load_recipe()
    tables = []
    for series, files in sample_series(args.folder).items():
        sid = sample_id(series)
        if wanted and sid not in wanted:
            continue
        if args.last is not None:
            files = [(t, p) for t, p in files if t <= args.last]
        out = args.out / sid
        out.mkdir(parents=True, exist_ok=True)
        # 1-2. first timepoint: cells (whole volume, no spots) and the cells of interest
        _r, seg = segment_cells(files[0][1], settings, reg=reg)
        named, interest = cells_of_interest(name_first(seg["cell_table"]).assign(sample=sid, timepoint=files[0][0]),
                                            settings, overrides)
        print(f"{sid}: cells {sorted(named['cell'])}; of interest {list(interest['cell'])}", flush=True)
        del seg
        # 3. every timepoint, spots in the ROI around the cells of interest
        cells, first, runs = track_series(files, interest, sample=sid, settings=settings, roi=roi,
                                          recipe=recipe, reg=reg, workers=args.workers or default_workers())
        cells = apply_identities(cells, identities)
        export_cells(cells, out / "cells_per_timepoint.csv", runs, settings, overrides, reg, roi)
        first_table = cells[cells["timepoint"] == cells["timepoint"].min()]
        title = f"{sid} ({len(files)} timepoints)"
        save_figure(bleaching_figure(cells, title), out / "bleaching.png")
        save_figure(mean_figure(cells, title), out / "bleaching_mean.png")
        if args.compare_counting:
            from .plots import counting_figure

            save_figure(counting_figure(cells, PlotOptions(kind="counting"), title), out / "counting_methods.png")
        save_figure(overlay_figure(first, first_table), out / "overlay_T0_cells.png")
        save_figure(overview_figure(first, first_table), out / "overlay_T0_head.png")
        del first
        tables.append(cells)
        print(f"{sid}: {len(files)} timepoints done", flush=True)
    if len(tables) > 1:
        everything = pd.concat(tables, ignore_index=True)
        everything.to_csv(args.out / "all_samples.csv", index=False)  # per-sample CSVs carry the provenance
        save_figure(batch_figure(everything), args.out / "batch_effect.png")


if __name__ == "__main__":
    main()
