"""Settings, the configured recipe, and the cell-exclusion rule -- shared by
the GUI, the series runner and the tests, so all three compute the same
numbers. Qt-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

RECIPE_PATH = Path(__file__).resolve().parents[2] / "recipes" / "timepoint.toml"   # cells + ROI spots
CELLS_RECIPE_PATH = Path(__file__).resolve().parents[2] / "recipes" / "cells.toml"  # cells only
SPOTS_RECIPE_PATH = Path(__file__).resolve().parents[2] / "recipes" / "spots.toml"  # stage 2, from cached cells
IMAGE_RECIPE_PATH = Path(__file__).resolve().parents[2] / "recipes" / "image.toml"  # one worm (group experiment)
CHANNELS = ("daf28", "ins4", "ins6")
GENE = {"daf28": "daf-28", "ins4": "ins-4", "ins6": "ins-6"}
CHANNEL_INDEX = {"daf28": 0, "ins4": 1, "ins6": 2, "dapi": 3}


@dataclass
class Settings:
    """What a user tunes; everything else stays as the recipe says."""

    background_radius: float = field(default=4.0, metadata={
        "label": "Background radius", "unit": "µm", "range": (0.5, 50.0), "step": 0.5,
        "help": "Broad Gaussian removed from daf-28 before thresholding (body autofluorescence)"})
    smooth_sigma: float = field(default=0.4, metadata={
        "label": "Smoothing sigma", "unit": "µm", "range": (0.0, 5.0), "step": 0.1})
    opening_radius: float = field(default=0.5, metadata={
        "label": "Opening radius", "unit": "µm", "range": (0.0, 5.0), "step": 0.1,
        "help": "Cuts thin necks to neighbouring bits before labelling"})
    split_distance: float = field(default=3.0, metadata={
        "label": "Split cells closer than", "unit": "µm", "range": (0.5, 20.0), "step": 0.5,
        "help": "Watershed seeds at least this far apart; touching somata are split"})
    min_volume: float = field(default=10.0, metadata={
        "label": "Smallest cell", "unit": "µm³", "range": (0.0, 1e4), "help": "Smaller objects are dropped"})
    spot_threshold: Literal["elbow", "noise"] = field(default="elbow", metadata={
        "label": "Spot threshold", "help": "elbow: automatic (Big-FISH style); noise: k x MAD noise"})
    exclude_channel: Literal["ins6", "daf28", "ins4", "none"] = field(default="ins6", metadata={
        "label": "Exclude cells without", "help": "Cells whose first-timepoint signal in this channel is "
                                                  "below the cutoff are excluded (overridable per cell)"})
    exclude_below: float = field(default=3.0, metadata={
        "label": "Exclusion cutoff", "unit": "a.u.", "range": (0.0, 255.0), "step": 0.5,
        "help": "Mean − background at the first timepoint"})
    max_distance: float = field(default=3.0, metadata={
        "label": "Track cells within", "unit": "µm", "range": (0.5, 20.0), "step": 0.5,
        "help": "A cell is matched to the previous timepoint's cell within this distance"})


@dataclass
class RoiSettings:
    """The loose box around the cells of interest in which spots are detected."""

    margin_xy: float = field(default=8.0, metadata={
        "label": "Margin x / y", "unit": "µm", "range": (0.0, 200.0), "step": 1.0,
        "help": "Box edge beyond the outermost cell of interest; generous, so isolated reference spots "
                "fall inside too"})
    margin_z: float = field(default=3.0, metadata={
        "label": "Margin z", "unit": "µm", "range": (0.0, 100.0), "step": 0.5})
    ref_margin_xy: float = field(default=25.0, metadata={
        "label": "Reference margin x / y", "unit": "µm", "range": (0.0, 500.0), "step": 1.0,
        "help": "A larger box around the same cells: spot detection, its threshold and the isolated "
                "single-spot reference (transcript estimate) use it, so they see enough isolated spots"})
    ref_margin_z: float = field(default=10.0, metadata={
        "label": "Reference margin z", "unit": "µm", "range": (0.0, 200.0), "step": 0.5})


def registry():
    from vanua.blocks import build_registry

    from .ops import OPS
    return build_registry(*OPS)


def load_recipe(path: "str | Path | None" = None):
    from vanua.core import recipe

    return recipe.load(path or RECIPE_PATH)


def load_cells_recipe():
    return load_recipe(CELLS_RECIPE_PATH)


def configured(recipe, source: "str | Path", settings: "Settings | None" = None, *, targets: str = "",
               roi: "RoiSettings | None" = None, drift=None):
    """The recipe (cells or timepoint) pointed at `source` with `settings`
    applied (with_params); for the timepoint recipe also the cells of
    interest (`targets`: their world centroids) and the ROI margins; for the
    image recipe (group experiments) the drift thresholds (`drift`: a
    :class:`hcrcells.qc.DriftSettings`)."""
    from vanua.core import recipe as recipe_mod

    s = settings or Settings()
    r = roi or RoiSettings()
    spot = {"threshold_mode": s.spot_threshold}
    overrides: dict[str, dict[str, Any]] = {
        "read_daf28": {"path": str(source)},
        "read_ins4": {"path": str(source)},
        "read_ins6": {"path": str(source)},
        "read_dapi": {"path": str(source)},
        "background": {"radius": s.background_radius},
        "smooth": {"sigma": s.smooth_sigma},
        "clean": {"opening_radius": s.opening_radius},
        "label": {"min_distance": s.split_distance},
        "filter": {"min_volume": s.min_volume},
        "spots_daf28": spot, "spots_ins4": spot, "spots_ins6": spot,
        "select": {"targets": targets, "max_distance": s.max_distance,
                   "margin_xy": r.margin_xy, "margin_z": r.margin_z,
                   "ref_margin_xy": r.ref_margin_xy, "ref_margin_z": r.ref_margin_z},
    }
    if drift is not None:
        overrides["drift"] = {"register_above": drift.register_above, "jump_above": drift.jump_above,
                              "exclude_above": drift.exclude_above}
        overrides["read_dapi"]["channel"] = drift.channel
    present = {step.id for step in recipe.steps}
    return recipe_mod.with_params(recipe, {k: v for k, v in overrides.items() if k in present})


def step_progress(recipe, progress, span=(0.0, 1.0), prefix: str = ""):
    """Adapt the runner's per-sub-step progress to one bar for the whole job.

    The runner calls ``progress(step_id, fraction, message)`` with `fraction`
    going 0 -> 1 *within each recipe step*, so passed straight to a progress
    bar it refilled once per sub-step. Here sub-step i of n at `fraction`
    maps to ``span[0] + (span[1] - span[0]) * (i + fraction) / n``, reported
    as ``progress(overall, message)`` -- never going backwards."""
    if progress is None:
        return None
    ids = [step.id for step in recipe.steps]
    n = max(len(ids), 1)
    lo, hi = span
    state = {"best": lo}

    def _progress(step_id, fraction, message=""):
        i = ids.index(step_id) if step_id in ids else 0
        f = min(max(float(fraction or 0.0), 0.0), 1.0)
        overall = max(state["best"], lo + (hi - lo) * (i + f) / n)
        state["best"] = overall
        progress(overall, f"{prefix}{i + 1}/{n} {step_id}" + (f": {message}" if message else ""))

    return _progress


def segment_cells(source, settings=None, *, recipe=None, reg=None, progress=None, cancel=None, log=None,
                  span=(0.0, 1.0), prefix: str = ""):
    """Cells only (whole volume, no spots): masks, centroids and per-cell
    intensity in every HCR channel -- the Segment and Select steps."""
    from vanua.core.runner import run

    reg = reg or registry()
    r = configured(recipe or load_cells_recipe(), source, settings)
    return r, run(r, registry=reg, progress=step_progress(r, progress, span, prefix), cancel=cancel, log=log)


def run_spots(source, labels_path, table_path, settings=None, *, targets: str = "",
              roi: "RoiSettings | None" = None, recipe=None, reg=None, cancel=None, log=None):
    """The parallel series runner's second stage: the timepoint recipe from
    cached cells (recipes/spots.toml). Same steps and parameters after the
    segmentation, so the same numbers as :func:`run_timepoint`."""
    from vanua.core import recipe as recipe_mod
    from vanua.core.runner import run

    reg = reg or registry()
    r = configured(recipe or load_recipe(SPOTS_RECIPE_PATH), source, settings, targets=targets, roi=roi)
    r = recipe_mod.with_params(r, {"load_cells": {"path": str(labels_path)}, "load_objects": {"path": str(table_path)}})
    return r, run(r, registry=reg, cancel=cancel, log=log)


#: Peak memory of one series worker on a 38 x 1024 x 1024 timepoint, measured
#: (2.44 GB); the default worker count keeps this much per worker free.
WORKER_MEMORY_BYTES = 2.5e9
#: Kept free for the GUI (a loaded sample is 2-3 GB) and the system: running
#: out of memory makes Windows page to disk and everything looks frozen.
RESERVED_MEMORY_BYTES = 3.0e9


def default_workers() -> int:
    """Worker processes for the series: half the logical CPUs, at most as many
    as the free memory holds (2.5 GB each, 3 GB kept free), capped at 4: on this
    machine 6 workers were no faster than 4 (117 s vs 120 s for 8 timepoints;
    memory bandwidth is the limit) and used 3.5 GB more."""
    import os

    cpus = max(1, (os.cpu_count() or 2) // 2)
    try:
        import psutil

        spare = psutil.virtual_memory().available - RESERVED_MEMORY_BYTES
        by_memory = max(1, int(spare / WORKER_MEMORY_BYTES))
    except Exception:  # noqa: BLE001
        by_memory = 2
    return max(1, min(cpus, by_memory, 4))


def run_timepoint(source, settings=None, *, targets: str = "", roi: "RoiSettings | None" = None, recipe=None,
                  reg=None, progress=None, cancel=None, log=None, span=(0.0, 1.0), prefix: str = ""):
    """One timepoint: cells on the whole volume, then spots and measurements
    in the ROI around the cells of interest (`targets`). Returns (the
    configured recipe, every produced value).

    `progress(fraction, message)` sees the progress of the whole run (or of
    `span` of a larger job, e.g. one timepoint of a series), not of each
    recipe sub-step."""
    from vanua.core.runner import run

    reg = reg or registry()
    r = configured(recipe or load_recipe(), source, settings, targets=targets, roi=roi)
    return r, run(r, registry=reg, progress=step_progress(r, progress, span, prefix), cancel=cancel, log=log)


def apply_exclusion(cells, settings: "Settings | None" = None, overrides: "dict | None" = None):
    """Set `excluded_by_rule` and `included` per (sample, cell).

    Rule: a cell whose first-timepoint `<exclude_channel>_mean_bgsub_au` is
    below `exclude_below` is excluded. `overrides` {(sample, cell): bool}
    forces a cell in (True) or out (False) whatever the rule says; it is
    recorded in the table as `override`."""
    s = settings or Settings()
    overrides = overrides or {}
    cells = cells.copy()
    keys = ["sample", "cell"]
    first = cells.sort_values("timepoint").groupby(keys).head(1).set_index(keys)
    if s.exclude_channel == "none":
        by_rule = {k: False for k in first.index}
    else:
        column = f"{s.exclude_channel}_mean_bgsub_au"
        by_rule = (first[column] < s.exclude_below).to_dict()
    index = list(zip(cells["sample"], cells["cell"]))
    cells["excluded_by_rule"] = [bool(by_rule.get(k, False)) for k in index]
    cells["override"] = [("" if k not in overrides else ("include" if overrides[k] else "exclude")) for k in index]
    unmatched = cells["cell"].astype(str).str.startswith("unmatched")
    cells["included"] = [(overrides[k] if k in overrides else not cells["excluded_by_rule"].iat[i]) and not unmatched.iat[i]
                         for i, k in enumerate(index)]
    return cells


IDENTITIES = ("ASIL", "ASIR", "ASJL", "ASJR")  # the four neurons of interest


def apply_identities(cells, identities: "dict | None" = None):
    """Set `identity` (next to `cell`): the neuron you named each cell,
    {(sample, cell): 'ASIL' | 'ASIR' | 'ASJL' | 'ASJR'}; '' where not named."""
    identities = identities or {}
    cells = cells.drop(columns=["identity"], errors="ignore")
    values = [identities.get((s, c), "") for s, c in zip(cells["sample"], cells["cell"])]
    cells.insert(list(cells.columns).index("cell") + 1, "identity", values)
    return cells


def shown_name(cell: str, identity: str = "") -> str:
    """'A · ASIL' once the cell is named, else 'A'."""
    return f"{cell} · {identity}" if identity else str(cell)


def settings_document(settings: "Settings | None" = None) -> dict:
    return asdict(settings or Settings())
