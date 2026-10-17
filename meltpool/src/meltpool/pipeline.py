"""One movie through the MeltPool recipe, and its export -- shared by the GUI,
the batch runner and the tests, so all three compute the same numbers.

Qt-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vanua.blocks import build_registry
from vanua.blocks.plot import axis_label, save_figure
from vanua.blocks.provenance import export_table
from vanua.core import recipe as recipe_mod
from vanua.core.runner import run

from . import boundary
from .ops import (OPS, BackgroundParams, CropParams, DynamicsParams, FlagParams, MedianParams, OpenParams,
                  OverlayParams, SolidBoundaryParams, StabiliseParams, TraceParams)

RECIPE_PATH = Path(__file__).resolve().parents[2] / "recipes" / "meltpool.toml"
SOLID_RECIPE_PATH = RECIPE_PATH.with_name("solid_boundary.toml")
LINE_COLOURS = {line: "#%02x%02x%02x" % rgb for line, rgb in boundary.COLOURS.items()}


@dataclass
class Calibration:
    pixel_um: float = OpenParams.pixel_um
    pixel_from_file: bool = OpenParams.pixel_from_file
    fps: float = OpenParams.fps
    flip_vertical: bool = OpenParams.flip_vertical
    flip_horizontal: bool = OpenParams.flip_horizontal
    dataset: str = OpenParams.dataset


@dataclass
class Settings:
    """Everything a user tunes, one object -> recipe overrides."""

    calibration: Calibration = field(default_factory=Calibration)
    stabilise: StabiliseParams = field(default_factory=StabiliseParams)
    crop: CropParams = field(default_factory=CropParams)
    background: BackgroundParams = field(default_factory=BackgroundParams)
    trace: TraceParams = field(default_factory=TraceParams)
    flags: FlagParams = field(default_factory=FlagParams)
    dynamics: DynamicsParams = field(default_factory=DynamicsParams)
    overlay: OverlayParams = field(default_factory=OverlayParams)


def registry():
    return build_registry(*OPS)


def load_recipe(path: "str | Path | None" = None):
    return recipe_mod.load(path or RECIPE_PATH)


def configured(recipe, source: "str | Path", settings: "Settings | None" = None,
               plot_path: "str | Path | None" = None, overlay: bool = True):
    s = settings or Settings()
    ids = {step.id for step in recipe.steps}
    overrides: dict[str, dict[str, Any]] = {
        "load": {"path": str(source), **asdict(s.calibration)},
        "stabilise": asdict(s.stabilise),
        "crop": asdict(s.crop),
        "background": asdict(s.background),
        "boundary": asdict(s.trace),
        "flags": asdict(s.flags),
        "dynamics": asdict(s.dynamics),
    }
    if plot_path is not None:
        overrides["plot"] = {"out_path": str(plot_path)}
    if "overlay" in ids:
        params = asdict(s.overlay)
        suffix = Path(s.overlay.out_path).suffix or ".mp4"
        out_dir = Path(plot_path).parent if plot_path is not None else Path(s.overlay.out_path).parent
        params["out_path"] = str(out_dir / f"{Path(str(source)).stem}_overlay{suffix}")
        overrides["overlay"] = params
    r = recipe_mod.with_params(recipe, {k: v for k, v in overrides.items() if k in ids})
    if not overlay and "overlay" in ids:
        import dataclasses

        r = dataclasses.replace(r, steps=tuple(st for st in r.steps if st.id != "overlay"))
    return r


def run_movie(source, settings=None, plot_path=None, *, recipe=None, reg=None, progress=None, cancel=None,
              log=None, overlay: bool = True) -> "tuple[Any, dict[str, Any]]":
    """Run one movie; returns (the configured recipe, every produced value).
    `overlay=False` skips the overlay video (it takes a while on long movies)."""
    reg = reg or registry()
    configured_recipe = configured(recipe or load_recipe(), source, settings, plot_path, overlay)
    values = run(configured_recipe, registry=reg, progress=progress, cancel=cancel, log=log)
    return configured_recipe, values


def export_movie(configured_recipe, values, out_dir: "str | Path", stem: str, reg=None) -> dict[str, Path]:
    """Per time point (`<stem>_pool.csv`), per run (`<stem>_summary.csv`), the traced
    lines (`_profiles.csv`), the spectra and their peaks (`_spectrum.csv`, `_peaks.csv`),
    each with its provenance sidecar (the movie's SHA-256 included); the series and
    spectrum plots; the overlay video if one was written."""
    reg = reg or registry()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = values["summary"].copy()
    summary.insert(0, "run", stem)
    table = values["table"]
    written = {
        "pool": export_table(table, out_dir / f"{stem}_pool.csv", configured_recipe, "table", reg),
        "summary": export_table(summary, out_dir / f"{stem}_summary.csv", configured_recipe, "summary", reg),
        "profiles": export_table(profiles_table(values["profiles"]), out_dir / f"{stem}_profiles.csv",
                                 configured_recipe, "profiles", reg),
    }
    unit = values["profiles"]["unit"]
    written["series_plot"] = save_figure(series_figure(table, unit), out_dir / f"{stem}_series.png")
    if values.get("spectrum") is not None:
        written["spectrum"] = export_table(values["spectrum"], out_dir / f"{stem}_spectrum.csv", configured_recipe,
                                           "spectrum", reg)
        written["peaks"] = export_table(values["peaks"], out_dir / f"{stem}_peaks.csv", configured_recipe, "peaks",
                                        reg)
        written["spectrum_plot"] = save_figure(spectrum_figure(values["spectrum"], values["peaks"], "depth", unit),
                                               out_dir / f"{stem}_spectrum.png")
    if values.get("overlay"):
        written["overlay"] = Path(values["overlay"])
    return written


def profiles_table(profiles: dict):
    """The traced lines in long format: one row per time point, line and angle (full-frame units)."""
    import pandas as pd

    py, px = profiles["pixel"]
    oy, ox = profiles["offset"]
    unit, th = profiles["unit"], profiles["theta"]
    parts = []
    for line in boundary.LINES:
        r = np.asarray(profiles["r"][line])
        n = len(profiles["frames"])
        parts.append(pd.DataFrame({
            "frame": np.repeat(profiles["frames"], len(th)), "time_ms": np.repeat(profiles["time_ms"], len(th)),
            "line": line, "theta_deg": np.tile(np.degrees(th), n),
            f"x_{unit}": ((profiles["cx"][:, None] + np.cos(th)[None, :] * r + ox) * px).ravel(),
            f"y_{unit}": ((profiles["surface"] + np.sin(th)[None, :] * r + oy) * py).ravel()}))
    return pd.concat(parts, ignore_index=True).dropna()


def configured_solid(source, settings: "Settings | None" = None, median: "MedianParams | None" = None,
                     solid: "SolidBoundaryParams | None" = None):
    """The liquid-solid recipe with the same calibration, stabilisation and crop as the main analysis."""
    s = settings or Settings()
    return recipe_mod.with_params(recipe_mod.load(SOLID_RECIPE_PATH), {
        "load": {"path": str(source), **asdict(s.calibration)},
        "stabilise": asdict(s.stabilise),
        "crop": asdict(s.crop),
        "median": asdict(median or MedianParams()),
        "solid": asdict(solid or SolidBoundaryParams()),
    })


def run_solid(source, settings=None, median=None, solid=None, *, reg=None, progress=None, cancel=None, log=None):
    reg = reg or registry()
    recipe = configured_solid(source, settings, median, solid)
    return recipe, run(recipe, registry=reg, progress=progress, cancel=cancel, log=log)


def series_figure(table, unit: str = "um"):
    """P1: depth, width and area against time -- the three lines (inner edge, band centre,
    outer edge) in their overlay colours, flagged (blank / start-up) time points shaded."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.5, 6.5), dpi=100, layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    t = table["time_ms"].to_numpy(float)
    flagged = table["flagged"].to_numpy(bool) if "flagged" in table else np.zeros(len(table), bool)
    for ax, (q, u) in zip(axes, (("depth", unit), ("width", unit), ("area", unit + "2"))):
        if flagged.any():
            ax.fill_between(t, 0, 1, where=flagged, transform=ax.get_xaxis_transform(), color="#ef4444",
                            alpha=0.10, linewidth=0)
        for line in boundary.LINES:
            c = f"{q}_{line}_{u}"
            if c in table:
                ax.plot(t, table[c], lw=0.9, color=LINE_COLOURS[line], label=line)
        ax.set_ylabel(axis_label(f"{q}_{u}"), fontsize=8)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=7, ncol=3, frameon=False)
    axes[-1].set_xlabel("time (ms)")
    axes[0].set_title("Liquid-gas boundary (shaded: blank / start-up)", fontsize=9)
    return fig


def spectrum_figure(spectrum, peaks, quantity: str = "depth", unit: str = "um"):
    """P2: amplitude spectrum of one quantity for the three lines; the dominant peak marked."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.5, 3.4), dpi=100, layout="constrained")
    ax = fig.add_subplot()
    f = spectrum["freq_hz"].to_numpy(float) / 1000.0
    for line in boundary.LINES:
        c = f"amp_{quantity}_{line}_{unit}"
        if c in spectrum:
            ax.plot(f[1:], spectrum[c].to_numpy(float)[1:], lw=0.9, color=LINE_COLOURS[line], label=line)
    if len(peaks):
        top = peaks[(peaks["series"] == f"{quantity}_centre_{unit}") & (peaks["rank"] == 1)]
        for fk in top["freq_hz"]:
            ax.axvline(fk / 1000.0, color="#374151", lw=0.8, ls="--")
            ax.annotate(f"{fk:.0f} Hz", (fk / 1000.0, 1.0), xycoords=("data", "axes fraction"), fontsize=8,
                        ha="left", va="top", xytext=(3, -2), textcoords="offset points")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel(f"amplitude ({unit})")
    ax.set_title(f"Spectrum of {quantity}, steady part (dashed: dominant, band centre)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    return fig
