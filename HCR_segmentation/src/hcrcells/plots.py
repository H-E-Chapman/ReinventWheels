"""Bleaching plots from the per-cell table -- Qt-free, shared by the GUI's plot
window and the headless series runner, so both draw the same figure.

Three kinds (:data:`KINDS`): per cell, mean of the included cells, and the
batch-effect check (each sample's mean separately). :class:`PlotOptions`
holds every choice the plot window's controls expose.

Y limits are set once, after everything is drawn, from the data actually
plotted (points and error bands) -- by default from 0 (or the lowest value,
if negative) to the highest, so no point is ever clipped. (Setting the limit
per panel while drawing froze shared axes at the first panel's range: the
"maxed at 1" clipping.)

Labels: every axis label and title has a *role* (:data:`LABEL_ROLES`); a
user's text for a role (``PlotOptions.labels``) replaces the automatic one
wherever that role appears, so it survives redraws and goes into exports.
Each figure records its automatic texts (``fig._hcr_auto``) and tags the
text artists with their role (``_hcr_role``) for editing in the GUI.

Fits (mean and batch plots): the mean curve is fitted with a linear and an
exponential decay to a plateau; ``fit="auto"`` keeps the one with the lower
AICc (:func:`fit_decay`). The fitted line is drawn black and dashed; its
equation goes in a box at the panel's "best" spot (least overlap with the
data, recomputed at every draw); the table of every fit is ``fig._hcr_fits``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import numpy as np

from .pipeline import CHANNELS, GENE

KINDS = {"cells": "Per cell", "mean": "Mean of cells", "batch": "Batch effect (samples separately)",
         "counting": "Counting methods: peaks vs Big-FISH"}
COUNTS = {"transcripts_est": "transcripts (estimate)", "spots": "resolved spots"}
INTENSITIES = {"mean_bgsub_au": "mean − background", "integrated_bgsub_au": "integrated − background"}
BANDS = {"sd": "± SD", "sem": "± SEM", "none": "none"}
Y_MODES = {"from_zero": "All data, from 0", "tight": "All data, tight", "manual": "Manual"}
EXPORT_FORMATS = {"PNG": "png", "TIFF": "tif", "PDF": "pdf", "SVG": "svg", "EPS": "eps", "JPEG": "jpg"}
RASTER = {"png", "tif", "jpg"}
FITS = {"off": "Off", "auto": "Auto (best of linear / exponential)", "linear": "Linear",
        "exponential": "Exponential decay"}
LABEL_ROLES = {"title": "Figure title", "x": "X axis", "y_intensity": "Y axis: intensity",
               "y_spots": "Y axis: spots", "y_mean": "Y axis: means", "y_counts": "Y axis: counts", "y_bigfish": "Y axis: Big-FISH (right)",
               "x_scatter": "X axis: peaks (scatter)", "y_scatter": "Y axis: Big-FISH (scatter)",
               **{f"panel_{ch}": f"Panel title: {GENE[ch]}" for ch in CHANNELS}}
_SAMPLE_COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2", "#be123c"]


@dataclass
class PlotOptions:
    kind: Literal["cells", "mean", "batch"] = "cells"
    channels: tuple = CHANNELS
    t_min: "int | None" = None
    t_max: "int | None" = None
    count: Literal["transcripts_est", "spots"] = "transcripts_est"
    intensity: Literal["mean_bgsub_au", "integrated_bgsub_au"] = "mean_bgsub_au"
    #: Spots (count) and intensity: normalized to each cell's first plotted
    #: timepoint (default), or absolute values.
    count_relative: bool = True
    intensity_relative: bool = True
    band: Literal["sd", "sem", "none"] = "sd"
    show_cells: bool = False
    y_mode: Literal["from_zero", "tight", "manual"] = "from_zero"
    y_min: float = 0.0
    y_max: float = 1.5
    share_y: bool = True
    log_y: bool = False
    markers: bool = True
    #: Fit the mean curve(s): off, auto (best by AICc), linear, exponential.
    fit: Literal["off", "auto", "linear", "exponential"] = "off"
    #: {role: {"text": user's text, "auto": the automatic text when it was set}}
    labels: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "PlotOptions":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in (data or {}).items() if k in known}
        if values.get("intensity_relative") is None and "intensity_relative" in values:
            values["intensity_relative"] = True  # sessions saved with the old "auto" choice
        if "channels" in values:
            values["channels"] = tuple(c for c in values["channels"] if c in CHANNELS) or CHANNELS
        if not isinstance(values.get("labels", {}), dict):
            values["labels"] = {}
        return cls(**values)

    def intensity_is_relative(self) -> bool:
        return bool(self.intensity_relative)


# -- helpers ---------------------------------------------------------------------------------


def relative_to_first(cells, column: str):
    """`column` divided by the same cell's value at its first plotted timepoint."""
    keys = ["sample", "cell"] if "sample" in cells.columns else ["cell"]
    first = cells.sort_values("timepoint").groupby(keys)[column].transform("first")
    with np.errstate(divide="ignore", invalid="ignore"):
        return cells[column] / first


def _select(cells, opts: PlotOptions):
    named = cells[cells["included"]].copy() if "included" in cells.columns else cells.copy()
    if opts.t_min is not None:
        named = named[named["timepoint"] >= opts.t_min]
    if opts.t_max is not None:
        named = named[named["timepoint"] <= opts.t_max]
    return named


def _series(named, column: str, relative: bool):
    return relative_to_first(named, column) if relative else named[column].astype(float)


def _count_label(opts) -> str:
    unit = "normalized to first T" if opts.count_relative else "per cell"
    return f"{COUNTS[opts.count]},\n{unit}"  # two lines: one ran into the next row's label


def _intensity_label(opts) -> str:
    unit = "normalized to first T" if opts.intensity_is_relative() else "(a.u.)"
    return f"{INTENSITIES[opts.intensity]}\n{unit}"


def _text(opts: PlotOptions, role: str, auto: str) -> str:
    """The user's text for `role` if set, else the automatic one."""
    override = (opts.labels or {}).get(role) or {}
    return override.get("text") or auto


def _register(fig, role: str, auto: str) -> None:
    if not hasattr(fig, "_hcr_auto"):
        fig._hcr_auto = {}
    fig._hcr_auto.setdefault(role, auto)


def _ylabel(ax, opts: PlotOptions, role: str, auto: str, **kwargs) -> None:
    _register(ax.figure, role, auto)
    ax.set_ylabel(_text(opts, role, auto), **kwargs)
    ax.yaxis.label._hcr_role = role


def _xlabel(ax, opts: PlotOptions, auto: str = "timepoint (repeat)") -> None:
    _register(ax.figure, "x", auto)
    ax.set_xlabel(_text(opts, "x", auto))
    ax.xaxis.label._hcr_role = "x"


def _panel_title(ax, opts: PlotOptions, ch: str) -> None:
    role = f"panel_{ch}"
    _register(ax.figure, role, GENE[ch])
    ax.set_title(_text(opts, role, GENE[ch]))
    ax.title._hcr_role = role


def _figure_title(fig, opts: PlotOptions, auto: str) -> None:
    _register(fig, "title", auto)
    text = _text(opts, "title", auto)
    if text:
        fig.suptitle(text)
        fig._suptitle._hcr_role = "title"


# -- fits ------------------------------------------------------------------------------------------


def _aicc(rss: float, n: int, k: int) -> float:
    """Akaike information criterion, corrected for small samples, of a least-squares fit."""
    if n - k - 1 <= 0 or rss <= 0:
        return float("inf") if rss > 0 or n - k - 1 <= 0 else float("-inf")
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def _tex_num(v: float) -> str:
    """A number for mathtext: 3 significant figures, powers of ten as 10^{n}."""
    s = f"{v:.3g}"
    if "e" in s:
        mantissa, exponent = s.split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return s


def _signed(v: float, tex: bool) -> str:
    """' + v' or ' − v' (minus sign; mathtext: '-')."""
    if tex:
        return f" {'-' if v < 0 else '+'} {_tex_num(abs(v))}"
    return f" {'−' if v < 0 else '+'} {_fmt(abs(v))}"


def fit_decay(t, y, model: str = "auto") -> "dict | None":
    """Fit y(t) with a linear decay, y = a + b·t, and an exponential decay to a
    plateau, y = c + (y0 − c)·e^(−t/τ); keep `model`, or with "auto" the one
    with the lower AICc (the exponential's extra parameter is penalised, so it
    wins only when the curvature is real). Returns the chosen fit with both
    models' AICc, R², the equation, and the half-life (exponential, t½ = ln2·τ)
    or slope (linear), or None with fewer than 3 points."""
    from scipy.optimize import curve_fit

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    n = len(t)
    if n < 3:
        return None
    tss = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    fits = {}
    b, a = np.polyfit(t, y, 1)
    rss = float(np.sum((y - (a + b * t)) ** 2))
    fits["linear"] = {"model": "linear", "params": {"a": float(a), "b": float(b)}, "rss": rss,
                      "aicc": _aicc(rss, n, 2), "r2": 1 - rss / tss, "slope_per_T": float(b),
                      "half_life_T": float("nan"),
                      "equation": f"y = {_fmt(a)}{_signed(b, False)}·t",
                      "latex": rf"y = {_tex_num(a)}{_signed(b, True)}\,t",
                      "predict": lambda x, a=a, b=b: a + b * np.asarray(x, float)}
    if n >= 4:
        def expo(x, c, amp, tau):
            return c + amp * np.exp(-(x - t[0]) / tau)

        span = float(t[-1] - t[0]) or 1.0
        try:
            (c, amp, tau), _cov = curve_fit(expo, t, y, p0=(float(y.min()), float(y[0] - y.min()) or 1e-3, span / 3),
                                            bounds=([-np.inf, -np.inf, 1e-3 * span], [np.inf, np.inf, 1e3 * span]),
                                            maxfev=20000)
            rss_e = float(np.sum((y - expo(t, c, amp, tau)) ** 2))
            shift = f"(t − {_fmt(t[0])})" if t[0] != 0 else "t"
            tex_shift = rf"(t - {_tex_num(t[0])})" if t[0] != 0 else "t"
            fits["exponential"] = {
                "model": "exponential", "params": {"c": float(c), "amp": float(amp), "tau": float(tau)},
                "rss": rss_e, "aicc": _aicc(rss_e, n, 3), "r2": 1 - rss_e / tss, "slope_per_T": float("nan"),
                "half_life_T": float(np.log(2) * tau),
                # Decay term first, the plateau (bias) c last.
                "equation": f"y = {'−' if amp < 0 else ''}{_fmt(abs(amp))}·e^(−{shift}/{_fmt(tau)}){_signed(c, False)}",
                "latex": rf"y = {'-' if amp < 0 else ''}{_tex_num(abs(amp))}\,e^{{-{tex_shift}/{_tex_num(tau)}}}"
                         + _signed(c, True),
                "predict": lambda x, c=c, amp=amp, tau=tau: expo(np.asarray(x, float), c, amp, tau)}
        except (RuntimeError, ValueError):
            pass  # no exponential fit: the linear one stands
    if model == "auto":
        chosen = min(fits.values(), key=lambda f: f["aicc"])
    else:
        chosen = fits.get(model)
        if chosen is None:
            return None
    out = dict(chosen)
    out["aicc_linear"] = fits["linear"]["aicc"]
    out["aicc_exponential"] = fits.get("exponential", {}).get("aicc", float("nan"))
    out["n_timepoints"] = n
    return out


def _draw_fit(ax, text_ax, ts, mean, opts, color, label, slot, record) -> "np.ndarray | None":
    """Fit the mean curve and draw the fitted line (black, dashed); the
    equation becomes an entry of the panel's equations box (:func:`_place_legends`),
    placed where it covers the data least. Returns the line's y."""
    if opts.fit == "off" or len(ts) < 3:
        return None
    res = fit_decay(ts, mean, opts.fit)
    if res is None:
        return None
    from matplotlib.lines import Line2D

    grid = np.linspace(float(np.min(ts)), float(np.max(ts)), 200)
    line = res["predict"](grid)
    ax.plot(grid, line, "--", color="black", lw=1.3, alpha=0.95, label="_fit", zorder=5)
    if res["model"] == "exponential":
        extra = rf",\ t_{{1/2}} = {_tex_num(res['half_life_T'])}\ \mathrm{{T}}"
    else:
        extra = rf",\ \mathrm{{slope}} = {_tex_num(res['slope_per_T'])}/\mathrm{{T}}"
    text = f"{label}, {res['model']}\n${res['latex']}$\n$R^2 = {res['r2']:.3f}{extra}$"
    # The entry's marker is a short line in the curve's colour: which curve this fit belongs to.
    handle = Line2D([], [], color=color, lw=2.5)
    entries = getattr(text_ax, "_hcr_fit_entries", [])
    entries.append((handle, text))
    text_ax._hcr_fit_entries = entries
    record.append({k: v for k, v in res.items() if k not in ("predict", "rss", "params")}
                  | {"params": res["params"]})
    return line


def _avoid(host, others) -> None:
    """Let `host`'s "best" legend placement avoid the curves and bands of the
    other axes in the same panel (a twin y axis): invisible copies of them,
    in their own coordinates, added without touching `host`'s limits."""
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    for other in others:
        for artist in list(other.lines):
            ghost = Line2D(artist.get_xdata(), artist.get_ydata(), transform=artist.get_transform(), alpha=0.0)
            host.add_artist(ghost)
        for artist in list(other.collections):
            if isinstance(artist, PolyCollection):
                ghost = PolyCollection([path.vertices for path in artist.get_paths()],
                                       transform=artist.get_transform(), alpha=0.0)
                host.add_artist(ghost)


def _place_legends(panels, series=None) -> None:
    """One box per panel with its fit equations, at matplotlib's "best"
    location -- the spot that covers the panel's data least, recomputed at
    every draw. `panels`: [(axes that hold the entries, host axes on top,
    the panel's other axes)]. `series`: (axes, handles, labels) -- the curve
    legend, merged into that panel's box so two boxes never collide."""
    series_ax = series[0] if series else None
    placed = False
    for entries_ax, host, others in panels:
        entries = getattr(entries_ax, "_hcr_fit_entries", [])
        handles = [h for h, _ in entries]
        labels = [s for _, s in entries]
        if entries_ax is series_ax:
            handles, labels = list(series[1]) + handles, list(series[2]) + labels
            placed = True
        if not handles:
            continue
        _avoid(host, others)
        legend = host.legend(handles, labels, loc="best", fontsize=6.5 if entries else 8, framealpha=0.95,
                             edgecolor="0.6", handlelength=1.4, borderpad=0.4, labelspacing=0.5)
        legend.set_zorder(20)
    if series and not placed:
        series[0].legend(series[1], series[2], fontsize=8, loc="best")


def _spread(values, band: str) -> "tuple[float, float, float]":
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    if band == "none" or len(v) < 2:
        return m, m, m
    s = float(v.std(ddof=1))
    if band == "sem":
        s /= np.sqrt(len(v))
    return m, m - s, m + s


class _Limits:
    """Collects the y-range actually drawn per axes, then sets every limit once."""

    def __init__(self) -> None:
        self.ranges: dict = {}
        self.groups: list = []

    def add(self, ax, *arrays) -> None:
        values = np.concatenate([np.asarray(a, float).ravel() for a in arrays]) if arrays else np.array([])
        values = values[np.isfinite(values)]
        if len(values):
            lo, hi = self.ranges.get(ax, (np.inf, -np.inf))
            self.ranges[ax] = (min(lo, values.min()), max(hi, values.max()))

    def group(self, axes) -> None:
        """Axes that share one y range when `share_y` is on (same quantity)."""
        self.groups.append(list(axes))

    def apply(self, opts: PlotOptions, reference_line: "dict | None" = None) -> None:
        reference_line = reference_line or {}
        targets = []
        for group in self.groups:
            ranges = [self.ranges[a] for a in group if a in self.ranges]
            if opts.share_y and ranges:
                shared = (min(r[0] for r in ranges), max(r[1] for r in ranges))
                targets += [(a, shared) for a in group]
            else:
                targets += [(a, self.ranges.get(a)) for a in group]
        for ax, rng in targets:
            if rng is None:
                continue
            lo, hi = rng
            if reference_line.get(ax) is not None:  # keep the "1.0 = first timepoint" line in view
                lo, hi = min(lo, reference_line[ax]), max(hi, reference_line[ax])
            if opts.log_y:
                ax.set_yscale("log")
                positive_lo = lo if lo > 0 else None
                if positive_lo is None:
                    continue  # matplotlib's own log limits; non-positive values cannot be shown
                ax.set_ylim(positive_lo / 1.15, hi * 1.15)
                continue
            if opts.y_mode == "manual":
                ax.set_ylim(opts.y_min, opts.y_max)
                continue
            span = (hi - lo) or abs(hi) or 1.0
            if opts.y_mode == "from_zero":
                bottom = min(0.0, lo - 0.05 * span)
                ax.set_ylim(bottom, hi + 0.06 * (hi - bottom or 1.0))
            else:
                ax.set_ylim(lo - 0.06 * span, hi + 0.06 * span)


def _style(ax, opts: PlotOptions, xlabel: bool) -> None:
    ax.set_box_aspect(1)  # square panels
    ax.spines[["top", "right"]].set_visible(False)
    if xlabel:
        _xlabel(ax, opts)
    if opts.t_min is not None or opts.t_max is not None:
        lo = opts.t_min if opts.t_min is not None else ax.get_xlim()[0]
        hi = opts.t_max if opts.t_max is not None else ax.get_xlim()[1]
        ax.set_xlim(lo - 0.5, hi + 0.5)


def _marker(opts) -> str:
    return "o-" if opts.markers else "-"


# -- figures ----------------------------------------------------------------------------------


def cells_figure(cells, opts: "PlotOptions | None" = None, title: str = ""):
    """Per cell: intensity (top) and the count (bottom), one panel per channel."""
    import matplotlib.figure

    opts = opts or PlotOptions(kind="cells")
    channels = [c for c in CHANNELS if c in opts.channels]
    # Square panels, two rows: about 3.4 in per panel plus room for the labels.
    fig = matplotlib.figure.Figure(figsize=(3.4 * max(len(channels), 1) + 1.0, 7.4), layout="constrained")
    axes = np.atleast_2d(fig.subplots(2, max(len(channels), 1), sharex=True, squeeze=False))
    named = _select(cells, opts)
    named["_key"] = (named["sample"].astype(str) + " " if "sample" in named.columns else "") + named["cell"].astype(str)
    if "identity" in named.columns:  # 'S1 A · ASIL' once the cell is named
        ident = named["identity"].fillna("").astype(str)
        named["_key"] = named["_key"].where(ident == "", named["_key"] + " · " + ident)
    limits, refs = _Limits(), {}
    for j, ch in enumerate(channels):
        top, bottom = axes[0, j], axes[1, j]
        named["_i"] = _series(named, f"{ch}_{opts.intensity}", opts.intensity_is_relative())
        named["_c"] = _series(named, f"{ch}_{opts.count}", opts.count_relative)
        for key, g in named.groupby("_key"):
            g = g.sort_values("timepoint")
            top.plot(g["timepoint"], g["_i"], _marker(opts), ms=2.5, lw=1, label=key)
            bottom.plot(g["timepoint"], g["_c"], _marker(opts), ms=2.5, lw=1, label=key)
            limits.add(top, g["_i"])
            limits.add(bottom, g["_c"])
        _panel_title(top, opts, ch)
        for ax, rel in ((top, opts.intensity_is_relative()), (bottom, opts.count_relative)):
            if rel:
                ax.axhline(1.0, color="0.7", lw=0.8)
                refs[ax] = 1.0
        _style(top, opts, xlabel=False)
        _style(bottom, opts, xlabel=True)
    limits.group(axes[0, :])
    limits.group(axes[1, :])
    limits.apply(opts, refs)
    _ylabel(axes[0, 0], opts, "y_intensity", _intensity_label(opts))
    _ylabel(axes[1, 0], opts, "y_spots", _count_label(opts))
    if channels:
        axes[0, -1].legend(title="cell", fontsize=7)
    _figure_title(fig, opts, title)
    fig._hcr_fits = []
    return fig


def mean_figure(cells, opts: "PlotOptions | None" = None, title: str = ""):
    """Mean over the included cells, per channel: count and intensity on one
    axis when both are relative (or both absolute); otherwise intensity on a
    second y axis. Optional faint individual cells behind the means."""
    import matplotlib.figure

    opts = opts or PlotOptions(kind="mean")
    channels = [c for c in CHANNELS if c in opts.channels]
    fig = matplotlib.figure.Figure(figsize=(3.4 * max(len(channels), 1) + 1.4, 4.1), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, max(len(channels), 1), sharex=True, squeeze=False)[0])
    named = _select(cells, opts)
    keys = ["sample", "cell"] if "sample" in named.columns else ["cell"]
    n = len(named.groupby(keys)) if len(named) else 0
    same_axis = opts.count_relative == opts.intensity_is_relative()
    limits, refs = _Limits(), {}
    count_axes, int_axes = [], []
    fits: list = []
    for ax, ch in zip(axes, channels):
        ax_i = ax if same_axis else ax.twinx()
        ax_i.set_box_aspect(1)  # a twin axis must keep the same square box
        for slot, (column, rel, label, color, target) in enumerate((
                (f"{ch}_{opts.count}", opts.count_relative, COUNTS[opts.count], "#2563eb", ax),
                (f"{ch}_{opts.intensity}", opts.intensity_is_relative(), INTENSITIES[opts.intensity], "#dc2626", ax_i))):
            named["_v"] = _series(named, column, rel)
            if opts.show_cells:
                for _key, g in named.groupby(keys):
                    g = g.sort_values("timepoint")
                    target.plot(g["timepoint"], g["_v"], "-", color=color, lw=0.6, alpha=0.25)
                    limits.add(target, g["_v"])
            stats = np.array([_spread(g["_v"], opts.band) for _t, g in named.groupby("timepoint")]).reshape(-1, 3)
            ts = np.array(sorted(named["timepoint"].unique()))
            if len(ts):
                target.plot(ts, stats[:, 0], _marker(opts), ms=2.5, color=color, label=label)
                if opts.band != "none":
                    target.fill_between(ts, stats[:, 1], stats[:, 2], color=color, alpha=0.18, linewidth=0)
                limits.add(target, stats)
                record = []
                line = _draw_fit(target, ax, ts, stats[:, 0], opts, color, label, slot, record)
                if line is not None:
                    limits.add(target, line)
                    fits += [r | {"plot": "mean", "sample": title, "channel": GENE[ch], "measure": label} for r in record]
            if rel:
                target.axhline(1.0, color="0.7", lw=0.8)
                refs[target] = 1.0
        _panel_title(ax, opts, ch)
        _style(ax, opts, xlabel=True)
        count_axes.append(ax)
        if not same_axis:
            ax_i.spines[["top"]].set_visible(False)
            ax_i.tick_params(axis="y", colors="#dc2626")
            int_axes.append(ax_i)
    limits.group(count_axes)
    if int_axes:
        limits.group(int_axes)
    limits.apply(opts, refs)
    band = "" if opts.band == "none" else f" {BANDS[opts.band]}"
    if same_axis:
        unit = "normalized to first T" if opts.count_relative else "per cell / a.u."
        _ylabel(axes[0], opts, "y_mean", f"mean{band}, {unit} (n={n} cells)")
    else:
        # Two lines each: one line ran off the figure with two y axes.
        _ylabel(axes[0], opts, "y_spots", f"{_count_label(opts)} (mean{band}, n={n})", color="#2563eb")
        _ylabel(int_axes[-1], opts, "y_intensity", f"{_intensity_label(opts)} (mean{band})", color="#dc2626")
    if channels:
        handles = [h for a in (count_axes[-1], *(int_axes[-1:] if int_axes else [])) for h in a.get_lines()
                   if not h.get_label().startswith("_")]
        # With a twin y axis the box goes on the twin (drawn on top) and avoids both axes' curves.
        panels = [(ax, int_axes[i] if int_axes else ax, [ax] if int_axes else [])
                  for i, ax in enumerate(count_axes)]
        if opts.fit == "off":
            count_axes[-1].legend(handles=handles, fontsize=8, loc="lower left")
        else:
            _place_legends(panels, (count_axes[-1], handles, [h.get_label() for h in handles]))
    _figure_title(fig, opts, title)
    fig._hcr_fits = fits
    return fig


def batch_figure(cells, opts: "PlotOptions | None" = None, title: str = "Batch-effect check: samples shown separately"):
    """Each sample's mean over its included cells: count (top) and intensity (bottom)."""
    import matplotlib.figure

    opts = opts or PlotOptions(kind="batch")
    channels = [c for c in CHANNELS if c in opts.channels]
    # Square panels, two rows: about 3.4 in per panel plus room for the labels.
    fig = matplotlib.figure.Figure(figsize=(3.4 * max(len(channels), 1) + 1.0, 7.4), layout="constrained")
    axes = np.atleast_2d(fig.subplots(2, max(len(channels), 1), sharex=True, squeeze=False))
    named = _select(cells, opts)
    limits, refs = _Limits(), {}
    fits: list = []
    rows = ((opts.count, opts.count_relative, _count_label(opts), "y_spots", COUNTS[opts.count]),
            (opts.intensity, opts.intensity_is_relative(), _intensity_label(opts), "y_intensity",
             INTENSITIES[opts.intensity]))
    for row, (measure, rel, ylabel, role, what) in enumerate(rows):
        for j, ch in enumerate(channels):
            ax = axes[row, j]
            named["_v"] = _series(named, f"{ch}_{measure}", rel)
            for k, (sample, g) in enumerate(named.groupby("sample")):
                color = _SAMPLE_COLORS[k % len(_SAMPLE_COLORS)]
                if opts.show_cells:
                    for _c, gc in g.groupby("cell"):
                        gc = gc.sort_values("timepoint")
                        ax.plot(gc["timepoint"], gc["_v"], "-", color=color, lw=0.6, alpha=0.25)
                        limits.add(ax, gc["_v"])
                stats = np.array([_spread(gt["_v"], opts.band) for _t, gt in g.groupby("timepoint")]).reshape(-1, 3)
                ts = np.array(sorted(g["timepoint"].unique()))
                ax.plot(ts, stats[:, 0], _marker(opts), ms=2.5, color=color, label=f"{sample} (n={g['cell'].nunique()})")
                if opts.band != "none":
                    ax.fill_between(ts, stats[:, 1], stats[:, 2], color=color, alpha=0.18, linewidth=0)
                limits.add(ax, stats)
                record = []
                line = _draw_fit(ax, ax, ts, stats[:, 0], opts, color, sample, k, record)
                if line is not None:
                    limits.add(ax, line)
                    fits += [r | {"plot": "batch", "sample": sample, "channel": GENE[ch], "measure": what}
                             for r in record]
            if rel:
                ax.axhline(1.0, color="0.7", lw=0.8)
                refs[ax] = 1.0
            if row == 0:
                _panel_title(ax, opts, ch)
            _style(ax, opts, xlabel=row == 1)
        _ylabel(axes[row, 0], opts, role, ylabel)
        limits.group(axes[row, :])
    limits.apply(opts, refs)
    if channels:
        if opts.fit == "off":
            axes[0, -1].legend(fontsize=8, loc="lower left")
        else:
            series_handles = [h for h in axes[0, -1].get_lines() if not h.get_label().startswith("_")]
            _place_legends([(ax, ax, []) for ax in axes.ravel()],
                           (axes[0, -1], series_handles, [h.get_label() for h in series_handles]))
    _figure_title(fig, opts, title)
    fig._hcr_fits = fits
    return fig


def counting_figure(cells, opts: "PlotOptions | None" = None, title: str = ""):
    """The two ways of counting transcripts per cell, compared.

    Top, per channel: mean ± band over the cells of (a) the number of spot
    peaks inside the cell -- resolved spots, `<ch>_spots` -- and (b) the
    Big-FISH estimate (cell signal ÷ one isolated spot's signal,
    `<ch>_transcripts_est`), normalized to the first timepoint (one axis) or
    absolute (peaks left, Big-FISH on a right-hand axis); fits apply to both.
    Bottom: every cell at every timepoint, peaks against the Big-FISH estimate
    (colour = timepoint), with Spearman's rank correlation -- where the peak
    count saturates in dense cells, the points bend away from a straight line."""
    import matplotlib.figure
    from scipy.stats import spearmanr

    opts = opts or PlotOptions(kind="counting")
    channels = [c for c in CHANNELS if c in opts.channels]
    n_ch = max(len(channels), 1)
    fig = matplotlib.figure.Figure(figsize=(3.4 * n_ch + 1.8, 7.8), layout="constrained")
    axes = np.atleast_2d(fig.subplots(2, n_ch, squeeze=False))
    named = _select(cells, opts)
    keys = ["sample", "cell"] if "sample" in named.columns else ["cell"]
    n = len(named.groupby(keys)) if len(named) else 0
    limits, refs, fits = _Limits(), {}, []
    scatter = None
    # Absolute counts: the dots (tens per cell) and the Big-FISH estimate
    # (hundreds to thousands) on one axis left the dots a flat line at zero.
    # So, absolute: dots on the left axis, Big-FISH on a right-hand one, each
    # panel to its own data.
    twin = not opts.count_relative
    twins = []
    for j, ch in enumerate(channels):
        top, bottom = axes[0, j], axes[1, j]
        right = top.twinx() if twin else top
        if twin:
            right.set_box_aspect(1)  # a twin axis must keep the same square box
            twins.append(right)
        sources = ((named, "spots", "peaks (resolved spots)", "#0f766e", top),
                   (named, "transcripts_est", "Big-FISH estimate", "#7c3aed", right))
        for slot, (table, measure, label, color, target) in enumerate(sources):
            column = f"{ch}_{measure}"
            if column not in table.columns or not len(table):
                continue
            table = table.copy()
            table["_v"] = _series(table, column, opts.count_relative)
            groups = table.groupby("timepoint")
            stats = np.array([_spread(g["_v"].dropna(), opts.band) for _t, g in groups]).reshape(-1, 3)
            ts = np.array([t for t, _g in groups])
            if not len(ts):
                continue
            target.plot(ts, stats[:, 0], _marker(opts), ms=2.5, color=color, label=label)
            if opts.band != "none":
                target.fill_between(ts, stats[:, 1], stats[:, 2], color=color, alpha=0.18, linewidth=0)
            limits.add(target, stats)
            record = []
            line = _draw_fit(target, top, ts, stats[:, 0], opts, color, label, slot, record)
            if line is not None:
                limits.add(target, line)
                fits += [r | {"plot": "counting", "sample": title, "channel": GENE[ch], "measure": label}
                         for r in record]
        if opts.count_relative:
            top.axhline(1.0, color="0.7", lw=0.8)
            refs[top] = 1.0
        _panel_title(top, opts, ch)
        _style(top, opts, xlabel=True)
        if twin:
            right.spines[["top"]].set_visible(False)
            right.tick_params(axis="y", colors="#7c3aed")
        # Bottom: peaks vs Big-FISH, every cell at every timepoint (absolute counts).
        x = named[f"{ch}_spots"].to_numpy(float) if f"{ch}_spots" in named else np.array([])
        y = named[f"{ch}_transcripts_est"].to_numpy(float) if f"{ch}_transcripts_est" in named else np.array([])
        ok = np.isfinite(x) & np.isfinite(y) if len(x) and len(y) else np.array([], bool)
        if ok.any():
            scatter = bottom.scatter(x[ok], y[ok], c=named["timepoint"].to_numpy(float)[ok], cmap="viridis", s=12,
                                     edgecolors="none", alpha=0.85)
            rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 2 else float("nan")
            bottom.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\nn = {int(ok.sum())} cell × timepoint",
                        transform=bottom.transAxes, ha="left", va="top", fontsize=7.5,
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9})
            bottom.set_xlim(left=0)
            bottom.set_ylim(bottom=0)
        bottom.set_box_aspect(1)
        bottom.spines[["top", "right"]].set_visible(False)
        _register(fig, "x_scatter", "peaks per cell (resolved spots)")
        bottom.set_xlabel(_text(opts, "x_scatter", "peaks per cell (resolved spots)"))
        bottom.xaxis.label._hcr_role = "x_scatter"
    if not twin:
        limits.group(axes[0, :])  # normalized: one scale for all, every curve starts at 1
    else:
        for ax in [*axes[0, :], *twins]:  # absolute: each axis its own scale (still from zero, etc.)
            limits.group([ax])
    limits.apply(opts, refs)
    band = "" if opts.band == "none" else f" {BANDS[opts.band]}"
    if twin:
        _ylabel(axes[0, 0], opts, "y_counts", f"dots per cell\n(mean{band}, n={n} cells)")
        _ylabel(twins[-1], opts, "y_bigfish", "Big-FISH estimate per cell", color="#7c3aed")
    else:
        _ylabel(axes[0, 0], opts, "y_counts", f"transcripts, normalized to first T\n(mean{band}, n={n} cells)")
    _ylabel(axes[1, 0], opts, "y_scatter", "Big-FISH estimate per cell")
    if scatter is not None:
        bar = fig.colorbar(scatter, ax=axes[1, -1], shrink=0.8)
        bar.set_label("timepoint (repeat)")
    if channels:
        handles = {}
        for ax in [*axes[0, :], *twins]:
            for h in ax.get_lines():
                if not h.get_label().startswith("_") and h.get_label() not in handles:
                    handles[h.get_label()] = h
        handles = list(handles.values())
        # One legend for the curves, below the panels; the fit equations stay in each panel.
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8, frameon=False)
        if opts.fit != "off":
            # With a twin axis the box goes on the twin (drawn on top) and avoids both axes' curves.
            _place_legends([(ax, twins[i] if twin else ax, [ax] if twin else []) for i, ax in enumerate(axes[0, :])])
    _figure_title(fig, opts, title)
    fig._hcr_fits = fits
    return fig


def fits_table(fig):
    """The figure's fits as a table: one row per fitted mean curve."""
    import pandas as pd

    rows = getattr(fig, "_hcr_fits", []) or []
    cols = ["plot", "sample", "channel", "measure", "model", "equation", "equation_latex", "r2", "half_life_T",
            "slope_per_T",
            "aicc_linear", "aicc_exponential", "n_timepoints", "params"]
    return pd.DataFrame([{c: r.get("latex" if c == "equation_latex" else c) for c in cols} for r in rows],
                        columns=cols)


def make_figure(cells, opts: PlotOptions, title: str = ""):
    if opts.kind == "counting":
        return counting_figure(cells, opts, title)
    if opts.kind == "mean":
        return mean_figure(cells, opts, title)
    if opts.kind == "batch":
        return batch_figure(cells, opts) if not title else batch_figure(cells, opts, title)
    return cells_figure(cells, opts, title)


def export_figure(fig, path, dpi: int = 300):
    """Write `fig` in the format given by `path`'s suffix (png, tif, pdf, svg, eps, jpg)."""
    from pathlib import Path

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig.canvas is None or not hasattr(fig.canvas, "print_figure"):
        FigureCanvasAgg(fig)
    fmt = path.suffix.lstrip(".").lower()
    fmt = {"tif": "tiff", "jpg": "jpeg"}.get(fmt, fmt)
    kwargs = {"dpi": dpi} if fmt in {"png", "tiff", "jpeg"} else {}
    fig.savefig(path, format=fmt, **kwargs)
    return path
