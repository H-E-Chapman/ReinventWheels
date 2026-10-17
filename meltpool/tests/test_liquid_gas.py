"""The liquid-gas boundary (the vapour-depression rim) on synthetic weld videos with a
known rim: the three lines, the profiles, the dynamics, stabilisation, the overlay
video, the main recipe and its provenance, and the batch runner."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import FPS, PIXEL_UM, weld_settings, weld_spec

from meltpool import boundary
from meltpool import movie as mv
from meltpool.synthetic import make_movie


def truth_lines(spec, k: float = 1.0):
    """(depth, half-width) in px of each line for size factor k: the rim band lies from
    rho = 1 to 1 + w/b, so along x it is w*a/b thick."""
    a, b, w = spec.a * k, spec.b * k, spec.rim_width
    return {"inner": (b, a), "centre": (b + w / 2, a + w / 2 * a / b), "outer": (b + w, a + w * a / b)}


@pytest.mark.parametrize("a,b,w", [(55, 45, 6), (55, 45, 3), (70, 35, 8)])
def test_m1_three_lines(a, b, w):
    """M1: depth and half-width of every line within 1.5 px; area within 5% of the half-ellipse it bounds."""
    spec = weld_spec(a=a, b=b, rim_width=w, osc_amp=0.0, height=200, n=80)
    movie = mv.Movie(mv.describe_array(make_movie(spec, upside_down=False)))
    image = movie.mean_of(60, 3)
    surface = mv.original_surface(image)
    s = boundary.TraceSettings()
    cx = boundary.recentre(image, surface, mv.depression_centre(image, surface), s)
    trace = boundary.trace_frame(image, surface, cx, s)
    got = boundary.measure(trace, image.shape, (1.0, 1.0), 1.0)
    for line, (depth, half) in truth_lines(spec).items():
        assert got[f"depth_{line}"] == pytest.approx(depth, abs=1.5), line
        assert got[f"width_{line}"] / 2 == pytest.approx(half, abs=1.5), line
        assert got[f"area_{line}"] == pytest.approx(np.pi * depth * half / 2, rel=0.05), line


def test_m1_boundary_recipe(weld_file, tmp_path):
    """Through the main recipe: the steady-state depth of each line within 1.5 px; the
    summary is the median of the steady time points; blank frames are not traced."""
    from meltpool.pipeline import run_movie

    _, values = run_movie(weld_file, weld_settings(), tmp_path / "depth.png", overlay=False)
    t = values["table"]
    spec = weld_spec()
    steady = t.loc[~t.flagged]
    for line, (depth, _half) in truth_lines(spec).items():
        assert steady[f"depth_{line}_um"].median() / PIXEL_UM == pytest.approx(depth, abs=1.5), line
    assert t["blank"].sum() == 0
    summary = values["summary"].iloc[0]
    assert summary["depth_centre_median_um"] == pytest.approx(steady["depth_centre_um"].median())


def test_m2_profiles(shaky_run, shaky_spec):
    """M2: profiles for every time point and line; the centre line within 2 px (median 1 px) of the synthetic rim."""
    from meltpool.pipeline import profiles_table

    _, values = shaky_run
    prof = values["profiles"]
    assert len(prof["frames"]) == len(values["table"])
    long = profiles_table(prof)
    assert set(long["line"]) == set(boundary.LINES)
    spec = shaky_spec
    frame0 = int(prof["frames"][0])
    k = 1 + spec.osc_amp * np.sin(2 * np.pi * frame0 / spec.osc_period)
    a, b, w = spec.a * k, spec.b * k, spec.rim_width
    rows = long[(long.frame == frame0) & (long.line == "centre")]
    x = rows["x_um"].to_numpy() / PIXEL_UM - spec.centre_x
    y = rows["y_um"].to_numpy() / PIXEL_UM - spec.surface
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    ra, rb = a + w / 2 * a / b, b + w / 2
    r_truth = 1.0 / np.sqrt((np.cos(th) / ra) ** 2 + (np.sin(th) / rb) ** 2)
    assert np.abs(r - r_truth).max() < 2.0 and np.median(np.abs(r - r_truth)) < 1.0


def test_m3_dominant_frequency(shaky_run):
    """M3: depth oscillating with a 20-frame period (5000 Hz at 100 kfps): found within one
    bin for every line, and in the summary."""
    _, values = shaky_run
    n = len(values["spectrum"]) * 2 - 2  # rfft of n points has n/2 + 1 bins
    bin_hz = FPS / max(n, 1)
    peaks = values["peaks"]
    for line in boundary.LINES:
        top = peaks[(peaks.series == f"depth_{line}_um") & (peaks["rank"] == 1)]
        assert top["freq_hz"].iloc[0] == pytest.approx(FPS / 20.0, abs=1.01 * bin_hz), line
    assert values["summary"].iloc[0]["depth_centre_dominant_hz"] == pytest.approx(FPS / 20.0, abs=1.01 * bin_hz)


def test_m3_stabilisation_keeps_the_oscillation(shaky_run, shaky_spec):
    """Stabilised on the substrate below the pool: the depth oscillation survives (amplitude within 15%)."""
    _, values = shaky_run
    spec = shaky_spec
    t = values["table"]
    d = t["depth_centre_um"].to_numpy() / PIXEL_UM
    phase = np.sin(2 * np.pi * t["frame"].to_numpy() / spec.osc_period)
    amp = np.polyfit(phase, d, 1)[0]
    assert amp == pytest.approx((spec.b + spec.rim_width / 2) * spec.osc_amp, rel=0.15)


def test_m4_overlay_movie(shaky_run, tmp_path):
    """M4: one video frame per time point, the lines in their three colours, a time stamp; GIF and MP4."""
    import av
    from PIL import Image

    _, values = shaky_run
    assert values["overlay"].endswith(".mp4")
    with av.open(values["overlay"]) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    assert len(frames) == len(values["table"])
    rgb = frames[len(frames) // 2].astype(int)
    for colour in boundary.COLOURS.values():  # lossy video: allow some colour error
        assert (np.abs(rgb - np.array(colour)).sum(axis=2) < 90).any(), colour
    corner = rgb[2:14, 4:60]
    assert (corner.min(axis=2) > 200).any() and (corner.max(axis=2) < 40).any()  # white text on black
    gif = boundary.write_overlay(frames[:3], tmp_path / "x.gif", 10)  # distinct frames (Pillow merges repeats)
    with Image.open(gif) as im:
        assert im.n_frames == 3


def test_m7_provenance(shaky_run, shaky_file, tmp_path):
    """M7: every CSV's sidecar names the movie's SHA-256, every op with its version and
    resolved parameters, and the units; plots and the overlay are written too."""
    import hashlib

    import pandas as pd

    from meltpool.pipeline import export_movie, registry

    recipe, values = shaky_run
    written = export_movie(recipe, values, tmp_path, "weld", registry())
    for key in ("pool", "summary", "profiles", "spectrum", "peaks", "series_plot", "spectrum_plot", "overlay"):
        assert written[key].is_file(), key
    doc = json.loads((tmp_path / "weld_pool.csv.provenance.json").read_text(encoding="utf-8"))
    assert [f["sha256"] for f in doc["input_files"]] == [hashlib.sha256(shaky_file.read_bytes()).hexdigest()]
    ops = {s["op"]: s for s in doc["steps"]}
    assert {"meltpool.open_movie", "meltpool.stabilise", "meltpool.crop", "meltpool.subtract_background",
            "meltpool.trace_boundary", "meltpool.flag_transient"} <= set(ops)
    assert ops["meltpool.stabilise"]["params"]["mode"] == "fixed" and ops["meltpool.stabilise"]["params"]["enabled"]
    assert ops["meltpool.crop"]["params"] == {"x0": 0, "x1": 0, "y0": 0, "y1": 0}
    assert all(s["op_version"] for s in doc["steps"])
    assert doc["units"]["depth_centre_um"] == "um" and doc["units"]["area_outer_um2"] == "um2"
    summary = pd.read_csv(written["summary"])
    assert summary.loc[0, "run"] == "weld" and "depth_centre_dominant_hz" in summary


def test_recipes_validate():
    from vanua.core import recipe

    from meltpool.pipeline import SOLID_RECIPE_PATH, load_recipe, registry

    assert recipe.validate(load_recipe(), registry()) == []
    assert recipe.validate(load_recipe(SOLID_RECIPE_PATH), registry()) == []


def test_headless_rerun_reproduces(weld_file, tmp_path):
    """The same settings give the same table (the exported recipe reproduces the numbers)."""
    import pandas as pd

    from meltpool.pipeline import run_movie

    s = weld_settings(stride=3)
    _, a = run_movie(weld_file, s, tmp_path / "a.png", overlay=False)
    _, b = run_movie(weld_file, s, tmp_path / "b.png", overlay=False)
    pd.testing.assert_frame_equal(a["table"], b["table"])


def test_batch(weld_file, tmp_path):
    from meltpool.batch import run_folder

    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "w.tif").write_bytes(weld_file.read_bytes())
    (tmp_path / "in" / "bad.avi").write_bytes(b"nope")
    table = run_folder(tmp_path / "in", tmp_path / "out", weld_settings(stride=2), overlay=False, log=lambda m: None)
    assert list(table["run"]) == ["bad", "w"] and table.loc[0, "error"] and table.loc[1, "error"] == ""
    for name in ("w_pool.csv", "w_summary.csv", "w_series.png", "w_spectrum.png", "summary.csv", "comparison.png",
                 "w_pool.csv.provenance.json"):
        assert (tmp_path / "out" / name).is_file(), name
