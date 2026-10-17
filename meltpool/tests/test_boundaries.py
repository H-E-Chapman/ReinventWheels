"""Sub-pixel stabilisation (M10), the rim method for the liquid-gas boundary (M11)
and the liquid-solid line on the stabilised median (M12), on synthetic movies with
a known shake, a known vapour depression and a known faint line."""

from __future__ import annotations

import numpy as np
import pytest

from meltpool import movie as mv
from meltpool.synthetic import SyntheticSpec, make_movie

#: Weld-like noise: substrate texture and camera stripes matched to the real
#: weld_0009 median (band-pass background sd ~0.007 here vs ~0.010 there).
WELD = dict(texture=0.012, pattern=0.005)


def residual_rms(est, spec, lit):
    d = est[lit] + spec.jitter()[lit]  # the correction undoes the applied shake
    d -= d.mean(axis=0)
    return np.sqrt((d ** 2).mean(axis=0))


def test_m10_stabilise_fixed_static_sample():
    """Static sample, 0.3 px shake: fixed mode recovers it to < 0.05 px (y) and < 0.1 px (x)."""
    spec = SyntheticSpec(n=200, velocity=0.0, depression=True, jitter_px=0.3)
    movie = mv.Movie(mv.describe_array(make_movie(spec, upside_down=False)))
    lit = np.arange(spec.n) >= spec.n_blank
    est = mv.estimate_shifts(movie, lit, (slice(70, 120), slice(0, 240)), mode="fixed", log=lambda m: None)
    ry, rx = residual_rms(est, spec, lit)
    assert ry < 0.05 and rx < 0.10, (ry, rx)


@pytest.mark.parametrize("velocity", [0.35, -0.35, 0.0])
def test_m10_stabilise_jitter_keeps_drift(velocity):
    """jitter mode: shake removed to < 0.12 px, both directions and a static sample;
    the steady drift survives stabilisation (measured again within 2%)."""
    spec = SyntheticSpec(n=200, velocity=velocity, depression=True, jitter_px=0.3)
    blob = mv.describe_array(make_movie(spec, upside_down=False))
    lit = np.arange(spec.n) >= spec.n_blank
    est = mv.estimate_shifts(mv.Movie(blob), lit, (slice(70, 120), slice(0, 240)), mode="jitter", log=lambda m: None)
    ry, rx = residual_rms(est, spec, lit)
    assert ry < 0.12 and rx < 0.12, (ry, rx)
    if velocity:
        stable = dict(blob, shifts=est.tolist())
        u = mv.estimate_velocity(mv.Movie(stable), lit, log=lambda m: None)["px_per_frame"]
        assert u == pytest.approx(velocity, rel=0.02)


def test_m10_stabilised_median_is_sharper():
    """The median of the stabilised movie is closer to the shake-free median than the unstabilised one."""
    base = dict(n=160, velocity=0.0, depression=True, a=40, b=30, **WELD)
    clean = make_movie(SyntheticSpec(**base), upside_down=False)
    shaky_spec = SyntheticSpec(**base, jitter_px=0.6)
    shaky = mv.describe_array(make_movie(shaky_spec, upside_down=False))
    frames = np.arange(20, 160)
    lit = np.arange(160) >= 20
    est = mv.estimate_shifts(mv.Movie(shaky), lit, (slice(95, 120), slice(0, 240)), mode="fixed", log=lambda m: None)
    truth = mv.median_image(mv.Movie(mv.describe_array(clean)), frames)
    raw = mv.median_image(mv.Movie(shaky), frames)
    stab = mv.median_image(mv.Movie(dict(shaky, shifts=est.tolist())), frames)
    inner = (slice(10, -10), slice(10, -10))
    err_raw = np.sqrt(((raw - truth)[inner] ** 2).mean())
    err_stab = np.sqrt(((stab - truth)[inner] ** 2).mean())
    assert err_stab < 0.6 * err_raw, (err_stab, err_raw)


def test_fourier_shift_is_exact_for_integer_shifts():
    a = np.random.default_rng(3).normal(100, 10, (40, 50)).astype(np.float32)
    b = mv.fourier_shift_frame(a, 2.0, -3.0)
    assert np.allclose(b[12:-12, 12:-12], np.roll(np.roll(a, 2, 0), -3, 1)[12:-12, 12:-12], atol=1e-3)


def _median_and_rim(spec):
    movie = mv.Movie(mv.describe_array(make_movie(spec, upside_down=False)))
    med = mv.median_image(movie, np.arange(spec.n_blank, spec.n))
    surface = float(np.nanmedian(mv.free_surface(med)))
    rx, ry = mv.rim_trace(med, surface, spec.centre_x)
    return med, surface, rx, ry


@pytest.mark.parametrize("seed", [0, 1])
def test_m12_auto_search_reports_nothing_without_a_line(seed):
    """No liquid-solid line in the movie: the automatic search must not report one."""
    spec = SyntheticSpec(n=150, height=200, width=720, centre_x=360, velocity=0.0, depression=True, a=60, b=45,
                         solid_line_contrast=0.0, seed=seed, **WELD)
    med, surface, rx, ry = _median_and_rim(spec)
    found = mv.solid_line_auto(med, surface, rx, ry, spec.centre_x)
    assert not found["detected"] and found["z"] < 3.0 and len(found["controls"]) >= 2


@pytest.mark.parametrize("seed", [0, 1])
def test_m12_auto_search_finds_a_5pc_line(seed):
    """A 5% dark line 20 px outside the depression: detected; depth within 2 px, width within 4 px."""
    spec = SyntheticSpec(n=150, height=200, width=720, centre_x=360, velocity=0.0, depression=True, a=60, b=45,
                         solid_line_contrast=0.05, solid_gap=20, seed=seed, **WELD)
    med, surface, rx, ry = _median_and_rim(spec)
    found = mv.solid_line_auto(med, surface, rx, ry, spec.centre_x)
    sa, sb = spec.solid_line()
    assert found["detected"], found["z"]
    assert found["y"].max() - surface == pytest.approx(sb, abs=2.0)
    assert found["x"].max() - found["x"].min() == pytest.approx(2 * sa, abs=4.0)


def test_m12_manual_line_measurement():
    """Hand-traced points (full-frame px) along a known half-ellipse, with a crop offset:
    depth and width exact, area within 3% of the half-ellipse, in frame units."""
    from vanua.core.datastore import WorldFrame

    from meltpool.ops import SolidBoundaryParams, _solid_boundary, crop_frame

    a, b, cx, surface = 50.0, 30.0, 120.0, 40.0
    th = np.linspace(0, np.pi, 25)
    pts = "; ".join(f"{cx + a * np.cos(t):.3f},{surface + b * np.sin(t):.3f}" for t in th)
    image = np.zeros((1, 100, 200), np.float32)
    blob = mv.crop_blob(mv.describe_array(np.zeros((3, 130, 260), np.uint16)), 10, 110, 30, 230)
    frame = crop_frame(WorldFrame.from_spacing((1.0, 2.0, 2.0)).as_document(), 10, 30)
    out = _solid_boundary(SolidBoundaryParams(points=pts, surface_y=int(surface)),
                          {"image": image, "frame": frame, "movie": blob}, lambda *x: None, None, lambda m: None)
    row = out["table"].iloc[0]
    assert row["method"] == "manual" and row["detected"] and row["n_points"] == 25
    assert row["depth_um"] == pytest.approx(b * 2.0, abs=1e-6)
    assert row["width_um"] == pytest.approx(2 * a * 2.0, abs=1e-6)
    assert row["area_um2"] == pytest.approx(np.pi * a * b / 2 * 4.0, rel=0.03)
    assert row["surface_y_um"] == pytest.approx(surface * 2.0)
