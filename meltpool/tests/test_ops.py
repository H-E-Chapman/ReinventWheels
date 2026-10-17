"""The ops on synthetic movies with a known answer: the reader, blank frames, start-up
flags, the frame round trip, the crop and the surface row."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import FPS, PIXEL_UM, weld_settings, weld_spec

from meltpool import movie as mv
from meltpool.synthetic import make_movie


def test_reader_memmap_matches_tifffile(steady_file):
    import tifffile

    blob = mv.describe_tiff(steady_file, flip_vertical=True)
    assert blob["memmap"] is not None and blob["n"] == 300
    movie = mv.Movie(blob)
    with tifffile.TiffFile(steady_file) as tif:
        for i in (0, 57, 299):
            assert np.array_equal(movie.raw(i), tif.pages[i].asarray()[::-1])


def test_m5_open_movie_flags_blank_frames(steady_file, steady_spec):
    from meltpool.ops import OpenParams, _open_movie

    out = _open_movie(OpenParams(path=str(steady_file)), {}, lambda *a: None, None, lambda m: None)
    blank = out["frames"]["blank"].to_numpy()
    assert blank[:steady_spec.n_blank].all() and not blank[steady_spec.n_blank:].any()
    assert out["frames"]["time_ms"].iloc[steady_spec.n_blank] == 0.0


def test_m5_flags_start_up(tmp_path):
    """M5: a depression growing from half to full size over 60 frames after beam-on: the
    growth is flagged as start-up, the time points after it are not; blank frames are not
    traced at all (time points start at the first lit frame)."""
    from meltpool.pipeline import run_movie
    from meltpool.synthetic import write_movie

    spec = weld_spec(n=200, ramp=60, osc_amp=0.0)
    path = write_movie(tmp_path / "ramp.tif", make_movie(spec, upside_down=False))
    s = weld_settings(stride=2)
    s.flags.rolling = 7
    _, values = run_movie(path, s, tmp_path / "d.png", overlay=False)
    t = values["table"]
    assert not t["blank"].any() and t["frame"].min() == spec.n_blank
    early = t[t.frame < spec.n_blank + 0.6 * spec.ramp]
    assert len(early) and early["transient"].all()
    late = t[t.frame > spec.n_blank + spec.ramp + 15]
    assert len(late) and not late["flagged"].any()
    assert (t["flag_reason"] == np.where(t.blank, "blank", np.where(t.transient, "transient", ""))).all()


def test_flag_table_blank_rows():
    """Blank rows are flagged 'blank' and never start the steady state."""
    import pandas as pd

    from meltpool.ops import FlagParams, flag_table

    n = 60
    area = np.r_[np.full(5, np.nan), np.linspace(100, 400, 15), np.full(40, 400.0)]
    table = pd.DataFrame({"frame": np.arange(n), "time_ms": np.arange(n) * 0.01, "blank": np.arange(n) < 5,
                          "area_centre_um2": area})
    t = flag_table(table, FlagParams(rolling=5))
    assert list(t.flag_reason[:5]) == ["blank"] * 5
    assert t.transient[5:12].all() and not t.flagged[25:].any()


def test_m6_frame_round_trip(tmp_path):
    """M6: an oblique, anisotropic, offset frame passes the recipe unchanged, and
    lengths use its spacing (3 um pixels here)."""
    from vanua.core.datastore import WorldFrame
    from vanua.core.ops import OpSpec
    from vanua.core.runner import run

    from meltpool.ops import frames_table
    from meltpool.pipeline import configured, load_recipe, registry

    spec = weld_spec(osc_amp=0.0, n=40, n_blank=0)
    arr = make_movie(spec, upside_down=False)
    rot = np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([2.0, 3.0, 3.0])
    affine[:3, 3] = (5.0, -7.0, 11.0)
    frame = WorldFrame(unit="um", affine=tuple(tuple(map(float, r)) for r in affine))

    @dataclasses.dataclass
    class NoParams:
        pass

    def _source(params, ctx, progress, cancel, log):
        blob = mv.describe_array(arr)
        blob["fps"] = FPS
        return {"movie": blob, "frame": frame.as_document(),
                "frames": frames_table(arr[:, ::8, ::8].reshape(len(arr), -1).mean(1), FPS, 0.5)}

    reg = registry()
    reg.register(OpSpec(name="test.source", summary="synthetic", inputs={},
                        outputs={"movie": "blob", "frame": "blob", "frames": "table"}, params=NoParams, run=_source))
    recipe = configured(load_recipe(), "unused", weld_settings(), tmp_path / "d.png", overlay=False)
    recipe = dataclasses.replace(recipe, steps=tuple(
        dataclasses.replace(st, op="test.source", params={}) if st.id == "load" else st for st in recipe.steps))
    values = run(recipe, registry=reg)
    assert values["frame"] == frame.as_document()
    t = values["table"]
    depth = spec.b + spec.rim_width / 2
    half = spec.a + spec.rim_width / 2 * spec.a / spec.b
    assert t["depth_centre_um"].median() == pytest.approx(depth * 3.0, abs=1.5 * 3.0)
    assert t["width_centre_um"].median() / 2 == pytest.approx(half * 3.0, abs=1.5 * 3.0)


def test_m8_crop_keeps_numbers_and_coordinates():
    """M8: tracing a ROI gives the same depth, width and full-frame centre as the full
    frame; the crop's frame is the full frame with its origin moved to the ROI's first pixel."""
    from vanua.core.datastore import WorldFrame

    from meltpool.ops import TraceParams, crop_frame, trace_one, trace_prepare

    spec = weld_spec(osc_amp=0.0, n=30, n_blank=0)
    arr = make_movie(spec, upside_down=False)
    full_blob = mv.describe_array(arr)
    frame = WorldFrame.from_spacing((1.0, PIXEL_UM, PIXEL_UM), origin_zyx=(0.0, 7.0, -3.0)).as_document()
    p = TraceParams(n_avg=1)

    def trace(blob, fr):
        movie = mv.Movie(blob)
        surface, cx = trace_prepare(movie, np.array([10]), p)
        out = trace_one(movie, 10, surface, cx, p, fr)
        oy, ox = mv.crop_offset(blob)
        return out["values"], out["trace"]["cx"] + ox, surface + oy

    y0, y1, x0, x1 = 20, 175, 30, 300
    roi_blob = mv.crop_blob(full_blob, y0, y1, x0, x1)
    roi_frame = crop_frame(frame, y0, x0)
    a, cx_a, s_a = trace(full_blob, frame)
    b, cx_b, s_b = trace(roi_blob, roi_frame)

    assert np.array_equal(mv.Movie(roi_blob).frame(10), arr[10, y0:y1, x0:x1].astype(np.float32))
    origin = np.asarray(WorldFrame.from_dict(roi_frame).affine)[:3, 3]
    assert np.allclose(origin, (0.0, 7.0 + y0 * PIXEL_UM, -3.0 + x0 * PIXEL_UM))
    assert cx_b == pytest.approx(cx_a, abs=0.5) and s_b == pytest.approx(s_a, abs=0.5)
    for key in ("depth_centre", "width_centre", "area_centre"):
        assert b[key] == pytest.approx(a[key], rel=0.02), key


def test_m8_crop_rejects_bad_bounds():
    blob = mv.describe_array(np.zeros((3, 50, 80), np.uint16))
    for bounds in ((10, 5, 0, 0), (0, 0, 70, 90), (0, 60, 0, 0)):
        with pytest.raises(ValueError):
            mv.crop_blob(blob, *bounds)
    assert mv.crop_blob(blob, 0, 0, 0, 0)["crop"] == [0, 50, 0, 80]


def test_manual_surface_row_and_centre_match_auto():
    """A surface row and centre column typed in (full-frame pixels) give the automatic
    result, with a crop too."""
    from meltpool.ops import TraceParams, trace_prepare

    spec = weld_spec(osc_amp=0.0, n=30, n_blank=0)
    blob = mv.describe_array(make_movie(spec, upside_down=False))
    surface, cx = trace_prepare(mv.Movie(blob), np.array([10]), TraceParams(n_avg=1))
    assert surface == pytest.approx(spec.surface, abs=1.0) and cx == pytest.approx(spec.centre_x, abs=1.5)
    roi = mv.crop_blob(blob, 20, 175, 30, 300)
    manual = TraceParams(n_avg=1, surface_y=spec.surface, center_x=spec.centre_x)
    s2, cx2 = trace_prepare(mv.Movie(roi), np.array([10]), manual)
    assert (s2 + 20, cx2 + 30) == pytest.approx((spec.surface, spec.centre_x))
