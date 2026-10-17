"""Offscreen: build the window, run each step's handler synchronously, check what it
fills. (`viewer.initialize()` -- the 3D tab -- is never called offscreen.)"""

from __future__ import annotations

import numpy as np
import pytest



@pytest.fixture()
def app_on(qapp, monkeypatch, weld_file):
    from PyQt5.QtCore import QSettings

    from meltpool import app as app_mod

    qapp.setOrganizationName("meltpool-tests")  # never touch the user's own demo-mode setting
    QSettings().setValue(app_mod.DEMO_KEY, False)
    monkeypatch.setenv("MELTPOOL_SAMPLE", str(weld_file))
    return app_mod


def _synthetic_settings(m):
    trace = m.trace_form.values()
    trace.n_avg, trace.stride = 1, 3
    m.trace_form.set_values(trace)
    flags = m.flag_form.values()
    flags.rolling = 5
    m.flag_form.set_values(flags)


def _no_flip(m):
    cal = m.calibration_form.values()
    cal.flip_vertical = False
    m.calibration_form.set_values(cal)


def test_workflow(app_on, weld_file, tmp_path):
    """Load, crop, preview the three lines, trace every time point, results and export (profiles too)."""
    from conftest import weld_spec

    m = app_on.build_app(synchronous=True)
    window = m.window
    assert not window.toolbox.isItemEnabled(1)
    _synthetic_settings(m)
    _no_flip(m)
    m.path_edit.setText(str(weld_file))
    m.load(weld_file)
    assert window.toolbox.isItemEnabled(4) and window.toolbox.isItemEnabled(5)

    from meltpool.ops import CropParams

    m.crop_form.set_values(CropParams(x0=20, x1=300, y0=20, y1=170))
    m.show_crop_outline()
    assert m._roi_item is not None and m._roi_item.isVisible()
    m.apply_crop()
    assert m.roi_blob["crop"] == [20, 170, 20, 300]

    m.slider.setValue(60)
    m.preview()
    assert "Depth" in m.boundary_status.text()
    assert len(m._line_items) == 3 and all(i.isVisible() for i in m._line_items.values())

    m.measure()
    assert window.current_step_index() == 0  # nothing moves on by itself
    assert m.table_view.model().rowCount() == len(m.table) > 20
    assert m.plot.figure is not None and len(m.plot.figure.axes) == 3
    assert "Steady-state medians" in m.summary_label.text() and "Hz" in m.summary_label.text()
    assert m.spectrum_plot.figure is not None
    spec = weld_spec()
    centre = m.table.loc[~m.table.flagged, "depth_centre_um"].median() / 4.44
    assert centre == pytest.approx(spec.b + spec.rim_width / 2, abs=2.0)
    written = m.export(tmp_path)
    assert {"pool", "summary", "profiles", "spectrum", "peaks", "series_plot", "spectrum_plot", "overlay"} <= set(written)
    assert all(p.is_file() for p in written.values())
    assert (tmp_path / "weld_pool.csv.provenance.json").is_file()


def test_demo_tour(app_on, tmp_path):
    """With demo mode on, clicking each pulsing control advances the tour to the end."""
    from PyQt5.QtCore import QSettings

    QSettings().setValue(app_on.DEMO_KEY, True)
    m = app_on.build_app(synchronous=True)
    tour, window = m.tour, m.window
    assert tour.enabled and tour.index == 0
    assert tour.banner.text().startswith("<b>Step 1 of 6</b>")
    assert m.trace_form.values().stride == 4  # demo preset

    m.run_button.click()                      # clicking ahead does nothing to the tour
    assert tour.index == 0
    _no_flip(m)
    m.sample_button.click()
    _synthetic_settings(m)
    assert tour.index == 1 and window.current_step_index() == 1
    m.stabilise_button.click()
    assert tour.index == 2 and window.current_step_index() == 2
    m.crop_button.click()
    assert tour.index == 3 and window.current_step_index() == 4  # the optional Transient page is skipped
    m.slider.setValue(60)
    m.preview_button.click()
    assert tour.index == 4
    window.set_current_step(5)
    m.run_button.click()
    assert tour.index == 5 and window.current_step_index() == 6
    m.out_edit.setText(str(tmp_path))
    m.export_button.click()
    assert tour.finished and "Done" in tour.banner.text()
    assert (tmp_path / "weld_pool.csv.provenance.json").is_file()

    tour.action.trigger()                     # View -> Tutorial / Demo mode off
    assert not tour.enabled and tour.banner.isHidden()
    QSettings().setValue(app_on.DEMO_KEY, False)


def test_contrast_panel(app_on, steady_file):
    """The histogram follows the frame shown; Auto/Full/typed values reach the view;
    a kept window survives scrubbing, an unkept one is re-stretched per frame."""
    m = app_on.build_app(synchronous=True)
    _synthetic_settings(m)
    m.load(steady_file)
    c, view = m.contrast, m.viewer.slice_viewer
    assert c.histogram._stats is not None  # histogram of the frame shown
    lo, hi = c.window()
    assert (view._level_min, view._level_max) == pytest.approx((lo, hi))

    c.full_range()
    plane = c._plane
    assert c.window() == pytest.approx((plane.min(), plane.max()), abs=0.05)  # boxes show 0.1
    c.min_box.setValue(900.0)
    c.max_box.setValue(1500.0)
    c.gamma_box.setValue(0.7)
    assert c.keep.isChecked()  # typing a value keeps it
    assert (view._level_min, view._level_max) == pytest.approx((900.0, 1500.0))

    m.slider.setValue(m.slider.value() - 40)
    m._show_current()
    assert c.window() == pytest.approx((900.0, 1500.0)) and c.gamma() == pytest.approx(0.7)
    c.keep.setChecked(False)
    m._show_current()
    assert c.window() != pytest.approx((900.0, 1500.0))  # auto-stretched again
    c.auto()
    assert c.window() == pytest.approx(tuple(np.percentile(c._plane, (0.5, 99.5))), abs=0.05)


def test_crop_by_drag(app_on, steady_file):
    """Draw mode: drag a box (right to left, past the frame edge), it fills the form clamped
    and sorted, Apply crops to it; the right-hand side can be cut; tool restored afterwards."""
    m = app_on.build_app(synchronous=True)
    _synthetic_settings(m)
    m.load(steady_file)
    view = m.viewer.slice_viewer
    m.draw_button.setChecked(True)
    assert view.left_tool() == "paint"
    m._on_drag(300.0, 100.5, "start", "left")   # beyond the right edge (frame is 240 wide)
    m._on_drag(150.0, 40.0, "move", "left")
    m._on_drag(60.2, 10.0, "end", "left")
    assert not m.draw_button.isChecked() and view.left_tool() == "crosshair"
    c = m.crop_form.values()
    assert (c.x0, c.x1, c.y0, c.y1) == (60, 240, 10, 101)
    m.apply_crop()
    assert m.roi_blob["crop"] == [10, 101, 60, 240]
    assert view._channel.shape == (1, 91, 180)

    from meltpool.ops import CropParams

    m.crop_form.set_values(CropParams(x0=0, x1=150, y0=0, y1=0))  # cut the right-hand side off
    m.apply_crop()
    assert m.roi_blob["crop"] == [0, 120, 0, 150]
    m.crop_form.set_values(CropParams(x0=200, x1=50, y0=0, y1=9999))  # typed backwards / too large
    m.apply_crop()
    assert m.roi_blob["crop"] == [0, 120, 50, 200]


def test_stabilise_median_and_liquid_solid_line(app_on, tmp_path):
    """Stabilise (fixed), median, automatic search finds the 5% line, a hand trace is
    measured, and both export with provenance (points on record)."""
    import json

    from meltpool.synthetic import SyntheticSpec, make_movie, write_movie

    spec = SyntheticSpec(n=150, height=200, width=720, centre_x=360, velocity=0.0, depression=True, a=60, b=45,
                         solid_line_contrast=0.05, solid_gap=20, jitter_px=0.3, texture=0.012, pattern=0.005)
    path = write_movie(tmp_path / "weldlike.tif", make_movie(spec))
    m = app_on.build_app(synchronous=True)
    m.load(path)
    assert m.window.toolbox.isItemEnabled(7)

    stab = m.stabilise_form.values()
    stab.mode, stab.y0, stab.y1 = "fixed", 90, 130
    m.stabilise_form.set_values(stab)
    m.stabilise()
    assert m.shifts is not None and "Stabilised" in m.stabilise_status.text()
    assert m.movie_blob.get("shifts") is not None

    m.compute_median()
    assert m.median is not None and m.median.shape == (200, 720)
    m.detail_check.setChecked(True)  # band-pass view
    m.measure_solid(manual=False)
    row = m.solid_result["table"].iloc[0]
    assert row["method"] == "auto" and row["detected"], row["z"]
    assert row["depth_um"] / 4.44 == pytest.approx(spec.solid_line()[1], abs=2.5)

    m.trace_button.setChecked(True)
    assert m.viewer.slice_viewer.left_tool() == "paint"
    sa, sb = spec.solid_line()
    # the movie is flipped on load: display rows = 199 - stored; the synthetic was built in display orientation
    # and stored upside down, so display coordinates equal the spec's
    for t in np.linspace(0, np.pi, 9):
        m._on_trace_click(360 + sa * np.cos(t), 60 + sb * np.sin(t), "start", "left")
    m.undo_point()
    m._on_trace_click(360 + sa * np.cos(np.pi), 60.0, "start", "left")
    assert len(m.solid_points) == 9
    m.measure_solid(manual=True)
    row = m.solid_result["table"].iloc[0]
    assert row["method"] == "manual" and row["width_um"] == pytest.approx(2 * sa * 4.44, abs=0.1)
    written = m.export_solid(tmp_path / "out")
    doc = json.loads((tmp_path / "out" / "weldlike_solid.csv.provenance.json").read_text(encoding="utf-8"))
    ops = {s["op"]: s for s in doc["steps"]}
    assert ops["meltpool.stabilise"]["params"]["enabled"] is True and ops["meltpool.stabilise"]["params"]["mode"] == "fixed"
    assert ops["meltpool.solid_boundary"]["params"]["points"].count(";") == 8
    assert written["figure"].is_file()


def test_volume_tabs(app_on, weld_file):
    """MIP, Ortho (and 3D on a display) show the movie as a (time, y, x) stack; it follows the ROI."""
    import numpy as np

    from meltpool.ops import CropParams

    m = app_on.build_app(synchronous=True)
    _no_flip(m)
    m.load(weld_file)
    tabs = m.viewer.as_widget()
    tabs.setCurrentIndex(1)
    n_lit = int((~m.frames["blank"].to_numpy(bool)).sum())
    assert m.volume_views.stack.shape == (n_lit, 180, 320) and m.volume_views.step == 1
    mip = np.asarray(m.viewer.mip_viewer._slice_viewer._channel)[0]
    assert np.array_equal(mip, m.volume_views.stack.max(axis=0))
    tabs.setCurrentIndex(2)
    assert m.viewer.ortho_viewer.xz._channel.shape == (180, n_lit, 320)  # x-t kymographs, scrolled along y
    m.crop_form.set_values(CropParams(x0=20, x1=300, y0=20, y1=170))
    m.apply_crop()
    assert m.volume_views.stack.shape == (n_lit, 150, 280)  # rebuilt for the ROI
    tabs.setCurrentIndex(0)


def test_overlay_video(app_on, weld_file, tmp_path):
    """Export overlay video: one frame per time point, brightness as on screen, settings on record."""
    import json

    import av

    m = app_on.build_app(synchronous=True)
    _synthetic_settings(m)
    _no_flip(m)
    m.load(weld_file)
    assert m.export_overlay(tmp_path) is None  # nothing traced yet
    m.measure()
    m._show_current()
    m.contrast.keep.setChecked(True)
    m.contrast.min_box.setValue(500.0)
    m.contrast.max_box.setValue(1500.0)
    path = m.export_overlay(tmp_path)
    assert path.name == "weld_overlay.mp4"
    with av.open(str(path)) as c:
        assert sum(1 for _ in c.decode(c.streams.video[0])) == len(m.table)
    assert (m._last_overlay.low, m._last_overlay.high) == (500.0, 1500.0)
    m.export(tmp_path)
    doc = json.loads((tmp_path / "weld_pool.csv.provenance.json").read_text(encoding="utf-8"))
    ops = {st["op"]: st for st in doc["steps"]}
    assert "meltpool.trace_boundary" in ops


def test_flip_buttons(app_on, weld_file, tmp_path):
    """Flip vertically / horizontally apply at once, flip the frame exactly, discard a stabilisation
    measured in the other orientation, and are recorded in the provenance."""
    import json

    import numpy as np

    from meltpool import movie as mv

    m = app_on.build_app(synchronous=True)
    _synthetic_settings(m)
    _no_flip(m)
    m.load(weld_file)
    raw = mv.Movie(dict(m.movie_blob)).raw(30).copy()
    m.toggle_flip("horizontal")
    assert m.movie_blob["flip_horizontal"] and np.array_equal(mv.Movie(m.movie_blob).raw(30), raw[:, ::-1])
    m.toggle_flip("vertical")
    assert m.movie_blob["flip_vertical"] and np.array_equal(mv.Movie(m.movie_blob).raw(30), raw[::-1, ::-1])
    m.stabilise()
    assert m.shifts is not None
    m.toggle_flip("vertical")
    assert m.shifts is None and m.movie_blob.get("shifts") is None  # measured in the other orientation
    m.toggle_flip("horizontal")
    m.measure()
    m.export(tmp_path)
    doc = json.loads((tmp_path / "weld_pool.csv.provenance.json").read_text(encoding="utf-8"))
    load = {st["op"]: st for st in doc["steps"]}["meltpool.open_movie"]["params"]
    assert load["flip_vertical"] is False and load["flip_horizontal"] is False


def test_draw_stabilisation_region(app_on, weld_file):
    """Draw the stabilisation region by dragging: the form gets it, it is shown, and stabilisation uses it;
    the automatic region lies in the substrate below the pool."""
    from conftest import weld_spec

    m = app_on.build_app(synchronous=True)
    _no_flip(m)
    m.load(weld_file)
    m.show_stabilisation_region()
    spec = weld_spec()
    auto = m.stabilise_status.text()
    assert "automatic" in auto
    top = int(auto.split("y ")[1].split("–")[0])
    assert top >= spec.surface + spec.b + spec.rim_width  # below the deepest point of the pool
    m.stab_draw_button.setChecked(True)
    assert m.viewer.slice_viewer.left_tool() == "paint"
    m.stab_drawer._on_paint(300.0, 175.0, "start", "left")  # dragged right-to-left, bottom-to-top
    m.stab_drawer._on_paint(10.0, 130.0, "end", "left")
    assert not m.stab_draw_button.isChecked() and m.viewer.slice_viewer.left_tool() == "crosshair"
    p = m.stabilise_form.values()
    assert (p.x0, p.y0, p.x1, p.y1) == (10, 130, 300, 175)
    assert m.stab_drawer._item.isVisible()
    logs = []
    m.window.append_log = logs.append
    m.stabilise()
    assert any("rows 130:175, columns 10:300" in line for line in logs)
    m.reset_stabilisation_region()
    p = m.stabilise_form.values()
    assert (p.x0, p.y0, p.x1, p.y1) == (0, 0, 0, 0)


def test_transient_step(app_on, weld_file, tmp_path):
    """Running-median subtraction after the crop: display only by default (the views change, the
    tracing does not); 'display and analysis' feeds the tracing; the setting is on record."""
    import json

    import numpy as np

    from meltpool import movie as mv
    from meltpool.ops import BackgroundParams

    m = app_on.build_app(synchronous=True)
    _synthetic_settings(m)
    _no_flip(m)
    m.load(weld_file)
    assert m.window.toolbox.isItemEnabled(3)
    m.slider.setValue(60)
    m.background_form.set_values(BackgroundParams(enabled=True, window=20, mode="centred", operation="subtract",
                                                  every=1))
    m.preview_transient()
    assert m.contrast._kind == "transient"
    m.apply_transient()
    assert m.display_blob.get("background") and not m.analysis_blob.get("background")
    assert m.window.current_step_index() == 0  # nothing moves on by itself
    shown = np.asarray(m.viewer.slice_viewer._channel)[0]
    raw = mv.Movie(m.roi_blob)
    disp = mv.Movie(m.display_blob)
    expected = np.mean([disp.raw(k) for k in range(60, 63)], axis=0)
    assert np.allclose(shown, expected, atol=1e-3)
    assert np.allclose(disp.raw(60), raw.frame(60) - disp.background(60), atol=1e-3)
    # the oscillating rim is transient, the static substrate is not
    t = np.abs(disp.raw(65))  # a quarter period after 60: the rim is furthest from its median position
    rim = np.percentile(t[95:120, 110:210], 99)  # where the rim moves
    substrate = np.percentile(t[150:175, 20:100], 99)
    assert rim > 3 * substrate, (rim, substrate)

    m.background_form.set_values(BackgroundParams(enabled=True, window=20, every=2, apply_to="display and analysis"))
    m.apply_transient()
    assert m.analysis_blob.get("background")
    m.measure()
    m.export(tmp_path)
    doc = json.loads((tmp_path / "weld_pool.csv.provenance.json").read_text(encoding="utf-8"))
    bg = {st["op"]: st for st in doc["steps"]}["meltpool.subtract_background"]["params"]
    assert bg["enabled"] is True and bg["apply_to"] == "display and analysis" and bg["window"] == 20


def test_3d_transparency_panel(app_on, weld_file):
    """The 3D transparency panel gets the stack's histogram; numbers make values outside
    [transparent below, transparent above] fully transparent and [opaque from, opaque to]
    opaque; presets hide dark or bright values."""
    import numpy as np

    m = app_on.build_app(synchronous=True)
    _no_flip(m)
    m.load(weld_file)
    m.viewer.as_widget().setCurrentIndex(3)  # 3D tab (no VTK offscreen): the stack and the panel still update
    panel = m.opacity_panel
    lo, hi = panel._range
    assert lo < hi and panel._sample is not None
    panel.set_trapezoid(800.0, 1000.0, 1200.0, 1400.0, 0.7)
    f = panel.function()
    for value, expected in ((700.0, 0.0), (800.0, 0.0), (900.0, 0.35), (1100.0, 0.7), (1300.0, 0.35), (1500.0, 0.0)):
        assert f.GetValue(value) == pytest.approx(expected, abs=1e-6), value
    assert panel.boxes["a"].value() == pytest.approx(800.0) and panel.max_opacity.value() == pytest.approx(0.7)
    panel.preset("bright")
    median = float(np.percentile(panel._sample, 50))
    assert panel.function().GetValue(median - 1.0) == 0.0 and panel.function().GetValue(hi) > 0
    panel.preset("dark")
    assert panel.function().GetValue(hi) == 0.0 and panel.function().GetValue(lo) > 0


def test_preview_not_black_with_transient_and_kept_window(app_on, weld_file):
    """Regression: with a movie-frame window kept and the transient frames used for the analysis,
    Preview showed a transient image through the frame window -- all black."""
    import numpy as np

    from meltpool.ops import BackgroundParams

    m = app_on.build_app(synchronous=True)
    _synthetic_settings(m)
    _no_flip(m)
    m.load(weld_file)
    m.slider.setValue(60)
    m._show_current()
    m.contrast.keep.setChecked(True)
    m.contrast.min_box.setValue(500.0)
    m.contrast.max_box.setValue(1500.0)  # a window for movie frames
    m.background_form.set_values(BackgroundParams(enabled=True, window=20, apply_to="display and analysis"))
    m.apply_transient()
    m.preview()
    v = m.viewer.slice_viewer
    plane = np.asarray(v._channel)[v.current_z].astype(float)
    inside = ((plane > v._level_min) & (plane < v._level_max)).mean()
    assert m.contrast._kind == "transient" and inside > 0.9, (v._level_min, v._level_max, inside)


def test_3d_time_axis_capped():
    """A long movie is not drawn as a needle in 3D: time at most the frame's larger side."""
    import numpy as np

    from meltpool.volume_views import MovieVolumeViews

    vv = MovieVolumeViews.__new__(MovieVolumeViews)
    vv.stack = np.zeros((334, 448, 896), np.uint8)  # 15214 frames, every 43rd
    assert vv._spacing_3d((43 * 4.44, 4.44, 4.44)) == pytest.approx((896 * 4.44 / 334, 4.44, 4.44))
    vv.stack = np.zeros((500, 328, 512), np.uint8)  # short enough: one frame = one pixel
    assert vv._spacing_3d((4.44, 4.44, 4.44)) == pytest.approx((4.44, 4.44, 4.44))


def test_transient_frame_cache_changes_nothing(weld_file):
    """Frames kept between running medians give the same transient frames as reading anew."""
    import numpy as np

    from meltpool import movie as mv

    blob = mv.describe_movie(weld_file)
    blob["background"] = {"window": 20, "mode": "centred", "operation": "subtract", "every": 5}
    cached = mv.Movie(blob)
    fresh = mv.Movie(blob)
    fresh._base_keep = 0
    for i in (0, 3, 7, 12, 40, 41, 90):
        assert np.array_equal(cached.raw(i), fresh.raw(i))
    assert 0 < len(cached._base_cache) <= cached._base_keep
