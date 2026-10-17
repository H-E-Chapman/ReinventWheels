from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))

from meltpool.synthetic import SyntheticSpec, make_movie, write_movie  # noqa: E402

PIXEL_UM = 4.44
FPS = 100000.0


@pytest.fixture(scope="session")
def qapp():
    import vanua.ui  # noqa: F401 -- before anything VTK
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="session")
def steady_spec():
    """A plain movie (moving textured solid, a liquid ellipse, blank frames first), stored
    upside down like the EHLA movies: for the readers, contrast and crop."""
    return SyntheticSpec(n=300)


@pytest.fixture(scope="session")
def steady_file(tmp_path_factory, steady_spec):
    """Stored upside down like the real movies, one page per frame."""
    return write_movie(tmp_path_factory.mktemp("syn") / "steady.tif", make_movie(steady_spec))


def weld_spec(**kw) -> SyntheticSpec:
    """A weld-like vapour depression: 6 px dark rim, size oscillating with a 20-frame period
    (5 kHz at 100 kfps), weld-like texture, stored the right way up (no flip)."""
    base = dict(n=140, height=180, width=320, centre_x=160, n_blank=10, velocity=0.0, depression=True, a=55, b=45,
                rim_width=6.0, rim_wobble=0.0, osc_amp=0.04, osc_period=20.0, texture=0.012, pattern=0.005)
    base.update(kw)
    return SyntheticSpec(**base)


@pytest.fixture(scope="session")
def weld_file(tmp_path_factory):
    return write_movie(tmp_path_factory.mktemp("weld") / "weld.tif", make_movie(weld_spec(), upside_down=False))


def weld_settings(**trace):
    """Settings for the synthetic weld movies: no flip, every frame, short flag window."""
    from meltpool.pipeline import Settings

    s = Settings()
    s.calibration.flip_vertical = False
    s.calibration.fps = FPS
    s.trace.n_avg = 1
    s.flags.rolling = 5
    for k, v in trace.items():
        setattr(s.trace, k, v)
    return s


@pytest.fixture(scope="session")
def shaky_spec():
    """No blank frames, a slight camera shake, a 5% oscillation: for stabilisation and dynamics."""
    return weld_spec(n=160, n_blank=0, osc_amp=0.05, jitter_px=0.2)


@pytest.fixture(scope="session")
def shaky_file(tmp_path_factory, shaky_spec):
    return write_movie(tmp_path_factory.mktemp("shaky") / "shaky.tif", make_movie(shaky_spec, upside_down=False))


@pytest.fixture(scope="session")
def shaky_run(shaky_file):
    """The whole recipe, stabilised (fixed), with the overlay video."""
    from meltpool.pipeline import run_movie

    s = weld_settings()
    s.stabilise.enabled, s.stabilise.mode = True, "fixed"
    return run_movie(shaky_file, s, shaky_file.parent / "out" / "depth.png")
