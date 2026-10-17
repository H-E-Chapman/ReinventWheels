"""Write the synthetic demo/test movie to data/synthetic_movie.tif: a weld-like vapour
depression with a 6 px dark rim, oscillating with a period of 25 frames, slight shake,
stored the right way up (build direction up: no flip).

Source: generated here (meltpool.synthetic) -- no external data, no licence needed.
The lab movies are never fetched, copied or committed.

    python scripts/make_synthetic.py [OUT]
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))

from meltpool.synthetic import SyntheticSpec, make_movie, write_movie  # noqa: E402

DEFAULT = APP_DIR / "data" / "synthetic_movie.tif"


def spec() -> SyntheticSpec:
    return SyntheticSpec(n=300, height=200, width=360, centre_x=180, n_blank=20, velocity=0.0, depression=True,
                         a=55, b=45, rim_width=6.0, rim_wobble=0.0, osc_amp=0.06, osc_period=25.0, jitter_px=0.2,
                         texture=0.012, pattern=0.005)


def make(path: "str | Path" = DEFAULT, spec_: "SyntheticSpec | None" = None) -> Path:
    return write_movie(path, make_movie(spec_ or spec(), upside_down=False))


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
