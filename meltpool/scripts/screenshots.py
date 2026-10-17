"""Offscreen screenshots of the GUI with Tutorial / Demo mode on (no 3D view offscreen).

    python scripts/screenshots.py OUT_DIR [MOVIE]

Writes 00-demo-load.png (step 1 pulsing), 01-preview.png (pool overlay on
MOVIE at 60% of its length, or the sample) and, for a small movie,
02-results.png. Screenshots of lab data stay under out/ (gitignored).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))


def main(out_dir: str, movie: "str | None" = None) -> None:
    import vanua.ui  # noqa: F401
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QApplication
    from vanua.ui.blocks import grab_window

    from meltpool import app as app_mod

    qapp = QApplication.instance() or QApplication([])
    qapp.setOrganizationName("meltpool-screenshots")
    QSettings().setValue(app_mod.DEMO_KEY, True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = app_mod.build_app(synchronous=True)
    m.window.resize(1500, 900)
    m.window.show()
    qapp.processEvents()
    grab_window(m.window).save(str(out / "00-demo-load.png"))
    if movie:
        m.path_edit.setText(movie)
        m.load(movie)
    else:
        m.load_sample()
    m.stabilise_button.click()
    m.crop_button.click()
    m.preview_button.click()
    m.viewer.slice_viewer.fit_to_view()
    qapp.processEvents()
    grab_window(m.window).save(str(out / "01-preview.png"))
    if m.movie_blob["n"] < 5000:
        m.run_button.click()
        qapp.processEvents()
        grab_window(m.window).save(str(out / "02-results.png"))
    print(m.calibrate_status.text())
    print(m.boundary_status.text())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
