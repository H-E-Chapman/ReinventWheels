"""Screenshots of HCR Cells on the REAL display (not offscreen).

    cd <app> && PYTHONPATH=src pixi run --manifest-path <vanua>/pixi.toml python scripts/screenshots.py

Writes into screenshots/:
  00-demo.png     demo mode: the pulsing Load sample button and the "Step 1 of 5" banner
  01-slice.png    Review step: daf-28 slice, cell outlines, spot markers, the cell table
  02-3d.png       3D tab: cell surfaces and the displayed channel's spots
  03-plots.png    Plots step after a short series (T=0..1)
Uses vanua.ui.blocks.grab_window so the VTK view is captured.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))

from vanua.ui import theme  # noqa: E402 -- vanua.ui before VTK
from vanua.ui.blocks import grab_window  # noqa: E402
from PyQt5.QtCore import QSettings  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from hcrcells.app import APP_NAME, DEMO_KEY, build_app, show  # noqa: E402

OUT = APP_DIR / "screenshots"


def settle(app: QApplication, seconds: float = 1.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def wait_for(app: QApplication, signal, timeout: float = 900.0) -> None:
    """Pump the event loop until `signal` fires: the work runs on worker
    threads, so the window keeps repainting (synchronous mode froze it)."""
    fired = []
    signal.connect(lambda *a: fired.append(True))
    end = time.monotonic() + timeout
    while not fired:
        if time.monotonic() > end:
            raise TimeoutError(f"no {signal} within {timeout:.0f} s")
        app.processEvents()
        time.sleep(0.02)


def save(hcr, name: str) -> None:
    path = OUT / name
    grab_window(hcr.window, hcr.viewer.viewer_3d).save(str(path))
    print(path, flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    theme.enable_high_dpi()
    app = QApplication(sys.argv[:1])
    app.setOrganizationName("hcrcells-screenshots")  # never touch the user's own demo-mode setting
    app.setApplicationName(APP_NAME)
    theme.install(app, app_id="org.vanua.hcrcells", app_name=APP_NAME)
    QSettings().setValue(DEMO_KEY, True)
    hcr = build_app()
    show(hcr)
    settle(app, 1.5)
    save(hcr, "00-demo.png")
    hcr.sample_button.click()
    wait_for(app, hcr.loaded)
    hcr.segment_button.click()
    wait_for(app, hcr.segmented)
    settle(app, 20.0)                    # surfaces are built on a worker after `segmented`
    hcr.review_button.click()
    hcr.window.set_current_step(2)
    hcr.cell_table.selectRow(0)          # jump to cell A
    settle(app, 1.0)
    save(hcr, "01-slice.png")
    hcr.viewer.as_widget().setCurrentIndex(3)
    settle(app, 1.5)
    save(hcr, "02-3d.png")
    hcr.viewer.as_widget().setCurrentIndex(0)
    hcr.out_edit.setText(tempfile.mkdtemp(prefix="hcrcells-shots-"))
    hcr.all_t.setChecked(False)
    hcr.last_t.setValue(1)
    hcr.run_button.click()
    wait_for(app, hcr.series_done)
    settle(app, 1.0)
    save(hcr, "03-plots.png")
    hcr.window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
