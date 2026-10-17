"""Fast start for the HCR Cells GUI: a splash paints at once while the heavy
imports run on a background thread (vanua.ui.splash.run_with_splash).

    pixi run gui            # = python -m hcrcells.launch originals

Imports nothing heavy at module level -- PyQt5 and vanua.ui.theme only.
numpy/scipy/pandas, scikit-image/tifffile, matplotlib and
:mod:`hcrcells.app` itself (which pulls all of them in at its own module
level) are imported in loader steps, off the main thread. Only widget
construction must happen on the main thread: the window is built after the
loader finishes and shown. VTK is not loaded at launch at all: the 3D view
is created the first time its tab is opened (hcrcells.app.initialize_3d).

`timings` (optional) receives perf_counter marks for scripts/bench_launch.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSlot
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication

# vanua.ui before anything that imports VTK (building.md §5); theme is light.
from vanua.ui import theme
from vanua.ui.splash import SplashLoader, SplashStep, render_splash, show_splash

APP_NAME = "HCR Cells"
APP_ID = "org.vanua.hcrcells"
_MIN_VISIBLE_S = 0.6  # as vanua's splash: long enough to read on a warm start


class _SharpSplash(QObject):
    """vanua.ui.splash.run_with_splash, with two fixes (built from its public
    pieces; vanua itself is not edited):

    * Every status frame is rendered at the screen's device-pixel ratio.
      run_with_splash renders only the first frame at it and every update at
      1x, which a 200 % display then stretches: the blurry logo and text.
    * Status updates arrive through this QObject, which lives on the main
      thread, so Qt queues them there. run_with_splash connects the loader's
      signal to a plain function, which runs on the loader thread -- painting
      and calling setPixmap off the GUI thread.
    """

    def __init__(self, app: QApplication, title: str, steps, build_window, icon) -> None:
        super().__init__(app)
        self.app, self.title, self.build_window, self.icon = app, title, build_window, icon
        self.messages = tuple(step.message for step in steps)
        screen = app.screenAt(QCursor.pos()) or app.primaryScreen()
        self.dpr = screen.devicePixelRatio() if screen is not None else 1.0
        self.splash = show_splash(app, title, self.messages[0] if self.messages else "", icon=icon,
                                  possible_messages=self.messages)
        self.shown_at = time.monotonic()
        self.loader = SplashLoader(steps)
        self.thread = QThread()
        self.loader.moveToThread(self.thread)
        self.thread.started.connect(self.loader.run)
        self.loader.progress.connect(self.on_progress)   # queued: receiver lives on the main thread
        self.loader.finished.connect(self.on_finished)
        self.loader.failed.connect(self.on_failed)
        self.thread.start()

    @pyqtSlot(str)
    def on_progress(self, message: str) -> None:
        self.splash.setPixmap(render_splash(self.title, message, icon=self.icon, dpr=self.dpr,
                                            possible_messages=self.messages))

    @pyqtSlot()
    def on_finished(self) -> None:
        self.thread.quit()
        remaining = _MIN_VISIBLE_S - (time.monotonic() - self.shown_at)
        QTimer.singleShot(max(0, round(remaining * 1000)), self._finish)

    def _finish(self) -> None:
        window = self.build_window()
        self.splash.finish(window)

    @pyqtSlot(str)
    def on_failed(self, message: str) -> None:
        from PyQt5.QtWidgets import QMessageBox

        self.thread.quit()
        self.splash.hide()
        QMessageBox.critical(None, self.title, f"Could not start:\n{message}")
        self.app.quit()


def _splash_icon(app: QApplication):
    """The largest available icon art (vanua's 1254 px PNG), so the splash
    logo is drawn from a big source at any display scale."""
    from PyQt5.QtGui import QIcon

    try:
        from vanua.core.resources import resolve_icon

        icon = QIcon(str(resolve_icon(None, prefer="png", app_name=APP_NAME)))
        if not icon.isNull():
            return icon
    except Exception:  # noqa: BLE001 -- fall back to the window icon
        pass
    return app.windowIcon()


def _import_core() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy.ndimage  # noqa: F401
    import scipy.optimize  # noqa: F401


def _import_imaging() -> None:
    import skimage.feature  # noqa: F401
    import skimage.measure  # noqa: F401
    import skimage.segmentation  # noqa: F401
    import tifffile  # noqa: F401


def _import_plotting() -> None:
    import matplotlib.figure  # noqa: F401
    from matplotlib.backends import backend_qt5agg  # noqa: F401


def launch(app: QApplication, folder: "str | Path | None" = None, timings: "dict | None" = None,
           on_ready=None, experiment: "str | None" = None) -> None:
    """Splash -> loader thread -> window (painted) -> 3D tab + folder scan."""
    mark = (lambda name: timings.__setitem__(name, time.perf_counter())) if timings is not None \
        else (lambda name: None)
    loaded: dict = {}

    def _prepare_interface() -> None:
        # The experiment type: given, or from the folder (series of timepoints:
        # photobleaching; date folders of single images: groups).
        kind = experiment
        if kind is None and folder:
            from .groups_app import detect_experiment

            kind = detect_experiment(folder)
        if kind == "groups":
            from . import groups_app as app_module
        else:
            from . import app as app_module

        loaded["app"] = app_module

    # No VTK step: the 3D view is created lazily, the first time its tab is
    # opened (hcrcells.app.initialize_3d), so launch never pays for it.
    steps = [
        SplashStep("Loading numpy, scipy, pandas…", _import_core),
        SplashStep("Loading scikit-image, tifffile…", _import_imaging),
        SplashStep("Loading matplotlib…", _import_plotting),
        SplashStep("Preparing interface…", _prepare_interface),
    ]

    def _build_window():
        mark("imports_done")
        app_module = loaded["app"]
        hcr = app_module.build_app()
        loaded["hcr"] = hcr
        mark("window_built")
        app_module.show(hcr, init_3d=False)
        hcr.window.append_log(f"{APP_NAME} started.")
        # Let the window paint first; the folder scan comes one turn later.
        QTimer.singleShot(0, _after_paint)
        return hcr.window

    def _after_paint() -> None:
        mark("window_painted")
        app_module, hcr = loaded["app"], loaded["hcr"]
        target = Path(folder) if folder else getattr(app_module, "SAMPLE_DIR", None)
        if target is not None:  # the groups app starts empty: choose the folder in step 1
            hcr.folder_edit.setText(str(target))
            hcr.scan(target)
        mark("ready")
        if on_ready:
            on_ready(hcr)

    # Kept on the app: the splash controller and its thread must outlive this function.
    app._hcrcells_splash = _SharpSplash(app, APP_NAME, steps, _build_window, _splash_icon(app))  # noqa: SLF001
    mark("splash_shown")


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    theme.enable_high_dpi()  # before the QApplication: Qt reads it once
    app = QApplication(argv)
    app.setOrganizationName("hcrcells")  # QSettings home of the demo-mode toggle
    app.setApplicationName(APP_NAME)
    theme.install(app, app_id=APP_ID, app_name=APP_NAME)
    # --groups / --bleaching: the experiment type, when the folder alone does not say.
    flags = {a for a in argv[1:] if a.startswith("--")}
    args = [a for a in argv[1:] if not a.startswith("--")]
    kind = "groups" if "--groups" in flags else "bleaching" if "--bleaching" in flags else None
    launch(app, args[0] if args else None, experiment=kind)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
