"""Launch benchmark: the old blocking start vs the splash launcher, on the REAL display.

    pixi run bench-launch            # 3 runs per mode
    pixi run bench-launch -- --runs 5

Each run is a fresh process (spawned by this script), so import costs are
real; modes alternate so the OS file cache favours neither. Every mark is
seconds since the parent spawned the child:

  first_paint     something is on screen (the splash; for 'direct', the window)
  window_painted  the main window has painted
  ready           the sample folder is scanned; the app takes input
                  ('direct' also creates the 3D view before this; 'splash'
                  creates it on first opening the 3D tab: open_3d_s, timed
                  separately after ready)

Writes out/bench_launch.json and prints medians. Windows open and close by
themselves (~10 s each).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
MARKS = ("python_up", "first_paint", "window_painted", "ready")


def _child(mode: str) -> None:
    t_spawn = float(os.environ["BENCH_T0"])
    offset = time.time() - time.perf_counter()  # perf_counter -> wall clock
    out: dict = {"python_up": time.time() - t_spawn}

    def wall(pc: float) -> float:
        return pc + offset - t_spawn

    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    from vanua.ui import theme

    theme.enable_high_dpi()
    app = QApplication(sys.argv[:1])
    app.setOrganizationName("hcrcells-bench")  # never touch the user's own settings
    app.setApplicationName("HCR Cells")
    theme.install(app, app_id="org.vanua.hcrcells", app_name="HCR Cells")
    folder = APP_DIR / "originals"

    if mode == "direct":
        from hcrcells import app as app_module  # every heavy import, on the main thread

        hcr = app_module.build_app()
        app_module.show(hcr, init_3d=True)       # the old start: VTK render window before the event loop
        app.processEvents()
        out["first_paint"] = out["window_painted"] = time.time() - t_spawn
        hcr.scan(folder)
        out["ready"] = time.time() - t_spawn
        QTimer.singleShot(300, app.quit)
    else:
        from hcrcells.launch import launch

        timings: dict = {}

        def _ready(hcr) -> None:
            def _open_3d() -> None:  # what the user pays later, on first opening the 3D tab
                t = time.perf_counter()
                hcr.viewer.as_widget().setCurrentIndex(3)
                timings["open_3d_s"] = time.perf_counter() - t
                QTimer.singleShot(300, app.quit)

            QTimer.singleShot(200, _open_3d)

        launch(app, folder, timings=timings, on_ready=_ready)
        app.processEvents()
        out["first_paint"] = time.time() - t_spawn
    app.exec_()
    if mode != "direct":
        out["window_painted"] = wall(timings["window_painted"])
        out["ready"] = wall(timings["ready"])
        out["open_3d_s"] = timings.get("open_3d_s", float("nan"))
    print("BENCH " + json.dumps(out), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0, help="Seconds before a launch counts as failed")
    parser.add_argument("--child", choices=("direct", "splash"))
    args = parser.parse_args()
    if args.child:
        _child(args.child)
        return 0

    results: dict = {"direct": [], "splash": []}
    env = dict(os.environ, PYTHONPATH=str(APP_DIR / "src"))
    for i in range(args.runs):
        for mode in ("direct", "splash"):
            env["BENCH_T0"] = repr(time.time())
            try:
                proc = subprocess.run([sys.executable, __file__, "--child", mode], env=env, cwd=APP_DIR,
                                      capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"run {i + 1} {mode:6s} FAILED: no ready window within {args.timeout:.0f} s", flush=True)
                continue
            line = next((l for l in proc.stdout.splitlines() if l.startswith("BENCH ")), None)
            if line is None:
                print(f"{mode} run {i + 1} failed:\n{proc.stderr[-2000:]}")
                continue
            marks = json.loads(line[6:])
            results[mode].append(marks)
            extra = f"  (3D tab opened later: {marks['open_3d_s']:.2f}s)" if "open_3d_s" in marks else ""
            print(f"run {i + 1} {mode:6s} " + "  ".join(f"{k} {marks[k]:5.2f}s" for k in MARKS) + extra, flush=True)

    print("\nmedian (s)      " + "  ".join(f"{k:>14s}" for k in MARKS))
    summary = {}
    for mode, runs in results.items():
        if runs:
            summary[mode] = {k: statistics.median(r[k] for r in runs) for k in MARKS}
            print(f"{mode:15s} " + "  ".join(f"{summary[mode][k]:14.2f}" for k in MARKS))
    out = APP_DIR / "out" / "bench_launch.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"runs": results, "median_s": summary}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
