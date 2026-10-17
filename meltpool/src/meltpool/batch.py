"""Headless MeltPool over a folder: the same recipe per movie.

    python -m meltpool.batch IN_DIR OUT_DIR [--pixel-um 4.44] [--fps 100000] [--crop X0,Y0,X1,Y1] ...

Writes per movie `<stem>_pool.csv` (the three lines per time point),
`<stem>_summary.csv` (steady-state medians, IQRs, dominant frequencies),
`<stem>_profiles.csv`, `<stem>_spectrum.csv`, `<stem>_peaks.csv`, the series and
spectrum plots and the overlay video; for the folder `summary.csv` (one row per
movie; a movie that fails is a row with the error, not a crash) and
`comparison.png`. Every CSV has a `.provenance.json` sidecar holding the movie's
SHA-256.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from vanua.blocks.plot import bar_figure, save_figure

from .movie import MOVIE_SUFFIXES as SUFFIXES
from .ops import CropParams, FlagParams, TraceParams
from .pipeline import Calibration, Settings, export_movie, load_recipe, registry, run_movie

COMPARED = ("depth_inner_median_um", "depth_centre_median_um", "depth_outer_median_um", "width_centre_median_um")


def comparison_figure(table):
    """Steady-state medians per run (lengths, µm)."""
    ok = table[table.get("error", "") == ""] if "error" in table else table
    series = {c.replace("_median_um", "").replace("_", " "): ok[c].tolist() for c in COMPARED if c in ok}
    return bar_figure(ok["run"].tolist(), series, ylabel="steady-state median (µm)",
                      title="Liquid-gas boundary per run (median of steady time points)")


def run_folder(in_dir, out_dir, settings: "Settings | None" = None, log=print, solid: bool = False,
               overlay: bool = True):
    import pandas as pd

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = settings or Settings()
    recipe, reg = load_recipe(), registry()
    rows = []
    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    # One recording often sits next to its exports (a .tif and an .avi of the same run): run stems must be unique.
    stems = [p.stem for p in files]
    files_stems = [(p, p.stem if stems.count(p.stem) == 1 else p.name.replace(".", "_")) for p in files]
    for i, (path, stem) in enumerate(files_stems, 1):
        log(f"[{i}/{len(files)}] {path.name}")
        try:
            configured, values = run_movie(path, settings, out_dir / f"{stem}_depth.png", recipe=recipe, reg=reg,
                                           log=lambda m: log(f"  {m}"), overlay=overlay)
            written = export_movie(configured, values, out_dir, stem, reg)
            row = pd.read_csv(written["summary"], keep_default_na=False).iloc[0].to_dict()
            if solid:
                from vanua.blocks.provenance import export_table

                from .pipeline import run_solid

                srecipe, svalues = run_solid(path, settings, reg=reg, log=lambda m: log(f"  {m}"))
                stable = svalues["solid"].copy()
                stable.insert(0, "run", stem)
                export_table(stable, out_dir / f"{stem}_solid.csv", srecipe, "solid", reg,
                             extra={"line": svalues["solid_line"]})
                r = stable.iloc[0]
                row.update({"solid_detected": bool(r["detected"]), "solid_z": float(r["z"]),
                            **{f"solid_{k}": r[k] for k in stable.columns if k.startswith(("depth_", "width_"))}})
            row["error"] = ""
        except Exception as exc:  # noqa: BLE001 -- one bad movie must not stop the batch
            log(f"  failed: {exc}")
            row = {"run": stem, "error": str(exc)}
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.csv.provenance.json").write_text(json.dumps({
        "schema_version": 1, "binding": "summary", "recipe": recipe.name, "settings": asdict(settings),
        "per_run_sidecars": [f"{r['run']}_summary.csv.provenance.json" for r in rows if not r.get("error")],
    }, indent=2, default=str), encoding="utf-8")
    if (table.get("error", "") == "").any():
        save_figure(comparison_figure(table), out_dir / "comparison.png")
    return table


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    cal, trace, flags = Calibration(), TraceParams(), FlagParams()
    parser.add_argument("--pixel-um", type=float, default=cal.pixel_um)
    parser.add_argument("--fps", type=float, default=cal.fps)
    parser.add_argument("--flip", action="store_true", help="flip vertically (recorded upside down)")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--dataset", default="", help="HDF5: path of the (time, y, x) dataset (default: the largest 3D)")
    parser.add_argument("--ignore-file-pixel", action="store_true",
                        help="use --pixel-um even when the file states a pixel size")
    parser.add_argument("--crop", default="", metavar="X0,Y0,X1,Y1",
                        help="ROI in pixels of the displayed frame, end exclusive, 0 = to the edge")
    parser.add_argument("--stabilise", choices=("off", "jitter", "fixed"), default="off",
                        help="sub-pixel stabilisation: fixed (static sample) or jitter (moving sample, keeps drift)")
    parser.add_argument("--transient", type=int, default=0, metavar="WINDOW",
                        help="trace on running-median-subtracted frames with this window (0: off)")
    parser.add_argument("--n-avg", type=int, default=trace.n_avg, help="frames averaged per traced time point")
    parser.add_argument("--stride", type=int, default=trace.stride, help="one time point per N frames")
    parser.add_argument("--r-max", type=float, default=trace.r_max, help="how far from the centre the rim may be (px)")
    parser.add_argument("--tolerance", type=float, default=flags.tolerance)
    parser.add_argument("--no-overlay", action="store_true", help="skip the overlay video of the traced lines")
    parser.add_argument("--overlay-format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--solid", action="store_true",
                        help="also: stabilised median and the automatic liquid-solid search (<stem>_solid.csv)")
    args = parser.parse_args(argv)
    settings = Settings()
    settings.calibration = Calibration(pixel_um=args.pixel_um, pixel_from_file=not args.ignore_file_pixel,
                                       fps=args.fps, flip_vertical=args.flip, flip_horizontal=args.flip_horizontal,
                                       dataset=args.dataset)
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
        settings.crop = CropParams(x0=x0, x1=x1, y0=y0, y1=y1)
    settings.stabilise.enabled = args.stabilise != "off"
    if settings.stabilise.enabled:
        settings.stabilise.mode = args.stabilise
    if args.transient:
        settings.background.enabled = True
        settings.background.window = args.transient
        settings.background.apply_to = "display and analysis"
    settings.trace.n_avg, settings.trace.stride, settings.trace.r_max = args.n_avg, args.stride, args.r_max
    settings.flags.tolerance = args.tolerance
    settings.overlay.out_path = f"out/overlay.{args.overlay_format}"
    table = run_folder(args.in_dir, args.out_dir, settings, solid=args.solid, overlay=not args.no_overlay)
    print(table.to_string(index=False))
    return 1 if table.get("error", "").astype(bool).any() else 0


if __name__ == "__main__":
    sys.exit(main())
