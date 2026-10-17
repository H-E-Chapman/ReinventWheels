# MeltPool

The liquid-gas boundary (the vapour-depression rim) of a melt pool in high-speed
X-ray video, frame by frame: size, profile, dynamics and an overlay movie, plus
the faint liquid-solid line on the stabilised median. Built on
[vanua](../../vanua). The spec, with every assumption and known limit, is
[app-spec.toml](app-spec.toml).

## Run

From this folder, with the environment in `../.venv` (Python 3.11, vanua editable):

```bash
# GUI -- or just `meltpool` with .venv active (Tutorial / Demo mode: View menu)
../.venv/Scripts/python -m meltpool.app [MOVIE]

# Batch over a folder: per-run CSVs (+ .provenance.json), plots, overlay videos, summary.csv, comparison.png
../.venv/Scripts/python -m meltpool.batch IN_DIR OUT_DIR --pixel-um 4.44 --fps 100000 --stabilise fixed

# Tests (synthetic data only)
../.venv/Scripts/python -m pytest tests -q
```

"Load sample" opens a synthetic weld video (`data/synthetic_movie.tif`, written by
`python scripts/make_synthetic.py` on first use); `MELTPOOL_SAMPLE=path` points it
at another file.

Batch options: `--flip` / `--flip-horizontal` (recorded upside down / mirrored),
`--crop X0,Y0,X1,Y1`, `--stabilise fixed|jitter`, `--transient WINDOW` (trace on
running-median-subtracted frames), `--n-avg`, `--stride`, `--r-max`,
`--no-overlay`, `--overlay-format mp4|gif`, `--solid` (also the liquid-solid
search), `--dataset` (HDF5).

## Workflow

1. **Load** a movie.
2. **Calibrate**: pixel size and frame rate; *Flip vertically / horizontally*
   (applied at once); *Stabilise now* registers every frame to 1/100 px on the
   substrate below the pool (*fixed* for a static camera on the pool, *jitter*
   for a moving sample). *Draw stabilisation region* to pick the region yourself
   -- it must not contain the moving boundary.
3. **Crop**: drag a box (*Draw ROI on image*) and *Apply crop*.
4. **Transient** (optional): subtract a running median to see what moves; for
   display only (views, overlay video) or for the tracing too.
5. **Liquid-gas boundary**: scrub to a frame and *Preview* -- the rim as three
   lines: inner edge (cyan, gas side), band centre (orange), outer edge (magenta,
   liquid side).
6. **Measure**: *Trace all frames*; the start-up is flagged, and the spectra of
   depth, width and area are computed over the steady part.
7. **Results**: steady-state medians with the dominant oscillation frequency,
   time series, spectrum, table; *Export* writes `<stem>_pool.csv`,
   `_summary.csv`, `_profiles.csv`, `_spectrum.csv`, `_peaks.csv` (each with a
   `.provenance.json` holding the movie's SHA-256, ops and parameters), the
   series and spectrum plots and the overlay video (`_overlay.mp4`).
8. **Liquid-solid**: stabilised median (optionally as fine detail), then click
   points along the line or run the automatic search, which only reports a line
   when it beats the same search where there is no pool.

Pages change only when you click them. The MIP / Ortho / 3D tabs show the movie
as a (time, y, x) stack; the Rendering dock holds the display contrast and the
3D transparency curve. Both are display only -- the analysis always uses the raw
intensities.

## Method

- **Surface**: the original (unmelted) surface, the 80th percentile of the
  per-column air/material edge.
- **Rim**: a dark-ridge filter (Sato) and a shortest path r(θ) around the
  depression centre (dynamic programming over angle), the centre re-found from
  the rim's two ends and tracked from frame to frame. Along each ray, the inner
  and outer edges are where the intensity is halfway from the band minimum to
  each side; the band centre is their midpoint.
- **Per line**: depth below the surface, width at the surface, area enclosed
  below the surface.
- **Dynamics**: amplitude spectrum (linear trend removed) of every depth, width
  and area series over the longest run of steady time points; the strongest peaks.

## Formats

TIFF (one frame per page, memory-mapped), HDF5/NeXus (`--dataset` for a 3D
(time, y, x) dataset; default the largest), `.npy`, and video (AVI, MP4, MOV,
MKV ... via PyAV, including palette AVI). The reader is chosen from the file's
content, not its extension. A pixel size stored in an ImageJ TIFF is used
(`--ignore-file-pixel` to override); a frame rate stored in a file is a playback
rate and is never used. Needs `pip install h5py av` (the `formats` extra) for
HDF5 and video.

Lab movies, tables and videos are never committed (`data/`, `out/` are ignored).
