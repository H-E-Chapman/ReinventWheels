"""Saved processing state per sample, so a half-processed dataset reopens where it was left.

Lives in ``<out>/<sample>/session/`` -- derived data next to that sample's
results, never beside the source files (read-only, I7):

  state.json      stage reached, source-file fingerprint, recipe hash, the
                  segmentation settings, review overrides, series run, and
                  the display settings (channels, overlays, spot colours, z, tab, plot)
  segmentation.npz  first-timepoint cell masks and their outlines (compressed)
  cell_table.csv    the first timepoint's cells (whole volume) with their intensities
  positions.json    each cell's centre per slice (for the cell labels)
  roi.json          the ROI box around the cells of interest (after spot detection)
  roi_cell_table.csv  the cells of interest measured in the ROI, with spots
  roi_spots_<ch>.csv  the first timepoint's spots in the ROI, per HCR channel

A saved state is trusted only for the same data: if any source file's name,
size or modification time changed, it is ignored. If the recipe changed, the
settings and display are restored but not the segmentation (it would no
longer be what this recipe computes). Every file is written to a temporary
name and renamed, so a crash mid-save never leaves a half-written state.

Qt-free: the GUI calls this from its worker threads.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
STAGES = ("loaded", "segmented", "reviewed", "roi_done", "series_done")
SPOT_CHANNELS = ("daf28", "ins4", "ins6")


def stage_index(stage: "str | None") -> int:
    return STAGES.index(stage) if stage in STAGES else -1


def fingerprint(files) -> list:
    """[[name, size, mtime_ns], ...] of every timepoint file of the sample."""
    out = []
    for _t, path in files:
        st = Path(path).stat()
        out.append([Path(path).name, st.st_size, st.st_mtime_ns])
    return out


def file_sha256(path) -> str:
    """SHA-256 of one file, or of several (a list: the cells and timepoint
    recipes together, so editing either invalidates a saved segmentation)."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    digest = hashlib.sha256()
    for p in paths:
        digest.update(Path(p).read_bytes())
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


class SessionStore:
    """The saved state of one sample under `out_dir`."""

    def __init__(self, out_dir: "str | Path", sample: str) -> None:
        self.sample = sample
        self.dir = Path(out_dir) / sample / "session"

    # -- state.json ----------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    def read(self) -> "dict | None":
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return state if state.get("schema") == SCHEMA and state.get("sample") == self.sample else None

    def update(self, **fields: Any) -> dict:
        """Merge `fields` into state.json (created if missing) and write it atomically."""
        self.dir.mkdir(parents=True, exist_ok=True)
        state = self.read() or {"schema": SCHEMA, "sample": self.sample}
        state.update(fields)
        state["saved"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write_text(self.state_path, json.dumps(state, indent=2, default=_json_default))
        return state

    def clear(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- segmentation ----------------------------------------------------------------

    def save_segmentation(self, values: dict, outlines: np.ndarray, positions: dict) -> None:
        """The first timepoint's results the GUI shows (not the images: they
        are re-read from the source, which is the ground truth)."""
        import io

        self.dir.mkdir(parents=True, exist_ok=True)
        cells = np.asarray(values["cells"])
        dtype = np.uint16 if cells.max(initial=0) < 65535 else np.int32
        buffer = io.BytesIO()
        np.savez_compressed(buffer, cells=cells.astype(dtype), outlines=np.asarray(outlines).astype(dtype))
        _atomic_write_bytes(self.dir / "segmentation.npz", buffer.getvalue())
        _atomic_write_text(self.dir / "cell_table.csv", values["cell_table"].to_csv(index=False))
        _atomic_write_text(self.dir / "positions.json",
                           json.dumps({str(z): rows for z, rows in positions.items()}))

    def load_segmentation(self, channels: dict, frame_doc: dict) -> "tuple[dict, np.ndarray, dict]":
        """(values, outlines, positions) as the Segment step produces them,
        with the images taken from the freshly read `channels`."""
        import pandas as pd

        with np.load(self.dir / "segmentation.npz") as npz:
            cells = npz["cells"].astype(np.int32)
            outlines = npz["outlines"].astype(np.int32)
        values = {"cells": cells, "frame": frame_doc,
                  "cell_table": pd.read_csv(self.dir / "cell_table.csv")}
        for ch in SPOT_CHANNELS:
            values[ch] = channels[ch]
        raw = json.loads((self.dir / "positions.json").read_text(encoding="utf-8"))
        positions = {int(z): [tuple(r) for r in rows] for z, rows in raw.items()}
        return values, outlines, positions


def save_roi(store: "SessionStore", values: dict) -> None:
    """The first timepoint's ROI run: box, measured cells of interest, spots."""
    store.dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(store.dir / "roi.json", json.dumps({"roi": values["roi"], "ref_roi": values.get("ref_roi")},
                                                          default=_json_default))
    _atomic_write_text(store.dir / "roi_cell_table.csv", values["cell_table"].to_csv(index=False))
    for ch in SPOT_CHANNELS:
        _atomic_write_text(store.dir / f"roi_spots_{ch}.csv", values[f"spots_{ch}"].to_csv(index=False))


def load_roi(store: "SessionStore", frame_doc: dict) -> dict:
    import pandas as pd

    boxes = json.loads((store.dir / "roi.json").read_text(encoding="utf-8"))
    values = {"roi": boxes["roi"], "ref_roi": boxes.get("ref_roi") or boxes["roi"], "frame": frame_doc,
              "cell_table": pd.read_csv(store.dir / "roi_cell_table.csv")}
    for ch in SPOT_CHANNELS:
        values[f"spots_{ch}"] = pd.read_csv(store.dir / f"roi_spots_{ch}.csv")
    return values


SessionStore.save_roi = save_roi
SessionStore.load_roi = load_roi


# -- every timepoint's segmentation (for browsing timepoints in the Display) ---------------------
#
# <session>/timepoints/T00012.npz  cell masks (all cells, compressed) and each HCR
#                                  channel's spots: whole-volume voxel indices and world µm
# <session>/timepoints/T00012.json tracked cell names per label, the ROI and reference
#                                  region, the source file
# Written by whoever holds that timepoint's results during Run series (the GUI
# for the first, the sequential loop, or a parallel stage-2 worker): ~0.3 MB each.

SPOT_COLUMNS = ("z_index", "y_index", "x_index", "z_um", "y_um", "x_um")


def timepoints_dir(session_dir) -> Path:
    return Path(session_dir) / "timepoints"


def clear_timepoints(session_dir) -> None:
    shutil.rmtree(timepoints_dir(session_dir), ignore_errors=True)


def save_timepoint(session_dir, t: int, cells: np.ndarray, spots: dict, names: dict, roi: dict,
                   ref_roi: "dict | None", source: str) -> Path:
    """One timepoint's segmentation. `spots`: {channel: table with whole-volume
    indices and world µm}; `names`: {label: tracked cell name}."""
    import io

    folder = timepoints_dir(session_dir)
    folder.mkdir(parents=True, exist_ok=True)
    cells = np.asarray(cells)
    arrays = {"cells": cells.astype(np.uint16 if cells.max(initial=0) < 65535 else np.int32)}
    for ch, table in spots.items():
        cols = [c for c in SPOT_COLUMNS if c in table.columns]
        arrays[f"spots_{ch}"] = table[cols].to_numpy(np.float64) if len(table) else np.zeros((0, len(cols)))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_write_bytes(folder / f"T{t:05d}.npz", buffer.getvalue())
    meta = {"t": int(t), "names": {str(int(k)): v for k, v in names.items()}, "roi": roi, "ref_roi": ref_roi,
            "source": source, "spot_columns": list(SPOT_COLUMNS)}
    _atomic_write_text(folder / f"T{t:05d}.json", json.dumps(meta, default=_json_default))
    return folder / f"T{t:05d}.npz"


def load_timepoint(session_dir, t: int) -> "dict | None":
    """{cells, spots: {channel: table}, names: {label: name}, roi, ref_roi, source}, or None."""
    import pandas as pd

    folder = timepoints_dir(session_dir)
    npz_path, meta_path = folder / f"T{t:05d}.npz", folder / f"T{t:05d}.json"
    if not (npz_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as npz:
            cells = npz["cells"].astype(np.int32)
            spots = {}
            for key in npz.files:
                if key.startswith("spots_"):
                    arr = npz[key]
                    table = pd.DataFrame(arr, columns=meta["spot_columns"][:arr.shape[1]] if arr.ndim == 2 else [])
                    for c in ("z_index", "y_index", "x_index"):
                        if c in table:
                            table[c] = table[c].astype(int)
                    spots[key[len("spots_"):]] = table
    except (OSError, KeyError, ValueError):
        return None
    return {"cells": cells, "spots": spots, "names": {int(k): v for k, v in meta["names"].items()},
            "roi": meta["roi"], "ref_roi": meta.get("ref_roi"), "source": meta.get("source", "")}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def restore_bundle(out_dir, sample: str, files, recipe_path, channels: dict, frame_doc: dict) -> dict:
    """Everything needed to restore `sample`, read off the GUI thread.

    Returns {"state": dict | None, "segmentation": (values, outlines, positions) | None,
    "results": DataFrame | None, "notes": [str]}; `state` is None when there
    is nothing usable (no saved state, or the source files changed)."""
    import pandas as pd

    store = SessionStore(out_dir, sample)
    bundle: dict = {"state": None, "segmentation": None, "roi": None, "results": None, "notes": []}
    state = store.read()
    if state is None:
        return bundle
    if state.get("sources") != fingerprint(files):
        bundle["notes"].append(f"{sample}: the source files changed since the saved state; starting fresh")
        return bundle
    bundle["state"] = state
    reached = stage_index(state.get("stage"))
    if reached >= stage_index("segmented"):
        if state.get("recipe_sha256") != file_sha256(recipe_path):
            bundle["notes"].append(f"{sample}: the recipe changed since the saved segmentation; "
                                   "settings and display restored, segment again")
            state["stage"] = "loaded"
        else:
            try:
                bundle["segmentation"] = store.load_segmentation(channels, frame_doc)
            except (OSError, KeyError, ValueError) as exc:
                bundle["notes"].append(f"{sample}: saved segmentation unreadable ({exc}); segment again")
                state["stage"] = "loaded"
    if stage_index(state.get("stage")) >= stage_index("roi_done"):
        try:
            bundle["roi"] = store.load_roi(frame_doc)
        except (OSError, KeyError, ValueError) as exc:
            bundle["notes"].append(f"{sample}: saved ROI spots unreadable ({exc}); detect them again")
            state["stage"] = "reviewed"
    if stage_index(state.get("stage")) >= stage_index("series_done"):
        csv = Path((state.get("series") or {}).get("results_csv", ""))
        if csv.is_file():
            bundle["results"] = pd.read_csv(csv)
        else:
            bundle["notes"].append(f"{sample}: series results not found ({csv}); run the series again")
            state["stage"] = "roi_done"
    return bundle
