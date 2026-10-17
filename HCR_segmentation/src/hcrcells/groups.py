"""Group experiments: transcripts per image and channel in worms of several
groups (strains, treatments), one image per worm. Qt-free; the GUI
(:mod:`hcrcells.groups_app`) and the command line share it.

    pixi run groups <folder> <out>      # = python -m hcrcells.groups

The folder holds one sub-folder per imaging day (its name has the date); in
it one TIFF per worm, the group at the start of the file name (``N2_...``,
``ins-6_...``, ``ins-6rescue_...``) and the worm's number at its end.

1. :func:`scan` -- every image: day, group, worm number. Misspelt groups are
   matched to the nearest known one (:func:`resolve_groups`); a name that
   matches none is flagged.
2. :func:`check_images` -- per image, a small DAPI projection (to find the
   same worm saved twice, e.g. under two numbers -- :func:`find_duplicates`)
   and the drift across the stack (:mod:`hcrcells.qc`). :func:`decide` then
   says which images are analysed and why the others are not; every problem
   is listed as a warning.
3. :func:`run_images` -- the image recipe on every analysed image (parallel):
   drift registration, cells, spots for every cell.
4. :func:`cell_results` -- the selection rule and your review (which cells
   count, which neuron each is); :func:`transcripts_per_image` -- the
   result: per image and channel, the transcripts in its cells of interest
   (:func:`write_results` writes it and the QC tables as CSV). Statistics,
   batch correction and group comparisons are left to the analysis in R.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .pipeline import CHANNELS, GENE, IMAGE_RECIPE_PATH, RoiSettings, Settings, configured, load_recipe, registry
from .qc import DriftSettings

IMAGE_SUFFIXES = (".tif", ".tiff")
QC_VERSION = 2  # the drift and fingerprint methods: cached checks of another version are redone
#: The worm's number: the last number of the file name, alone or as worm3 / w3.
WORM_PATTERN = r"(?:^|[_\s.\-])(?:worm|w)?0*(\d+)$"
_TIMEPOINT_SUFFIX = re.compile(r"\s*-\s*T=\d+$")
_DATES = (
    (re.compile(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)"), (1, 2, 3)),  # 2026-03-14
    (re.compile(r"(?<!\d)(0[1-9]|[12]\d|3[01])[-_.](0[1-9]|1[0-2])[-_.](20\d{2})(?!\d)"), (3, 2, 1)),     # 14.03.2026
    (re.compile(r"(?<!\d)(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)"), (1, 2, 3)),                # 260314
)


@dataclass
class GroupSettings:
    """How images are named and when two images count as the same worm."""

    groups: str = field(default="", metadata={
        "label": "Groups", "help": "The group names, comma-separated (e.g. 'N2, ins-6, ins-6rescue'); "
                                   "file names are matched to the nearest. Empty: learnt from the file names"})
    worm_pattern: str = field(default=WORM_PATTERN, metadata={
        "label": "Worm number pattern", "help": "Regular expression; its first group is the worm's number"})
    max_typo: int = field(default=2, metadata={
        "label": "Typos corrected up to", "range": (0, 5), "help": "Letters wrong, missing, extra or swapped "
                                                                  "(names of 4 letters or more)"})
    duplicate_above: float = field(default=0.90, metadata={
        "label": "Same worm above", "range": (0.0, 1.0), "step": 0.01,
        "help": "Similarity of two DAPI projections above which they are the same worm (same worm "
                "re-imaged: 0.97-1.00; two worms: about 0.45)"})
    possible_above: float = field(default=0.75, metadata={
        "label": "Possibly the same worm above", "range": (0.0, 1.0), "step": 0.01,
        "help": "Above this (and below 'same worm') the pair is only flagged"})


# -- names ------------------------------------------------------------------------------------


def normalize(name: str) -> str:
    """Case, spaces, hyphens and underscores do not distinguish groups."""
    return re.sub(r"[\s\-_]+", "", name).lower()


def edit_distance(a: str, b: str) -> int:
    """Letters changed, missing, extra or swapped (optimal string alignment)."""
    d = np.zeros((len(a) + 1, len(b) + 1), int)
    d[:, 0] = np.arange(len(a) + 1)
    d[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + 1)
    return int(d[-1, -1])


def image_stem(path: "str | Path") -> str:
    """The file name without its extensions and a ' - T=n' suffix
    ('N2_x_3.czi - T=0.tif' -> 'N2_x_3')."""
    stem = Path(path).name
    for suffix in IMAGE_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = _TIMEPOINT_SUFFIX.sub("", stem)
    return re.sub(r"\.czi$", "", stem, flags=re.IGNORECASE)


def group_token(stem: str) -> str:
    """The group: the file name up to its first underscore."""
    return stem.split("_", 1)[0].strip()


def worm_number(stem: str, pattern: str = WORM_PATTERN) -> "int | None":
    m = re.search(pattern, stem, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def resolve_groups(tokens, known=(), max_typo: int = 2) -> dict:
    """{token as written: (group, how)}; how is 'exact', 'case' (only case,
    spaces or hyphens differ), 'typo' (within `max_typo` edits of exactly
    one group, names of 4+ letters) or 'unknown' (group None).

    Without `known` groups they are learnt: the most frequent spellings
    first, and a rarer spelling within `max_typo` of one of them joins it."""
    from collections import Counter

    counts = Counter(tokens)
    canon = [g.strip() for g in known if g.strip()]
    learn = not canon
    out = {}
    for token, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        group, how = _match(token, canon, max_typo)
        if group is None and learn:
            canon.append(token)
            group, how = token, "exact"
        out[token] = (group, how)
    return out


def _match(token: str, canon, max_typo: int):
    if token in canon:
        return token, "exact"
    norm = normalize(token)
    for g in canon:
        if normalize(g) == norm:
            return g, "case"
    if len(norm) < 4 or not canon:
        return None, "unknown"
    dist = sorted((edit_distance(norm, normalize(g)), g) for g in canon)
    best, group = dist[0]
    runner_up = dist[1][0] if len(dist) > 1 else None
    if best <= max_typo and (runner_up is None or runner_up > best):
        return group, "typo"
    return None, "unknown"


def day_of(path: Path, root: Path) -> "tuple[str, str]":
    """(day: the date folder's name, ISO date or '') -- the nearest folder
    between `root` and the file whose name has a date; else the parent's name."""
    rel = path.relative_to(root)
    folders = list(rel.parents)[:-1]  # nearest first, root excluded
    for folder in folders:
        iso = parse_date(folder.name)
        if iso:
            return str(folder).replace("\\", "/"), iso
    return (str(rel.parent).replace("\\", "/") if rel.parent != Path(".") else ""), ""


def parse_date(name: str) -> str:
    for pattern, order in _DATES:
        m = pattern.search(name)
        if m:
            y, mo, d = (m.group(i) for i in order)
            y = y if len(y) == 4 else f"20{y}"
            return f"{y}-{mo}-{d}"
    return ""


def image_id(day: str, stem: str) -> str:
    """A file-system-safe id, unique in the experiment."""
    allowed = set(string.ascii_letters + string.digits + "-_.")
    clean = lambda s: "".join(c if c in allowed else "-" for c in s)  # noqa: E731
    return f"{clean(day) or 'root'}__{clean(stem)}"


def scan(root: "str | Path", settings: "GroupSettings | None" = None):
    """One row per image: id, path, day, date, group (as written and
    resolved, and how), worm number."""
    import pandas as pd

    s = settings or GroupSettings()
    root = Path(root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    for p in paths:
        stem = image_stem(p)
        day, iso = day_of(p, root)
        rows.append({"image": image_id(day, stem), "path": str(p), "file": p.name, "day": day, "date": iso,
                     "group_raw": group_token(stem), "worm": worm_number(stem, s.worm_pattern)})
    table = pd.DataFrame(rows, columns=["image", "path", "file", "day", "date", "group_raw", "worm"])
    known = [g for g in s.groups.split(",") if g.strip()]
    resolved = resolve_groups(table["group_raw"].tolist(), known, s.max_typo)
    table["group"] = [resolved[t][0] or "" for t in table["group_raw"]]
    table["group_match"] = [resolved[t][1] for t in table["group_raw"]]
    table["worm"] = table["worm"].astype("Int64")
    return table


def naming_warnings(table) -> list:
    """Warnings about names and numbers: corrected / unknown groups, images
    outside a date folder, missing worm numbers, a number used twice in one
    group on one day."""
    out = []
    for raw, g in table.groupby("group_raw"):
        how, group = g["group_match"].iat[0], g["group"].iat[0]
        if how == "typo":
            out.append({"kind": "group corrected", "images": list(g["image"]),
                        "message": f"'{raw}' read as group '{group}' ({len(g)} image(s))"})
        elif how == "unknown":
            out.append({"kind": "unknown group", "images": list(g["image"]),
                        "message": f"'{raw}' matches no group; {len(g)} image(s) left out until you choose one"})
    for group, g in table[table["group"] != ""].groupby("group"):
        if len(g) == 1 and table["group"].nunique() > 1:
            out.append({"kind": "single image", "images": list(g["image"]),
                        "message": f"group '{group}' has a single image -- a misspelt name?"})
    nodate = table[table["date"] == ""]
    if len(nodate):
        out.append({"kind": "no date", "images": list(nodate["image"]),
                    "message": f"{len(nodate)} image(s) not in a folder named with a date"})
    nonum = table[table["worm"].isna()]
    if len(nonum):
        out.append({"kind": "no worm number", "images": list(nonum["image"]),
                    "message": f"{len(nonum)} image(s) without a worm number at the end of the name"})
    numbered = table[table["worm"].notna() & (table["group"] != "")]
    for (day, group, worm), g in numbered.groupby(["day", "group", "worm"]):
        if len(g) > 1:
            out.append({"kind": "number used twice", "images": list(g["image"]),
                        "message": f"{day}: {group} worm {worm} appears {len(g)} times ({', '.join(g['file'])})"})
    return out


# -- image checks (drift, duplicates) -----------------------------------------------------------


def _stat_key(path: Path) -> list:
    st = path.stat()
    return [st.st_size, int(st.st_mtime)]


def check_image(path: "str | Path", cache_dir: "str | Path | None" = None, drift_channel: int = 3) -> dict:
    """The drift table and summary, and the DAPI fingerprint, of one image
    (only the DAPI pages are read). Cached in `cache_dir` while the file is unchanged."""
    from .lazy import TiffStack, tiff_frame
    from .qc import drift_summary, estimate_z_drift, fingerprint

    import pandas as pd

    path = Path(path)
    import hashlib

    # Named by the file's path (hashlib: the same in every process, unlike hash()).
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    cache = Path(cache_dir) / f"{image_id('', image_stem(path))}_{digest}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            if list(f["key"]) == _stat_key(path) and int(f["channel"]) == drift_channel                     and int(f["version"]) == QC_VERSION if "version" in f.files else False:
                drift = pd.DataFrame({k[6:]: f[k] for k in f.files if k.startswith("drift_")})
                return {"fingerprint": f["fingerprint"], "drift": drift, "summary": json.loads(str(f["summary"]))}
    stack = TiffStack(path)
    try:
        dapi = np.stack([stack.read(z, drift_channel) for z in range(stack.n_z)])
    finally:
        stack.close()
    spacing = np.abs(np.diag(np.asarray(tiff_frame(path).as_document()["affine"], float)[:3, :3]))
    drift = estimate_z_drift(dapi, spacing)
    summary = drift_summary(drift) | {"n_z": int(dapi.shape[0])}
    fp = fingerprint(dapi)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, key=np.asarray(_stat_key(path)), channel=drift_channel, version=QC_VERSION,
                            fingerprint=fp,
                            summary=json.dumps(summary), **{f"drift_{c}": drift[c].to_numpy() for c in drift.columns})
    return {"fingerprint": fp, "drift": drift, "summary": summary}


def check_images(table, cache_dir=None, drift_channel: int = 3, workers: int = 1, progress=None, cancel=None) -> dict:
    """{image id: check_image(...)} for every image; in `workers` processes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out, n = {}, len(table)
    items = list(zip(table["image"], table["path"]))
    if workers <= 1:
        for i, (iid, path) in enumerate(items):
            if cancel is not None and cancel():
                break
            out[iid] = check_image(path, cache_dir, drift_channel)
            if progress:
                progress((i + 1) / max(n, 1), f"checked {i + 1}/{n}")
        return out
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(check_image, path, cache_dir, drift_channel): iid for iid, path in items}
        for i, fut in enumerate(as_completed(futures)):
            out[futures[fut]] = fut.result()
            if progress:
                progress((i + 1) / max(n, 1), f"checked {i + 1}/{n}")
            if cancel is not None and cancel():
                for f in futures:
                    f.cancel()
                break
    return out


def find_duplicates(table, checks: dict, settings: "GroupSettings | None" = None):
    """Pairs of images of one day whose DAPI projections are alike: one row
    per pair with its `similarity` and `verdict` ('same worm' / 'possibly the
    same worm')."""
    import pandas as pd

    from .qc import similarity

    s = settings or GroupSettings()
    rows = []
    for day, g in table.groupby("day"):
        ids = [i for i in g["image"] if i in checks]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                sim = similarity(checks[ids[a]]["fingerprint"], checks[ids[b]]["fingerprint"])
                if np.isfinite(sim) and sim >= s.possible_above:
                    rows.append({"day": day, "image_a": ids[a], "image_b": ids[b], "similarity": sim,
                                 "verdict": "same worm" if sim >= s.duplicate_above else "possibly the same worm"})
    return pd.DataFrame(rows, columns=["day", "image_a", "image_b", "similarity", "verdict"])


def decide(table, checks: dict, duplicates, drift: "DriftSettings | None" = None, user: "dict | None" = None):
    """Which images are analysed: the table with `drift_um`, `max_jump_um`,
    `drift_decision`, `duplicate_of`, `use` and `reason`.

    Left out: an unknown group; drift above the limit; the second (and later)
    image of a worm saved more than once -- the first by worm number, then
    name, is kept; and both, when one worm was saved under two groups.
    `user` {image id: {"use": bool, "group": str}} overrides any of it."""
    from .qc import drift_decision

    d = drift or DriftSettings()
    user = user or {}
    t = table.copy()
    for iid, choice in user.items():
        if choice.get("group") and iid in set(t["image"]):
            t.loc[t["image"] == iid, ["group", "group_match"]] = [choice["group"], "chosen"]
    summaries = [checks.get(i, {}).get("summary") for i in t["image"]]
    t["drift_um"] = [s["drift_um"] if s else np.nan for s in summaries]
    t["max_jump_um"] = [s["max_step_um"] if s else np.nan for s in summaries]
    t["drift_decision"] = [drift_decision(s, d) if s else "not checked" for s in summaries]
    t["duplicate_of"] = ""
    reasons = {i: [] for i in t["image"]}
    for i, g, dd, um in zip(t["image"], t["group"], t["drift_decision"], t["drift_um"]):
        if not g:
            reasons[i].append("unknown group")
        if dd == "excluded":
            reasons[i].append(f"drift {um:.1f} µm (limit {d.exclude_above:g})")
    same = duplicates[duplicates["verdict"] == "same worm"] if len(duplicates) else duplicates
    order = {i: k for k, i in enumerate(t.sort_values(["day", "group", "worm", "file"])["image"])}
    group_of = dict(zip(t["image"], t["group"]))
    for cluster in _clusters(same):
        groups = {group_of[i] for i in cluster}
        ranked = sorted(cluster, key=order.get)
        if len(groups) > 1:
            for i in ranked:
                others = ", ".join(sorted(x for x in cluster if x != i))
                reasons[i].append(f"same worm as {others}, under another group: choose one")
                t.loc[t["image"] == i, "duplicate_of"] = others
        else:
            for i in ranked[1:]:
                reasons[i].append(f"same worm as {ranked[0]}")
                t.loc[t["image"] == i, "duplicate_of"] = ranked[0]
    t["use"] = [not reasons[i] for i in t["image"]]
    t["reason"] = ["; ".join(reasons[i]) for i in t["image"]]
    for iid, choice in user.items():
        if "use" in choice and iid in reasons:
            t.loc[t["image"] == iid, "use"] = bool(choice["use"])
            t.loc[t["image"] == iid, "reason"] = ("included by you" if choice["use"] else "left out by you") + \
                (f" (was: {'; '.join(reasons[iid])})" if reasons[iid] else "")
    return t


def _clusters(pairs) -> list:
    """Connected images (a worm saved three times is one cluster)."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b in zip(pairs["image_a"], pairs["image_b"]) if len(pairs) else []:
        parent[find(a)] = find(b)
    groups: dict = {}
    for x in list(parent):
        groups.setdefault(find(x), set()).add(x)
    return [g for g in groups.values() if len(g) > 1]


def check_warnings(decided, duplicates) -> list:
    """Warnings from the image checks, for the user."""
    out = []
    for _, r in duplicates.iterrows():
        a, b = r["image_a"], r["image_b"]
        out.append({"kind": r["verdict"], "images": [a, b],
                    "message": f"{a} and {b} look like {'the same worm' if r['verdict'] == 'same worm' else 'possibly the same worm'} "
                               f"(similarity {r['similarity']:.2f})"
                               + ("" if r["verdict"] == "same worm" else ": both kept -- check them")})
    for _, r in decided[decided["drift_decision"].isin(["registered", "excluded"])].iterrows():
        what = "corrected" if r["drift_decision"] == "registered" else "too large: left out"
        out.append({"kind": f"drift {r['drift_decision']}", "images": [r["image"]],
                    "message": f"{r['image']}: drift {r['drift_um']:.1f} µm, largest jump {r['max_jump_um']:.1f} µm -- {what}"})
    return out


# -- the run ------------------------------------------------------------------------------------


def run_image(path, settings: "Settings | None" = None, roi: "RoiSettings | None" = None,
              drift: "DriftSettings | None" = None, save_dir: "str | Path | None" = None, image: str = "",
              recipe=None, reg=None, cancel=None, log=None) -> dict:
    """The image recipe on one worm: {"cells": per-cell table, "drift":
    summary, "roi", "ref_roi"}; with `save_dir` also its cells (masks),
    spots and drift for the review."""
    from vanua.core.runner import run

    reg = reg or registry()
    r = configured(recipe or load_recipe(IMAGE_RECIPE_PATH), path, settings, roi=roi, drift=drift or DriftSettings())
    values = run(r, registry=reg, cancel=cancel, log=log)
    cells = values["cell_table"].copy()
    cells.insert(0, "image", image)
    out = {"cells": cells, "drift": values["drift_summary"], "roi": values["roi"], "ref_roi": values["ref_roi"]}
    if save_dir is not None:
        d = Path(save_dir) / image
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "cells.npz", labels=np.asarray(values["cells"]).astype(np.int32))
        cells.to_csv(d / "cells.csv", index=False)
        values["drift"].to_csv(d / "drift.csv", index=False)
        for ch in CHANNELS:
            values[f"spots_{ch}"].to_csv(d / f"spots_{ch}.csv", index=False)
        (d / "run.json").write_text(json.dumps({"path": str(path), "drift": values["drift_summary"],
                                                "roi": values["roi"], "ref_roi": values["ref_roi"]},
                                               indent=2, default=float), encoding="utf-8")
    return out


def _run_one(args):
    path, settings, roi, drift, save_dir, image = args
    out = run_image(path, settings, roi, drift, save_dir, image)
    return image, out


def run_images(decided, settings=None, roi=None, drift=None, save_dir=None, workers: int = 1, progress=None,
               cancel=None, log=None) -> dict:
    """run_image on every image to be analysed: {image id: result}."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    todo = decided[decided["use"]]
    jobs = [(p, settings, roi, drift, save_dir, i) for i, p in zip(todo["image"], todo["path"])]
    out, n = {}, len(jobs)
    if workers <= 1:
        for k, job in enumerate(jobs):
            if cancel is not None and cancel():
                break
            image, res = _run_one(job)
            out[image] = res
            if progress:
                progress((k + 1) / max(n, 1), f"{k + 1}/{n} images: {image}")
        return out
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for k, fut in enumerate(as_completed(futures)):
            try:
                image, res = fut.result()
                out[image] = res
            except Exception as exc:  # noqa: BLE001 -- one bad image must not stop the others
                if log:
                    log(f"an image failed: {exc}")
            if progress:
                progress((k + 1) / max(n, 1), f"{k + 1}/{n} images")
            if cancel is not None and cancel():
                for f in futures:
                    f.cancel()
                break
    return out


def load_runs(save_dir: "str | Path", images) -> dict:
    """Saved per-image results (the cells table and the run summary) of
    `images`, as far as they exist."""
    import pandas as pd

    out = {}
    for image in images:
        d = Path(save_dir) / image
        if (d / "cells.csv").exists() and (d / "run.json").exists():
            run = json.loads((d / "run.json").read_text(encoding="utf-8"))
            out[image] = {"cells": pd.read_csv(d / "cells.csv"), "drift": run["drift"], "roi": run["roi"],
                          "ref_roi": run["ref_roi"]}
    return out


# -- results ---------------------------------------------------------------------------------


def cell_results(decided, runs: dict, settings: "Settings | None" = None, overrides: "dict | None" = None,
                 identities: "dict | None" = None):
    """One row per cell of every analysed image: image, day, group, worm,
    cell (A, B, ... by depth), identity, the selection rule and your review
    (`included`), every measurement."""
    import pandas as pd

    from .pipeline import apply_exclusion, apply_identities
    from .series import name_first

    s = settings or Settings()
    meta = decided.set_index("image")
    tables = []
    for image, res in runs.items():
        if image not in meta.index or not bool(meta.at[image, "use"]):
            continue
        cells = res["cells"]
        if not len(cells):
            continue
        cells = name_first(cells.drop(columns=["cell"], errors="ignore"))
        cells = cells.assign(sample=image, timepoint=0, day=meta.at[image, "day"], date=meta.at[image, "date"],
                             group=meta.at[image, "group"], worm=meta.at[image, "worm"],
                             drift_um=meta.at[image, "drift_um"], registered=meta.at[image, "drift_decision"] != "ok")
        tables.append(cells)
    if not tables:
        return pd.DataFrame()
    cells = pd.concat(tables, ignore_index=True)
    cells = apply_exclusion(cells, s, overrides)
    cells = apply_identities(cells, identities)
    front = ["image", "day", "date", "group", "worm", "cell", "identity", "included", "excluded_by_rule", "override"]
    cells = cells[[c for c in front if c in cells] + [c for c in cells.columns if c not in front]]
    return cells.drop(columns=["sample", "timepoint"]).rename(columns={})


IMAGE_COLUMNS = ["image", "date", "day", "group", "worm", "file", "drift_um", "drift_decision"]


def transcripts_per_image(cells, decided):
    """The result: one row per analysed image (worm) -- date, group, worm,
    file, drift, `n_cells` (cells of interest) and per channel
    `<ch>_dots` (spot peaks) and `<ch>_transcripts` (Big-FISH estimate),
    summed over the cells of interest; once cells are named, per neuron too
    (`ASIL_ins6_dots`, `ASIL_ins6_transcripts`, ...). An image without a cell
    of interest has n_cells 0 and empty counts (not zeros)."""
    import pandas as pd

    base = decided[decided["use"]][IMAGE_COLUMNS].reset_index(drop=True)
    if not len(cells):
        return base.assign(n_cells=0)
    inc = cells[cells["included"]]
    count = {f"{ch}_spots": f"{ch}_dots" for ch in CHANNELS} | \
        {f"{ch}_transcripts_est": f"{ch}_transcripts" for ch in CHANNELS}
    cols = [c for c in count if c in inc]
    totals = inc.groupby("image")[cols].sum(min_count=1).rename(columns=count)
    out = base.merge(totals, left_on="image", right_index=True, how="left")
    out.insert(len(IMAGE_COLUMNS), "n_cells", out["image"].map(inc.groupby("image").size()).fillna(0).astype(int))
    # Per neuron: always every neuron's columns, in the same order (empty where
    # no cell of the image is named so), so the table's layout never changes.
    from .pipeline import IDENTITIES

    named = inc[inc["identity"].fillna("").astype(str) != ""] if "identity" in inc else inc.iloc[0:0]
    order = [f"{i}_{count[c]}" for i in IDENTITIES for c in cols]
    if len(named):
        wide = named.pivot_table(index="image", columns="identity", values=cols, aggfunc="sum")
        wide.columns = [f"{ident}_{count[c]}" for c, ident in wide.columns]
        out = out.merge(wide[[c for c in order if c in wide]], left_on="image", right_index=True, how="left")
    for c in order:
        if c not in out:
            out[c] = np.nan
    return out[[c for c in out.columns if c not in order] + order]


def rule_warnings(cells, settings: "Settings | None" = None) -> list:
    """The selection rule leaves out far more cells in one group than in the
    others -- e.g. ins-6 as the rule's channel in an ins-6 mutant."""
    s = settings or Settings()
    if not len(cells) or s.exclude_channel == "none":
        return []
    share = cells.groupby("group")["excluded_by_rule"].mean()
    out = []
    for g, frac in share.items():
        if frac - share.drop(g).median() > 0.3 if len(share) > 1 else False:
            out.append({"kind": "selection rule", "images": [],
                        "message": f"the rule ({GENE.get(s.exclude_channel, s.exclude_channel)} mean − background "
                                   f"< {s.exclude_below:g}) leaves out {100 * frac:.0f} % of the cells in '{g}' "
                                   f"(others: {100 * share.drop(g).median():.0f} %). If '{g}' lacks "
                                   f"{GENE.get(s.exclude_channel, s.exclude_channel)}, select on another channel "
                                   "or review its cells"})
    return out


# -- command line ---------------------------------------------------------------------------


def write_results(out: Path, decided, cells, warnings, settings_doc: dict) -> dict:
    """The CSVs: transcripts_per_image.csv (the result), cells.csv (one row
    per cell), images_qc.csv (every image: drift, duplicates, used or why
    not), warnings.json, settings.json. Returns {name: path}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"transcripts_per_image": out / "transcripts_per_image.csv", "cells": out / "cells.csv",
             "images_qc": out / "images_qc.csv", "warnings": out / "warnings.json", "settings": out / "settings.json"}
    transcripts_per_image(cells, decided).to_csv(paths["transcripts_per_image"], index=False)
    cells.to_csv(paths["cells"], index=False)
    qc = decided.drop(columns=["path"]).rename(columns={"use": "analysed"})
    qc.to_csv(paths["images_qc"], index=False)
    paths["warnings"].write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["settings"].write_text(json.dumps(settings_doc, indent=2, default=str), encoding="utf-8")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Group experiment: transcripts per image and channel")
    parser.add_argument("folder", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--groups", default="", help="Group names, e.g. 'N2,ins-6,ins-6rescue' (default: learnt)")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0: automatic)")
    parser.add_argument("--exclude-channel", default="ins6", choices=[*CHANNELS, "none"],
                        help="Selection rule's channel (cells without its signal are left out)")
    parser.add_argument("--check-only", action="store_true", help="Only the name and image checks")
    args = parser.parse_args(argv)
    from .pipeline import default_workers

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    gs = GroupSettings(groups=args.groups)
    settings = Settings(exclude_channel=args.exclude_channel)
    drift = DriftSettings()
    workers = args.workers or default_workers()
    table = scan(args.folder, gs)
    print(f"{len(table)} images, {table['day'].nunique()} days, groups: "
          f"{', '.join(f'{g} ({n})' for g, n in table['group'].value_counts().items())}", flush=True)
    checks = check_images(table, out / "qc_cache", drift.channel, workers)
    dups = find_duplicates(table, checks, gs)
    decided = decide(table, checks, dups, drift)
    warnings = naming_warnings(table) + check_warnings(decided, dups)
    for w in warnings:
        print(f"WARNING [{w['kind']}] {w['message']}", flush=True)
    print(f"{int(decided['use'].sum())} of {len(decided)} images to analyse", flush=True)
    doc = {"groups": asdict(gs), "settings": asdict(settings), "drift": asdict(drift), "roi": asdict(RoiSettings())}
    if args.check_only:
        decided.drop(columns=["path"]).to_csv(out / "images_qc.csv", index=False)
        return
    runs = run_images(decided, settings, RoiSettings(), drift, out / "images", workers,
                      progress=lambda f, m: print(m, flush=True))
    cells = cell_results(decided, runs, settings)
    warnings += rule_warnings(cells, settings)
    paths = write_results(out, decided, cells, warnings, doc)
    print(f"wrote {paths['transcripts_per_image']}", flush=True)


if __name__ == "__main__":
    main()
