"""Figures for README.md: the image after each processing step, on real data.

    pixi run python scripts/make_readme_figures.py [originals] [out]

Runs the app's own recipes (recipes/cells.toml, recipes/timepoint.toml) on
the first timepoint of sample 1 and sample 2 and draws the intermediate
images the recipes produce, plus a few charts (threshold histogram, elbow
curve, single-spot reference, tracking, fits) from the same values and from
the series results in out/<sample>/cells_per_timepoint.csv. Writes PNGs to
docs/images/.

The images are derived from the lab's unpublished data: keep them with the
data's own restrictions (do not publish or commit them without consent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))
OUT = APP_DIR / "docs" / "images"

MAGENTA, GREEN, CYAN = (1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0)


def _norm(img, lo_pct=0.5, hi_pct=99.8):
    img = np.asarray(img, float)
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    return np.clip((img - lo) / ((hi - lo) or 1.0), 0, 1)


def _composite(planes_colors):
    rgb = 0
    for plane, color in planes_colors:
        rgb = rgb + _norm(plane)[..., None] * np.asarray(color)
    return np.clip(rgb, 0, 1)


def _square(y0, y1, x0, x1, shape):
    """Widen the shorter side of a crop so the panel is square (within the image)."""
    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = int(min(max(cy - side // 2, 0), max(shape[0] - side, 0)))
    x0 = int(min(max(cx - side // 2, 0), max(shape[1] - side, 0)))
    return slice(y0, min(y0 + side, shape[0])), slice(x0, min(x0 + side, shape[1]))


def _crop_box(labels, margin=60):
    mip = np.asarray(labels).max(0) > 0
    ys, xs = np.nonzero(mip)
    return _square(max(ys.min() - margin, 0), min(ys.max() + margin, mip.shape[0]),
                   max(xs.min() - margin, 0), min(xs.max() + margin, mip.shape[1]), mip.shape)


def _outline(ax, mask, color="yellow", lw=0.8):
    ax.contour(np.asarray(mask, float), levels=[0.5], colors=[color], linewidths=lw)


def _panel(ax, img, title, cmap="gray", **kw):
    ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _scalebar(ax, um, px_per_um, label=None):
    h, w = ax.get_images()[0].get_array().shape[:2]
    x0, y0 = w * 0.05, h * 0.93
    ax.plot([x0, x0 + um * px_per_um], [y0, y0], color="white", lw=2.5, solid_capstyle="butt")
    ax.text(x0, y0 - h * 0.03, label or f"{um:g} µm", color="white", fontsize=7)


def run(name, path, extra=None):
    from hcrcells.pipeline import registry
    from vanua.core import recipe as R
    from vanua.core.runner import run as run_recipe

    r = R.load(APP_DIR / "recipes" / f"{name}.toml")
    over = {s.id: {"path": str(path)} for s in r.steps if s.op == "io.read_image"}
    for k, v in (extra or {}).items():
        over.setdefault(k, {}).update(v)
    return run_recipe(R.with_params(r, over), registry=registry())


def main(argv=None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import ndimage
    from skimage.measure import label as cc_label

    from hcrcells.ops import elbow_threshold, format_targets
    from hcrcells.plots import PlotOptions, make_figure
    from hcrcells.series import cell_names, sample_series, sample_id

    argv = list(sys.argv[1:] if argv is None else argv)
    folder = Path(argv[0]) if argv else APP_DIR / "originals"
    results_dir = Path(argv[1]) if len(argv) > 1 else APP_DIR / "out"
    OUT.mkdir(parents=True, exist_ok=True)
    series = {sample_id(k): v for k, v in sample_series(folder).items()}
    s1_path, s2_path = series["S1"][0][1], series["S2"][0][1]
    saved = []

    def save(fig, name):
        fig.savefig(OUT / name, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(name)
        print(OUT / name, flush=True)

    # -- sample 1, first timepoint: cells --------------------------------------------------
    v = run("cells", s1_path)
    frame = v["frame"]
    spacing = np.linalg.norm(np.asarray(frame["affine"])[:3, :3], axis=0)  # z, y, x µm per voxel
    px_um = 1.0 / spacing[1]
    cells, table = v["cells"], v["cell_table"]
    names = cell_names(table)
    crop = _crop_box(cells)
    big = table.sort_values("volume_um3").iloc[-1]
    z = int(round(big["centroid_z_um"] / spacing[0]))
    sl = (z,) + crop

    # 1. the four channels
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, key, title, color in zip(axes, ("daf28", "ins4", "ins6"), ("C0 daf-28 (HCR)", "C1 ins-4 (HCR)", "C2 ins-6 (HCR)"),
                                     (MAGENTA, GREEN, CYAN)):
        _panel(ax, _norm(v[key][sl])[..., None] * np.asarray(color), title)
    dapi = run_dapi = None
    from vanua.blocks.io import read_volume

    dapi, _ = read_volume(s1_path, 3, log=lambda *_: None)
    _panel(axes[3], _norm(dapi[sl]), "C3 DAPI (nuclei)")
    _panel(axes[4], _composite([(v["daf28"][sl], MAGENTA), (v["ins4"][sl], GREEN), (v["ins6"][sl], CYAN)]),
           "composite (C0–C2)")
    _scalebar(axes[4], 10, px_um)
    fig.suptitle(f"One slice (z = {z}, {z * spacing[0]:.1f} µm deep) of the head, sample 1, first timepoint", fontsize=10)
    save(fig, "01_channels.png")

    # 2. segmentation steps
    background = ndimage.gaussian_filter(np.asarray(v["daf28"], np.float32), 4.0 / spacing)
    steps = [
        (_norm(v["daf28"][sl]), "1  daf-28, as acquired", "gray"),
        (_norm(background[sl]), "2  background estimate\n(very strong blur, 4 µm)", "gray"),
        (_norm(v["daf28_bgsub"][sl]), "3  background removed\n(1 minus 2)", "gray"),
        (_norm(v["daf28_smooth"][sl]), "4  smoothed\n(gentle blur, 0.4 µm)", "gray"),
        (v["cell_mask"][sl].astype(float), "5  threshold (Otsu):\nbright = 'cell'", "gray"),
        (v["cell_mask_clean"][sl].astype(float), "6  cleaned: thin necks cut,\nholes filled", "gray"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.2), layout="constrained")
    axes = axes.ravel()
    for ax, (img, title, cmap) in zip(axes, steps):
        _panel(ax, img, title, cmap=cmap)
    lab = v["cells_all"][sl]
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(3)
    colors = np.vstack([[0, 0, 0], rng.uniform(0.35, 1.0, (int(lab.max()) + 1, 3))])
    _panel(axes[6], colors[lab], "7  split into objects\n(watershed), one colour each")
    keep = v["cells"][sl]
    axes[7].imshow(_norm(v["daf28"][sl]), cmap="gray")
    for label in np.unique(keep[keep > 0]):
        _outline(axes[7], keep == label, "yellow", 1.0)
        yy, xx = ndimage.center_of_mass(keep == label)
        axes[7].text(xx, yy, names[int(label)], color="white", fontsize=10, ha="center", va="center", weight="bold",
                     bbox={"boxstyle": "round,pad=0.2", "fc": (0, 0, 0, 0.5), "ec": "none"})
    axes[7].set_title("8  small objects dropped;\ncells named A, B, … by depth", fontsize=9)
    axes[7].set_xticks([]); axes[7].set_yticks([])
    _scalebar(axes[0], 10, px_um)
    fig.suptitle("Finding the cells on the daf-28 channel (same slice as above)", fontsize=10)
    save(fig, "02_segmentation_steps.png")

    # 3. Otsu histogram
    smooth = np.asarray(v["daf28_smooth"])
    from skimage.filters import threshold_otsu

    thr = threshold_otsu(smooth)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_box_aspect(1)
    sample = smooth[:, ::4, ::4].ravel()
    ax.hist(sample, bins=200, color="0.4", log=True)
    ax.axvline(thr, color="crimson", lw=2)
    ax.text(thr, ax.get_ylim()[1] * 0.3, f"  Otsu threshold = {thr:.1f}", color="crimson", fontsize=9)
    ax.set_xlabel("smoothed, background-removed daf-28 intensity (a.u.)")
    ax.set_ylabel("number of voxels (log)")
    ax.set_title("Otsu picks the cut that best separates the two groups of voxels:\n"
                 "dim background (left, the vast majority) and bright cell signal (right)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "03_otsu_histogram.png")

    # -- sample 2: the watershed split of two touching cells ------------------------------------
    w = run("cells", s2_path)
    crop2 = _crop_box(w["cells"], margin=40)
    raw_mask = np.asarray(w["cell_mask"])            # straight from the threshold
    objects = cc_label(raw_mask)                    # connected pieces, in 3D
    split = np.asarray(w["cells_all"])              # after the watershed split
    # the threshold object that the split divided in two (or more)
    pieces = {int(o): len(np.unique(split[(objects == o) & (split > 0)])) for o in np.unique(objects[split > 0]) if o}
    merged_id = max(pieces, key=pieces.get)
    region = objects == merged_id
    ys, xs = np.nonzero(region.max(0))
    crop2 = _square(max(ys.min() - 30, 0), ys.max() + 30, max(xs.min() - 30, 0), xs.max() + 30, region.shape[1:])
    dist = ndimage.distance_transform_edt(np.asarray(w["cell_mask_clean"]) & region, sampling=spacing)
    palette = np.vstack([[0, 0, 0], rng.uniform(0.35, 1, (int(max(objects.max(), split.max())) + 1, 3))])
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout="constrained")
    _panel(axes[0], _norm(np.asarray(w["daf28"]).max(0)[crop2]), "daf-28 (all slices projected):\ntwo cells touching")
    _panel(axes[1], palette[np.where(region, objects, 0).max(0)[crop2]],
           f"threshold alone: ONE object\n(one colour; {pieces[merged_id]} cells inside)")
    _panel(axes[2], dist.max(0)[crop2], "distance to the object's edge:\neach cell has its own peak", cmap="magma")
    pieces_mip = np.where(region, split, 0).max(0)[crop2]
    relabel = np.zeros(int(pieces_mip.max()) + 1, int)
    for i, lab_id in enumerate(np.unique(pieces_mip[pieces_mip > 0]), start=1):
        relabel[lab_id] = i
    two_colours = np.array([[0, 0, 0], [0.95, 0.45, 0.75], [0.55, 0.45, 0.95], [0.4, 0.85, 0.6], [1.0, 0.7, 0.3]])
    _panel(axes[3], two_colours[np.minimum(relabel[pieces_mip], 4)],
           f"watershed from the peaks:\n{pieces[merged_id]} cells (one colour each)")
    fig.suptitle("Splitting touching cells (sample 2, first timepoint; the seeds must be at least 3 µm apart)",
                 fontsize=10)
    save(fig, "04_watershed_split.png")

    # -- cells of interest, ROI and reference region, spots (sample 1) --------------------------
    sel = table[table["ins6_mean_bgsub_au"] >= 3.0]
    targets = format_targets(sel[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy())
    tp = run("timepoint", s1_path, {"select": {"targets": targets}})
    roi, ref = tp["roi"], tp["ref_roi"]
    mip = lambda a: np.asarray(a).max(0)  # noqa: E731
    shape2 = np.asarray(v["cells"]).shape[1:]
    big_crop = _square(max(ref["y"][0] - 30, 0), ref["y"][1] + 30, max(ref["x"][0] - 30, 0), ref["x"][1] + 30, shape2)
    zoom_crop = _square(max(roi["y"][0] - 10, 0), roi["y"][1] + 10, max(roi["x"][0] - 10, 0), roi["x"][1] + 10, shape2)
    included = set(sel["label"].astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 0.55]})
    for ax, crop_, show_names in ((axes[0], big_crop, False), (axes[1], zoom_crop, True)):
        ax.imshow(_composite([(mip(v["daf28"])[crop_], MAGENTA), (mip(v["ins4"])[crop_], GREEN),
                              (mip(v["ins6"])[crop_], CYAN)]))
        y0, x0 = crop_[0].start, crop_[1].start
        placed = []  # badge positions, so two cells at different depths do not stack their names
        for label, name in names.items():
            m = mip(v["cells"] == label)[crop_]
            keep_it = label in included
            _outline(ax, m, "yellow" if keep_it else "0.65", 1.1)
            if show_names:
                yy, xx = ndimage.center_of_mass(m)
                while any(abs(xx - px) < 16 and abs(yy - py) < 14 for px, py in placed):
                    xx += 18
                placed.append((xx, yy))
                ax.text(xx, yy, name, color="black" if keep_it else "white", fontsize=10, weight="bold",
                        ha="center", va="center",
                        bbox={"boxstyle": "round,pad=0.25", "fc": (1, 0.9, 0.2) if keep_it else (0.35, 0.35, 0.4),
                              "ec": "none"})
        for box, color, text in ((roi, (1, 0.77, 0), "ROI"), (ref, (0.75, 0.75, 0.8), "reference region")):
            if crop_ is zoom_crop and box is ref:
                continue  # the reference region is larger than the zoom
            ax.add_patch(plt.Rectangle((box["x"][0] - x0, box["y"][0] - y0), box["x"][1] - box["x"][0],
                                       box["y"][1] - box["y"][0], fill=False, ls="--", lw=1.5, ec=color))
            ax.text(box["x"][0] - x0 + 4, box["y"][0] - y0 + 12, text, color=color, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        _scalebar(ax, 10 if crop_ is big_crop else 5, px_um)
    axes[0].set_title("Overview: the ROI (amber) and the larger\nreference region (grey), all slices projected", fontsize=9)
    axes[1].set_title("Zoom on the ROI: yellow = cells of interest,\ngrey = left out by the rule", fontsize=9)
    rows = [[n, f"{float(table.loc[table['label'] == l, 'ins6_mean_bgsub_au'].iloc[0]):.1f}",
             "yes" if l in included else "no (below 3)"] for l, n in sorted(names.items(), key=lambda kv: kv[1])]
    axes[2].axis("off")
    tab = axes[2].table(cellText=rows, colLabels=["cell", "ins-6 mean −\nbackground (a.u.)", "of interest?"],
                        loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.8)
    axes[2].set_title("Default rule: cells with ins-6\nbelow 3 a.u. are left out (editable)", fontsize=9)
    save(fig, "05_selection_roi.png")

    # spot detection steps on ins-4 in the reference region
    from hcrcells.ops import detect_spots

    ref_img = np.asarray(tp["roi_ins4"], np.float32)
    rframe = tp["roi_frame"]
    s1v = np.array([0.35, 0.12, 0.12]) / spacing
    dog = ndimage.gaussian_filter(ref_img, s1v) - ndimage.gaussian_filter(ref_img, 3 * s1v)
    spots, threshold, noise = detect_spots(ref_img, rframe)
    # A 200 x 200 px window centred on the ROI (where the cells and most spots are),
    # on the slice with the most final spots inside it.
    cy = (roi["y"][0] + roi["y"][1]) // 2 - ref["y"][0]
    cx = (roi["x"][0] + roi["x"][1]) // 2 - ref["x"][0]
    zoom = (slice(max(cy - 100, 0), max(cy - 100, 0) + 200), slice(max(cx - 100, 0), max(cx - 100, 0) + 200))
    inwin = spots[spots["y_index"].between(zoom[0].start, zoom[0].stop - 1)
                  & spots["x_index"].between(zoom[1].start, zoom[1].stop - 1)]
    zc = int(inwin["z_index"].value_counts().idxmax()) if len(inwin) else ref_img.shape[0] // 2
    dog_hi = float(np.percentile(dog[zc][zoom], 99.7))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), layout="constrained")
    _panel(axes[0], _norm(ref_img[zc][zoom]), "1  ins-4, next to the cells", cmap="gray")
    _panel(axes[1], dog[zc][zoom], "2  spot filter (difference of\nGaussians): spot-sized blobs pop out",
           cmap="magma", vmin=0, vmax=dog_hi)
    from skimage.feature import peak_local_max

    cand = peak_local_max(dog[zc], min_distance=2, threshold_abs=float(np.median(dog)) + 3 * noise)
    axes[2].imshow(dog[zc][zoom], cmap="magma", vmin=0, vmax=dog_hi)
    inzoom = cand[(cand[:, 0] >= zoom[0].start) & (cand[:, 0] < zoom[0].stop)
                  & (cand[:, 1] >= zoom[1].start) & (cand[:, 1] < zoom[1].stop)]
    axes[2].scatter(inzoom[:, 1] - zoom[1].start, inzoom[:, 0] - zoom[0].start, s=10, facecolors="none",
                    edgecolors="white", lw=0.6)
    axes[2].set_title(f"3  every local peak above 3 × noise:\ncandidates ({len(inzoom)} in this window)", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    near = inwin[abs(inwin["z_index"] - zc) <= 1]
    axes[3].imshow(_norm(ref_img[zc][zoom]), cmap="gray")
    axes[3].scatter(near["x_index"] - zoom[1].start, near["y_index"] - zoom[0].start, s=30, facecolors="none",
                    edgecolors=(1, 0, 1), lw=1.1)
    axes[3].set_title(f"4  kept: above the elbow threshold\n(final spots: {len(near)} on this slice ±1)", fontsize=9)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    _scalebar(axes[0], 5, px_um)
    fig.suptitle("Spot detection (ins-4, one slice of the reference region, sample 1 first timepoint)", fontsize=10)
    save(fig, "06_spot_steps.png")

    # elbow curve
    vals = dog[tuple(peak_local_max(dog, footprint=np.ones((3, 5, 5), bool),
                                    threshold_abs=float(np.median(dog)) + 3 * noise, exclude_border=False).T)]
    ts = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99.99), 200)
    counts = np.array([(vals > t).sum() for t in ts])
    elbow = elbow_threshold(vals)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_box_aspect(1)
    ax.semilogy(ts, np.maximum(counts, 1), color="0.3")
    ax.axvline(elbow, color="crimson", lw=2)
    ax.text(elbow, counts.max() / 3, f"  elbow = {elbow:.2f}", color="crimson", fontsize=9)
    ax.set_xlabel("threshold on the spot filter")
    ax.set_ylabel("candidates above it (log)")
    ax.set_title("The automatic threshold: noise makes the steep left part (many weak peaks);\n"
                 "real spots make the flat tail. The elbow between them is the cut.", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "07_elbow.png")

    # background shell
    cells_ref = np.asarray(tp["roi_cells"])
    cell_lab = int(sel.iloc[0]["label"])
    zz = int(round(float(sel.iloc[0]["centroid_z_um"]) / spacing[0])) - ref["z"][0]
    dist_c, near_idx = ndimage.distance_transform_edt(cells_ref == 0, sampling=spacing, return_indices=True)
    owner = cells_ref[tuple(near_idx)]
    shell = (dist_c > 0.5) & (dist_c <= 2.0) & (owner == cell_lab)
    ys, xs = np.nonzero((cells_ref[zz] == cell_lab) | shell[zz])
    win = (slice(max(ys.min() - 15, 0), ys.max() + 15), slice(max(xs.min() - 15, 0), xs.max() + 15))
    ins6_ref = np.asarray(tp["roi_ins6"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), layout="constrained")
    _panel(axes[0], _norm(ins6_ref[zz][win])[..., None] * np.asarray(CYAN), "ins-6 around one cell")
    axes[1].imshow(_norm(ins6_ref[zz][win]), cmap="gray")
    over = np.zeros(cells_ref[zz][win].shape + (4,))
    over[cells_ref[zz][win] == cell_lab] = (1, 0.85, 0, 0.45)
    over[shell[zz][win]] = (0.2, 0.6, 1, 0.55)
    axes[1].imshow(over)
    axes[1].set_title("yellow: the cell (summed);  blue: its background\nshell, 0.5–2 µm outside (median)", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.suptitle("Per-cell intensity: signal inside the cell minus the local background around it", fontsize=10)
    save(fig, "08_background_shell.png")

    # single-spot reference
    from hcrcells.ops import reference_spot

    ref_value, n_ref = reference_spot(ref_img, spots, cells_ref, rframe)
    z_i, y_i, x_i = (spots[c].to_numpy(int) for c in ("z_index", "y_index", "x_index"))
    isolated = dist_c[z_i, y_i, x_i] > 3.0
    bg = ndimage.gaussian_filter(ref_img, 2.0 / spacing)
    sums = [float(ref_img[max(z_i[i] - 1, 0):z_i[i] + 2, max(y_i[i] - 2, 0):y_i[i] + 3, max(x_i[i] - 2, 0):x_i[i] + 3].sum()
                  - bg[z_i[i], y_i[i], x_i[i]] * ref_img[max(z_i[i] - 1, 0):z_i[i] + 2, max(y_i[i] - 2, 0):y_i[i] + 3,
                                                       max(x_i[i] - 2, 0):x_i[i] + 3].size)
            for i in np.nonzero(isolated)[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    axes[0].imshow(_norm(mip(ref_img)), cmap="gray")
    _outline(axes[0], mip(cells_ref) > 0, "yellow", 0.9)
    axes[0].scatter(x_i[isolated], y_i[isolated], s=14, facecolors="none", edgecolors=(0.2, 1, 0.2), lw=0.8)
    axes[0].scatter(x_i[~isolated], y_i[~isolated], s=4, c="0.6")
    axes[0].set_title(f"ins-4 spots in the reference region: green = isolated\n(more than 3 µm from any cell, {int(isolated.sum())} spots)",
                      fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].set_box_aspect(1)
    axes[1].hist(sums, bins=40, color="0.5")
    axes[1].axvline(ref_value, color="crimson", lw=2)
    axes[1].text(ref_value, axes[1].get_ylim()[1] * 0.85, f"  median = {ref_value:.0f} a.u.\n  = one transcript", color="crimson", fontsize=9)
    axes[1].set_xlabel("signal of one isolated spot (a.u., background removed)")
    axes[1].set_ylabel("number of spots")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("The single-spot reference: transcripts in a cell ≈ cell signal ÷ one spot's signal", fontsize=10)
    save(fig, "09_reference_spot.png")

    # -- across timepoints (series results) ---------------------------------------------------------
    csv = results_dir / "S1" / "cells_per_timepoint.csv"
    if csv.is_file():
        c = pd.read_csv(csv)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), layout="constrained")
        for name, g in c.groupby("cell"):
            g = g.sort_values("timepoint")
            axes[0].plot(g["centroid_x_um"], g["centroid_y_um"], "-o", ms=2.5, label=name)
            axes[0].annotate("T0", (g["centroid_x_um"].iloc[0], g["centroid_y_um"].iloc[0]), fontsize=7)
            start = g[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].iloc[0].to_numpy()
            drift = np.linalg.norm(g[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy() - start, axis=1)
            axes[1].plot(g["timepoint"], drift, "-o", ms=2.5, label=name)
        axes[0].set_xlabel("x (µm)"); axes[0].set_ylabel("y (µm)"); axes[0].invert_yaxis()
        axes[0].set_title("Each cell's centre over the 40 repeats (top view)", fontsize=9)
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].set_box_aspect(1)
        axes[1].set_box_aspect(1)
        axes[1].set_xlabel("timepoint (repeat)"); axes[1].set_ylabel("distance from its T0 position (µm)")
        axes[1].axhline(3.0, color="crimson", ls="--", lw=1)
        axes[1].text(0, 3.05, "matching limit between two timepoints (3 µm)", color="crimson", fontsize=8)
        axes[1].set_title("Drift: small steps from one timepoint to the next", fontsize=9)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(title="cell", fontsize=8)
        fig.suptitle("Tracking the cells of interest across timepoints (sample 1)", fontsize=10)
        save(fig, "10_tracking.png")
        fig = make_figure(c, PlotOptions(kind="mean", fit="auto"), "Sample 1: mean of the 4 cells, fitted")
        fig.savefig(OUT / "11_fit.png", dpi=110, bbox_inches="tight", facecolor="white")
        saved.append("11_fit.png")
        print(OUT / "11_fit.png")
    print(f"{len(saved)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
