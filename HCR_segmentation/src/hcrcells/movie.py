"""A z-sweep movie of one timepoint, to identify cells by eye.

    python -m hcrcells.movie "originals/... T=0.tif" out/zsweep_T0.mp4

Left: HCR composite (daf-28 magenta, ins-4 green, ins-6 cyan); right: DAPI.
Each segmented cell's outline is drawn on the slices it occupies, with its
letter (the same letters as the series table: T=0 cells ordered by depth).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pipeline import CHANNELS, run_timepoint
from .series import cell_names, head_crop

_COLORS = {"daf28": (1, 0, 1), "ins4": (0, 1, 0), "ins6": (0, 1, 1)}


def _scale(img: np.ndarray, crop) -> float:
    return float(np.percentile(img[(slice(None),) + crop], 99.9)) or 1.0


def write_movie(path: Path, out: Path, crop=None, fps: float = 3.0, dapi_channel: int = 3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from scipy import ndimage
    from skimage.segmentation import find_boundaries

    from vanua.blocks.io import read_volume

    _, values = run_timepoint(path)
    labels = values["cells"]
    crop = crop or head_crop(labels)
    table = values["cell_table"]
    names = cell_names(table)
    dapi, _ = read_volume(path, dapi_channel, log=lambda *_: None)
    spacing_z = float(np.linalg.norm(np.asarray(values["frame"]["affine"])[:3, 0]))
    hcr = {c: np.asarray(values[c], float) for c in CHANNELS}
    hi = {c: _scale(hcr[c], crop) for c in CHANNELS}
    dapi_hi = _scale(dapi, crop)
    centroids = dict(zip(table["label"].astype(int),
                         ndimage.center_of_mass(np.ones_like(labels), labels, table["label"].astype(int))))
    ext = (crop[1].start, crop[1].stop, crop[0].stop, crop[0].start)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), layout="constrained")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    def draw(z):
        for ax in axes:
            ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        rgb = sum(np.clip(hcr[c][z][crop] / hi[c], 0, 1)[..., None] * np.array(_COLORS[c]) for c in CHANNELS)
        axes[0].imshow(np.clip(rgb, 0, 1), extent=ext)
        axes[1].imshow(dapi[z][crop], cmap="gray", vmin=0, vmax=dapi_hi, extent=ext)
        plane = labels[z][crop]
        for lbl in np.unique(plane[plane > 0]):
            ey, ex = np.nonzero(find_boundaries(plane == lbl, mode="inner"))
            yy, xx = centroids[int(lbl)][1:]
            for ax in axes:
                ax.scatter(ex + crop[1].start, ey + crop[0].start, s=0.5, c="yellow")
                ax.text(xx + 20, yy, names[int(lbl)], color="yellow", fontsize=14, weight="bold")
        axes[0].set_title("HCR: daf-28 magenta, ins-4 green, ins-6 cyan", fontsize=10)
        axes[1].set_title("DAPI", fontsize=10)
        fig.suptitle(f"{path.name}   z = {z} / {labels.shape[0] - 1}   ({z * spacing_z:.1f} µm)", fontsize=11)

    anim = animation.FuncAnimation(fig, draw, frames=range(labels.shape[0]), interval=1000 / fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=80)
    else:
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=4000), dpi=100)
    plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args(argv)
    print(write_movie(args.path, args.out, fps=args.fps))


if __name__ == "__main__":
    main()
