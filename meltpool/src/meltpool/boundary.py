"""Liquid-gas boundary tracing, Qt-free: the liquid-gas boundary of a weld pool (the dark
phase-contrast rim of the vapour depression), frame by frame.

The rim is a dark band a few pixels thick. Around the depression centre on the
surface it is traced as a smooth path r(theta) through the band's centre (dynamic
programming on a dark-ridge filter -- meltpool.movie.polar_trace). Along each ray
the band's inner edge (gas side) and outer edge (liquid side) are where the ridge
response falls to `edge_fraction` of its value on the centre line. Each line is
measured below the original surface: depth, width, area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import movie as mv

LINES = ("inner", "centre", "outer")
N_THETA = 181


@dataclass
class TraceSettings:
    r_min: float = 15.0
    r_max: float = 150.0
    edge_fraction: float = 0.5
    sigmas: "tuple[float, ...]" = (1.0, 2.0, 3.0)


def thetas() -> np.ndarray:
    return np.linspace(0.05, np.pi - 0.05, N_THETA)


def trace_frame(image: np.ndarray, surface: float, cx: float, s: TraceSettings) -> dict:
    """The three lines on one (averaged) frame, as radii on the common angle grid
    (NaN where a ray is inside the surface band or leaves the image)."""
    from scipy import ndimage as ndi

    ridge = mv.dark_ridges(image, s.sigmas)
    blind = max(0, int(np.ceil(surface)) + 4)
    ridge[:blind] = 0.0
    x, y, _ = mv.polar_trace(ridge, cx, surface, s.r_min, s.r_max)
    th = thetas()
    r_c = np.hypot(x - cx, y - surface)
    # Edges from the intensity across the band, along every ray: the band is darker than
    # the gas inside and the liquid outside; each edge is where the intensity is
    # `edge_fraction` of the way from the band minimum back up to that side's plateau.
    dr = 0.25
    rs = np.arange(0.0, s.r_max + 25.0, dr)
    yy = surface + np.outer(np.sin(th), rs)
    xx = cx + np.outer(np.cos(th), rs)
    smooth = ndi.gaussian_filter(np.asarray(image, dtype=np.float64), 0.7)
    prof = ndi.map_coordinates(smooth, [yy, xx], order=1, mode="nearest")
    r_in = np.full(N_THETA, np.nan)
    r_out = np.full(N_THETA, np.nan)
    reach = int(round(15.0 / dr))  # plateaus looked for within 15 px of the band centre
    for k in range(N_THETA):
        j = int(round(r_c[k] / dr))
        if j <= reach // 3 or j >= len(rs) - 2:
            continue
        lo_j = max(0, j - 8)
        jm = lo_j + int(np.argmin(prof[k, lo_j: j + 9]))  # band minimum near the centre line
        vmin = prof[k, jm]
        inner_side = prof[k, max(0, jm - reach): jm + 1]
        outer_side = prof[k, jm: jm + reach + 1]
        lvl_in = vmin + s.edge_fraction * (inner_side.max() - vmin)
        lvl_out = vmin + s.edge_fraction * (outer_side.max() - vmin)
        up_in = np.flatnonzero(prof[k, :jm + 1][::-1] >= lvl_in)
        up_out = np.flatnonzero(prof[k, jm:] >= lvl_out)
        if len(up_in) and up_in[0] > 0:
            i0 = jm - up_in[0]  # first sample back at the level, going inward
            a, b = prof[k, i0], prof[k, i0 + 1]
            r_in[k] = rs[i0] + (a - lvl_in) / (a - b) * dr if a != b else rs[i0]
        if len(up_out) and up_out[0] > 0:
            o0 = jm + up_out[0]
            a, b = prof[k, o0 - 1], prof[k, o0]
            r_out[k] = rs[o0 - 1] + (lvl_out - a) / (b - a) * dr if a != b else rs[o0]
    r_in = ndi.median_filter(np.where(np.isfinite(r_in), r_in, r_c), 5, mode="nearest")
    r_out = ndi.median_filter(np.where(np.isfinite(r_out), r_out, r_c), 5, mode="nearest")
    seen = (surface + np.sin(th) * r_c) >= blind
    r_in, r_out = np.minimum(r_in, r_c), np.maximum(r_out, r_c)
    # the band centre is midway between its edges (the traced path can ride one edge of a band
    # wider than the ridge filter's scales)
    radii = {"inner": r_in, "centre": 0.5 * (r_in + r_out), "outer": r_out}
    for line in LINES:
        radii[line] = np.where(seen, radii[line], np.nan)
    return {"cx": float(cx), "surface": float(surface), "r": radii}


def line_xy(trace: dict, line: str) -> "tuple[np.ndarray, np.ndarray]":
    th, r = thetas(), trace["r"][line]
    ok = np.isfinite(r)
    return trace["cx"] + np.cos(th[ok]) * r[ok], trace["surface"] + np.sin(th[ok]) * r[ok]


def measure(trace: dict, shape, pixel: "tuple[float, float]", pixel_area: float) -> dict:
    """Depth below the surface, width (x extent) and area below the surface of each line."""
    py, px = pixel
    out = {}
    for line in LINES:
        x, y = line_xy(trace, line)
        if len(x) < 3:
            out.update({f"depth_{line}": np.nan, f"width_{line}": np.nan, f"area_{line}": np.nan})
            continue
        mask = mv.region_below(shape, x, y, trace["surface"])
        out[f"depth_{line}"] = float((y.max() - trace["surface"]) * py)
        out[f"width_{line}"] = float((x.max() - x.min()) * px)
        out[f"area_{line}"] = float(mask.sum() * pixel_area)
    return out


def recentre(image, surface, cx, s: TraceSettings, iterations: int = 2) -> float:
    """Move the centre to the midpoint of the traced rim's ends (bowl symmetry)."""
    for _ in range(iterations):
        x, _y = mv.rim_trace(image, surface, cx, r_min=s.r_min, r_max=s.r_max, sigmas=s.sigmas)
        cx = float((x.min() + x.max()) / 2)
    return cx


# -- dynamics -------------------------------------------------------------------------------

def spectrum(t_ms: np.ndarray, values: np.ndarray, detrend: bool = True) -> "tuple[np.ndarray, np.ndarray]":
    """One-sided amplitude spectrum (same unit as `values`) of an evenly sampled series;
    NaNs filled by interpolation, linear trend removed, Hann window (amplitude-corrected)."""
    t = np.asarray(t_ms, float) / 1000.0
    v = np.asarray(values, float).copy()
    ok = np.isfinite(v)
    if ok.sum() < 8:
        return np.zeros(0), np.zeros(0)
    v[~ok] = np.interp(t[~ok], t[ok], v[ok])
    if detrend:
        v -= np.polyval(np.polyfit(t, v, 1), t)
    else:
        v -= v.mean()
    dt = float(np.median(np.diff(t)))
    w = np.hanning(len(v))
    amp = np.abs(np.fft.rfft(v * w)) * 2.0 / w.sum()
    return np.fft.rfftfreq(len(v), dt), amp


def dominant(freq: np.ndarray, amp: np.ndarray, n: int = 3) -> "list[tuple[float, float]]":
    """The n strongest local maxima above 0 Hz, sub-bin refined (parabolic)."""
    if len(amp) < 4:
        return []
    peaks = [k for k in range(1, len(amp) - 1) if amp[k] >= amp[k - 1] and amp[k] > amp[k + 1]]
    peaks = sorted(peaks, key=lambda k: amp[k], reverse=True)[:n]
    out = []
    df = freq[1] - freq[0]
    for k in peaks:
        a, b, c = amp[k - 1], amp[k], amp[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
        out.append((float(freq[k] + d * df), float(b)))
    return out


# -- overlay movie ---------------------------------------------------------------------------

COLOURS = {"inner": (34, 211, 238), "centre": (245, 158, 11), "outer": (217, 70, 239)}


def overlay_frame(image: np.ndarray, trace: "dict | None", window: "tuple[float, float]",
                  lines=LINES, label: str = "", gamma: float = 1.0) -> np.ndarray:
    """RGB uint8 frame with the chosen lines drawn (inner cyan, centre orange, outer
    magenta) and an optional text label (e.g. the time) in the top-left corner."""
    from skimage.draw import line as draw_line

    lo, hi = window
    g = np.clip((np.asarray(image, float) - lo) / max(hi - lo, 1e-9), 0, 1)
    if gamma != 1.0:
        g = g ** (1.0 / gamma)
    rgb = (np.repeat(g[..., None], 3, axis=2) * 255).astype(np.uint8)
    if label:
        rgb = _stamp(rgb, label)
    if trace is None:
        return rgb
    h, w = g.shape
    for line in lines:
        x, y = line_xy(trace, line)
        for k in range(len(x) - 1):
            rr, cc = draw_line(int(round(y[k])), int(round(x[k])), int(round(y[k + 1])), int(round(x[k + 1])))
            # 2 px thick: video stores colour at half resolution (4:2:0), a 1 px line loses its colour
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                r2, c2 = rr + dy, cc + dx
                ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
                rgb[r2[ok], c2[ok]] = COLOURS[line]
    return rgb


def _stamp(rgb: np.ndarray, text: str) -> np.ndarray:
    """White text on a dark box, top left (Pillow's built-in font, scaled to the frame)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    size = max(10, rgb.shape[0] // 22)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: no size argument
        font = ImageFont.load_default()
    x0, y0, x1, y1 = draw.textbbox((4, 3), text, font=font)
    draw.rectangle((x0 - 3, y0 - 2, x1 + 3, y1 + 2), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return np.array(img)  # a writable copy: the lines are drawn onto it next


def write_overlay(frames_rgb, path, fps: float = 20.0) -> "str":
    """GIF (Pillow) or, for .mp4, H.264/MPEG-4 via PyAV."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = list(frames_rgb)
    if path.suffix.lower() == ".gif":
        from PIL import Image

        images = [Image.fromarray(f) for f in frames_rgb]
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000 / fps)), loop=0)
        return str(path)
    import av

    h, w = frames_rgb[0].shape[:2]
    with av.open(str(path), "w") as container:
        stream = None
        for codec in ("libx264", "h264", "mpeg4"):
            try:
                stream = container.add_stream(codec, rate=int(round(fps)))
                break
            except Exception:  # noqa: BLE001 -- encoder not in this FFmpeg build
                continue
        stream.width, stream.height = w - w % 2, h - h % 2
        stream.pix_fmt = "yuv420p"
        for f in frames_rgb:
            f = np.ascontiguousarray(f[: stream.height, : stream.width])
            # convert RGB to the encoder's YUV explicitly: handing it RGB bytes tints every frame
            vf = av.VideoFrame.from_ndarray(f, format="rgb24").reformat(format="yuv420p")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)
