"""Every reader returns the source frames exactly (lossless files), the reader is
chosen by content not extension, and the numbers do not depend on the container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import PIXEL_UM, weld_settings, weld_spec

from meltpool import movie as mv
from meltpool.synthetic import make_movie, to_uint8, write_hdf5, write_movie, write_video


@pytest.fixture(scope="module")
def spec():
    return weld_spec(n=300)


@pytest.fixture(scope="module")
def stack(spec):
    return make_movie(spec)  # stored upside down, like the EHLA movies


@pytest.fixture(scope="module")
def files(tmp_path_factory, stack):
    d = tmp_path_factory.mktemp("formats")
    np.save(d / "movie.npy", stack)
    import tifffile

    tifffile.imwrite(d / "mislabelled.avi", to_uint8(stack), imagej=True, resolution=(1 / PIXEL_UM, 1 / PIXEL_UM),
                     metadata={"axes": "TYX", "unit": "um", "fps": 10.0})
    return {
        "tiff": write_movie(d / "movie.tif", stack),
        "hdf5": write_hdf5(d / "movie.h5", stack),
        "npy": d / "movie.npy",
        "mkv16": write_video(d / "movie16.mkv", stack, codec="ffv1"),
        "avi8": write_video(d / "movie8.avi", to_uint8(stack)),
        "imagej_as_avi": d / "mislabelled.avi",
    }


@pytest.mark.parametrize("key,kind", [("tiff", "tiff"), ("hdf5", "hdf5"), ("npy", "npy"), ("mkv16", "video"),
                                      ("avi8", "video"), ("imagej_as_avi", "tiff")])
def test_reader_by_content(files, key, kind):
    assert mv.sniff(files[key]) == kind


@pytest.mark.parametrize("key", ["tiff", "hdf5", "npy", "mkv16", "avi8", "imagej_as_avi"])
def test_frames_exact(files, stack, key):
    blob = mv.describe_movie(files[key], flip_vertical=True, log=lambda m: None)
    movie = mv.Movie(blob)
    source = to_uint8(stack) if key in ("avi8", "imagej_as_avi") else stack
    assert movie.n == len(stack) and movie.shape == stack.shape[1:]
    for i in (0, 1, 150, 299, 7, 298, 120):  # forwards, then jumps back and forth (video seeking)
        assert np.array_equal(movie.raw(i), source[i][::-1]), (key, i)
    movie.close()


def test_hdf5_picks_largest_dataset_or_the_named_one(files):
    log: list[str] = []
    blob = mv.describe_movie(files["hdf5"], log=log.append)
    assert blob["dataset"] == "entry/data/data" and any("largest" in m for m in log)
    assert mv.describe_movie(files["hdf5"], dataset="entry/instrument/dark")["n"] == 2
    with pytest.raises(ValueError):
        mv.describe_movie(files["hdf5"], dataset="nope")


def test_imagej_pixel_and_playback_rate(files):
    log: list[str] = []
    blob = mv.describe_movie(files["imagej_as_avi"], log=log.append)
    assert blob["file_meta"]["pixel_um"] == pytest.approx(PIXEL_UM, rel=1e-4)
    assert blob["file_meta"]["fps_in_file"] == 10.0
    assert any("content is TIFF" in m for m in log)


def _flipped_settings(stride: int = 4):
    s = weld_settings(stride=stride)
    s.calibration.flip_vertical = True  # stored upside down
    return s


@pytest.mark.parametrize("key", ["hdf5", "npy", "mkv16"])
def test_lossless_containers_give_identical_numbers(files, key):
    from meltpool.pipeline import run_movie

    _, tif = run_movie(files["tiff"], _flipped_settings(), files["tiff"].parent / "t.png", overlay=False)
    _, other = run_movie(files[key], _flipped_settings(), files[key].parent / f"{key}.png", overlay=False)
    pd.testing.assert_frame_equal(tif["table"], other["table"])


def test_8bit_video_depth(files, spec):
    """An 8-bit export loses grey levels; the steady-state depth of the band centre is still within 1.5 px."""
    from meltpool.pipeline import run_movie

    _, values = run_movie(files["avi8"], _flipped_settings(), files["avi8"].parent / "a8.png", overlay=False)
    t = values["table"]
    depth = t.loc[~t.flagged, "depth_centre_um"].median() / PIXEL_UM
    assert depth == pytest.approx(spec.b + spec.rim_width / 2, abs=1.5)


PALETTE_AVI = __import__("pathlib").Path(
    "C:/Users/HECha/Downloads/EHLA/MS7_u05_03_074-125w-1mms-20A_Ti_L1_C1S0001_S0001.avi")


@pytest.mark.skipif(not PALETTE_AVI.is_file(), reason="lab palette (pal8) AVI not on this machine")
def test_palette_video_not_black_after_seeking():
    """pal8 video: the palette comes with the first packet only; frames read after
    a seek must still be grey values (they were all 0 = black before the fix)."""
    import av

    straight = {}
    with av.open(str(PALETTE_AVI)) as c:
        for k, fr in enumerate(c.decode(c.streams.video[0])):
            if k in (0, 999, 1999):
                straight[k] = fr.to_ndarray()[0]  # palette indices; the palette is a grey ramp
            if k == 1999:
                break
    blob = mv.describe_movie(PALETTE_AVI, flip_vertical=False, log=lambda m: None)
    assert blob["file_meta"]["palette"] == "grey ramp"
    movie = mv.Movie(blob)
    for i in (999, 0, 1999):
        a = movie.raw(i)
        assert a.max() > 0 and np.array_equal(a, straight[i])
