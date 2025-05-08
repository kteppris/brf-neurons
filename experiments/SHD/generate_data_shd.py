from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import ssl
import sys
import urllib.request as urlreq
from pathlib import Path
from typing import Dict, Tuple, Union

try:
    from tensorflow.keras.utils import get_file  # type: ignore
    _HAS_TF = True
except ModuleNotFoundError:  # keep the script usable without TF
    _HAS_TF = False

import tables
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------------------------------------------------------#
# 1.  CONFIGURATION                                                            #
# -----------------------------------------------------------------------------#
BASE_URL  = "https://compneuro.net/datasets"
FILES     = ["shd_test.h5.gz", "shd_train.h5.gz"]
MD5_LIST  = f"{BASE_URL}/md5sums.txt"

# Root directory for data (env var wins; falls back to script/../data)
DATA_ROOT = Path(os.environ.get("SHD_DATA_ROOT", Path(__file__).resolve().parent / "data"))
CACHE_DIR = DATA_ROOT / "hdspikes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------#
# 2.  HELPER FUNCTIONS                                                         #
# -----------------------------------------------------------------------------#
def _read_remote_md5s() -> Dict[str, str]:
    """Return {filename: md5} mapping downloaded from md5sums.txt."""
    with urlreq.urlopen(MD5_LIST) as resp:
        text = resp.read().decode("utf-8")
    pairs = [ln.split()[:2] for ln in text.splitlines() if ln.strip()]
    return {fname: md5 for md5, fname in pairs}


def _md5(path: Path, chunk: int = 1 << 16) -> str:
    """MD5 checksum of a local file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        while (blk := f.read(chunk)):
            h.update(blk)
    return h.hexdigest()


def _download(url: str, target: Path) -> None:
    """Download *url* to *target* (atomic-ish)."""
    tmp = target.with_suffix(".tmp")
    with urlreq.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(target)


def _get_file(url: str, fname: str, expected_md5: str) -> Path:
    """Download *fname* to CACHE_DIR, verify MD5, return Path to the .gz file."""
    target = CACHE_DIR / fname
    if target.exists() and _md5(target) == expected_md5:
        return target

    print(f"Downloading {fname} …")
    if _HAS_TF:
        tmp = get_file(
            fname,
            url,
            cache_dir=str(CACHE_DIR),   # absolute path – prevents ~/.keras hijack
            cache_subdir=".",           # we already gave the correct dir
            md5_hash=expected_md5,
        )
        # tf.keras returns the path as str
        target = Path(tmp)
    else:
        _download(url, target)
        if _md5(target) != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {target}")
    return target


def _gunzip(src: Path) -> Path:
    """Return Path to un-gzipped file, inflating if necessary."""
    dst = src.with_suffix("")  # drop .gz
    if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
        return dst
    print(f"Decompressing {src.name} …")
    with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dst


# -----------------------------------------------------------------------------#
# 3.  DOWNLOAD DATA                                                            #
# -----------------------------------------------------------------------------#
print(f"[INFO] Using data root: {DATA_ROOT}")
md5_dict = _read_remote_md5s()

h5_paths = []
for fn in FILES:
    gz_path = _get_file(f"{BASE_URL}/{fn}", fn, md5_dict[fn])
    h5_paths.append(_gunzip(gz_path))

# -----------------------------------------------------------------------------#
# 4.  PRE-PROCESSING (unchanged public API)                                    #
# -----------------------------------------------------------------------------#
def binary_image_readout(times: np.ndarray,
                         units: np.ndarray,
                         dt: float = 1e-3,
                         n_units: int = 700) -> np.ndarray:
    """
    Convert a pair of (times, units) arrays into a 2-D binary image.

    Parameters
    ----------
    times : array_like
        Spike times in seconds.
    units : array_like
        Unit indices. The original SHD IDs run 1…700.
    dt : float, optional
        Temporal bin size in seconds (default 1 ms).
    n_units : int, optional
        Number of channels/units (default 700).

    Returns
    -------
    np.ndarray
        Shape (T, n_units) binary matrix.
    """
    times = times.copy()
    units = units.copy()
    n_steps = int(1 / dt)
    img = np.zeros((n_steps, n_units), dtype=np.uint8)

    step_edges = np.arange(n_steps) * dt
    for step in range(n_steps):
        idxs = np.flatnonzero(times <= step_edges[step])
        active_units = units[idxs]
        active_units = active_units[active_units > 0]
        img[step, n_units - active_units] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
    return img


def generate_dataset(h5_file: Union[str, Path],
                     dt: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) numpy arrays for an SHD .h5 file."""
    h5_file = Path(h5_file)
    with tables.open_file(h5_file, mode="r") as h5:
        units_arr = h5.root.spikes.units
        times_arr = h5.root.spikes.times
        labels    = h5.root.labels

        print(f"[INFO] {h5_file.name}: {len(times_arr)} samples")
        X = [binary_image_readout(times_arr[i], units_arr[i], dt) for i in range(len(times_arr))]
        y = labels.read()[:]
    return np.stack(X, axis=0), y


# -----------------------------------------------------------------------------#
# 5.  CONVERT AND SAVE NPYS                                                   #
# -----------------------------------------------------------------------------#
def _save(split: str, X: np.ndarray, y: np.ndarray, dt_ms: int) -> None:
    np.save(DATA_ROOT / f"{split}X_{dt_ms}ms.npy", X)
    np.save(DATA_ROOT / f"{split}Y_{dt_ms}ms.npy", y)


if __name__ == "__main__":
    DT = 4e-3                   # seconds
    dt_ms = int(DT * 1000)

    test_X, test_y   = generate_dataset(h5_paths[0], dt=DT)
    train_X, train_y = generate_dataset(h5_paths[1], dt=DT)

    _save("test",  test_X,  test_y,  dt_ms)
    _save("train", train_X, train_y, dt_ms)

    print("[DONE] Saved npy files to:", DATA_ROOT.resolve())
