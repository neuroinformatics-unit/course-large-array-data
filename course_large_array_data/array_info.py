from pathlib import Path

import numpy as np


def print_in_raw_gb(array):
    gib = in_raw_gb(array)
    print(f"{gib:.2f} GiB")


def in_raw_gb(array):
    shape = array.shape
    dtype = array.dtype
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    gib = nbytes / 1024**3
    return gib


def zarr_disk_size(path):
    assert str(path).endswith(".zarr"), "Not a Zarr file"
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
