from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
import remfile, h5py
import numpy as np

def get_s3_url(dandiset_id: str, asset_path: str) -> str:
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id).get_asset_by_path(asset_path)
        return asset.get_content_url(follow_redirects=1, strip_query=True)

def open_remote_file(dandiset_id: str, asset_path: str):
    s3_url = get_s3_url(dandiset_id, asset_path)
    print(f"Resolved S3 URL: {s3_url}")
    rfile = remfile.File(s3_url)
    h5f = h5py.File(rfile, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, io


def in_raw_gb(array):
    shape = array.shape
    dtype = array.dtype
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    gib = nbytes / 1024**3
    print(f"{gib:.2f} GiB")
    return gib