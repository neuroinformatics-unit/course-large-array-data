import h5py
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO


def _get_s3_url(dandiset_id: str, asset_path: str) -> str:
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id).get_asset_by_path(asset_path)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def open_remote_file(dandiset_id: str, asset_path: str):
    s3_url = _get_s3_url(dandiset_id, asset_path)
    print(f"Resolved S3 URL: {s3_url}")
    rfile = remfile.File(s3_url)
    h5f = h5py.File(rfile, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, io
