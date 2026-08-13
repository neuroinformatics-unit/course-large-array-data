from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dandi.dandiapi import DandiAPIClient
import h5py
import remfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "img" / "raw_voltage_snippet.png"
DANDISET_ID = "000409"
SUBJECT_ID = "sub-NYU-39"
SESSION_ID = "ses-6ed57216-498d-48a6-b48b-a243a34710ea"


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {remaining_seconds:02d}s"
    return f"{remaining_seconds:d}s"


def find_asset(dandiset) -> object:
    assets = [asset for asset in dandiset.get_assets() if SESSION_ID in asset.path and "ecephys" in asset.path]
    for prefix in ("desc-raw", "desc-processed"):
        for asset in assets:
            if prefix in asset.path:
                return asset

    if assets:
        return assets[0]

    raise RuntimeError(f"No ecephys asset found for session {SESSION_ID}")


def find_voltage_series(root: h5py.Group) -> h5py.Group:
    candidates: list[tuple[tuple[int, int, int], str, h5py.Group]] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group):
            return
        data = obj.get("data")
        if not isinstance(data, h5py.Dataset):
            return
        if data.ndim != 2:
            return
        if data.shape[1] <= 1:
            return

        score = (
            0 if obj.attrs.get("neurodata_type") == "ElectricalSeries" else 1,
            0 if "ElectricalSeries" in name else 1,
            -int(data.shape[0] * data.shape[1]),
        )
        candidates.append((score, name, obj))

    root.visititems(visitor)
    if not candidates:
        raise RuntimeError("No 2D voltage series with a data dataset was found")

    _, name, series = sorted(candidates, key=lambda item: item[0])[0]
    print(f"Selected series group: {name}")
    return series


def main() -> None:
    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID, "draft")

    asset = find_asset(dandiset)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    rem_file = remfile.File(s3_url)
    h5_file = h5py.File(rem_file, "r")

    electrical_series = find_voltage_series(h5_file)
    data = electrical_series["data"]
    rate = float(electrical_series.attrs.get("rate", 30_000.0))
    total_duration_s = data.shape[0] / rate

    n_seconds = 60.0
    n_samples = int(n_seconds * rate)
    start_sample = int(rate * 10)
    stop_sample = min(start_sample + n_samples, data.shape[0])

    display_rate = 200.0
    time_stride = max(1, int(round(rate / display_rate)))

    snippet = np.asarray(data[start_sample:stop_sample:time_stride, :], dtype=np.float32)
    conversion = float(data.attrs.get("conversion", 1.0))
    offset = float(data.attrs.get("offset", 0.0))
    unit = str(data.attrs.get("unit", "volts"))
    snippet = snippet * conversion + offset
    if unit == "volts":
        snippet *= 1e6
        colorbar_label = "Voltage (uV)"
    else:
        colorbar_label = f"Voltage ({unit})"
    snippet = snippet.T

    plotted_rate = rate / time_stride
    t = start_sample / rate + np.arange(snippet.shape[1]) / plotted_rate
    channel_count = snippet.shape[0]

    vmin, vmax = np.percentile(snippet, [2, 98])
    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(
        snippet,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t[0], t[-1], 0.5, channel_count + 0.5],
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label(colorbar_label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_yticks(np.linspace(1, channel_count, num=min(7, channel_count), dtype=int))
    ax.set_title(
        "Raw voltage from DANDI:000409\n"
        f"All {channel_count} channels, {n_seconds:.0f}s window starting at t={start_sample / rate:.1f}s"
    )
    ax.text(
        0.01,
        0.98,
        f"Recording duration: {format_duration(total_duration_s)}\n"
        f"Window shown: {format_duration(n_seconds)} starting at {start_sample / rate:.1f}s",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4},
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Using asset: {asset.path}")
    print(f"Shape: {data.shape}, dtype: {data.dtype}, rate: {rate} Hz")
    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()