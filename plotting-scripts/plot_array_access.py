import matplotlib.figure
import matplotlib.pyplot as plt

import numpy as np


DEFAULT_ACCESS_STYLE: dict[str, object] = {
    "all_color": "#E4E3F7",
    "all_alpha": 0.35,
    "requested_color": "#1F6B47",
    "requested_alpha": 0.95,
    "loaded_color": "#B3E2CD",
    "loaded_alpha": 0.45,
    "edge_color": "black",
    "edge_linewidth": 0.5,
    "legend_all": "All data",
    "legend_requested": "Requested data ({count} voxels)",
    "legend_loaded": "Loaded data ({count} voxels)",
    "title_fontsize": 24,
    "legend_fontsize": 18,
}
DPI = 300


def _merge_style(overrides: dict[str, object] | None) -> dict[str, object]:
    style = DEFAULT_ACCESS_STYLE.copy()
    if overrides:
        style.update(overrides)
    return style


def _normalize_requested_slices(
    array_shape: tuple[int, ...],
    requested: tuple[slice, ...],
) -> tuple[slice, ...]:
    if len(requested) != len(array_shape):
        raise ValueError("requested must contain one slice per array axis")

    normalized: list[slice] = []
    for axis, (axis_len, axis_slice) in enumerate(zip(array_shape, requested, strict=True)):
        start, stop, step = axis_slice.indices(axis_len)
        if step != 1:
            raise ValueError(f"requested slice step must be contiguous (axis {axis})")

        normalized.append(slice(start, stop, 1))

    return tuple(normalized)


def _compute_requested_and_loaded_masks(
    array_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    requested: tuple[slice, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if len(array_shape) not in (2, 3):
        raise ValueError("array_shape must be 2D or 3D")
    if any(size <= 0 for size in array_shape):
        raise ValueError("array_shape values must be positive")
    if len(chunk_shape) != len(array_shape):
        raise ValueError("chunk_shape dimensionality must match array_shape")
    if any(c < 0 for c in chunk_shape):
        raise ValueError("chunk_shape values must be positive")

    req = _normalize_requested_slices(array_shape, requested)

    requested_mask = np.zeros(array_shape, dtype=bool)
    requested_mask[req] = True

    loaded_mask = np.zeros(array_shape, dtype=bool)

    if chunk_shape[0]>0:
        chunk_starts_per_axis: list[range] = [
            range(0, size, chunk)
            for size, chunk in zip(array_shape, chunk_shape, strict=True)
        ]

        for chunk_origin in np.ndindex(*(len(r) for r in chunk_starts_per_axis)):
            chunk_slices: list[slice] = []
            intersects = True
            for axis, chunk_idx in enumerate(chunk_origin):
                start = chunk_starts_per_axis[axis][chunk_idx]
                stop = min(start + chunk_shape[axis], array_shape[axis])
                chunk_slices.append(slice(start, stop))

                req_start = req[axis].start
                req_stop = req[axis].stop
                if stop <= req_start or start >= req_stop:
                    intersects = False
                    break

            if intersects:
                loaded_mask[tuple(chunk_slices)] = True

    return requested_mask, loaded_mask


def _plot_array_access_2d(
    *,
    array_shape: tuple[int, int],
    requested_mask: np.ndarray,
    loaded_mask: np.ndarray,
    chunk_shape: tuple[int, int],
    style: dict[str, object],
    title: str | None,
) -> tuple[matplotlib.figure.Figure, object]:
    fig, ax = plt.subplots()
    has_chunk_legend = chunk_shape[0] > 0
    title_fontsize = float(style["title_fontsize"])
    legend_fontsize = float(style["legend_fontsize"])
    width_in = 4.6
    data_height_in = width_in * (array_shape[0] / array_shape[1])
    title_text = title or f"Array shape = {array_shape}" + (f"\nChunk shape = {chunk_shape}" if has_chunk_legend else "")
    title_lines = title_text.count("\n") + 1
    top_pad_in = 0.08 + (title_fontsize / 72.0) * (title_lines + 0.6)
    bottom_pad_in = 0.8 + ((legend_fontsize / 72.0) * 3.4 if has_chunk_legend else 0.0)
    height_in = data_height_in + top_pad_in + bottom_pad_in
    fig.set_size_inches(width_in, height_in)

    base_layer = np.zeros((*array_shape, 4), dtype=float)
    base_layer[..., :3] = plt.matplotlib.colors.to_rgb(style["all_color"])
    base_layer[..., 3] = float(style["all_alpha"])
    ax.imshow(base_layer)

    loaded_layer = np.zeros((*array_shape, 4), dtype=float)
    loaded_layer[..., :3] = plt.matplotlib.colors.to_rgb(style["loaded_color"])
    loaded_layer[..., 3] = loaded_mask.astype(float) * float(style["loaded_alpha"])
    ax.imshow(loaded_layer)

    requested_layer = np.zeros((*array_shape, 4), dtype=float)
    requested_layer[..., :3] = plt.matplotlib.colors.to_rgb(style["requested_color"])
    requested_layer[..., 3] = requested_mask.astype(float) * float(style["requested_alpha"])
    ax.imshow(requested_layer)

    # Draw per-pixel outlines on top of the color layers.
    ax.pcolormesh(
        np.arange(array_shape[1] + 1) - 0.5,
        np.arange(array_shape[0] + 1) - 0.5,
        np.zeros(array_shape, dtype=float),
        shading="flat",
        facecolors="none",
        edgecolors=style["edge_color"],
        linewidth=0.35,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.5, array_shape[1] - 0.5)
    ax.set_ylim(array_shape[0] - 0.5, -0.5)
    ax.margins(0)
    ax.set_title(title_text, fontsize=title_fontsize)

    from matplotlib.patches import Patch, Rectangle

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            array_shape[1],
            array_shape[0],
            fill=False,
            edgecolor=style["edge_color"],
            linewidth=1.2,
        )
    )

    if has_chunk_legend:
        ax.legend(
            [
                Patch(facecolor=style["all_color"], alpha=float(style["all_alpha"])),
                Patch(facecolor=style["requested_color"], alpha=float(style["requested_alpha"])),
                Patch(facecolor=style["loaded_color"], alpha=float(style["loaded_alpha"])),
            ],
            [
                str(style["legend_all"]),
                str(style["legend_requested"]).format(count=int(requested_mask.sum())),
                str(style["legend_loaded"]).format(count=int(loaded_mask.sum())),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=1,
            borderaxespad=0.0,
            fontsize=legend_fontsize,
        )

    bottom_frac = bottom_pad_in / height_in
    top_frac = top_pad_in / height_in
    ax.set_position([0.0, bottom_frac, 1.0, 1.0 - bottom_frac - top_frac])

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    return fig, ax


def _plot_array_access_3d(
    *,
    array_shape: tuple[int, int, int],
    requested_mask: np.ndarray,
    loaded_mask: np.ndarray,
    chunk_shape: tuple[int, int, int],
    style: dict[str, object],
    title: str | None,
) -> tuple[matplotlib.figure.Figure, object]:
    has_chunk_legend = chunk_shape[0] > 0
    title_fontsize = float(style["title_fontsize"])
    legend_fontsize = float(style["legend_fontsize"])
    title_text = title or f"Array shape = {array_shape}" + (f"\nChunk shape = {chunk_shape}" if has_chunk_legend else "")
    title_lines = title_text.count("\n") + 1

    width_in = 4.8
    data_height_in = 4.8
    top_pad_in = 0.08 + (title_fontsize / 72.0) * (title_lines + 0.6)
    bottom_pad_in = 0.8 + ((legend_fontsize / 72.0) * 3.4 if has_chunk_legend else 0.0)
    height_in = data_height_in + top_pad_in + bottom_pad_in

    # Use a taller canvas so the 3D axes can fill width while leaving room for legend below.
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_subplot(projection="3d")

    all_mask = np.ones(array_shape, dtype=bool)

    all_vox = ax.voxels(
        all_mask,
        alpha=float(style["all_alpha"]),
        edgecolors=style["edge_color"],
        linewidths=float(style["edge_linewidth"]),
        facecolors=style["all_color"],
        shade=False,
    )
    loaded_vox = ax.voxels(
        loaded_mask,
        edgecolors=style["edge_color"],
        linewidths=float(style["edge_linewidth"]),
        facecolors=style["loaded_color"],
        alpha=float(style["loaded_alpha"]),
        shade=False,
    )
    requested_vox = ax.voxels(
        requested_mask,
        edgecolors=style["edge_color"],
        linewidths=float(style["edge_linewidth"]),
        facecolors=style["requested_color"],
        alpha=float(style["requested_alpha"]),
        shade=False,
    )

    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_box_aspect(array_shape)
    ax.set_title(title_text, fontsize=title_fontsize)

    from matplotlib.patches import Patch

    if has_chunk_legend:
        ax.legend(
            [
                Patch(facecolor=style["all_color"], alpha=float(style["all_alpha"])),
                Patch(facecolor=style["requested_color"], alpha=float(style["requested_alpha"])),
                Patch(facecolor=style["loaded_color"], alpha=float(style["loaded_alpha"])),
            ],
            [
                str(style["legend_all"]),
                str(style["legend_requested"]).format(count=int(requested_mask.sum())),
                str(style["legend_loaded"]).format(count=int(loaded_mask.sum())),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=1,
            borderaxespad=0.0,
            fontsize=legend_fontsize,
        )

    # Expand the 3D axes itself to remove side whitespace.
    bottom_frac = bottom_pad_in / height_in
    top_frac = top_pad_in / height_in
    ax.set_position([0.0, bottom_frac, 1.0, 1.0 - bottom_frac - top_frac])
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    return fig, ax


def plot_array_access(
    *,
    array_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    requested: tuple[slice, ...],
    style: dict[str, object] | None = None,
    title: str | None = None,
) -> tuple[matplotlib.figure.Figure, object]:
    merged_style = _merge_style(style)
    requested_mask, loaded_mask = _compute_requested_and_loaded_masks(
        array_shape=array_shape,
        chunk_shape=chunk_shape,
        requested=requested,
    )

    if len(array_shape) == 2:
        return _plot_array_access_2d(
            array_shape=(array_shape[0], array_shape[1]),
            requested_mask=requested_mask,
            loaded_mask=loaded_mask,
            chunk_shape=(chunk_shape[0], chunk_shape[1]),
            style=merged_style,
            title=title,
        )

    return _plot_array_access_3d(
        array_shape=(array_shape[0], array_shape[1], array_shape[2]),
        requested_mask=requested_mask,
        loaded_mask=loaded_mask,
        chunk_shape=(chunk_shape[0], chunk_shape[1], chunk_shape[2]),
        style=merged_style,
        title=title,
    )

def color_chunk_figure(*, image_shape: tuple[int, int, int], chunk_shape: tuple[int, int, int]) -> matplotlib.figure.Figure:
    if len(image_shape) != 3:
        raise ValueError("image_shape must be 3D")
    if len(chunk_shape) != 3:
        raise ValueError("chunk_shape must be 3D")
    if any(size <= 0 for size in image_shape):
        raise ValueError("image_shape values must be positive")
    if any(size <= 0 for size in chunk_shape):
        raise ValueError("chunk_shape values must be positive")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for x in range(0, image_shape[0], chunk_shape[0]):
        for y in range(0, image_shape[1], chunk_shape[1]):
            for z in range(0, image_shape[2], chunk_shape[2]):
                voxels = np.zeros(image_shape)
                voxels[x:x+chunk_shape[0], y:y+chunk_shape[1], z:z+chunk_shape[2]] = 1
                ax.voxels(voxels, edgecolors='black', linewidths=0.5, alpha=0.9, shade=False);

    ax.set_aspect("equal")
    ax.axis('off');
    ax.set_title(f'Image shape = {image_shape}\nChunk shape = {chunk_shape}')
    return fig


from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "img"

if __name__ == "__main__":
    array_shape = (5,10,20)
    chunks = (5,5,5)
    requested = (slice(2,3), slice(7,9), slice(5,11))

    for i, dimension in enumerate([2,3]):
        print(dimension)
        fig, ax = plot_array_access(array_shape=array_shape[:dimension], chunk_shape=[0]*dimension, requested=[slice(0,0)]*dimension)
        fig.tight_layout()
        fig.savefig(OUTPUT_PATH / f"array_{dimension}d.png", dpi=DPI)
        plt.close(fig)

        fig, ax = plot_array_access(array_shape=array_shape[:dimension], chunk_shape=chunks[:dimension], requested=requested[:dimension])
        fig.tight_layout()
        fig.savefig(OUTPUT_PATH / f"access_{dimension}d.png", dpi=DPI)
        plt.close(fig)

