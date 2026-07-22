import matplotlib.figure
import matplotlib.pyplot as plt

import numpy as np


DEFAULT_ACCESS_STYLE: dict[str, object] = {
    "all_color": "tab:blue",
    "all_alpha": 0.18,
    "requested_color": "tab:red",
    "requested_alpha": 0.95,
    "loaded_color": "tab:orange",
    "loaded_alpha": 0.35,
    "edge_color": "black",
    "edge_linewidth": 0.5,
    "legend_all": "All data",
    "legend_requested": "Requested data ({count} voxels)",
    "legend_loaded": "Loaded data ({count} voxels)",
}


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

        if stop <= start:
            raise ValueError(f"requested region is empty after normalization on axis {axis}")

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
    if any(c <= 0 for c in chunk_shape):
        raise ValueError("chunk_shape values must be positive")

    req = _normalize_requested_slices(array_shape, requested)

    requested_mask = np.zeros(array_shape, dtype=bool)
    requested_mask[req] = True

    loaded_mask = np.zeros(array_shape, dtype=bool)

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

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"Array shape = {array_shape}\nChunk shape = {chunk_shape}")

    from matplotlib.patches import Patch

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
    )

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
    fig = plt.figure()
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
    ax.set_title(title or f"Array shape = {array_shape}\nChunk shape = {chunk_shape}")

    custom_lines = [
        list(all_vox.values())[0],
        list(requested_vox.values())[0],
        list(loaded_vox.values())[0],
    ]
    ax.legend(
        custom_lines,
        [
            str(style["legend_all"]),
            str(style["legend_requested"]).format(count=int(requested_mask.sum())),
            str(style["legend_loaded"]).format(count=int(loaded_mask.sum())),
        ],
    )

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


def access_figure() -> tuple[tuple[matplotlib.figure.Figure, object], tuple[matplotlib.figure.Figure, object]]:
    first = plot_array_access(
        array_shape=(10, 10, 20),
        chunk_shape=(10, 10, 1),
        requested=(slice(2, 4), slice(3, 5), slice(2, 4)),
        title="Chunk shape = (10, 10, 1)",
    )

    second = plot_array_access(
        array_shape=(10, 10, 20),
        chunk_shape=(2, 2, 3),
        requested=(slice(2, 4), slice(3, 5), slice(2, 4)),
        title="Chunk shape = (2, 2, 3)",
    )

    return first, second