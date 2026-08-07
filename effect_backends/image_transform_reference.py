"""Reference image transform helpers backed by OpenCV.

These functions define the quality and border contract for future CPU/GPU
backends. They intentionally stay close to the current core.py behavior.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


_INTERPOLATION_FLAGS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def interpolation_flag(interpolation: str | int) -> int:
    if isinstance(interpolation, int):
        return interpolation
    try:
        return _INTERPOLATION_FLAGS[interpolation]
    except KeyError as exc:
        valid = ", ".join(sorted(_INTERPOLATION_FLAGS))
        raise ValueError(f"unsupported interpolation: {interpolation!r}; expected one of {valid}") from exc


def _pad_to_canvas(
    image: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    top = int(offset_y)
    left = int(offset_x)
    bottom = int(canvas_height) - (top + image.shape[0])
    right = int(canvas_width) - (left + image.shape[1])

    if top < 0 or left < 0 or bottom < 0 or right < 0:
        raise ValueError(
            "resized image must fit inside canvas: "
            f"image={image.shape[:2]}, canvas=({canvas_width}, {canvas_height}), "
            f"offset=({offset_x}, {offset_y})"
        )

    pad_width = ((top, bottom), (left, right))
    if image.ndim == 3:
        pad_width = (*pad_width, (0, 0))
    return np.pad(image, pad_width, mode="constant")


def fit_crop_to_canvas(
    image: np.ndarray,
    source_rect: Sequence[int | float],
    canvas_width: int,
    canvas_height: int,
    draw_width: int,
    draw_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
    interpolation: str | int = "area",
) -> np.ndarray:
    """Crop a source rectangle, resize it, and place it on a zero canvas."""

    x, y, width, height = source_rect
    x = int(x)
    y = int(y)
    width = max(1, int(width))
    height = max(1, int(height))
    draw_width = max(1, int(draw_width))
    draw_height = max(1, int(draw_height))
    canvas_width = max(1, int(canvas_width))
    canvas_height = max(1, int(canvas_height))

    cropped = image[y : y + height, x : x + width]
    resized = cv2.resize(
        cropped,
        (draw_width, draw_height),
        interpolation=interpolation_flag(interpolation),
    )
    return _pad_to_canvas(resized, canvas_width, canvas_height, int(offset_x), int(offset_y))


def transform_to_canvas(
    image: np.ndarray,
    matrix: Sequence[Sequence[int | float]] | np.ndarray,
    canvas_width: int,
    canvas_height: int,
    transform_type: str = "affine",
    interpolation: str | int = "linear",
    border_mode: str = "reflect",
) -> np.ndarray:
    """Transform an image into a canvas using OpenCV as the reference."""

    flags = interpolation_flag(interpolation)
    border = cv2.BORDER_REFLECT if border_mode == "reflect" else cv2.BORDER_CONSTANT
    matrix_arr = np.asarray(matrix, dtype=np.float64)
    output_size = (max(1, int(canvas_width)), max(1, int(canvas_height)))

    if transform_type == "perspective" or matrix_arr.shape == (3, 3):
        return cv2.warpPerspective(image, matrix_arr, output_size, flags=flags, borderMode=border)
    if matrix_arr.shape != (2, 3):
        raise ValueError(f"affine matrix must be 2x3, got {matrix_arr.shape}")
    return cv2.warpAffine(image, matrix_arr, output_size, flags=flags, borderMode=border)


def transform_crop_to_canvas(
    image: np.ndarray,
    matrix: Sequence[Sequence[int | float]] | np.ndarray,
    source_rect: Sequence[int | float],
    transform_width: int,
    transform_height: int,
    canvas_width: int,
    canvas_height: int,
    draw_width: int,
    draw_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
    transform_type: str = "affine",
    interpolation: str | int = "linear",
    border_mode: str = "reflect",
    lens_strength: float = 0.0,
    lens_scale: float = 1.0,
    mesh_map_x: np.ndarray | None = None,
    mesh_map_y: np.ndarray | None = None,
) -> np.ndarray:
    """Reference fused transform + crop + canvas placement."""

    if abs(float(lens_strength)) > 1.0e-6 or abs(float(lens_scale) - 1.0) > 0.01:
        from cores.distortion_correction import correct_lens_distortion

        image = correct_lens_distortion(
            image,
            strength=float(lens_strength),
            scale=float(lens_scale),
            interpolation="bilinear" if interpolation in {"area", "linear"} else "bicubic",
            grid_size=4,
        )

    transformed = transform_to_canvas(
        image,
        matrix,
        transform_width,
        transform_height,
        transform_type=transform_type,
        interpolation=interpolation,
        border_mode=border_mode,
    )
    if mesh_map_x is not None and mesh_map_y is not None:
        # mesh_map_* は絶対座標ではなくキャンバス座標からの「変位」。絶対座標を
        # bicubic 拡大すると Keys(a=-0.75) が 1 次を再現できず画像全体が波打つため
        # (cores.distortion_correction.warp_correction.upsample_coarse_mesh_map 参照)。
        full_map_x = cv2.resize(
            np.asarray(mesh_map_x, dtype=np.float32),
            (int(transform_width), int(transform_height)),
            interpolation=cv2.INTER_CUBIC,
        )
        full_map_y = cv2.resize(
            np.asarray(mesh_map_y, dtype=np.float32),
            (int(transform_width), int(transform_height)),
            interpolation=cv2.INTER_CUBIC,
        )
        full_map_x += np.arange(int(transform_width), dtype=np.float32)[None, :]
        full_map_y += np.arange(int(transform_height), dtype=np.float32)[:, None]
        transformed = cv2.remap(
            transformed,
            full_map_x,
            full_map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    return fit_crop_to_canvas(
        transformed,
        source_rect,
        canvas_width,
        canvas_height,
        draw_width,
        draw_height,
        offset_x,
        offset_y,
        interpolation,
    )
