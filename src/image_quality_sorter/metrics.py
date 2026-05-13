"""Sharpness metrics used by the image quality sorter."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

MetricName = Literal["laplacian-variance", "tenengrad"]
FloatArray = npt.NDArray[np.float64]

GRAYSCALE_NDIM: Final[int] = 2

SUPPORTED_METRICS: Final[tuple[MetricName, ...]] = (
    "laplacian-variance",
    "tenengrad",
)


class ImageLoadError(ValueError):
    """Raised when a path cannot be decoded as an image."""

    def __init__(self, path: str | Path, reason: str) -> None:
        """Create an image loading error.

        Args:
            path: File that failed to load.
            reason: Human-readable error description.
        """
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"Cannot load image '{self.path}': {reason}")


def supported_metrics() -> tuple[MetricName, ...]:
    """Return the names accepted by the CLI and scoring functions."""
    return SUPPORTED_METRICS


def load_grayscale(path: str | Path) -> FloatArray:
    """Load an image file as a two-dimensional grayscale array.

    Args:
        path: Path to a file readable by Pillow.

    Returns:
        A copy of the image intensities as ``float64`` values in ``[0, 255]``.

    Raises:
        ImageLoadError: If the file is missing or cannot be decoded as an image.
    """
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            return np.array(grayscale, dtype=np.float64, copy=True)
    except FileNotFoundError as error:
        raise ImageLoadError(image_path, "file does not exist") from error
    except (OSError, UnidentifiedImageError) as error:
        raise ImageLoadError(image_path, str(error)) from error


def _ensure_grayscale_array(image: npt.ArrayLike) -> FloatArray:
    """Validate and convert an input array to a grayscale ``float64`` array."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != GRAYSCALE_NDIM:
        raise ValueError("Expected a two-dimensional grayscale image array.")
    if array.size == 0:
        raise ValueError("Expected a non-empty image array.")
    return array


def laplacian_response(image: npt.ArrayLike) -> FloatArray:
    """Compute a four-neighbour discrete Laplacian response.

    The kernel is equivalent to::

        [[ 0,  1, 0],
         [ 1, -4, 1],
         [ 0,  1, 0]]

    Args:
        image: Two-dimensional grayscale image.

    Returns:
        Laplacian response with the same shape as the input image.
    """
    gray = _ensure_grayscale_array(image)
    padded = np.pad(gray, pad_width=1, mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - (4.0 * gray)
    )


def variance_of_laplacian(image: npt.ArrayLike) -> float:
    """Return the variance of the Laplacian response.

    Higher values usually indicate many high-contrast edges and, therefore, a
    sharper image. Very low values are typical for defocused or strongly blurred
    images.

    Args:
        image: Two-dimensional grayscale image.

    Returns:
        Variance of the discrete Laplacian response.
    """
    return float(np.var(laplacian_response(image)))


def _sobel_components(image: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Compute Sobel-like horizontal and vertical gradients."""
    gray = _ensure_grayscale_array(image)
    padded = np.pad(gray, pad_width=1, mode="edge")
    top_left = padded[:-2, :-2]
    top = padded[:-2, 1:-1]
    top_right = padded[:-2, 2:]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    bottom_left = padded[2:, :-2]
    bottom = padded[2:, 1:-1]
    bottom_right = padded[2:, 2:]

    gradient_x = (
        -top_left
        - (2.0 * left)
        - bottom_left
        + top_right
        + (2.0 * right)
        + bottom_right
    )
    gradient_y = (
        -top_left
        - (2.0 * top)
        - top_right
        + bottom_left
        + (2.0 * bottom)
        + bottom_right
    )
    return gradient_x, gradient_y


def tenengrad_sharpness(image: npt.ArrayLike) -> float:
    """Return the mean squared Sobel gradient magnitude.

    Args:
        image: Two-dimensional grayscale image.

    Returns:
        Tenengrad-style focus score. Higher values indicate sharper gradients.
    """
    gradient_x, gradient_y = _sobel_components(image)
    return float(np.mean((gradient_x * gradient_x) + (gradient_y * gradient_y)))


def score_array(image: npt.ArrayLike, metric: MetricName) -> float:
    """Score a grayscale array with the selected metric.

    Args:
        image: Two-dimensional grayscale image.
        metric: Name of the metric to compute.

    Returns:
        Sharpness score where larger values mean a sharper image.

    Raises:
        ValueError: If the metric name is unsupported.
    """
    if metric == "laplacian-variance":
        return variance_of_laplacian(image)
    if metric == "tenengrad":
        return tenengrad_sharpness(image)
    raise ValueError(f"Unsupported metric: {metric}")


def score_image(path: str | Path, metric: MetricName) -> float:
    """Load and score one image file.

    Args:
        path: Image path.
        metric: Name of the sharpness metric.

    Returns:
        Sharpness score where larger values mean a sharper image.
    """
    return score_array(load_grayscale(path), metric)
