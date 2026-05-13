from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from image_quality_sorter.metrics import (
    ImageLoadError,
    laplacian_response,
    load_grayscale,
    score_array,
    score_image,
    supported_metrics,
    tenengrad_sharpness,
    variance_of_laplacian,
)


def save_checkerboard(path: Path, *, size: int = 32) -> None:
    indexes = np.indices((size, size)).sum(axis=0)
    array = ((indexes % 2) * 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_supported_metrics_are_exposed() -> None:
    assert supported_metrics() == ("laplacian-variance", "tenengrad")


def test_laplacian_and_tenengrad_scores_distinguish_edges() -> None:
    flat = np.zeros((5, 5), dtype=np.float64)
    edge = flat.copy()
    edge[:, 3:] = 255.0

    assert np.all(laplacian_response(flat) == 0.0)
    assert variance_of_laplacian(flat) == 0.0
    assert variance_of_laplacian(edge) > variance_of_laplacian(flat)
    assert tenengrad_sharpness(edge) > tenengrad_sharpness(flat)


def test_score_array_dispatches_metrics() -> None:
    image = np.array([[0.0, 255.0], [255.0, 0.0]])

    assert score_array(image, "laplacian-variance") == variance_of_laplacian(image)
    assert score_array(image, "tenengrad") == tenengrad_sharpness(image)

    with pytest.raises(ValueError, match="Unsupported metric"):
        score_array(image, "unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_image",
    [np.empty((0, 0)), np.zeros((2, 2, 3), dtype=np.float64)],
)
def test_metrics_reject_non_grayscale_arrays(bad_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Expected"):
        variance_of_laplacian(bad_image)


def test_load_grayscale_and_score_image(tmp_path: Path) -> None:
    sharp_path = tmp_path / "sharp.png"
    blurred_path = tmp_path / "blurred.png"
    save_checkerboard(sharp_path)
    Image.open(sharp_path).filter(ImageFilter.GaussianBlur(radius=4)).save(blurred_path)

    gray = load_grayscale(sharp_path)

    assert gray.shape == (32, 32)
    assert gray.dtype == np.float64
    assert score_image(sharp_path, "laplacian-variance") > score_image(
        blurred_path,
        "laplacian-variance",
    )


def test_load_grayscale_reports_missing_and_invalid_files(tmp_path: Path) -> None:
    invalid_path = tmp_path / "not-an-image.png"
    invalid_path.write_text("not actually an image", encoding="utf-8")

    with pytest.raises(ImageLoadError, match="file does not exist"):
        load_grayscale(tmp_path / "missing.png")

    with pytest.raises(ImageLoadError, match="Cannot load image"):
        load_grayscale(invalid_path)
