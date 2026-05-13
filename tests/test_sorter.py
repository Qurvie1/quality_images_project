from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from image_quality_sorter.sorter import (
    ImageDirectoryError,
    ImageQuality,
    build_output_name,
    collect_image_paths,
    copy_ranked_images,
    default_threshold,
    normalize_extensions,
    quality_label,
    rank_images,
    rank_scores,
    rows_for_csv,
    write_csv,
)


def save_block_image(path: Path, value: int = 255) -> None:
    array = np.zeros((8, 8), dtype=np.uint8)
    array[:, 4:] = value
    Image.fromarray(array, mode="L").save(path)


def test_collect_image_paths_filters_and_sorts(tmp_path: Path) -> None:
    root_image = tmp_path / "B.PNG"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_image = nested_dir / "a.jpg"
    text_file = tmp_path / "notes.txt"
    root_image.write_bytes(b"fake")
    nested_image.write_bytes(b"fake")
    text_file.write_text("ignore", encoding="utf-8")

    assert collect_image_paths(tmp_path) == (root_image,)
    assert collect_image_paths(tmp_path, recursive=True) == (root_image, nested_image)
    assert normalize_extensions(["jpg", ".PNG"]) == frozenset({".jpg", ".png"})


def test_collect_image_paths_rejects_bad_input(tmp_path: Path) -> None:
    file_path = tmp_path / "file.png"
    file_path.write_bytes(b"fake")

    with pytest.raises(ImageDirectoryError, match="does not exist"):
        collect_image_paths(tmp_path / "missing")

    with pytest.raises(ImageDirectoryError, match="not a directory"):
        collect_image_paths(file_path)


def test_rank_scores_descending_and_ascending(tmp_path: Path) -> None:
    high = tmp_path / "high.png"
    low = tmp_path / "low.png"
    scored = [(low, 1.0), (high, 10.0)]

    descending = rank_scores(scored, "laplacian-variance", threshold=5.0)
    ascending = rank_scores(
        scored, "laplacian-variance", threshold=5.0, descending=False
    )

    assert [result.path for result in descending] == [high, low]
    assert [result.label for result in descending] == ["sharp", "blurry"]
    assert [result.path for result in ascending] == [low, high]
    assert default_threshold("laplacian-variance") == 100.0
    assert quality_label(100.0, 100.0) == "sharp"
    assert quality_label(99.0, 100.0) == "blurry"


def test_rank_images_scores_real_images(tmp_path: Path) -> None:
    sharp = tmp_path / "sharp.png"
    flat = tmp_path / "flat.png"
    save_block_image(sharp)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(flat)

    results = rank_images([flat, sharp], "laplacian-variance")

    assert results[0].path == sharp
    assert results[0].rank == 1
    assert results[1].path == flat


def test_output_names_copying_and_csv(tmp_path: Path) -> None:
    source = tmp_path / "weird name.PNG"
    source.write_bytes(b"content")
    result = ImageQuality(
        rank=1,
        path=source,
        score=12.3456,
        metric="laplacian-variance",
        label="blurry",
    )
    output_dir = tmp_path / "out"

    assert build_output_name(result) == "0001_blurry_12_35_weird-name.png"

    with patch("image_quality_sorter.sorter.shutil.copy2") as copy2:
        targets = copy_ranked_images([result], output_dir)

    assert targets == (output_dir / "0001_blurry_12_35_weird-name.png",)
    copy2.assert_called_once_with(source, targets[0])

    with patch("image_quality_sorter.sorter.shutil.copy2") as copy2:
        dry_targets = copy_ranked_images([result], output_dir / "dry", dry_run=True)

    assert dry_targets == (output_dir / "dry" / "0001_blurry_12_35_weird-name.png",)
    copy2.assert_not_called()

    csv_path = tmp_path / "reports" / "results.csv"
    write_csv([result], csv_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert rows == rows_for_csv([result])


def test_output_name_falls_back_for_empty_stem() -> None:
    result = ImageQuality(
        rank=2,
        path=Path(),
        score=0.0,
        metric="tenengrad",
        label="blurry",
    )

    assert build_output_name(result) == "0002_blurry_0_00_image"
