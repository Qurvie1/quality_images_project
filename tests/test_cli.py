from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from image_quality_sorter.cli import main


def save_cli_image(path: Path, *, sharp: bool) -> None:
    array = np.zeros((12, 12), dtype=np.uint8)
    if sharp:
        array[:, 6:] = 255
    Image.fromarray(array, mode="L").save(path)


def test_cli_dry_run_csv_and_output_listing(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    save_cli_image(input_dir / "sharp.png", sharp=True)
    save_cli_image(input_dir / "flat.png", sharp=False)
    csv_path = tmp_path / "out" / "scores.csv"
    output_dir = tmp_path / "ranked"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(input_dir),
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--csv",
            str(csv_path),
            "--threshold",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "rank  label" in result.output
    assert "planned/written files" in result.output
    assert csv_path.exists()
    assert not output_dir.exists()


def test_cli_without_output_dir_prints_results(tmp_path: Path) -> None:
    save_cli_image(tmp_path / "flat.png", sharp=False)

    runner = CliRunner()
    result = runner.invoke(
        main, [str(tmp_path), "--metric", "tenengrad", "--ascending"]
    )

    assert result.exit_code == 0
    assert "flat.png" in result.output
    assert "planned/written files" not in result.output


def test_cli_rejects_empty_directory(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [str(tmp_path)])

    assert result.exit_code != 0
    assert "No supported images" in result.output


def test_cli_rejects_missing_directory(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [str(tmp_path / "missing")])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_cli_fails_or_skips_unreadable_images(tmp_path: Path) -> None:
    bad_image = tmp_path / "bad.png"
    bad_image.write_text("not an image", encoding="utf-8")
    save_cli_image(tmp_path / "good.png", sharp=True)
    runner = CliRunner()

    failing_result = runner.invoke(main, [str(tmp_path)])
    skipping_result = runner.invoke(main, [str(tmp_path), "--skip-errors"])

    assert failing_result.exit_code != 0
    assert "Cannot load image" in failing_result.output
    assert skipping_result.exit_code == 0
    assert "Skipped unreadable files" in skipping_result.output


def test_cli_reports_when_every_candidate_fails(tmp_path: Path) -> None:
    bad_image = tmp_path / "bad.png"
    bad_image.write_text("not an image", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, [str(tmp_path), "--skip-errors"])

    assert result.exit_code != 0
    assert "All candidate image files failed" in result.output
