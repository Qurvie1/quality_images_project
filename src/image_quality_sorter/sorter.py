"""Filesystem and ranking helpers for image quality assessment."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from image_quality_sorter.metrics import MetricName, score_image

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
DEFAULT_THRESHOLDS: Final[dict[MetricName, float]] = {
    "laplacian-variance": 100.0,
    "tenengrad": 1_000.0,
}
CSV_HEADERS: Final[tuple[str, ...]] = ("rank", "label", "score", "metric", "path")


class ImageDirectoryError(ValueError):
    """Raised when an input directory is missing or invalid."""


@dataclass(frozen=True, slots=True)
class ImageQuality:
    """Ranked quality information for a single image.

    The record stores the original path, one-based rank, numeric score, metric
    name, and a human-readable ``sharp`` or ``blurry`` label.
    """

    rank: int
    path: Path
    score: float
    metric: MetricName
    label: str


def normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    """Normalize extensions to lowercase strings starting with a dot.

    Args:
        extensions: File suffixes such as ``"jpg"`` or ``".png"``.

    Returns:
        Normalized extension set.
    """
    normalized = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    return frozenset(normalized)


def collect_image_paths(
    input_dir: str | Path,
    *,
    recursive: bool = False,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> tuple[Path, ...]:
    """Collect supported images from a directory.

    Args:
        input_dir: Directory to scan.
        recursive: Whether to scan nested directories.
        extensions: Supported image suffixes.

    Returns:
        Sorted tuple of image paths.

    Raises:
        ImageDirectoryError: If ``input_dir`` does not point to a directory.
    """
    directory = Path(input_dir)
    if not directory.exists():
        raise ImageDirectoryError(f"Input directory does not exist: {directory}")
    if not directory.is_dir():
        raise ImageDirectoryError(f"Input path is not a directory: {directory}")

    normalized_extensions = normalize_extensions(extensions)
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    images = (
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )
    return tuple(sorted(images, key=lambda path: path.as_posix().lower()))


def default_threshold(metric: MetricName) -> float:
    """Return the default sharp/blurry threshold for a metric."""
    return DEFAULT_THRESHOLDS[metric]


def quality_label(score: float, threshold: float) -> str:
    """Return a human-readable label for a quality score."""
    return "sharp" if score >= threshold else "blurry"


def rank_scores(
    scored_images: Iterable[tuple[str | Path, float]],
    metric: MetricName,
    *,
    threshold: float | None = None,
    descending: bool = True,
) -> tuple[ImageQuality, ...]:
    """Rank already scored images.

    Args:
        scored_images: Pairs of original image path and metric score.
        metric: Metric used to create the scores.
        threshold: Optional label threshold. Defaults to a metric-specific value.
        descending: If true, highest quality is rank 1. If false, lowest
            quality is rank 1.

    Returns:
        Ranked immutable results.
    """
    label_threshold = default_threshold(metric) if threshold is None else threshold
    normalized_scores = [(Path(path), float(score)) for path, score in scored_images]
    if descending:
        ordered_scores = sorted(
            normalized_scores,
            key=lambda item: (-item[1], item[0].as_posix().lower()),
        )
    else:
        ordered_scores = sorted(
            normalized_scores,
            key=lambda item: (item[1], item[0].as_posix().lower()),
        )

    return tuple(
        ImageQuality(
            rank=index,
            path=path,
            score=score,
            metric=metric,
            label=quality_label(score, label_threshold),
        )
        for index, (path, score) in enumerate(ordered_scores, start=1)
    )


def rank_images(
    paths: Iterable[str | Path],
    metric: MetricName,
    *,
    threshold: float | None = None,
    descending: bool = True,
) -> tuple[ImageQuality, ...]:
    """Load, score, and rank image paths.

    Args:
        paths: Image files to evaluate.
        metric: Sharpness metric name.
        threshold: Optional label threshold.
        descending: Whether rank 1 is the sharpest image.

    Returns:
        Ranked immutable results.
    """
    scored_images = ((Path(path), score_image(path, metric)) for path in paths)
    return rank_scores(
        scored_images,
        metric,
        threshold=threshold,
        descending=descending,
    )


def build_output_name(result: ImageQuality) -> str:
    """Build a deterministic rank-prefixed output filename."""
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in result.path.stem
    )
    if not safe_stem:
        safe_stem = "image"
    score_text = f"{result.score:.2f}".replace(".", "_")
    suffix = result.path.suffix.lower()
    return f"{result.rank:04d}_{result.label}_{score_text}_{safe_stem}{suffix}"


def copy_ranked_images(
    results: Sequence[ImageQuality],
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> tuple[Path, ...]:
    """Copy ranked images to a target directory.

    Args:
        results: Ranked results to copy.
        output_dir: Destination directory.
        dry_run: If true, return the planned targets without touching the filesystem.

    Returns:
        Target paths in rank order.
    """
    target_dir = Path(output_dir)
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = []
    for result in results:
        target = target_dir / build_output_name(result)
        targets.append(target)
        if not dry_run:
            shutil.copy2(result.path, target)
    return tuple(targets)


def rows_for_csv(results: Sequence[ImageQuality]) -> list[dict[str, str]]:
    """Convert results to CSV-ready rows."""
    return [
        {
            "rank": str(result.rank),
            "label": result.label,
            "score": f"{result.score:.6f}",
            "metric": result.metric,
            "path": result.path.as_posix(),
        }
        for result in results
    ]


def write_csv(results: Sequence[ImageQuality], destination: str | Path) -> None:
    """Write ranked image scores as CSV.

    Args:
        results: Ranked results.
        destination: CSV output path.
    """
    csv_path = Path(destination)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows_for_csv(results))
