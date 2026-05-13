"""Command-line interface for image-quality-sorter."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from image_quality_sorter.metrics import (
    ImageLoadError,
    MetricName,
    score_image,
    supported_metrics,
)
from image_quality_sorter.sorter import (
    ImageDirectoryError,
    ImageQuality,
    collect_image_paths,
    copy_ranked_images,
    rank_scores,
    write_csv,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def format_results(
    results: tuple[ImageQuality, ...], copied_paths: tuple[Path, ...]
) -> str:
    """Format ranked results as a small text table."""
    lines = ["rank  label   score       image"]
    lines.extend(
        f"{result.rank:<5} {result.label:<7} {result.score:<11.2f} {result.path}"
        for result in results
    )
    if copied_paths:
        lines.append("")
        lines.append("planned/written files:")
        lines.extend(str(path) for path in copied_paths)
    return "\n".join(lines)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "input_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Copy ranked images to this directory.",
)
@click.option(
    "-m",
    "--metric",
    type=click.Choice(list(supported_metrics()), case_sensitive=False),
    default="laplacian-variance",
    show_default=True,
    help="Sharpness metric to compute.",
)
@click.option("-r", "--recursive", is_flag=True, help="Scan nested directories.")
@click.option(
    "--descending/--ascending",
    default=True,
    show_default=True,
    help="Sort best images first, or worst images first with --ascending.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Override the metric-specific sharp/blurry threshold.",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a CSV report.",
)
@click.option("--dry-run", is_flag=True, help="Show output filenames without copying.")
@click.option(
    "--skip-errors/--fail-on-error",
    default=False,
    show_default=True,
    help="Continue when a file with an image extension cannot be decoded.",
)
def main(
    input_dir: Path,
    output_dir: Path | None,
    metric: str,
    recursive: bool,
    descending: bool,
    threshold: float | None,
    csv_path: Path | None,
    dry_run: bool,
    skip_errors: bool,
) -> None:
    """Sort INPUT_DIR images by quality and optionally copy them by rank."""
    metric_name = cast(MetricName, metric)
    try:
        image_paths = collect_image_paths(input_dir, recursive=recursive)
    except ImageDirectoryError as error:
        raise click.ClickException(str(error)) from error

    if not image_paths:
        raise click.ClickException("No supported images found in the input directory.")

    scored_images: list[tuple[Path, float]] = []
    failed_paths: list[str] = []
    for image_path in image_paths:
        try:
            scored_images.append((image_path, score_image(image_path, metric_name)))
        except ImageLoadError as error:
            if not skip_errors:
                raise click.ClickException(str(error)) from error
            failed_paths.append(str(error.path))

    if not scored_images:
        raise click.ClickException("All candidate image files failed to load.")

    results = rank_scores(
        scored_images,
        metric_name,
        threshold=threshold,
        descending=descending,
    )
    copied_paths: tuple[Path, ...] = ()
    if output_dir is not None:
        copied_paths = copy_ranked_images(results, output_dir, dry_run=dry_run)
    if csv_path is not None:
        write_csv(results, csv_path)

    click.echo(format_results(results, copied_paths))
    if failed_paths:
        click.echo(f"Skipped unreadable files: {', '.join(failed_paths)}", err=True)
