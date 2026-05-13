"""Generate figures used by the LaTeX performance report."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


def laplacian_variance(image: np.ndarray) -> float:
    """Compute the same Laplacian-variance metric used in the package."""
    padded = np.pad(image, pad_width=1, mode="edge")
    response = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - (4.0 * image)
    )
    return float(np.var(response))


def synthetic_image(size: int) -> np.ndarray:
    """Create a deterministic high-frequency image for timing."""
    indexes = np.indices((size, size)).sum(axis=0)
    return ((indexes % 2) * 255).astype(np.float64)


def timed_score(size: int, repeats: int = 7) -> float:
    """Return median scoring time for one image size."""
    image = synthetic_image(size)
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        laplacian_variance(image)
        timings.append(perf_counter() - start)
    return float(np.median(timings))


def main() -> None:
    """Generate ``figures/performance.png``."""
    sizes = [128, 256, 512, 1024, 2048]
    times = [timed_score(size) for size in sizes]

    figure_dir = Path(__file__).parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, times, marker="o")
    plt.xlabel("Image side, pixels")
    plt.ylabel("Median time, seconds")
    plt.title("Laplacian-variance scoring time")
    plt.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "performance.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
