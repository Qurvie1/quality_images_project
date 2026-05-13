"""Small benchmark for local experimentation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from image_quality_sorter.metrics import variance_of_laplacian


def synthetic_image(size: int) -> np.ndarray:
    """Return a deterministic checkerboard-like image."""
    indexes = np.indices((size, size)).sum(axis=0)
    return ((indexes % 2) * 255).astype(np.float64)


def main() -> None:
    """Run the benchmark and print elapsed times."""
    output_dir = Path("benchmark-output")
    output_dir.mkdir(exist_ok=True)
    for size in (128, 256, 512, 1024):
        image = synthetic_image(size)
        start = perf_counter()
        score = variance_of_laplacian(image)
        elapsed = perf_counter() - start
        Image.fromarray(image.astype(np.uint8), mode="L").save(
            output_dir / f"{size}.png"
        )
        print(f"{size:4d}x{size:<4d} score={score:10.2f} elapsed={elapsed:.6f}s")


if __name__ == "__main__":
    main()
