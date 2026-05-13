"""Profile only the laplacian_variance function on large images."""

from generate_figures import laplacian_variance, synthetic_image


def main() -> None:
    """Run performance test for laplacian_variance on large synthetic images."""
    size = 4096
    repeats = 100
    image = synthetic_image(size)
    for _ in range(repeats):
        laplacian_variance(image)


if __name__ == "__main__":
    main()
