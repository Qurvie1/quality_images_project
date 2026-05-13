# Algorithm

The application measures blur rather than general photographic quality. It does
not know whether a composition is beautiful; it estimates whether strong local
edges survived image capture and preprocessing.

## Variance of Laplacian

1. Convert the image to grayscale.
2. Apply a discrete Laplacian kernel:

   ```text
   [[ 0,  1, 0],
    [ 1, -4, 1],
    [ 0,  1, 0]]
   ```

3. Compute the variance of the response.
4. Sort by that value. Higher is sharper, lower is blurrier.

This metric is fast and easy to explain. Its main limitation is that thresholds
are dataset-dependent: texture, resolution, noise, compression, and lighting can
change the score.

## Tenengrad

Tenengrad is included as an alternative. It computes Sobel-like horizontal and
vertical gradients and averages the squared gradient magnitude. It is also a
sharpness metric where higher is better.

## Labels

The project uses metric-specific default thresholds:

| Metric | Default threshold |
| --- | ---: |
| `laplacian-variance` | `100.0` |
| `tenengrad` | `1000.0` |

For real data, run the sorter on a small labelled sample and tune `--threshold`.
