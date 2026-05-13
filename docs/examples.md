# Examples

## Dry run

```bash
uv run iqsort examples/sample_input --dry-run
```

Example output:

```text
rank  label   score       image
1     sharp   1021490.82  examples/sample_input/sharp.png
2     blurry  15.91       examples/sample_input/medium.png
3     blurry  8.74        examples/sample_input/blur.png
```

## Copy sorted files

```bash
uv run iqsort examples/sample_input --output-dir sorted --csv results.csv
```

The output directory receives deterministic rank-prefixed filenames:

```text
sorted/
├── 0001_sharp_1021490_82_sharp.png
├── 0002_blurry_15_91_medium.png
└── 0003_blurry_8_74_blur.png
```

The CSV report contains:

```text
rank,label,score,metric,path
1,sharp,1021490.820000,laplacian-variance,examples/sample_input/sharp.png
2,blurry,15.910000,laplacian-variance,examples/sample_input/medium.png
3,blurry,8.740000,laplacian-variance,examples/sample_input/blur.png
```

## Recursive scan

```bash
uv run iqsort dataset --recursive --metric tenengrad --ascending
```

This command puts the blurriest images first, which is useful when you want to
inspect or remove bad images manually.
