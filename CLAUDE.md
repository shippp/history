# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**History** is a research toolkit for comparing stereo reconstruction workflows applied to historical imagery (aerial, KH-9 PC, KH-9 MC) over two study sites: Casa Grande (Arizona) and Iceland. It covers two pipelines:

1. **Preprocessing** – downloading and preparing raw historical images for stereo reconstruction (via Jupyter notebooks + `hipp`/`usgsxplore`)
2. **Post-processing** – validating, organizing, and evaluating stereo reconstruction submissions from experiment participants

The repository does **not** implement any stereo pipeline itself.

## Installation & Setup

```bash
conda env create -f environment.yml
conda activate history
pip install -e .
```

Then launch JupyterLab and select the `history` kernel:
```bash
jupyter lab
```

Key dependencies: `xdem`, `geoutils`, `rasterio`, `pdal`, `laspy`, `hipp==0.2.0`, `usgsxplore`, `py7zr`, `contextily`.

## Code Style & Linting

```bash
ruff check src/
ruff format src/
```

Line length is 120 (`[tool.ruff]` in `pyproject.toml`). There are no automated tests.

## CLI Entry Point

After `pip install -e .`, the `history-postprocess` command is available in the active environment. It sets up an output directory with the post-processing notebook and a ready-to-submit SLURM job script:

```bash
history-postprocess <output_dir>
```

The command copies `notebooks/postprocessing/post_process_workflow.ipynb` into `<output_dir>`, generates `run.slurm.sh` using the current Python executable (so no `conda activate` is needed at runtime), then prints step-by-step instructions:

1. Adapt the notebook configuration to your data
2. Review/adjust SLURM resources in `run.slurm.sh` (memory, time, CPUs, partition)
3. Submit with `sbatch <output_dir>/run.slurm.sh`

> **Note:** Run the command from within the `history` conda environment — the generated script embeds the absolute path to that environment's Python and jupytext binaries.

## Scripts

Both post-processing scripts require `--config PATH` pointing to a JSON config file (see `scripts/postprocessing/config.exemple.json` for a template).

```bash
# Check a submission directory for valid filenames and mandatory files
python scripts/check_submissions.py <path/to/submission_dir>

# Test filename parsing logic
python scripts/check_submissions.py test

# Extract compressed archives from <base_dir>/raw/ into <base_dir>/extracted/
python scripts/postprocessing/extraction.py --config <path/to/config.json> [--overwrite] [--max-workers N]

# Convert dense point clouds in the symlinks dir to DEMs via PDAL
python scripts/postprocessing/point2dem.py --config <path/to/config.json> [--overwrite] [--pdal-exec-path pdal] [--max-workers 4] [--dry-run]
```

## Architecture

### `src/history/`

- **`postprocessing/pipeline.py`** – core batch-processing logic: archive extraction, symlink creation, point-cloud→DEM conversion (via PDAL subprocess), DEM coregistration (Nuth–Kaab + vertical shift using `xdem`), dDEM generation, STD DEM computation. Supports parallel execution via `ThreadPoolExecutor`. Errors are logged without stopping batch runs.
- **`postprocessing/io.py`** – filename parsing (`parse_filename`), `ReferencesData` loader, and several helper functions used in notebooks: `analyze_submissions`, `combine_intrinsics_files`, `combine_extrinsics_files`, `filter_experiment_data`, `mirror_as_symlinks`, `get_filepaths_df`. Defines `FILE_CODE_MAPPING` and `FILENAME_PATTERN` for the submission naming convention.
- **`postprocessing/statistics.py`** – landcover-stratified statistics on dDEMs.
- **`postprocessing/plotting.py`** / **`visualization.py`** – figure generation for analysis outputs.
- **`aux_data/download_tools.py`** – helpers to download Copernicus DEM tiles and auxiliary geospatial data.
- **`utils.py`** – `log_to_file` context manager for adding timestamped file handlers to loggers.

### Submission Filename Convention

Submissions follow a strict `_`-separated code pattern:

```
AUTHOR_SITE_DATASET_IMAGES_CALIB_GEOREF_PCOREG_MTP[_VN]_SUFFIX.ext
```

| Segment | Options |
|---|---|
| AUTHOR | 3–5 alphanumeric chars |
| SITE | `CG` (Casa Grande) / `IL` (Iceland) |
| DATASET | `AI` (Aerial) / `MC` (KH-9 MC) / `PC` (KH-9 PC) |
| IMAGES | `PP` (preprocessed) / `RA` (raw) |
| CALIB | `CY` / `CN` (camera calibration used) |
| GEOREF | `GM` / `GA` / `GN` (GCP: manual / automated / none) |
| PCOREG | `PY` / `PN` |
| MTP | `MY` / `MN` |
| version | optional `_V1`, `_V2`, … (uppercase V) |
| SUFFIX | `dense_pointcloud`, `sparse_pointcloud`, `extrinsics`, `intrinsics` (mandatory); `dem.tif`, `orthoimage.tif` (optional) |

Parsing is implemented in both `src/history/postprocessing/io.py` (used by the post-processing pipeline) and `scripts/check_submissions.py` (standalone checker). The two implementations have slightly different regex details—keep them consistent when modifying the convention.

### Post-processing Configuration

The config JSON (`scripts/postprocessing/config.exemple.json`) has three top-level keys:
- `base_dir` – root output directory
- `custom_paths` – optional zoom reference DEMs
- `references_data_mapping` – nested `{site: {dataset: {ref_dem, ref_dem_mask, landcover}}}` required by `point2dem.py` and `ReferencesData`

Valid site values: `"casa_grande"`, `"iceland"`. Valid dataset values: `"aerial"`, `"kh9mc"`, `"kh9pc"`. `ReferencesData` requires all 6 (site, dataset) combinations to be present.

The scripts derive this directory layout from `base_dir`:

```
<base_dir>/
├── raw/                              ← compressed submission archives (input)
├── extracted/                        ← extracted submission folders
└── processing/
    ├── symlinks/dense_pointclouds/   ← symlinks created by indexing step
    └── raw_dems/                     ← DEMs generated by point2dem script
```

### Notebooks

- `notebooks/preprocessing/` – per-dataset preprocessing workflows (aerial fiducial detection, KH-9 PC joining/restitution, KH-9 MC reseau correction).
- `notebooks/postprocessing/post_process_workflow.ipynb` – main 8-step evaluation workflow driving `history.postprocessing`: extract archives → analyze & symlink → point cloud→DEM → coregister → dDEM → landcover stats → STD DEMs → STD DEM stats.
