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

After `pip install -e .`, the `history-postprocess` command is available in the active environment. It exposes two subcommands.

### `history-postprocess create <output_dir>`

Scaffolds a working directory and copies the config template into it:

```bash
history-postprocess create my_run/
# → creates my_run/config.toml from the built-in template
```

Edit `config.toml` to set your paths before running any step.

### `history-postprocess run <STEP> --config <path/to/config.toml>`

Executes one or all pipeline steps:

```bash
history-postprocess run all --config my_run/config.toml
history-postprocess run point2dem --config my_run/config.toml --overwrite
```

Available steps (executed in this order when using `all`):

| Step | Description |
|---|---|
| `uncompress` | Extract compressed submission archives into `extracted_dir` |
| `symlinks` | Parse filenames and create typed symlink directories |
| `point2dem` | Convert dense point clouds to DEMs via PDAL; integrate user-provided DEMs |
| `coregister` | Coregister raw DEMs to the reference (Nuth–Kaab + vertical shift) |
| `ddem` | Compute differential DEMs before and after coregistration |
| `std_dem` | Build one STD DEM per (site, dataset) group from all coregistered DEMs |
| `landcover` | Compute and plot landcover-stratified statistics on dDEMs and STD DEMs |
| `all` | Run all steps above in order |

**Common flags** (override the config file values):

| Flag | Description |
|---|---|
| `--overwrite` | Recompute existing outputs |
| `--dry-run` | Build PDAL pipelines without executing them |
| `--no-plots` | Skip plot generation for this step |
| `--max-workers N` | Number of parallel worker threads |
| `-v` / `-vv` | Increase verbosity (INFO / DEBUG) |

## Scripts

```bash
# Check a submission directory for valid filenames and mandatory files
python scripts/check_submissions.py <path/to/submission_dir>

# Test filename parsing logic
python scripts/check_submissions.py test
```

> `scripts/postprocessing/extraction.py` and `point2dem.py` are legacy scripts superseded by `history-postprocess run uncompress` and `history-postprocess run point2dem`.

## Architecture

### `src/history/`

- **`postprocessing/cli.py`** – `history-postprocess` entry point. Two subcommands: `create` (scaffold a directory with `config.toml`) and `run` (dispatch one or all pipeline steps). CLI flags override config-file values.
- **`postprocessing/config.py`** – `Config` and `ProcConfig` dataclasses. `Config.from_toml_file` loads the TOML config; `ProcConfig.from_base_dir` derives the full processing directory layout from a single root path.
- **`postprocessing/pipeline.py`** – core batch-processing logic: archive extraction, symlink creation, point-cloud→DEM conversion (via PDAL subprocess), DEM coregistration (Nuth–Kaab + vertical shift using `xdem`), dDEM generation, STD DEM computation. Each main step has a matching `plot_*` function. Supports parallel execution via `ThreadPoolExecutor`; errors are logged without stopping batch runs.
- **`postprocessing/io.py`** – filename parsing (`parse_filename`), `ReferencesData` loader, and several helper functions used in notebooks: `analyze_submissions`, `combine_intrinsics_files`, `combine_extrinsics_files`, `filter_experiment_data`, `mirror_as_symlinks`, `get_filepaths_df`. Defines `FILE_CODE_MAPPING` and `FILENAME_PATTERN` for the submission naming convention.
- **`postprocessing/statistics.py`** – statistics on DEMs and point clouds: basic raster stats, coregistration shifts, landcover-stratified dDEM statistics.
- **`postprocessing/visualization.py`** – figure generation for all pipeline steps (mosaics, barplots, shift scatter plots, landcover boxplots, STD DEM maps).
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

The config file (`config.exemple.toml`, copied by `history-postprocess create`) uses TOML format with the following keys:

| Key | Description |
|---|---|
| `raw_dir` | Directory containing compressed submission archives (input) |
| `extracted_dir` | Where archives are extracted |
| `proc_dir` | Root of the processing directory tree (see layout below) |
| `plot_dir` | Where all output plots are saved |
| `overwrite` | Recompute existing outputs (default `false`) |
| `dry_run` | Prepare PDAL pipelines without executing (default `false`) |
| `pdal_exec_path` | Path or name of the PDAL executable (default `"pdal"`) |
| `max_workers` | Parallel worker threads (default `4`) |
| `[references_data_mapping.<site>.<dataset>]` | `ref_dem`, `ref_dem_mask`, `landcover` paths for each combination |

Valid site values: `casa_grande`, `iceland`. Valid dataset values: `aerial`, `kh9mc`, `kh9pc`. All 6 (site, dataset) combinations must be present in `references_data_mapping`.

`ProcConfig.from_base_dir` derives this layout from `proc_dir`:

```
<proc_dir>/
├── symlinks/
│   ├── dense_pointclouds/            ← symlinks to .las/.laz files
│   ├── sparse_pointclouds/
│   ├── extrinsics/
│   ├── intrinsics/
│   └── dems/                         ← user-provided DEM symlinks
├── raw_dems/                         ← *-DEM.tif from point2dem step
├── coregistered_dems/                ← *-DEM.tif from coregister step
├── ddems/
│   ├── before_coregistration/        ← *-DDEM.tif
│   └── after_coregistration/         ← *-DDEM.tif
└── std_dems/                         ← <site>_<dataset>_std_dem.tif
```

### Notebooks

- `notebooks/preprocessing/` – per-dataset preprocessing workflows (aerial fiducial detection, KH-9 PC joining/restitution, KH-9 MC reseau correction).
- `notebooks/postprocessing/post_process_workflow.ipynb` – main 8-step evaluation workflow driving `history.postprocessing`: extract archives → analyze & symlink → point cloud→DEM → coregister → dDEM → landcover stats → STD DEMs → STD DEM stats.
