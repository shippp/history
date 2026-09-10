# History

**Historical Images for Surface Topography Reconstruction over the Last 50 Years**

This repository supports an experiment designed to compare stereo workflows applied to historical imagery.  
It is organized into two main steps:  
1. preparing the historical image dataset,  
2. post-processing the results.

Participants can use the provided dataset to run their own stereo pipelines and submit outputs (point clouds, DEMs, camera calibration files, etc.). These results are then post-processed to evaluate and compare the different workflows.

This repository only provides tools for downloading and preprocessing the dataset, as well as for post-processing the submitted results.  
**It does not implement any stereo pipeline.**

## Preparing the Historical Image Dataset

Two main packages are used to prepare the dataset:

- [`usgsxplore`](https://github.com/adehecq/usgs_explorer/), used to download raw images from USGS (requires valid [user credentials](https://github.com/adehecq/usgs_explorer?tab=readme-ov-file#credentials)),
- [`hipp`](https://github.com/shippp/hipp), used for the preprocessing of raw imagery.

**Note:**  
Some of the images required for this experiment are not available from USGS.  
Additionally, KH-9 MC image preprocessing is not currently supported by [`hipp`](https://github.com/shippp/hipp), meaning the full workflow is not completely reproducible.

The Historical Images Dataset includes three types of imagery collected over two study sites.
The table below summarizes all available datasets:

| Dataset Name | Date | Images Count  | Images Provider| Preprocessed images size | Raw images size| Bounding Box |
|--|--|--|--|--|--|--|
| Casa Grande Aerial | 1978/09/06 | 37 | [USGS EE](https://earthexplorer.usgs.gov/), [single frame dataset](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-aerial-photo-single-frames?qt-science_center_objects=0#qt-science_center_objects)| 2.6 Go | 7.2 Go |-111.913862   32.699015 -111.685857   32.942928|
| Casa Grande KH-9 PC | 1978/03/25 | 6  | [USGS EE](https://earthexplorer.usgs.gov/), [Declass 3](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-3) | 43 Go | 47 Go |-113.535   32.28  -109.812   33.25|
| Casa Grande KH-9 MC | 1978/03/25 | 4 |  [USGS EE](https://earthexplorer.usgs.gov/), [Declass 2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-2?qt-science_center_objects=0#qt-science_center_objects) | 8.2 Go | 8.9 Go |-112.645   30.782 -110.574   35.053|
| Iceland Aerial | 1980/08/22 | 125 | FTP | 14 Go | 12 Go |-19.74345739  63.42948958 -17.68651768  63.86200756|
| Iceland KH-9 PC | 1980/08/22 | 6 | [USGS EE](https://earthexplorer.usgs.gov/), [Declass 3](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-3) | 33 Go | 30 Go |-23.362  62.923 -15.674  64.405|
| Iceland KH-9 MC | 1980/08/22 | 4  | [USGS EE](https://earthexplorer.usgs.gov/), [Declass 2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-2?qt-science_center_objects=0#qt-science_center_objects) | 8.1 Go | 6.4 Go |-21.946  61.43  -16.633  66.089|

For more details on how the Historical Image Dataset was prepared, see this [README](notebooks/preprocessing/README.md).

## Post-Processing the results

The goal of the Post-Processing workflow is to compare, evaluate, and analyse the outcomes of the stereo reconstruction submissions.  
It takes all user submissions as input, validates and organizes them, and then processes the data to generate a comprehensive set of analytical figures, statistical summaries, and cross-comparisons between methods and configurations.

The `history-postprocess` CLI (available after `pip install -e .`) drives the full pipeline through three subcommands.

**1. Scaffold a working directory**

```bash
history-postprocess create my_run/
```

This creates `my_run/` and copies a `config.toml` template into it. Edit that file to set the paths to your submission archives, reference DEMs, landcover rasters, and output directories.

**2. Run the pipeline**

```bash
history-postprocess run all --config my_run/config.toml
```

The pipeline runs the following steps in order:

| Step | Description |
|---|---|
| `uncompress` | Extract compressed submission archives |
| `symlinks` | Parse filenames and create typed symlink directories |
| `check_planned` | Check extracted experiments against planned experiment |
| `sparse_viz` | Generate sparse point cloud mosaics colored by elevation difference with the reference DEM |
| `provided_dem` | Plot a mosaic of user-provided DEMs against the reference DEM |
| `point2dem` | Convert dense point clouds to DEMs via PDAL |
| `coregister` | Coregister DEMs to the reference (Nuth–Kaab + vertical shift) |
| `ddem` | Compute differential DEMs before and after coregistration |
| `generate_pdf` | Assemble all pipeline output plots into a single PDF report |

Each step can also be run individually, or as a subset (e.g. `history-postprocess run point2dem coregister --config …`) — steps always run in the pipeline order above, regardless of the order given on the command line. Use `--overwrite` to recompute existing outputs, `--no-plots` to skip figures, and `-v`/`-vv` to increase log verbosity.

**3. Check progress**

```bash
history-postprocess status --config my_run/config.toml
```

Prints a quick file-count overview of every processing directory (extracted submissions, symlinks, raw/coregistered DEMs, dDEMs, STD DEMs, plots), to see at a glance how far the pipeline has progressed.

For more details on the Post-Processing workflow, see this [README](notebooks/postprocessing/README.md).

## Analysis

Once `history-postprocess` has produced coregistered DEMs, the `history-analysis` CLI computes the derived statistics and plots on top of them, reusing the same `config.toml`:

```bash
history-analysis run all --config my_run/config.toml
```

| Step | Description |
|---|---|
| `std_dem` | Build one standard-deviation DEM per (site, dataset) group from coregistered DEMs |
| `landcover` | Compute and plot landcover-stratified statistics on dDEMs and STD DEMs |

Each step can also be run individually, or as a subset (e.g. `history-analysis run std_dem landcover --config …`), with the same `--overwrite`/`--no-plots`/`-v` flags as `history-postprocess`.

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/shippp/history.git
cd history
conda env create -f environment.yml
pip install -e .
```

Once the environment is created, launch Jupyter and select the history kernel to run the notebooks.


