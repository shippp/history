"""
This module orchestrates the batch processing workflow for DEM generation, coregistration,
differencing, statistics computation, and visualization across multiple datasets and sites.

It defines Prefect flows and tasks to automate the full processing chain — from raw point clouds
to final DEM quality assessment and post-processing visualization. Each flow encapsulates a specific
processing stage, and higher-level flows coordinate their sequential execution.

Main Features
-------------
- **Archive Extraction**: Decompresses and organizes raw submissions into a structured workspace.
- **Point Cloud Processing**: Converts point clouds into DEMs using PDAL.
- **DEM Coregistration**: Aligns DEMs spatially using reference surfaces and masks.
- **dDEM Computation**: Generates DEM differences before and after coregistration.
- **Statistics Computation**: Computes global and landcover-based statistics for DEMs and dDEMs.
- **Standard Deviation DEMs**: Aggregates DEMs to create STD surfaces (with and without inlier filtering).
- **Postprocessing Visualizations**: Produces global and per-dataset plots summarizing performance metrics.
- **Batch Execution**: Manages concurrent processing of multiple datasets within a site.

Structure Overview
------------------
Flows:
    - `uncompress_all_submissions`: Extracts all compressed submissions in parallel.
    - `run_postprocessing`: Runs the entire processing chain for all available sites/datasets.
    - `process_group`: Orchestrates all processing steps for a single site/dataset.
    - `process_pointclouds_to_dems`: Converts point clouds into DEMs using PDAL.
    - `process_coregister_dems`: Coregisters DEMs with reference surfaces.
    - `process_generate_ddems`: Generates DEM differences (before and after coregistration).
    - `process_compute_statistics`: Computes DEM and dDEM statistics and identifies inliers.
    - `process_compute_landcover_statistics`: Computes per-landcover statistics on DEMs.
    - `process_generate_std_dems`: Generates standard deviation DEMs across a dataset group.
    - `generate_postprocessing_plots`: Creates visual summaries and quality plots.

Dependencies
------------
- Prefect: Task orchestration and parallelization.
- Pandas: DataFrame manipulation and CSV export.
- GeoUtils / xDEM: Raster and terrain analysis.
- PDAL: Point cloud to DEM conversion.
- Matplotlib / custom viz module: Visualization of metrics and results.

Notes
-----
All flows are designed to be modular, reproducible, and fault-tolerant.
Intermediate outputs (DEMs, statistics, plots) are cached and reused unless `overwrite=True`.

"""

import warnings
from pathlib import Path

import geoutils as gu
import pandas as pd
from prefect import flow
from prefect.context import get_run_context

import history.postprocessing.visualization as viz
from history.postprocessing.core import (
    compute_raster_statistics,
    compute_raster_statistics_by_landcover,
    convert_pointcloud_to_dem,
    coregister_dem,
    create_std_dem,
    extract_archive,
    generate_ddem,
    get_pointcloud_metadatas,
    get_raster_coregistration_shifts,
    get_raster_statistics,
    get_raster_statistics_by_landcover,
    is_existing_std_dem,
)
from history.postprocessing.io import FILE_CODE_MAPPING_V1, parse_filename
from history.postprocessing.processing_directory import ProcessingDirectory, SubProcessingDirectory


@flow(log_prints=True)
def uncompress_all_submissions(
    input_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """
    Uncompresses all supported archive submissions from an input directory into an output directory.

    This flow scans the input directory for compressed archives (ZIP, 7z, and TAR variants)
    and extracts each one into a corresponding folder within the output directory.
    It runs extractions in parallel using Prefect tasks and supports controlled overwriting.

    Supported archive formats:
        - .zip
        - .7z
        - .tgz
        - .tar.gz
        - .tar.bz2
        - .tar.xz

    Args:
        input_dir (str | Path): Path to the directory containing compressed submissions.
        output_dir (str | Path): Path to the directory where archives will be extracted.
        overwrite (bool, optional): Whether to overwrite existing extracted folders. Defaults to False.

    Returns:
        None: Extracted contents are saved in `output_dir`, one folder per archive.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    supported_extensions = [".zip", ".7z", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz"]

    archives = [
        (fp, output_dir / fp.name[: -len(ext)]) for ext in supported_extensions for fp in input_dir.glob(f"*{ext}")
    ]

    tasks = []
    for input_path, output_path in archives:
        if output_path.exists() and not overwrite:
            print(f"Skipping extraction (folder exists): {output_path}")
            continue
        tasks.append(
            extract_archive.with_options(name=f"extract_archive_{input_path.name}").submit(
                input_path, output_path, overwrite
            )
        )

    for t in tasks:
        try:
            t.result()  # bloque jusqu’à la fin
        except Exception as e:
            print(f"[!] Error while processing: {e}")


@flow(log_prints=True)
def run_postprocessing(
    input_dir: str | Path | ProcessingDirectory,
    output_dir: str | Path,
    pdal_exec_path: str = "pdal",
    verbose: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Executes the complete postprocessing pipeline for all site-dataset pairs in a directory.

    This flow orchestrates the entire postprocessing sequence for multiple datasets by:
        1. Extracting and organizing point clouds into structured processing directories.
        2. Running the full DEM generation and analysis workflow (`process_group`) for each site/dataset.
        3. Handling execution context and parallelization across subflows.

    It acts as a global controller to automate DEM generation, coregistration, differential DEM creation,
    and computation of related statistics and landcover analyses for all datasets found in the input directory.

    Args:
        input_dir (str | Path): Root directory containing the input point cloud files for all sites and datasets.
        output_dir (str | Path): Destination directory where processed outputs will be stored.
        pdal_exec_path (str, optional): Path to the PDAL executable. Defaults to "pdal".
        verbose (bool, optional): Whether to display detailed processing logs. Defaults to True.
        overwrite (bool, optional): Whether to overwrite existing outputs. Defaults to False.
        dry_run (bool, optional): If True, simulates execution without performing actual computations. Defaults to False.

    Returns:
        None: Results are saved under `output_dir`, organized by site and dataset.
    """

    output_dir = Path(output_dir)

    # get the current task runner to propagate it on subflows
    current_task_runner = get_run_context().task_runner

    proc_dir = ProcessingDirectory(input_dir)

    for (site, dataset), sub_dir in proc_dir.sub_dirs.items():
        try:
            process_group.with_options(task_runner=current_task_runner, name=f"process_{site}_{dataset}")(
                sub_dir.base_dir, pdal_exec_path, verbose, overwrite, dry_run
            )
        except Exception as e:
            print(f"Skip {site} - {dataset} : {e}")
            continue


@flow(log_prints=True)
def process_group(
    base_dir: str | Path,
    pdal_exec_path: str = "pdal",
    verbose: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Executes the full DEM processing workflow for a given base directory.

    This flow sequentially runs all major processing steps required to transform point clouds into
    coregistered DEMs, compute elevation differences, derive quality metrics, and generate
    landcover-based and standard deviation DEM statistics. It acts as a single entry point to
    process an entire dataset group.

    Workflow steps:
        1. Convert dense point clouds to raw DEMs.
        2. Coregister DEMs to a reference DEM.
        3. Generate differential DEMs (before and after coregistration).
        4. Compute raster statistics (e.g., NMAD, mean, voids).
        5. Compute DEM statistics per landcover class.
        6. Generate standard deviation DEMs for quality assessment.
        7. Compute landcover statistics on standard deviation DEMs.

    Args:
        base_dir (str | Path): Root directory containing dataset-specific subdirectories.
        pdal_exec_path (str, optional): Path to the PDAL executable. Defaults to "pdal".
        verbose (bool, optional): Whether to print detailed logs during processing. Defaults to True.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        dry_run (bool, optional): If True, simulate the workflow without performing operations. Defaults to False.

    Returns:
        None: Results are written to the corresponding subdirectories under `base_dir`.
    """

    process_pointclouds_to_dems.fn(base_dir, pdal_exec_path, verbose, overwrite, dry_run)
    process_coregister_dems.fn(base_dir, verbose, overwrite)
    process_generate_ddems.fn(base_dir, verbose, overwrite)
    process_compute_statistics.fn(base_dir)
    process_compute_landcover_statistics.fn(base_dir)
    process_generate_std_dems.fn(base_dir, overwrite)
    process_compute_landcover_statistics_on_std_dems.fn(base_dir)


@flow(log_prints=True)
def process_pointclouds_to_dems(
    base_dir: str | Path,
    pdal_exec_path: str = "pdal",
    verbose: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Processes all point clouds in a base directory and converts them into DEMs using PDAL.

    This flow automates the conversion of multiple point cloud files (LAS/LAZ) into Digital
    Elevation Models (DEMs). Each point cloud is matched to a reference DEM for alignment and
    interpolation. The resulting DEMs are organized within the processing directory structure.
    The function supports asynchronous task submission and optional dry runs.

    Args:
        base_dir (str | Path): Base directory containing the point cloud data and reference DEM.
        pdal_exec_path (str, optional): Path to the PDAL executable. Defaults to "pdal".
        verbose (bool, optional): Whether to print detailed process information. Defaults to True.
        overwrite (bool, optional): Whether to overwrite existing DEMs. Defaults to False.
        dry_run (bool, optional): If True, generates PDAL pipelines without executing them. Defaults to False.

    Returns:
        None: The function processes and generates DEMs for all detected point clouds.
    """

    proc_dir = SubProcessingDirectory(base_dir)
    future_dems = {}

    ref_dem_path = proc_dir.get_reference_dem()

    for file in proc_dir.get_pointclouds():
        try:
            code, metadatas = parse_filename(str(file))

            output_dem_path = proc_dir.raw_dems_dir / f"{code}-DEM.tif"

            if output_dem_path.exists() and not overwrite:
                if verbose:
                    print(f"Skip point2dem for {code}: output already exists.")
                continue

            future_dems[code] = convert_pointcloud_to_dem.with_options(name=f"point2dem_{code}").submit(
                file, ref_dem_path, output_dem_path, pdal_exec_path, None, overwrite, dry_run
            )

        except Exception as e:
            print(f"Error {file.name} : {e}")
            continue

    for code, f in future_dems.items():
        try:
            f.result()
        except Exception as e:
            print(f"Point2dem error for {code}: {e}")


@flow(log_prints=True)
def process_coregister_dems(base_dir: str | Path, verbose: bool = True, overwrite: bool = False) -> None:
    """
    Coregisters all DEMs within a processing directory to a reference DEM.

    This flow automates the coregistration of multiple raw DEMs to a common reference DEM
    using both horizontal (Nuth-Kaab) and vertical shift corrections. It handles asynchronous
    task submission for parallel processing and organizes the results in a dedicated
    coregistered DEMs directory.

    Args:
        base_dir (str | Path): Base directory containing raw DEMs and the reference DEM.
        verbose (bool, optional): Whether to print detailed progress information. Defaults to True.
        overwrite (bool, optional): Whether to overwrite existing coregistered DEMs. Defaults to False.

    Returns:
        None: The function processes all DEMs in the directory and saves the coregistered outputs.
    """

    proc_dir = SubProcessingDirectory(base_dir)
    future_dems = {}

    ref_dem_path = proc_dir.get_reference_dem()
    ref_dem_mask_path = proc_dir.get_reference_dem_mask()

    for file in proc_dir.get_raw_dems():
        try:
            code, _ = parse_filename(str(file))

            output_dem_path = proc_dir.coreg_dems_dir / f"{code}-DEM.tif"

            if output_dem_path.exists() and not overwrite:
                if verbose:
                    print(f"Skip coregistration for {code}: output already exists.")
                continue

            future_dems[code] = coregister_dem.with_options(name=f"coreg_{code}").submit(
                file, ref_dem_path, ref_dem_mask_path, output_dem_path, overwrite
            )

        except Exception as e:
            print(f"Error {file.name} : {e}")
            continue

    for code, f in future_dems.items():
        try:
            f.result()
        except Exception as e:
            print(f"Coregistration error for {code}: {e}")


@flow(log_prints=True)
def process_generate_ddems(base_dir: str | Path, verbose: bool = True, overwrite: bool = False) -> None:
    """
    Generates differential DEMs (dDEMs) before and after coregistration for all DEMs in a processing directory.

    This flow creates dDEMs by subtracting each DEM (raw and coregistered) from the reference DEM.
    It processes all DEMs found in the pipeline directory structure, organizes the outputs in
    dedicated folders for before and after coregistration, and supports asynchronous parallel execution.

    Args:
        base_dir (str | Path): Base directory containing raw and coregistered DEMs along with the reference DEM.
        verbose (bool, optional): Whether to print detailed process information. Defaults to True.
        overwrite (bool, optional): Whether to overwrite existing dDEM outputs. Defaults to False.

    Returns:
        None: The function generates and saves dDEMs for all matching DEMs in the directory.
    """

    proc_dir = SubProcessingDirectory(base_dir)
    files_mapping = {
        "ddem_before": (proc_dir.get_raw_dems(), proc_dir.ddems_before_dir),
        "ddem_after": (proc_dir.get_coreg_dems(), proc_dir.ddems_after_dir),
    }

    future_dems = []

    ref_dem_path = proc_dir.get_reference_dem()

    for name, (files, output_dir) in files_mapping.items():
        for f in files:
            try:
                code, _ = parse_filename(f)

                output_dem_path = output_dir / f"{code}-DDEM.tif"

                if output_dem_path.exists() and not overwrite:
                    if verbose:
                        print(f"Skip {name} for {code}: output already exists.")
                    continue

                future_dems.append(
                    generate_ddem.with_options(name=f"{name}_{code}").submit(
                        f, ref_dem_path, output_dem_path, overwrite
                    )
                )

            except Exception as e:
                print(f"Error {f.name} : {e}")
                continue

    for f in future_dems:
        try:
            f.result()
        except Exception as e:
            print(f"ddem error: {e}")


@flow(log_prints=True)
def process_compute_statistics(base_dir: str | Path, nmad_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Computes and aggregates raster and point cloud statistics for all datasets in a processing directory.

    This flow automates the computation of descriptive statistics for all DEMs (raw, coregistered,
    and dDEMs) as well as metadata extraction for point clouds. It stores all results in a unified
    DataFrame and applies an inlier filter based on the NMAD distribution to flag consistent datasets.

    Args:
        base_dir (str | Path): Base directory containing the processing structure (DEMs, point clouds, etc.).
        nmad_multiplier (float, optional): Multiplier applied to the NMAD value for filtering inliers.
            Defaults to 3.0.

    Returns:
        pd.DataFrame: A DataFrame containing computed statistics, point cloud metadata,
        and coregistration parameters for all datasets.
    """

    proc_dir = SubProcessingDirectory(base_dir)
    df = proc_dir.get_filepaths_df()

    dem_colnames = ["raw_dem_file", "coreg_dem_file", "ddem_before_file", "ddem_after_file"]
    future_stats_dict = {}

    # Compute raster stats
    for colname in dem_colnames:
        prefix = colname.replace("file", "")
        for code, dem_file in df[colname].dropna().items():
            stats = get_raster_statistics(dem_file)
            if stats is None:
                future_stats_dict[(code, prefix)] = compute_raster_statistics.with_options(
                    name=f"compute_stats_{prefix}_{code}"
                ).submit(dem_file)
            else:
                for key, value in stats.items():
                    df.at[code, prefix + key] = value

    # Retrieve results from futures
    for (code, prefix), fut in future_stats_dict.items():
        try:
            stats = fut.result()
            for key, value in stats.items():
                df.at[code, prefix + key] = value
        except Exception as e:
            print(f"[WARNING] Could not compute stats for {prefix} {code}: {e}")

    # Pointcloud metadata
    for code, pc_file in df["pointcloud_file"].dropna().items():
        for key, value in get_pointcloud_metadatas(pc_file).items():
            df.at[code, "pointcloud_" + key] = value

    # coregistration shifts
    for code, dem_file in df["coreg_dem_file"].dropna().items():
        for key, value in get_raster_coregistration_shifts(dem_file).items():
            df.at[code, key] = value

    # add a inliers filter based on ddems_after_coreg
    if "ddem_after_nmad" in df.columns and df["ddem_after_nmad"].notna().any():
        nmad = df["ddem_after_nmad"].dropna()
        threshold = nmad.median() + nmad_multiplier * gu.stats.nmad(nmad)
        df["inliers"] = df["ddem_after_nmad"] <= threshold
    else:
        df["inliers"] = pd.NA
    nmad = df["ddem_after_nmad"]

    df.to_csv(proc_dir.statistics_file, index=True)

    print(f"✅ Statistics DataFrame saved to: {proc_dir.statistics_file}")
    return df


@flow(log_prints=True)
def process_compute_landcover_statistics(base_dir: str | Path) -> None:
    """
    Computes landcover-based statistics for all coregistered DEMs in a processing directory.

    This flow calculates descriptive statistics (mean, median, quartiles, NMAD, etc.) for each
    landcover class within every coregistered DEM, using a reference landcover raster. Results
    are collected asynchronously, flattened into a unified table, and exported as a CSV file.

    Args:
        base_dir (str | Path): Base directory containing coregistered DEMs and the reference landcover raster.

    Returns:
        None: The function saves a CSV file containing landcover-based statistics for all DEMs.
    """

    proc_dir = SubProcessingDirectory(base_dir)
    landcover_path = proc_dir.get_reference_landcover()
    stats_dict = {}
    future_stats_dict = {}

    for f in proc_dir.get_coreg_dems():
        try:
            code, metadata = parse_filename(f)
            key = (code, metadata["site"], metadata["dataset"])
            stats = get_raster_statistics_by_landcover(f)
            if stats is None:
                future_stats_dict[key] = compute_raster_statistics_by_landcover.with_options(
                    name=f"landcover_stat_{code}"
                ).submit(f, landcover_path)
            else:
                stats_dict[key] = stats
        except Exception as e:
            print(f"Error {f.name} : {e}")
            continue

    for key, future in future_stats_dict.items():
        try:
            stats_dict[key] = future.result()
        except Exception as e:
            print(f"[WARNING] Could not compute stats for {key[0]}: {e}")

    # Step 3: flatten results
    records = []
    for (code, site, dataset), stats in stats_dict.items():
        records += [{"code": code, "site": site, "dataset": dataset, **elem} for elem in stats]

    df = pd.DataFrame(records)
    df.to_csv(proc_dir.landcover_statistics_file, index=False)


@flow(log_prints=True)
def process_generate_std_dems(base_dir: str | Path, overwrite: bool = False) -> None:
    """
    Generates standard deviation DEMs (std DEMs) from sets of raw and coregistered DEMs.

    This flow computes pixel-wise standard deviation maps across multiple DEMs to assess
    terrain consistency and variability. It produces separate std DEMs for all datasets and
    for inlier-only subsets (based on NMAD filtering). The function supports conditional
    overwriting and parallel task execution.

    Args:
        base_dir (str | Path): Base directory containing processed DEMs and statistics.
        overwrite (bool, optional): Whether to overwrite existing std DEMs. Defaults to False.

    Returns:
        None: The function saves the generated std DEMs to the standard deviation directory.
    """

    proc_dir = SubProcessingDirectory(base_dir)
    df = proc_dir.get_statistics()
    futures = []

    colnames = ["raw_dem_file", "coreg_dem_file"]

    for colname in colnames:
        prefix = colname.replace("dem_file", "")

        # Prepare the two DEM sets: all and inliers
        dem_sets = {
            "": df[colname].dropna().to_list(),
            "_inliers": df.loc[df["inliers"], colname].dropna().to_list(),
        }

        for suffix, dem_files in dem_sets.items():
            std_dem_file = proc_dir.std_dems_dir / f"{prefix}std_dem{suffix}.tif"

            # Skip if there are not enough DEMs
            if len(dem_files) <= 1:
                warnings.warn(
                    f"Only {len(dem_files)} DEM file(s) for '{colname}{suffix}'. "
                    f"Skipping generation of {std_dem_file.name}.",
                    UserWarning,
                )
                continue

            # Skip if std_dem already exists and overwrite=False
            if is_existing_std_dem(dem_files, std_dem_file) and not overwrite:
                print(f"Skip {std_dem_file.name}: output already exists.")
                continue

            # Otherwise, submit creation task
            futures.append(create_std_dem.submit(dem_files, std_dem_file))

    # Wait for all tasks to complete
    for f in futures:
        try:
            f.result()
        except Exception as e:
            print(f"Error: {e}")


@flow(log_prints=True)
def process_compute_landcover_statistics_on_std_dems(base_dir: str | Path) -> None:
    proc_dir = SubProcessingDirectory(base_dir)

    landcover_path = proc_dir.get_reference_landcover()

    future_stats_dict = {}
    stats_dict = {}
    for dem_type in ["coreg", "raw"]:
        for inliers in [True, False]:
            try:
                dem_file = proc_dir.std_dems_dir / f"{dem_type}_std_dem.tif"
                if inliers:
                    dem_file = dem_file.with_stem(dem_file.stem + "_inliers")
                key = (dem_type, inliers)

                stats = get_raster_statistics_by_landcover(dem_file)

                if stats is None:
                    future_stats_dict[key] = compute_raster_statistics_by_landcover.with_options(
                        name=f"landcover_stat_{dem_file.stem}"
                    ).submit(dem_file, landcover_path)
                else:
                    stats_dict[key] = stats

            except Exception as e:
                print(f"Error {dem_file.name} : {e}")
                continue

    for key, future in future_stats_dict.items():
        try:
            stats_dict[key] = future.result()
        except Exception as e:
            print(f"[WARNING] Could not compute stats for {key}: {e}")

    # Step 3: flatten results
    records = []
    for (dem_type, inliers), stats in stats_dict.items():
        records += [
            {"dem_type": dem_type, "inliers": inliers, "site": proc_dir.site, "dataset": proc_dir.dataset, **elem}
            for elem in stats
        ]
    if records:
        df = pd.DataFrame(records)
        df.to_csv(proc_dir.std_landcover_statistics_file, index=False)
        print(f"Saved std landcover stats to {proc_dir.std_landcover_statistics_file}.")
    else:
        print("No landcover statistics were computed — no std_dem files available or all computations failed.")


@flow(log_prints=True)
def generate_postprocessing_plots(input_dir: str | Path, output_dir: str | Path) -> None:
    """
    Generates all post-processing plots and visual summaries for DEM and point cloud analysis results.

    This flow consolidates statistics and visualization tasks from multiple processing directories
    (site/dataset pairs) and produces a full suite of plots, including global summaries, per-dataset
    comparisons, and mosaics. It visualizes DEM quality, NMAD before/after coregistration, coregistration
    shifts, landcover-based metrics, and slope/hillshade mosaics for better spatial interpretation.

    Args:
        input_dir (str | Path): Root directory containing processed datasets organized by site and dataset.
        output_dir (str | Path): Directory where all generated plots and summaries will be saved.

    Returns:
        None: The function saves a comprehensive set of plots (global, per-site, and per-dataset)
        in the specified output directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    proc_dirs = {
        (site_dir.name, dataset_dir.name): SubProcessingDirectory(dataset_dir)
        for site_dir in input_dir.iterdir()
        for dataset_dir in site_dir.iterdir()
    }

    stats_dict = {key: proc_dir.get_statistics() for key, proc_dir in proc_dirs.items()}
    lc_stats_dict = {key: proc_dir.get_landcover_statistics() for key, proc_dir in proc_dirs.items()}
    std_lc_stats_dict = {
        key: proc_dir.get_std_landcover_statistics()
        for key, proc_dir in proc_dirs.items()
        if proc_dir.std_landcover_statistics_file.exists()
    }
    global_stats_df = pd.concat(stats_dict.values())
    global_std_lc_stats_df = pd.concat(std_lc_stats_dict.values())
    global_std_lc_stats_df = global_std_lc_stats_df[global_std_lc_stats_df["dem_type"] == "coreg"]
    global_std_lc_stats_df_inliers = global_std_lc_stats_df[global_std_lc_stats_df["inliers"]]

    # ======================================================================================
    #                               GENERATE GLOBAL PLOTS
    # ======================================================================================

    viz.barplot_var(global_stats_df, output_dir / "pointcloud_point_count.png", "pointcloud_point_count", "Point count")
    viz.barplot_var(
        global_stats_df,
        output_dir / "nmad_after_coregistration.png",
        "ddem_after_nmad",
        "NMAD of dDEM after coregistration",
    )
    viz.barplot_var(
        global_stats_df, output_dir / "raw_dem_voids.png", "raw_dem_percent_nodata", "Raw DEM nodata percent"
    )
    viz.generate_landcover_grouped_boxplot_from_std_dems(
        global_std_lc_stats_df, output_dir / "landcover_boxplot_from_std_dems.png"
    )
    viz.generate_landcover_grouped_boxplot_from_std_dems(
        global_std_lc_stats_df_inliers, output_dir / "landcover_boxplot_from_std_dems_inliers.png"
    )

    # ======================================================================================
    #                            GENERATE STATS PLOTS FOR EACH GROUP
    # ======================================================================================
    for (site, dataset), stats in stats_dict.items():
        sub_dir = output_dir / site / dataset
        viz.generate_plot_nmad_before_vs_after(stats, sub_dir / "nmad_before_vs_after_coregistration.png")
        viz.generate_plot_coreg_shifts(stats, sub_dir / "coregistration_shifts.png")

        # generate also with inliers only
        viz.generate_plot_nmad_before_vs_after(
            stats.loc[stats["inliers"]], sub_dir / "nmad_before_vs_after_coregistration_inliers.png"
        )
        viz.generate_plot_coreg_shifts(stats.loc[stats["inliers"]], sub_dir / "coregistration_shifts_inliers.png")

        # landcover plots
        lc_stats = lc_stats_dict[(site, dataset)]
        viz.generate_landcover_grouped_boxplot(lc_stats, sub_dir / "landcover_grouped_boxplot.png")
        viz.generate_landcover_boxplot(lc_stats, sub_dir / "landcover_boxplot.png")
        viz.generate_landcover_nmad(lc_stats, sub_dir / "landcover_nmad.png")

        # landcove plots inliers
        lc_stats_inliers = lc_stats[lc_stats["code"].isin(stats.index[stats["inliers"]])]
        viz.generate_landcover_grouped_boxplot(lc_stats_inliers, sub_dir / "landcover_grouped_boxplot_inliers.png")
        viz.generate_landcover_boxplot(lc_stats_inliers, sub_dir / "landcover_boxplot_inliers.png")
        viz.generate_landcover_nmad(lc_stats_inliers, sub_dir / "landcover_nmad_inliers.png")

    # ======================================================================================
    #                            GENERATE FOR EACH GROUP MOSAIC PLOTS
    # ======================================================================================
    futures = []
    for (site, dataset), stats in stats_dict.items():
        mosaic_dir = output_dir / site / dataset / "mosaic"
        coreg_output_dir = output_dir / site / dataset / "coregistration_individual_plots"

        futures.append(viz.generate_coregistration_individual_plots.submit(stats, coreg_output_dir))

        for colname in ["raw_dem_file", "coreg_dem_file"]:
            futures.append(viz.generate_dems_mosaic.submit(stats, mosaic_dir / f"mosaic_{colname[:-5]}.png", colname))

        for colname in ["ddem_before_file", "ddem_after_file"]:
            futures.append(
                viz.generate_ddems_mosaic.submit(stats, mosaic_dir / f"mosaic_{colname[:-5]}_coreg.png", colname)
            )

            futures.append(
                viz.generate_slopes_mosaic.submit(
                    stats, mosaic_dir / f"mosaic_slopes_{colname[:-5]}_coreg.png", colname
                )
            )
            futures.append(
                viz.generate_hillshades_mosaic.submit(
                    stats, mosaic_dir / f"mosaic_hillshades_{colname[:-5]}_coreg.png", colname
                )
            )

    for fut in futures:
        try:
            fut.result()
        except Exception as e:
            print(f"Error of plot generating : {e}")


def create_pointcloud_symlinks(input_dir: str | Path, output_dir: str | Path) -> None:
    """
    Finds and organizes point cloud files by creating symbolic links in a structured output directory.

    This function searches recursively for LAS and LAZ point cloud files in the input directory,
    parses their filenames to extract site and dataset metadata, and creates symbolic links for
    each point cloud under the output directory, organized by site and dataset. For each (site, dataset)
    pair, the function prints the number of point clouds found.

    Args:
        input_dir (str | Path): Root directory containing point cloud files.
        output_dir (str | Path): Directory where symbolic links will be created, structured as
            <output_dir>/<site>/<dataset>/pointclouds/.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    pointcloud_files = list(input_dir.rglob("*_dense_pointcloud.las")) + list(input_dir.rglob("*_dense_pointcloud.laz"))

    # Group pointclouds by (site, dataset)
    grouped_pointclouds = {
        (site, dataset): []
        for site in FILE_CODE_MAPPING_V1["site"].values()
        for dataset in FILE_CODE_MAPPING_V1["dataset"].values()
    }

    for pc_path in pointcloud_files:
        _, metadata = parse_filename(str(pc_path))
        site, dataset = str(metadata["site"]), str(metadata["dataset"])
        grouped_pointclouds[(site, dataset)].append(pc_path)

    # Create symlinks and print summary
    for (site, dataset), files in grouped_pointclouds.items():
        for pc_path in files:
            link = output_dir / site / dataset / "pointclouds" / pc_path.name
            link.parent.mkdir(exist_ok=True, parents=True)

            # Remove existing link if it already exists
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(pc_path)

        print(f" {site} - {dataset} → {len(files)} pointcloud(s) linked.")
