import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geoutils as gu
import pandas as pd

import history.postprocessing.visualization as viz
from history.postprocessing.io import parse_filename
from history.postprocessing.processing_directory import ProcessingDirectory

from . import core


def uncompress_all_submissions(
    input_dir: str | Path, output_dir: str | Path, overwrite: bool = False, max_workers: int | None = None
) -> None:
    """
    Uncompress all supported archive submissions from an input directory into an output directory
    using Python's ProcessPoolExecutor for parallel extraction.

    This function scans the input directory for compressed archives (ZIP, 7z, and TAR variants),
    determines their corresponding target extraction folders, and extracts each archive in
    parallel using multiple worker processes.

    Supported archive formats:
        - .zip
        - .7z
        - .tgz
        - .tar.gz
        - .tar.bz2
        - .tar.xz

    Args:
        input_dir (str | Path): Directory containing compressed submissions.
        output_dir (str | Path): Directory where the extracted folders will be created.
        overwrite (bool, optional): Overwrite the output folder if it already exists. Defaults to False.
        max_workers (int | None, optional): Maximum number of worker processes. Defaults to the CPU count.

    Returns:
        None: All extraction results are written to the filesystem.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    supported_extensions = [".zip", ".7z", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz"]

    archives = [
        (fp, output_dir / fp.name[: -len(ext)]) for ext in supported_extensions for fp in input_dir.glob(f"*{ext}")
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for input_path, output_path in archives:
            if output_path.exists() and not overwrite:
                print(f"Skipping extraction (folder exists): {output_path}")
                continue
            futures.append(executor.submit(core.extract_archive, input_path, output_path, overwrite))

        for fut in as_completed(futures):
            try:
                fut.result()  # bloque jusqu’à la fin
            except Exception as e:
                print(f"[!] Error while processing: {e}")


def index_submissions_and_link_files(input_dir: str | Path, output_dir: str | Path | ProcessingDirectory) -> None:
    """
    Index files contained in a submissions directory and create organized symbolic links
    inside the structure of a ProcessingDirectory.

    Iterates through each subfolder of `input_dir`, attempts to extract a `code` and
    metadata via `parse_filename`, collects paths for expected files (dense/sparse
    pointclouds, extrinsics/intrinsics, optional DEM), and warns if mandatory files
    are missing. For each (site, dataset) pair, creates symbolic links in the output
    directory at `output_dir.sub_dirs[(site, dataset)].symlinks_dir`. Existing links
    are overwritten.

    Args:
        input_dir (str | Path): Directory containing submissions (each submission in a subfolder).
        output_dir (str | Path | ProcessingDirectory): Processing directory or ProcessingDirectory object.

    Returns:
        None
    """
    input_dir = Path(input_dir)
    output_dir = ProcessingDirectory(output_dir)

    # Define regex patterns mapping to dataframe column names
    mandatory_patterns = {
        "dense_pointcloud_file": r"_dense_pointcloud\.(las|laz)$",
        "sparse_pointcloud_file": r"_sparse_pointcloud\.(las|laz)$",
        "extrinsics_file": r"_extrinsics\.csv$",
        "intrinsics_file": r"_intrinsics\.csv$",
    }
    optional_patterns = {"dem_file": r".*dem.*\.tif$"}

    patterns = {**mandatory_patterns, **optional_patterns}

    df = pd.DataFrame()
    df.index.name = "code"
    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue

        all_files = list(subdir.rglob("*"))

        for file in all_files:
            try:
                code, metadatas = parse_filename(file)
            except ValueError:
                continue

            for column, pattern in patterns.items():
                if re.search(pattern, file.name, re.IGNORECASE):
                    df.at[code, "submission"] = subdir.name
                    for k, v in metadatas.items():
                        df.at[code, k] = v
                    df.at[code, column] = str(file)

                    break  # Stop at first match

    for code, row in df.iterrows():
        missing_files = [c for c in mandatory_patterns if pd.isna(row[c])]
        if missing_files:
            print(f"[WARNING] {code} - Missing the following mandatory file(s) : {missing_files}")

        # creating symlinks
        site, dataset = str(row["site"]), str(row["dataset"])

        mapping = {
            "dense_pointcloud_file": output_dir.sub_dirs[(site, dataset)].symlinks_dir.dense_pointclouds_dir,
            "sparse_pointcloud_file": output_dir.sub_dirs[(site, dataset)].symlinks_dir.sparse_pointclouds_dir,
            "extrinsics_file": output_dir.sub_dirs[(site, dataset)].symlinks_dir.extrinsics_dir,
            "intrinsics_file": output_dir.sub_dirs[(site, dataset)].symlinks_dir.intrinsics_dir,
            "dem_file": output_dir.sub_dirs[(site, dataset)].symlinks_dir.raw_dems_dir,
        }

        for c, p in mapping.items():
            if not pd.isna(row[c]):
                link = p / Path(row[c]).name
                link.parent.mkdir(exist_ok=True, parents=True)

                # Remove existing link if it already exists
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(row[c])


def process_pointclouds_to_dems(
    processing_directory: str | Path | ProcessingDirectory,
    pdal_exec_path: str = "pdal",
    overwrite: bool = False,
    dry_run: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Converts all point clouds within a processing directory into DEMs for each site and dataset.

    This flow iterates over all subdirectories of the given processing directory, retrieves
    the reference DEM for each dataset, and converts each point cloud into a DEM using the
    PDAL pipeline. Tasks are executed asynchronously and logged through Prefect for monitoring.

    If a DEM already exists and `overwrite` is False, the conversion is skipped. Errors encountered
    for individual files or datasets are logged without interrupting the overall execution.

    Args:
        processing_directory (str | Path | ProcessingDirectory):
            Root processing directory containing subdirectories for each site and dataset.
        pdal_exec_path (str, optional):
            Path to the PDAL executable. Defaults to "pdal".
        overwrite (bool, optional):
            If True, existing DEMs will be overwritten. Defaults to False.
        dry_run (bool, optional):
            If True, simulate execution without performing real computations. Defaults to False.

    Returns:
        None: All generated DEMs are written under each subdirectory of the processing directory.
    """

    processing_directory = ProcessingDirectory(processing_directory)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_dems = {}

        for (site, dataset), sub_dir in processing_directory.sub_dirs.items():
            # Get the reference DEM, skip group if an exception occurs
            try:
                ref_dem_path = sub_dir.get_reference_dem()
            except Exception as e:
                print(f"[WARNING] Skip group {site} - {dataset}: {e}")
                continue

            for file in sub_dir.symlinks_dir.get_dense_pointclouds():
                try:
                    code, metadatas = parse_filename(file)

                    output_dem_path = sub_dir.raw_dems_dir / f"{code}-DEM.tif"

                    if output_dem_path.exists() and not overwrite:
                        print(f"[INFO] Skip point2dem for {code}: output already exists.")
                        continue

                    future = executor.submit(
                        core.convert_pointcloud_to_dem,
                        file,
                        ref_dem_path,
                        output_dem_path,
                        pdal_exec_path,
                        None,
                        overwrite,
                        dry_run,
                    )
                    future_dems[future] = code

                except Exception as e:
                    print(f"[ERROR] Error processing {file.name}: {e}")
                    continue

        # Wait for all point2dem tasks to finish
        for fut in as_completed(future_dems):
            code = future_dems[fut]
            try:
                fut.result()
                print(f"[OK] point2dem complete for {code}")
            except Exception as e:
                print(f"[ERROR] Point2dem error for {code}: {e}")


def process_coregister_dems(
    processing_directory: str | Path | ProcessingDirectory, overwrite: bool = False, max_workers: int | None = None
) -> None:
    """
    Coregisters all DEMs within a processing directory to their corresponding reference DEMs.

    This flow iterates over each site and dataset in the given processing directory, retrieves
    the reference DEM and its mask, and aligns all raw DEMs to this reference using the
    coregistration pipeline. Each coregistration task is submitted asynchronously and monitored
    through Prefect logging.

    If a coregistered DEM already exists and `overwrite` is False, it will be skipped.
    Errors for individual DEMs or datasets are logged without interrupting the entire process.

    Args:
        processing_directory (str | Path | ProcessingDirectory):
            Root processing directory containing subdirectories for each site and dataset.
        overwrite (bool, optional):
            If True, overwrite existing coregistered DEMs. Defaults to False.

    Returns:
        None: All generated coregistered DEMs are written under each subdirectory of the processing directory.
    """
    processing_directory = ProcessingDirectory(processing_directory)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_dems = {}

        for (site, dataset), sub_dir in processing_directory.sub_dirs.items():
            try:
                ref_dem_path = sub_dir.get_reference_dem()
                ref_dem_mask_path = sub_dir.get_reference_dem_mask()
            except Exception as e:
                print(f"[WARNING] Skip group {site} - {dataset}: {e}")
                continue

            for file in sub_dir.get_raw_dems():
                try:
                    code, _ = parse_filename(file)

                    output_dem_path = sub_dir.coreg_dems_dir / f"{code}-DEM.tif"

                    if output_dem_path.exists() and not overwrite:
                        print(f"[INFO] Skip coregister for {code}: output already exists.")
                        continue

                    future = executor.submit(
                        core.coregister_dem, file, ref_dem_path, ref_dem_mask_path, output_dem_path, overwrite
                    )
                    future_dems[future] = code

                except Exception as e:
                    print(f"[ERROR] Error processing {file.name}: {e}")
                    continue

        for fut in as_completed(future_dems):
            code = future_dems[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] Coregistration error for {code}: {e}")


def process_generate_ddems(
    processing_directory: str | Path | ProcessingDirectory, overwrite: bool = False, max_workers: int | None = None
) -> None:
    """
    Generates differential DEMs (DDEMs) before and after coregistration for each dataset in the processing directory.

    This flow processes all site/dataset subdirectories, computing DDEMs by differencing each DEM
    against its corresponding reference DEM. Two sets of DDEMs are created:
        - `ddem_before`: using raw DEMs before coregistration.
        - `ddem_after`: using coregistered DEMs after alignment.

    Each DDEM generation task is executed asynchronously and logged through Prefect.
    If a DDEM already exists and `overwrite` is False, it will be skipped. Errors encountered for
    individual files or datasets are logged without interrupting the overall workflow.

    Args:
        processing_directory (str | Path | ProcessingDirectory):
            Root processing directory containing subdirectories for each site and dataset.
        overwrite (bool, optional):
            If True, overwrite existing DDEMs. Defaults to False.

    Returns:
        None: All generated DDEMs are saved under the corresponding subdirectories.
    """

    processing_directory = ProcessingDirectory(processing_directory)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_dems = []
        for (site, dataset), sub_dir in processing_directory.sub_dirs.items():
            files_mapping = {
                "ddem_before": (sub_dir.get_raw_dems(), sub_dir.ddems_before_dir),
                "ddem_after": (sub_dir.get_coreg_dems(), sub_dir.ddems_after_dir),
            }
            try:
                ref_dem_path = sub_dir.get_reference_dem()
            except Exception as e:
                print(f"[WARNING] Skip group {site} - {dataset}: {e}")
                continue

            for name, (files, output_dir) in files_mapping.items():
                for f in files:
                    try:
                        code, _ = parse_filename(f)

                        output_dem_path = output_dir / f"{code}-DDEM.tif"

                        if output_dem_path.exists() and not overwrite:
                            print(f"[INFO] Skip {name} for {code}: output already exists.")
                            continue

                        future_dems.append(
                            executor.submit(core.generate_ddem, f, ref_dem_path, output_dem_path, overwrite)
                        )

                    except Exception as e:
                        print(f"Error {f.name} : {e}")
                        continue

        for fut in as_completed(future_dems):
            try:
                fut.result()
            except Exception as e:
                print(f"ddem error: {e}")


def process_compute_statistics(
    processing_directory: str | Path | ProcessingDirectory, nmad_multiplier: float = 3.0, max_workers: int | None = None
) -> None:
    """
    Computes statistics for all DEMs, DDEMs, and point clouds within each site and dataset in the processing directory.

    This flow performs the following operations for each subdirectory:
        1. Computes raster statistics for raw DEMs, coregistered DEMs, and DDEMs, either retrieving
           existing statistics or submitting asynchronous computation tasks.
        2. Extracts point cloud metadata for all point cloud files.
        3. Computes coregistration shifts for each coregistered DEM.
        4. Adds an inliers filter based on NMAD of the DDEMs after coregistration, using the
           specified `nmad_multiplier`.
        5. Saves the resulting statistics DataFrame to a CSV file within the subdirectory.

    All tasks are logged through Prefect. Errors in individual computations are captured without
    interrupting the processing of other datasets.

    Args:
        processing_directory (str | Path | ProcessingDirectory):
            Root directory containing site/dataset subdirectories with DEMs and point clouds.
        nmad_multiplier (float, optional):
            Multiplier applied to the NMAD to determine inlier thresholds. Defaults to 3.0.

    Returns:
        None: The computed statistics are saved as CSV files under each subdirectory's statistics file path.
    """

    processing_directory = ProcessingDirectory(processing_directory)

    # ------------------------------------------------
    # Raster statistics (parallel)
    # ------------------------------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for (site, dataset), sub_dir in processing_directory.sub_dirs.items():
            df = sub_dir.get_filepaths_df()

            dem_colnames = ["raw_dem_file", "coreg_dem_file", "ddem_before_file", "ddem_after_file"]
            future_stats_dict = {}

            # Compute raster stats
            for colname in dem_colnames:
                prefix = colname.replace("file", "")
                for code, dem_file in df[colname].dropna().items():
                    stats = core.get_raster_statistics(dem_file)
                    if stats is None:
                        fut = executor.submit(core.compute_raster_statistics, dem_file)
                        future_stats_dict[fut] = (code, prefix)
                    else:
                        for key, value in stats.items():
                            df.at[code, prefix + key] = value

            # Retrieve results from futures
            for fut in as_completed(future_stats_dict):
                code, prefix = future_stats_dict[fut]
                try:
                    stats = fut.result()
                    for key, value in stats.items():
                        df.at[code, prefix + key] = value
                except Exception as e:
                    print(f"[WARNING] Could not compute stats for {prefix} {code}: {e}")

            # ------------------------------------------------
            # Pointcloud metadata (NOT parallel)
            # ------------------------------------------------
            for code, pc_file in df["dense_pointcloud_file"].dropna().items():
                for key, value in core.get_pointcloud_metadatas(pc_file).items():
                    df.at[code, "pointcloud_" + key] = value

            # ------------------------------------------------
            # Coregistration shifts (NOT parallel)
            # ------------------------------------------------
            for code, dem_file in df["coreg_dem_file"].dropna().items():
                for key, value in core.get_raster_coregistration_shifts(dem_file).items():
                    df.at[code, key] = value

            # ------------------------------------------------
            # Compute inliers from NMAD
            # ------------------------------------------------
            if "ddem_after_nmad" in df.columns and df["ddem_after_nmad"].notna().any():
                nmad = df["ddem_after_nmad"].dropna()
                threshold = nmad.median() + nmad_multiplier * gu.stats.nmad(nmad)
                df["inliers"] = df["ddem_after_nmad"] <= threshold
            else:
                df["inliers"] = pd.NA
            nmad = df["ddem_after_nmad"]

            df.to_csv(sub_dir.statistics_file, index=True)

            print(f"[INFO] ({site} - {dataset}) Statistics DataFrame saved to: {sub_dir.statistics_file}")


def process_compute_landcover_statistics(
    processing_directory: str | Path | ProcessingDirectory, max_workers: int | None = None
) -> None:
    """
    Computes landcover-based statistics for all coregistered dDEMs within each site and dataset.

    This flow processes each subdirectory in the given processing directory by:
        1. Retrieving the reference landcover map associated with the dataset.
        2. Computing raster statistics stratified by landcover for each coregistered dDEM.
           Existing statistics are reused if available; otherwise, asynchronous computation
           tasks are submitted through Prefect.
        3. Aggregating and flattening all per-dDEM results into a single DataFrame per dataset.
        4. Saving the resulting landcover statistics as a CSV file in the corresponding subdirectory.

    All computation tasks are logged through Prefect. Errors for individual dDEMs or datasets
    are handled gracefully and do not interrupt the global workflow.

    Args:
        processing_directory (str | Path | ProcessingDirectory):
            Root directory containing site/dataset subdirectories with coregistered dDEMs.

    Returns:
        None: The computed landcover statistics are saved under each subdirectory's
        `landcover_statistics_file`.
    """

    processing_directory = ProcessingDirectory(processing_directory)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for (site, dataset), sub_dir in processing_directory.sub_dirs.items():
            try:
                landcover_path = sub_dir.get_reference_landcover()
            except Exception as e:
                print(f"[WARNING] Skip group {site} - {dataset}: {e}")
                continue

            stats_dict = {}
            future_stats_dict = {}

            for f in sub_dir.get_ddems_after():
                try:
                    code, metadata = parse_filename(f)
                    key = (code, metadata["site"], metadata["dataset"])
                    stats = core.get_raster_statistics_by_landcover(f)
                    if stats is None:
                        fut = executor.submit(core.compute_raster_statistics_by_landcover, f, landcover_path)
                        future_stats_dict[fut] = key
                    else:
                        stats_dict[key] = stats
                except Exception as e:
                    print(f"[ERROR] Error {f.name} : {e}")
                    continue

            for fut in as_completed(future_stats_dict):
                key = future_stats_dict[fut]
                try:
                    stats_dict[key] = fut.result()
                except Exception as e:
                    print(f"[ERROR] Could not compute stats for {key[0]}: {e}")

            # Step 3: flatten results
            records = []
            for (code, site, dataset), stats in stats_dict.items():
                records += [{"code": code, "site": site, "dataset": dataset, **elem} for elem in stats]

            df = pd.DataFrame(records)
            df.to_csv(sub_dir.landcover_statistics_file, index=False)
            print(
                f"[INFO] ({site} - {dataset}) Landcover statistics DataFrame saved to: {sub_dir.landcover_statistics_file}"
            )


def process_generate_std_dems(
    processing_directory: str | Path | ProcessingDirectory, overwrite: bool = False, max_workers: int | None = None
) -> None:
    """
    Generates standard deviation DEMs (std DEMs) for raw and coregistered DEMs within each site and dataset.

    This flow processes each subdirectory in the given processing directory by:
        1. Retrieving the statistics DataFrame to identify available DEM files.
        2. Preparing two DEM sets for each type: all DEMs and inliers-only DEMs.
        3. Computing standard deviation DEMs for each set, skipping sets with insufficient DEMs
           or if the output already exists and `overwrite` is False.
        4. Submitting asynchronous tasks to generate the std DEMs and logging progress via Prefect.

    Warnings are issued for sets with insufficient DEMs, and errors in individual tasks
    are captured without halting the overall workflow.

    Args:
        processing_directory (str | Path | ProcessingDirectory):
            Root directory containing site/dataset subdirectories with DEMs and statistics.
        overwrite (bool, optional):
            If True, existing std DEMs will be overwritten. Defaults to False.

    Returns:
        None: All generated standard deviation DEMs are saved under each subdirectory's `std_dems_dir`.
    """

    processing_directory = ProcessingDirectory(processing_directory)
    futures = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for (site, dataset), sub_dir in processing_directory.sub_dirs.items():
            df = sub_dir.get_statistics()

            colnames = ["raw_dem_file", "coreg_dem_file"]

            for colname in colnames:
                prefix = colname.replace("dem_file", "")

                # Prepare the two DEM sets: all and inliers
                dem_sets = {
                    "all": df[colname].dropna().to_list(),
                    "inliers": df.loc[df["inliers"], colname].dropna().to_list(),
                }

                for suffix, dem_files in dem_sets.items():
                    std_dem_file = sub_dir.std_dems_dir / f"{prefix}std_dem_{suffix}.tif"

                    # Skip if there are not enough DEMs
                    if len(dem_files) <= 1:
                        print(
                            f"[WARNING] ({site} - {dataset}) "
                            f"Only {len(dem_files)} DEM file(s) for '{colname} {suffix}'. "
                            f"Skipping generation of {std_dem_file.name}."
                        )
                        continue

                    # Skip if std_dem already exists and overwrite=False
                    if core.is_existing_std_dem(dem_files, std_dem_file) and not overwrite:
                        print(f"[INFO] ({site} - {dataset}) Skip {std_dem_file.name}: output already exists.")
                        continue

                    # Otherwise, submit creation task
                    futures.append(executor.submit(core.create_std_dem, dem_files, std_dem_file))

        # Wait for all tasks to complete
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[EROOR] Error : {e}")


def process_compute_landcover_statistics_on_std_dems(
    processing_directory: str | Path | ProcessingDirectory, max_workers: int | None = None
) -> None:
    """
    Computes landcover-based statistics for standard deviation DEMs (std DEMs) within each site and dataset.

    This flow iterates over each subdirectory in the processing directory and performs the following steps:
        1. Retrieves the reference landcover map for the dataset.
        2. For both raw and coregistered DEM types, and for each subset ('all' and 'inliers'),
           computes raster statistics stratified by landcover. Existing statistics are reused
           when available; otherwise, asynchronous tasks are submitted.
        3. Aggregates and flattens the results into a single DataFrame per dataset.
        4. Saves the resulting std landcover statistics as a CSV file in the corresponding subdirectory.

    All tasks and errors are logged through Prefect. If no std DEMs are available or all computations
    fail, a warning is issued without interrupting the workflow.

    Args:
        processing_directory (str | Path | ProcessingDirectory):
            Root directory containing site/dataset subdirectories with standard deviation DEMs.

    Returns:
        None: The computed std DEM landcover statistics are saved under each subdirectory's
        `std_landcover_statistics_file`.
    """

    processing_directory = ProcessingDirectory(processing_directory)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for (site, dataset), sub_dir in processing_directory.sub_dirs.items():
            try:
                landcover_path = sub_dir.get_reference_landcover()
            except Exception as e:
                print(f"[WARNING] Skip group {site} - {dataset}: {e}")

            future_stats_dict = {}
            stats_dict = {}

            for dem_type in ["coreg", "raw"]:
                for subset in ["all", "inliers"]:
                    try:
                        dem_file = sub_dir.std_dems_dir / f"{dem_type}_std_dem_{subset}.tif"
                        key = (dem_file, dem_type, subset)

                        stats = core.get_raster_statistics_by_landcover(dem_file)

                        if stats is None:
                            fut = executor.submit(core.compute_raster_statistics_by_landcover, dem_file, landcover_path)
                            future_stats_dict[fut] = key
                        else:
                            stats_dict[key] = stats

                    except Exception as e:
                        print(f"[ERROR] ({site} - {dataset}) Error {dem_file.name} : {e}")
                        continue

            for fut in as_completed(future_stats_dict):
                key = future_stats_dict[fut]
                try:
                    stats_dict[key] = fut.result()
                except Exception as e:
                    print(f"[ERROR] ({site} - {dataset}) Could not compute stats for {key}: {e}")

            # Step 3: flatten results
            records = []
            for (dem_file, dem_type, subset), stats in stats_dict.items():
                records += [
                    {
                        "dem_file": dem_file,
                        "dem_type": dem_type,
                        "subset": subset,
                        "site": sub_dir.site,
                        "dataset": sub_dir.dataset,
                        **elem,
                    }
                    for elem in stats
                ]
            if records:
                df = pd.DataFrame(records)
                df.to_csv(sub_dir.std_landcover_statistics_file, index=False)
                print(
                    f"[INFO] ({site} - {dataset}) Saved std landcover stats to {sub_dir.std_landcover_statistics_file}."
                )
            else:
                print(
                    f"[WARNING] ({site} - {dataset}) No landcover statistics were computed — no std_dem files available or all computations failed."
                )


def generate_postprocessing_plots(
    input_dir: str | Path | ProcessingDirectory, output_dir: str | Path, max_workers: int | None = None
) -> None:
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

    input_dir = ProcessingDirectory(input_dir)
    output_dir = Path(output_dir)

    global_stats_df = input_dir.get_statistics()
    global_lc_stats_df = input_dir.get_landcover_statistics()
    global_std_lc_stats_df = input_dir.get_std_landcover_statistics()

    # process the global_std_lc_stats_df
    global_std_lc_stats_df = global_std_lc_stats_df.loc[global_std_lc_stats_df["dem_type"] == "coreg"]
    global_std_lc_stats_df_all = global_std_lc_stats_df.loc[global_std_lc_stats_df["subset"] == "all"]
    global_std_lc_stats_df_inliers = global_std_lc_stats_df.loc[global_std_lc_stats_df["subset"] == "inliers"]

    # ======================================================================================
    #                               GENERATE GLOBAL PLOTS
    # ======================================================================================

    viz.barplot_var(
        global_stats_df,
        output_dir / "pointcloud_point_count.png",
        "pointcloud_point_count",
        "Point count in dense point-cloud file",
    )
    viz.barplot_var(
        global_stats_df,
        output_dir / "nmad_after_coregistration.png",
        "ddem_after_nmad",
        "NMAD of Altitude differences with ref DEM after coregistration by code",
    )
    viz.barplot_var(
        global_stats_df, output_dir / "raw_dem_voids.png", "raw_dem_percent_nodata", "Raw DEM nodata percent"
    )
    viz.generate_landcover_grouped_boxplot_from_std_dems(
        global_std_lc_stats_df_all, output_dir / "landcover_boxplot_from_std_dems.png"
    )
    viz.generate_landcover_grouped_boxplot_from_std_dems(
        global_std_lc_stats_df_inliers, output_dir / "landcover_boxplot_from_std_dems_inliers.png"
    )

    # ======================================================================================
    #                            GENERATE STATS PLOTS FOR EACH GROUP
    # ======================================================================================
    for (site, dataset), stats in global_stats_df.groupby(["site", "dataset"]):
        sub_dir = output_dir / site / dataset
        title_prefix = f"({site} - {dataset})"
        viz.generate_plot_nmad_before_vs_after(
            stats,
            sub_dir / "nmad_before_vs_after_coregistration.png",
            title=f"{title_prefix} NMAD of DEM differences before vs after coregistration",
        )
        viz.generate_plot_coreg_shifts(
            stats,
            sub_dir / "coregistration_shifts.png",
            title=f"{title_prefix} Coregistration shifts",
        )

        # generate also with inliers only
        viz.generate_plot_nmad_before_vs_after(
            stats.loc[stats["inliers"]],
            sub_dir / "nmad_before_vs_after_coregistration_inliers.png",
            title=f"{title_prefix} NMAD of DEM differences before vs after coregistration (inliers only)",
        )
        viz.generate_plot_coreg_shifts(
            stats.loc[stats["inliers"]],
            sub_dir / "coregistration_shifts_inliers.png",
            title=f"{title_prefix} Coregistration shifts (inliers only)",
        )

        # landcover plots
        lc_stats = global_lc_stats_df.loc[
            (global_lc_stats_df["site"] == site) & (global_lc_stats_df["dataset"] == dataset)
        ]
        viz.generate_landcover_grouped_boxplot(
            lc_stats,
            sub_dir / "landcover_grouped_boxplot.png",
            title=f"{title_prefix} Boxplot of Altitude difference with ref DEM by code/landcover",
        )
        # viz.generate_landcover_boxplot(lc_stats, sub_dir / "landcover_boxplot.png")
        viz.generate_landcover_nmad(
            lc_stats,
            sub_dir / "landcover_nmad.png",
            title=f"{title_prefix} NMAD of Altitude difference with ref DEM by code/landcover",
        )

        # landcove plots inliers
        lc_stats_inliers = lc_stats[lc_stats["code"].isin(stats.index[stats["inliers"]])]
        viz.generate_landcover_grouped_boxplot(
            lc_stats_inliers,
            sub_dir / "landcover_grouped_boxplot_inliers.png",
            title=f"{title_prefix} Boxplot of Altitude difference with ref DEM by code/landcover (inliers only)",
        )
        # viz.generate_landcover_boxplot(lc_stats_inliers, sub_dir / "landcover_boxplot_inliers.png")
        viz.generate_landcover_nmad(
            lc_stats_inliers,
            sub_dir / "landcover_nmad_inliers.png",
            title=f"{title_prefix} NMAD of Altitude difference with ref DEM by code/landcover (inliers only)",
        )

    # ======================================================================================
    #                            GENERATE FOR EACH GROUP STD DEMS
    # ======================================================================================
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for (dem_file, site, dataset), stats in global_std_lc_stats_df.groupby(["dem_file", "site", "dataset"]):
            output_path = output_dir / site / dataset / Path(dem_file).with_suffix(".png").stem
            futures.append(executor.submit(viz.generate_std_dem_plots, dem_file, output_path))

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Error of plot generating : {e}")

    # ======================================================================================
    #                            GENERATE FOR EACH GROUP MOSAIC PLOTS
    # ======================================================================================

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for (site, dataset), stats in global_stats_df.groupby(["site", "dataset"]):
            group_dir = output_dir / site / dataset
            mosaic_dir = group_dir / "mosaic"
            coreg_output_dir = group_dir / "coregistration_individual_plots"
            title_prefix = f"({site} - {dataset})"

            # add the individual coregistration plots generation
            futures.append(executor.submit(viz.generate_coregistration_individual_plots, stats, coreg_output_dir))

            for colname in ["raw_dem_file", "coreg_dem_file"]:
                futures.append(
                    executor.submit(
                        viz.generate_dems_mosaic,
                        stats,
                        mosaic_dir / f"mosaic_{colname[:-5]}.png",
                        colname,
                        title=f"{title_prefix} Mosaic {colname}",
                    )
                )

            for colname in ["ddem_before_file", "ddem_after_file"]:
                futures.append(
                    executor.submit(
                        viz.generate_ddems_mosaic,
                        stats,
                        mosaic_dir / f"mosaic_{colname[:-5]}_coreg.png",
                        colname,
                        title=f"{title_prefix} Mosaic {colname}",
                    )
                )

                futures.append(
                    executor.submit(
                        viz.generate_slopes_mosaic,
                        stats,
                        mosaic_dir / f"mosaic_slopes_{colname[:-5]}_coreg.png",
                        colname,
                        title=f"{title_prefix} Mosaic slopes {colname}",
                    )
                )
                futures.append(
                    executor.submit(
                        viz.generate_hillshades_mosaic,
                        stats,
                        mosaic_dir / f"mosaic_hillshades_{colname[:-5]}_coreg.png",
                        colname,
                        title=f"{title_prefix} Mosaic hillshades {colname}",
                    )
                )

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Error of plot generating : {e}")
