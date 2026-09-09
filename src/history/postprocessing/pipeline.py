"""
Module for processing and organizing DEM and point cloud submissions in a reproducible
evaluation workflow.

This module provides high-level utilities to:
- Extract and clean compressed submission archives.
- Index submission folders, parse metadata from filenames, and create structured
  symbolic-link directories.
- Convert point clouds to DEMs using PDAL, including CRS detection and alignment to
  reference DEMs.
- Integrate externally provided DEMs by reprojecting them onto reference grids.
- Coregister DEMs using Nuth–Kaab horizontal shifts and vertical shift correction.
- Generate differential DEMs (dDEMs) and standard-deviation DEMs from multiple DEM inputs.
- Inspect existing STD DEMs via embedded metadata and infer their associated source files.

Most functions support parallel execution through ``ThreadPoolExecutor`` and are
designed to fail gracefully: errors are logged without interrupting batch processing.
All I/O operations rely on ``geoutils``, ``rasterio``, ``laspy``, and related geospatial
libraries, ensuring consistent handling of CRS, raster grids, and metadata.
"""

from collections import defaultdict
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
import geoutils as gu
from history.postprocessing.config import Config
import humanize
import laspy
import numpy as np
import pandas as pd
import py7zr
import rasterio
import xdem
from pyproj import CRS as ProjCRS
from pyproj import Transformer
from rasterio.windows import Window
from shapely import box, transform
from tqdm import tqdm

import history.postprocessing.io as io
import history.postprocessing.sankey as sankey
import history.postprocessing.statistics as stats
import history.postprocessing.visualization as viz
from history.postprocessing.io import ReferencesData, is_output_up_to_date, parse_filename

logger = logging.getLogger(__name__)

#######################################################################################################################
##                                                  MAIN FUNCTIONS
#######################################################################################################################


def uncompress_all_submissions(
    input_dir: str | Path, output_dir: str | Path, overwrite: bool = False, max_workers: int | None = None
) -> None:
    """
    Uncompress all supported archive submissions from an input directory into an output directory
    using Python's ThreadPoolExecutor for parallel extraction.

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

    args_list = []
    for input_path in input_dir.iterdir():
        if input_path.suffix in [".docx", ".pdf", ".odt"]:
            logger.debug(f"Ignoring file {input_path} which is not an archive")
            continue

        output_path = output_dir / input_path.name.split(".")[0]
        if not overwrite and is_output_up_to_date(input_path, output_path):
            logger.info(f"Skipping extraction (folder up to date): {output_path}")
            continue
        args_list.append((input_path, output_path))

    if not args_list:
        return

    __estimate_extraction_time([f for f, _ in args_list], max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_archive, *args): args for args in args_list}

        for fut in tqdm(as_completed(futures), desc="Extraction", total=len(futures)):
            input_path, output_path = futures[fut]
            try:
                fut.result()
                logger.info(f"OK: {input_path} extracted to {output_path}.")
            except Exception as e:
                logger.error(f"Extraction error for {input_path}: {e}")
                continue


def create_symlinks(df: pd.DataFrame, output_dir: str | Path, overwrite: bool = False) -> None:
    """Create typed symlink directories from a scanned submissions DataFrame.

    Each file column in *df* is mapped to a subdirectory under *output_dir*
    (e.g. ``dense_pointcloud_file`` → ``dense_pointclouds/``). Existing symlinks at
    the target path are replaced without following them (avoids permission errors on
    inaccessible mounts). Non-symlink files at the target path are left untouched.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`io.scan_submissions`, indexed by submission code.
    output_dir:
        Root directory where symlink subdirectories will be created.
    overwrite:
        If True, *output_dir* is deleted entirely before creating new links.
    """
    output_dir = Path(output_dir)

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    for _code, row in df.iterrows():
        for col, subdir_name in io._FILE_COL_TO_SUBDIR.items():
            if col not in row or pd.isna(row[col]):
                continue
            name_col = col.removesuffix("_file") + "_name"
            link_name = row[name_col] if (name_col in row and pd.notna(row.get(name_col))) else Path(row[col]).name
            link = output_dir / subdir_name / link_name
            link.parent.mkdir(exist_ok=True, parents=True)
            if link.is_symlink():
                link.unlink()
            link.symlink_to(row[col])


def report_symlinks(config: Config) -> None:
    """Scan extracted_dir and raw_dir to create symlinks for all report founds"""
    output_dir = config.proc_dir.symlinks_dir / "reports"

    # remove old symlinks dir and recreate it
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # find all report 
    files = list(config.extracted_dir.rglob("*report*")) + list(config.raw_dir.rglob("*report*"))

    # link all files
    for f in files:
        link = output_dir / f.name
        n = 1
        while link.exists():
            link = output_dir / f"{f.stem} ({n}){f.suffix}"
            n += 1
        link.symlink_to(f)

    # extensions
    extensions = list(set(f.suffix for f in files))

    logger.info(f"Saved {len(files)} reports to {output_dir} ({len(extensions)} format(s): {', '.join(extensions) or 'none'})")
        

def index_submissions_and_link_files(input_dir: str | Path, output_dir: str | Path, overwrite: bool = False) -> None:
    """Scan, validate, and create symlinks for all submissions in *input_dir*.

    Convenience wrapper combining :func:`io.scan_submissions`,
    :func:`io.validate_submissions`, and :func:`create_symlinks`.
    Kept for backwards compatibility with existing notebooks.
    """
    df = io.scan_submissions(input_dir)
    io.validate_submissions(df)
    create_symlinks(df, output_dir, overwrite=overwrite)


def check_planned_submissions(
    submission_codes: list[str],
    planned_outfile: str | Path,
) -> None:
    """Compare received submission codes against the planned list from the shared Google Sheet.

    Downloads and updates the planned-submissions CSV at *planned_outfile*, then logs
    which planned submissions are missing and which received submissions were unplanned.

    Parameters
    ----------
    submission_codes:
        Experiment codes to compare (e.g. ``df.index.tolist()`` from :func:`io.scan_submissions`).
    planned_outfile:
        Path where the downloaded/updated planned-submissions CSV is saved.
    """
    planned_outfile = Path(planned_outfile)
    planned_df = io.download_planned_submissions(planned_outfile)

    set_received = set(submission_codes)
    set_planned = set(planned_df["Submission code"])

    planned_df["received"] = planned_df["Submission code"].map(lambda x: x in set_received)

    unsubmitted = planned_df.loc[~planned_df["received"], "Submission code"].tolist()
    unplanned = sorted(set_received - set_planned)

    logger.info(f"{len(planned_df) - len(unsubmitted)}/{len(planned_df)} planned submissions received")

    if unsubmitted:
        logger.warning(f"{len(unsubmitted)} planned submissions not yet received:")
        for code in sorted(unsubmitted):
            logger.info(f"  {code}")

    if unplanned:
        logger.warning(f"{len(unplanned)} unplanned submissions received:")
        for code in unplanned:
            logger.info(f"  {code}")

    planned_df.to_csv(planned_outfile)


def plot_planned_submissions(config: Config, planned_outfile: str | Path) -> None:
    """Generate a Sankey diagram of planned submissions by software, dataset, and georef strategy."""
    planned_outfile = Path(planned_outfile)
    output_path = config.plot_dir / "planned_submissions_sankey.png"
    if not config.overwrite_plots and is_output_up_to_date(planned_outfile, output_path):
        logger.info(f"Skip {output_path.name}: output is up to date.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sankey.save_sankey(
        planned_outfile,
        output_path,
        columns=["software", "dataset", "georef"],
        columns_label=["Software", "Dataset", "Georef strategy"],
        force_color={"georef": sankey.light_grey},
        title="Planned Submissions",
    )
    logger.info(f"Updated planned submissions saved to {planned_outfile}.")


def save_pointcloud_diff(pointcloud_path: Path, ref_dem: gu.Raster, output_path: Path) -> None:
    """Compute the elevation difference between a point cloud and a reference DEM, and save it as a LAS file."""
    sparse_pc = gu.PointCloud(str(pointcloud_path))
    sparse_pc.reproject(ref_dem, inplace=True)

    ref_z = ref_dem.interp_points(sparse_pc, as_array=True)
    pc_diff: gu.PointCloud = sparse_pc - ref_z
    pc_diff.to_las(str(output_path))

def generate_pointcloud_diff(
    pointcloud_files: dict[str, Path],
    ref_dem: gu.Raster,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Cache, for each point cloud, a copy whose Z values hold the elevation difference with ``ref_dem``.

    Outputs are cached LAS files under ``output_dir``, one per input file, skipped when already up to date.

    Returns:
        Mapping from code to output path, for the files successfully cached.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    diff_files: dict[str, Path] = {}
    for code, f in tqdm(pointcloud_files.items(), desc="Point cloud diff"):
        output_path = output_dir / f.name

        if not overwrite and is_output_up_to_date(f, output_path):
            logger.debug(f"File {output_path} is up to date -> skipping.")
            diff_files[code] = output_path
            continue

        try:
            save_pointcloud_diff(f, ref_dem, output_path)
            diff_files[code] = output_path
        except Exception as e:
            logger.error(f"Error computing point cloud diff for {f.name}: {e}")
            continue

    return diff_files


def generate_sparse_pointcloud_viz(config: Config) -> None:
    """Cache sparse point cloud vs. reference DEM diffs and plot one mosaic per (site, dataset) group."""
    input_dir = config.proc_dir.symlinks_dir / "sparse_pointclouds"

    files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
    logger.info(f"Found {len(files)} sparse point cloud(s) to plot in {input_dir}")

    # the first step is to group all sparse point cloud files by site, dataset
    grouped_files: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for f in files:
        try:
            code, metadatas = parse_filename(f)
            grouped_files[(metadatas["site"], metadatas["dataset"])][code] = f
        except ValueError as e:
            logger.warning(f"Skipping unparseable sparse point cloud filename {f.name}: {e}")

    # then create a mosaic for each group
    for (site, dataset), pc_files_dict in grouped_files.items():
        logger.debug(f"Plotting sparse point cloud mosaic **** {site} - {dataset} **** ({len(pc_files_dict)} files)")
        ref_dem_path = config.references_data_mapping.get_ref_dem(site, dataset)
        ref_dem = gu.Raster(ref_dem_path)

        diff_files_dict = generate_pointcloud_diff(
            pc_files_dict, ref_dem, config.proc_dir.cache.pc_diff_dir, config.overwrite
        )
        output_path = config.plot_dir / f"{site}_{dataset}" / "mosaic" / "mosaic_sparse_pointcloud_diff.png"

        viz.generate_sparse_pointclouds_mosaic(
            diff_files_dict,
            output_path,
            title=f"({site} {dataset}) Mosaic of sparse point cloud \n altitude difference vs reference DEM",
            overwrite=config.overwrite_plots,
        )


def process_pointclouds_to_dems(
    pointcloud_files: list[str | Path],
    output_directory: str | Path,
    references_data: ReferencesData,
    pdal_exec_path: str = "pdal",
    overwrite: bool = False,
    dry_run: bool = False,
    max_workers: int | None = None,
    suffix: str = "-DEM",
) -> None:
    """
    Convert a list of point cloud files into DEMs using PDAL, aligning each output
    to its corresponding reference DEM.

    This function prepares and executes a set of ``convert_pointcloud_to_dem`` tasks,
    distributing them across multiple workers using a ``ThreadPoolExecutor``.
    For each point cloud file, the workflow is:

    1. Parse filename to extract ``code``, ``site``, and ``dataset``.
    2. Retrieve the matching reference DEM from ``references_data``.
    3. Build the output DEM path inside ``output_directory`` using the configured ``suffix``.
    4. Skip processing if the output DEM already exists and ``overwrite=False``.
    5. Otherwise, schedule the PDAL DEM generation task.
    6. Display progress using a ``tqdm`` progress bar, while logs are redirected in
       a way that does not break the bar (see ``TqdmLogHandler``).

    Parameters
    ----------
    pointcloud_files : list[str | Path]
        List of point cloud file paths to be converted.
    output_directory : str or Path
        Directory where the generated DEM files will be written.
    references_data : ReferencesData
        Object capable of providing the correct reference DEM for each (site, dataset).
    pdal_exec_path : str, optional
        Path to the PDAL executable. Default is ``"pdal"``.
    overwrite : bool, optional
        If ``True``, existing DEMs are re-generated. Default is ``False``.
    dry_run : bool, optional
        If ``True``, PDAL commands are constructed but not executed. Default is ``False``.
    suffix : str, optional
        Suffix appended to the output DEM filename (before ``.tif``).
        Default is ``"-DEM"``.
    max_workers : int or None, optional
        Maximum number of parallel worker threads. If ``None``, Python selects a default.

    Returns
    -------
    None
        The function performs file generation as a side effect and returns nothing.

    Notes
    -----
    - Each point cloud is processed independently and in parallel.
    - Errors for individual files are logged but do not interrupt the overall pipeline.
    - Output DEM filenames follow the pattern: ``<code><suffix>.tif``.
    - A ``tqdm`` progress bar is displayed for the parallel tasks.
    """
    output_directory = Path(output_directory)
    pointcloud_files: list[Path] = [Path(f) for f in pointcloud_files]

    args_dict = {}
    for file in pointcloud_files:
        try:
            code, metadatas = parse_filename(file)

            # get the corresponding reference DEM
            ref_dem_path = references_data.get_ref_dem(metadatas["site"], metadatas["dataset"])

            # create the output DEM path
            output_dem_path = output_directory / f"{code}{suffix}.tif"

            # avoid overwriting existing DEM
            if not overwrite and is_output_up_to_date([file, ref_dem_path], output_dem_path):
                logger.info(f"Skip point2dem for {code}: output is up to date.")
                continue

            args_dict[code] = [file, ref_dem_path, output_dem_path]

        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")
            continue

    if len(args_dict) > 0:

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(convert_pointcloud_to_dem, *args, pdal_exec_path=pdal_exec_path, dry_run=dry_run): code
                for code, args in args_dict.items()
            }

            # Wait for all point2dem tasks to finish
            for fut in tqdm(as_completed(futures), total=len(futures), desc="point2dem"):
                code = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Point2dem error for {code}: {e}")
                    continue

    # Check that no results from deleted submissions exist
    raw_dem_files = list(output_directory.glob("*-DEM.tif"))
    dem_prefixes = [f.stem[:-4] for f in raw_dem_files]
    pc_prefixes = [f.stem[:-17] for f in pointcloud_files]
    if len(dem_prefixes) != len(pc_prefixes):
        unexpected_exp = list(set(dem_prefixes) - set(pc_prefixes))
        unexpected_str = ", ".join(list(unexpected_exp))

        logger.warning(f"Found the following experiments in the raw DEM folder: {unexpected_str}")
        logger.warning("Consider deleting the following files:")

        for code in unexpected_exp:
            found_files = list(output_directory.parent.glob(f"**/*{code}*"))
            print("command: rm " + str(output_directory.parent) + f"/**/*{code}*")
            for f in found_files:
                print(f)


def save_dem_diff(dem_path: Path, ref_dem: gu.Raster, output_path: Path) -> None:
    """Compute the elevation difference between a DEM and a reference DEM, and save it as a raster."""
    dem = gu.Raster(dem_path).reproject(ref_dem)
    ddem = ref_dem - dem
    ddem.save(str(output_path))


def generate_dem_diff(
    dem_files: dict[str, Path],
    ref_dem: gu.Raster,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Cache, for each DEM, its elevation difference with ``ref_dem``.

    Outputs are cached GeoTIFF files under ``output_dir``, one per input file, skipped when
    already up to date.

    Returns:
        Mapping from code to output path, for the files successfully cached.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    diff_files: dict[str, Path] = {}
    for code, f in tqdm(dem_files.items(), desc="DEM diff"):
        output_path = output_dir / f.name

        if not overwrite and is_output_up_to_date(f, output_path):
            logger.debug(f"File {output_path} is up to date -> skipping.")
            diff_files[code] = output_path
            continue

        try:
            save_dem_diff(f, ref_dem, output_path)
            diff_files[code] = output_path
        except Exception as e:
            logger.error(f"Error computing DEM diff for {f.name}: {e}")
            continue

    return diff_files


def generate_provided_dem_viz(config: Config) -> None:
    """Cache provided DEM vs. reference DEM diffs and plot one mosaic per (site, dataset) group."""
    input_dir = config.proc_dir.symlinks_dir / "dems"

    files = list(input_dir.glob("*.tif"))
    logger.info(f"Found {len(files)} provided DEM(s) to plot in {input_dir}")

    # group all DEMs per site, dataset
    grouped_files: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for f in files:
        try:
            code, metadatas = parse_filename(f)
            grouped_files[(metadatas["site"], metadatas["dataset"])][code] = f
        except ValueError as e:
            logger.warning(f"Skipping unparseable provided DEM filename {f.name}: {e}")

    # create a mosaic for each group
    for (site, dataset), dem_files_dict in grouped_files.items():
        logger.debug(f"Plotting provided DEMs mosaic **** {site} - {dataset} **** ({len(dem_files_dict)} files)")
        ref_dem_path = config.references_data_mapping.get_ref_dem(site, dataset)
        ref_dem = gu.Raster(ref_dem_path)

        diff_files_dict = generate_dem_diff(dem_files_dict, ref_dem, config.proc_dir.cache.dem_diff_dir, config.overwrite)
        output_path = config.plot_dir / f"{site}_{dataset}" / "mosaic" / "mosaic_provided_ddem.png"

        viz.generate_provided_dems_mosaic(
            diff_files_dict,
            output_path,
            title=f"({site} {dataset}) Mosaic of provided DEM(s) \n altitude difference vs reference DEM",
            overwrite=config.overwrite_plots
        )

def add_provided_dems(
    dem_files: list[str | Path],
    output_dir: str | Path,
    references_data: ReferencesData,
    overwrite: bool = False,
    suffix: str = "-DEM",
) -> None:
    """
    Integrate externally provided DEMs into the processing workflow by
    reprojecting them onto their corresponding reference DEM grid.

    This function parses metadata from each input DEM filename, retrieves the
    appropriate reference DEM through `references_data`, reprojects the provided
    DEM onto the reference raster grid, and saves the aligned result in
    `output_dir`. Output filenames follow the pattern ``<code><suffix>.tif``.

    Existing outputs are skipped unless ``overwrite`` is ``True``. Any error
    encountered during file handling, reprojection, or saving is logged and does
    not interrupt processing of the remaining DEMs.

    Parameters
    ----------
    dem_files : list of str or Path
        List of user-provided DEM file paths to process.
    output_dir : str or Path
        Directory where reprojected DEMs will be written. Created if necessary.
    references_data : ReferencesData
        Object used to retrieve reference DEMs based on metadata extracted from
        input filenames (e.g., ``site`` and ``dataset`` fields).
    overwrite : bool, optional
        If ``True``, overwrite existing output files. If ``False`` (default),
        files already present in ``output_dir`` are skipped.
    suffix : str, optional
        Suffix appended to output filenames before the ``.tif`` extension.
        Default is ``"-DEM"``.

    Returns
    -------
    None
        This function performs processing and file generation as side effects.

    Notes
    -----
    - Filenames must comply with the conventions expected by ``parse_filename()``.
    - Reprojection is performed using ``geoutils.Raster`` operations.
    - Errors are logged individually and never halt the batch processing.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            # avoid overwriting existing files
            output_path = output_dir / f"{code}{suffix}.tif"
            if not overwrite and is_output_up_to_date(file, output_path):
                logger.info(f"Skip {code} output already exists.")
                continue

            # extract corresponding reference DEM
            ref_dem_path = references_data.get_ref_dem(metadatas["site"], metadatas["dataset"])

            # read the provided DEM and reproject it on the reference DEM
            raster = gu.Raster(file).reproject(gu.Raster(ref_dem_path))

            # save the raster
            raster.save(output_path)
            logger.info(f"DEM successfully reprojected and saved at {output_path}.")
        except Exception as e:
            logger.error(f"Error while processing {file}: {e}")
            continue


def coregister_dems(
    dem_files: Iterable[str | Path],
    output_dir: str | Path,
    references_data: ReferencesData,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Coregister a list of DEM files using their corresponding reference DEM and
    reference DEM mask.

    Each DEM is matched to its reference dataset through metadata extracted
    from its filename. A parallel execution pool is used to accelerate the
    processing. For each DEM, the function:
    - Parses the filename to retrieve ``code``, ``site``, and ``dataset``.
    - Retrieves the associated reference DEM and reference DEM mask from
      ``references_data``.
    - Creates the output DEM path inside ``output_dir``.
    - Runs the ``core.coregister_dem`` routine, unless an output already exists
      and ``overwrite`` is False.

    Parameters
    ----------
    dem_files : Iterable of str or Path
        Iterable of DEM file paths to be coregistered.
    output_dir : str or Path
        Directory where coregistered DEMs will be saved.
    references_data : ReferencesData
        Object providing reference DEMs and masks for each (site, dataset) pair.
    overwrite : bool, optional
        If ``True``, overwrite existing output DEMs. Default is ``False``.
    max_workers : int or None, optional
        Maximum number of parallel workers used by ``ThreadPoolExecutor``.
        Default uses the system's default.

    Returns
    -------
    None
        The function performs processing for its side effects (file creation)
        and does not return a value.

    Notes
    -----
    - Errors encountered for individual DEMs are logged and do not stop the
      processing of remaining files.
    - Output DEMs preserve the original filename.
    - Reference DEMs and masks are retrieved automatically based on the
      metadata parsed from each input filename.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    output_dir = Path(output_dir)

    args_dict = {}

    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            output_dem_path = output_dir / file.name

             # extract corresponding ref dem and mask with site and dataset
            ref_dem_path = references_data.get_ref_dem(metadatas["site"], metadatas["dataset"])
            ref_dem_mask_path = references_data.get_ref_dem_mask(metadatas["site"], metadatas["dataset"])

            # avoid overwriting existing files
            if not overwrite and is_output_up_to_date([file, ref_dem_path, ref_dem_mask_path], output_dem_path):
                logger.info(f"Skip coregistration for {code}, output already exists.")
                continue

            args_dict[code] = [file, ref_dem_path, ref_dem_mask_path, output_dem_path]

        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")
            continue

    # return if no coregistration needed
    if not args_dict:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for code, args in args_dict.items():
            fut = executor.submit(coregister_dem, *args)
            futures[fut] = code

        for fut in tqdm(as_completed(futures), desc="Coregistration", total=len(futures)):
            code = futures[fut]
            try:
                fut.result()
                logger.info(f"Coregistration complete for {code}")
            except Exception as e:
                logger.error(f"Coregistration error for {code}: {e}")
                continue


def generate_ddems(
    dem_files: Iterable[str | Path],
    output_dir: str | Path,
    references_data: ReferencesData,
    overwrite: bool = False,
    max_workers: int | None = None,
    suffix: str = "-DDEM",
) -> None:
    """
    Generate differential DEMs (DDEMs) for a list of DEM files by subtracting
    each DEM from its corresponding reference DEM.

    Each DEM is paired with a reference dataset according to metadata extracted
    from its filename. The processing is parallelized using ``ThreadPoolExecutor``.
    For every DEM in the input list, this function:

    - Parses the filename to extract ``code``, ``site``, and ``dataset``.
    - Retrieves the matching reference DEM through ``references_data``.
    - Builds the output filename as ``<code><suffix>.tif`` inside ``output_dir``.
    - Runs ``generate_ddem`` unless the output file already exists and
      ``overwrite`` is ``False``.

    Parameters
    ----------
    dem_files : Iterable[str or Path]
        Iterable of DEM file paths to process.
    output_dir : str or Path
        Directory where DDEM files will be written. Created if missing.
    references_data : ReferencesData
        Object that maps (site, dataset) pairs to their reference DEM files.
    overwrite : bool, optional
        If ``True``, existing output files are replaced. Default is ``False``.
    suffix : str, optional
        String appended to the ``code`` to generate the output filename.
        Default is ``"-DDEM"``.
    max_workers : int or None, optional
        Maximum number of parallel workers. Defaults to the system's choice.

    Returns
    -------
    None
        This function performs file generation as a side effect and returns nothing.

    Notes
    -----
    - Errors in individual computations are logged but do not interrupt the
      entire processing pipeline.
    - Output filenames follow the pattern: ``<code><suffix>.tif``.
    - Reference DEM selection is fully automatic based on filename metadata.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    args_dict = {}
    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            output_path = output_dir / f"{code}{suffix}.tif"

            # get corresponding reference DEM with site and dataset
            ref_dem_path = references_data.get_ref_dem(metadatas["site"], metadatas["dataset"])

            # avoid overwriting existing files
            if not overwrite and is_output_up_to_date([file, ref_dem_path], output_path):
                logger.info(f"Skip DDEM {code}, output is up to date.")
                continue

            args_dict[code] = [file, ref_dem_path, output_path]
        except Exception as e:
            logger.error(f"Error while processing {file}: {e}")

    if not args_dict:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_ddem, *args): code for code, args in args_dict.items()}

        for fut in tqdm(as_completed(futures), desc="ddem", total=len(futures)):
            code = futures[fut]
            try:
                fut.result()
                logger.info(f"DDEM successfully generated for {code}")
            except Exception as e:
                logger.error(f"DDEM error for {code}: {e}")
                continue


def create_std_dem(
    dem_files: list[str | Path],
    output_path: str | Path,
    overwrite: bool = False,
    block_size: int = 256,
    metadata_key: str = "dem_files",
) -> None:
    """
    Generates a standard deviation Digital Elevation Model (DEM) from a list of input DEM files.

    This function computes the pixel-wise standard deviation across multiple DEMs, processing
    the rasters block by block to efficiently handle large datasets.

    Args:
        dem_files (list[str | Path]): List of paths to input DEM raster files.
        output_path (str | Path): Path to the output standard deviation DEM file.
        block_size (int, optional): Size of the processing block in pixels. Defaults to 256.
        metadata_key (str, optional): Metadata tag name used to store the list of input DEM files
            in the output raster. Defaults to "dem_files".

    Returns:
        None: The function writes the resulting standard deviation DEM to the specified output path.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    output_path = Path(output_path)

    if len(dem_files) <= 1:
        logger.warning(f"Need at least 2 DEMs for computing the STD DEM: {output_path.name}.")
        return

    if not overwrite and io.is_output_up_to_date(dem_files, output_path) and is_existing_std_dem(dem_files, output_path):
        logger.info(f"Skip {output_path.name}: output is up to date.")
        return

    # first open the first raster of the list to have a reference profile
    with rasterio.open(dem_files[0]) as src_ref:
        profile = src_ref.profile.copy()
        width, height = src_ref.width, src_ref.height

    profile.update(dtype="float32", count=1)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        # Loop through the raster by windows
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                win = Window(
                    col_off=x,
                    row_off=y,
                    width=min(block_size, width - x),
                    height=min(block_size, height - y),
                )

                # Read the corresponding window from each DEM
                block_stack = []
                for dem_path in dem_files:
                    with rasterio.open(dem_path) as src:
                        data = src.read(1, window=win, masked=True).filled(np.nan)
                        block_stack.append(data)

                # Compute std for this block
                block_stack = np.stack(block_stack, axis=0)

                # Avoid computing std on empty slices
                if np.all(np.isnan(block_stack)):
                    block_std = np.full(block_stack.shape[1:], np.nan, dtype="float32")
                else:
                    block_std = np.nanstd(block_stack, axis=0).astype("float32")

                # Write the result
                dst.write(block_std, 1, window=win)

        # Add metadata tags
        dem_files_str = [str(p) for p in dem_files]
        dst.update_tags(1, **{metadata_key: json.dumps(dem_files_str)})

    logger.info(f"STD DEM generated at {output_path}")


def create_std_dems(
    dem_files: Iterable[str | Path],
    output_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """
    Group coregistered DEMs by (site, dataset) and compute one STD DEM per group.

    Parses each filename to extract site and dataset metadata, groups files
    accordingly, and calls ``create_std_dem`` for each group. Output filenames
    follow the pattern ``<site>_<dataset>_std_dem.tif`` inside ``output_dir``.
    Files whose names cannot be parsed are skipped with a warning.

    Parameters
    ----------
    dem_files : Iterable of str or Path
        Iterable of coregistered DEM file paths to process.
    output_dir : str or Path
        Directory where the STD DEM files will be written. Created if missing.
    overwrite : bool, optional
        If True, existing STD DEMs are recomputed. Default is False.
    """
    dem_files = [Path(f) for f in dem_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    groups: dict[tuple[str, str], list[Path]] = {}
    for file in dem_files:
        try:
            _, metadatas = parse_filename(file)
            key = (metadatas["site"], metadatas["dataset"])
            groups.setdefault(key, []).append(file)
        except ValueError:
            logger.warning(f"Cannot parse filename for std_dem grouping: {file.name}")

    for (site, dataset), files in groups.items():
        logger.info(f"Creating std DEM for {site} / {dataset}")
        output_path = output_dir / f"{site}_{dataset}_std_dem.tif"
        create_std_dem(dem_files=files, output_path=output_path, overwrite=overwrite)


#######################################################################################################################
##                                                  CLEANUP FUNCTIONS
#######################################################################################################################


def cleanup_orphaned_extracted(raw_dir: Path, extracted_dir: Path) -> None:
    """Remove extracted folders whose source archive no longer exists in raw_dir."""
    if not extracted_dir.exists():
        return
    archive_stems = {p.name.split(".")[0] for p in raw_dir.iterdir() if p.suffix not in [".docx", ".pdf", ".odt"]}
    for folder in extracted_dir.iterdir():
        if folder.is_dir() and folder.name not in archive_stems:
            logger.info(f"Removing orphaned extracted folder (source archive gone): {folder.name}")
            shutil.rmtree(folder)


def cleanup_orphaned_raw_dems(raw_dems_dir: Path, symlinks_dir: Path) -> None:
    """Remove raw DEMs whose source pointcloud or provided DEM symlink no longer exists."""
    if not raw_dems_dir.exists():
        return
    valid_codes: set[str] = set()
    for subdir in ["dense_pointclouds", "dems"]:
        src_dir = symlinks_dir / subdir
        if src_dir.exists():
            for f in src_dir.iterdir():
                try:
                    code, _ = parse_filename(f)
                    valid_codes.add(code)
                except ValueError:
                    pass
    for dem_file in raw_dems_dir.glob("*-DEM.tif"):
        try:
            code, _ = parse_filename(dem_file)
            if code not in valid_codes:
                logger.info(f"Removing orphaned raw DEM (source gone): {dem_file.name}")
                dem_file.unlink()
        except ValueError:
            pass


def cleanup_orphaned_coreg_dems(coreg_dems_dir: Path, raw_dems_dir: Path) -> None:
    """Remove coregistered DEMs whose source raw DEM no longer exists."""
    if not coreg_dems_dir.exists():
        return
    raw_codes: set[str] = set()
    if raw_dems_dir.exists():
        for f in raw_dems_dir.glob("*-DEM.tif"):
            try:
                raw_codes.add(parse_filename(f)[0])
            except ValueError:
                pass
    for dem_file in coreg_dems_dir.glob("*-DEM.tif"):
        try:
            code, _ = parse_filename(dem_file)
            if code not in raw_codes:
                logger.info(f"Removing orphaned coregistered DEM (source gone): {dem_file.name}")
                dem_file.unlink()
        except ValueError:
            pass


def cleanup_orphaned_ddems(ddems_dir: Path, source_dems_dir: Path) -> None:
    """Remove dDEMs whose source DEM no longer exists."""
    if not ddems_dir.exists():
        return
    source_codes: set[str] = set()
    if source_dems_dir.exists():
        for f in source_dems_dir.glob("*-DEM.tif"):
            try:
                source_codes.add(parse_filename(f)[0])
            except ValueError:
                pass
    for ddem_file in ddems_dir.glob("*-DDEM.tif"):
        try:
            code, _ = parse_filename(ddem_file)
            if code not in source_codes:
                logger.info(f"Removing orphaned dDEM (source gone): {ddem_file.name}")
                ddem_file.unlink()
        except ValueError:
            pass


#######################################################################################################################
##                                                  OTHERS FUNCTIONS
#######################################################################################################################


def convert_pointcloud_to_dem(
    pointcloud_path: str | Path,
    reference_dem_path: str | Path,
    output_dem_path: str | Path,
    pdal_exec_path: str = "pdal",
    output_pipeline_path: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Converts a point cloud file (LAS/LAZ) into a DEM raster using a reference DEM for alignment.

    This function generates a PDAL pipeline to read the point cloud, reproject it to the reference
    DEM's CRS if necessary, and interpolate it into a raster using IDW. The pipeline can be saved
    to a JSON file, and the output DEM is optionally overwritten or skipped if it already exists.

    Args:
        pointcloud_path (str | Path): Path to the input point cloud file.
        reference_dem_path (str | Path): Path to the reference DEM for CRS and resolution.
        output_dem_path (str | Path): Path to the output DEM file.
        pdal_exec_path (str, optional): Path to the PDAL executable. Defaults to "pdal".
        output_pipeline_path (str | Path | None, optional): Path to save the PDAL pipeline JSON.
            Defaults to a "processing_pipelines" folder near the output DEM.
        dry_run (bool, optional): If True, only writes the pipeline JSON without executing it. Defaults to False.

    Returns:
        Path: Path to the generated DEM file.
    """

    pointcloud_path = Path(pointcloud_path)
    output_dem_path = Path(output_dem_path)

    output_pipeline_path = (
        output_dem_path.parent / "processing_pipelines" / f"pdal_pipeline_{output_dem_path.stem}.json"
        if output_pipeline_path is None
        else Path(output_pipeline_path)
    )
    output_pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(pointcloud_path) as las_reader:
        pc_crs = las_reader.header.parse_crs()

    ref_dem = gu.Raster(reference_dem_path)
    ref_crs = ref_dem.crs
    if ref_crs is None:
        raise ValueError(f"The reference dem {reference_dem_path} has no CRS.")
    ref_box = box(*ref_dem.bounds)

    # if no crs found in pc_crs, tests with a list of CRS
    if pc_crs is None:
        test_crs_list = [str(ref_crs), "EPSG:4326"]

        logger.warning(f"{pointcloud_path.name} : No CRS found, try CRSs : {test_crs_list}")

        # open the real bounding box of the pointcloud file
        las = laspy.read(pointcloud_path)
        pc_box = box(float(las.x.min()), float(las.y.min()), float(las.x.max()), float(las.y.max()))

        # buffered of 10% of area the ref_dem bounding box
        ref_box_buffered = ref_box.buffer(np.sqrt(ref_box.area) * 0.1)

        for tested_crs in test_crs_list:
            transformer = Transformer.from_crs(tested_crs, ref_crs, always_xy=True)
            pc_box_reprojected = transform(pc_box, transformer.transform, interleaved=False)
            if pc_box_reprojected.within(ref_box_buffered):
                pc_crs = tested_crs

    if pc_crs is None:
        raise ValueError(f"{pointcloud_path.name} : Can't find a valid CRS")

    if not ProjCRS.from_user_input(pc_crs).equals(ProjCRS.from_user_input(ref_crs)):
        logger.warning(
            f"{pointcloud_path.name}: CRS mismatch — point cloud CRS is {ProjCRS.from_user_input(pc_crs).to_epsg() or pc_crs},"
            f" reference CRS is {ref_crs}."
        )

    # --- PDAL pipeline definition ---
    pipeline_dict = {
        "pipeline": [
            {"type": "readers.las", "filename": str(pointcloud_path)},
            {
                "type": "filters.reprojection",
                "in_srs": str(pc_crs),
                "out_srs": str(ref_crs),
            },
            {
                "type": "writers.gdal",
                "filename": str(output_dem_path),
                "resolution": ref_dem.res[0],
                "output_type": "idw",  # Interpolation like point2dem
                "data_type": "float32",
                "gdaldriver": "GTiff",
                "nodata": -9999,
                "origin_x": ref_dem.bounds.left,
                "origin_y": ref_dem.bounds.bottom,
                "width": ref_dem.shape[1],
                "height": ref_dem.shape[0],
            },
        ]
    }

    # write the pipeline in a json file
    with open(output_pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_dict, f, ensure_ascii=False, indent=4)

    if not dry_run:
        logger.info(f"Start Processing {pointcloud_path.name}.")

        start = time.perf_counter()

        cmd = [pdal_exec_path, "pipeline", output_pipeline_path]
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)

        elapsed = time.perf_counter() - start

        # --- Add metadata tags using rasterio ---
        try:
            with rasterio.open(output_dem_path, "r+") as dst:
                dst.update_tags(1, pdal_generated_time=f"{elapsed:.3f}")
        except Exception as e:
            logger.warning(f"Could not write metadata tags to {output_dem_path} : {e}")

        human_eta = humanize.naturaldelta(elapsed)
        logger.info(f"[OK] DEM successfully generated for {pointcloud_path.name} (execution time : {human_eta})")

    return output_dem_path


def coregister_dem(
    dem_path: str | Path,
    ref_dem_path: str | Path,
    ref_dem_mask_path: str | Path,
    output_dem_path: str | Path,
) -> Path:
    """
    Coregisters a DEM to a reference DEM using horizontal and vertical adjustment methods.

    This function reprojects the input DEM to match the reference DEM's grid, applies a horizontal
    coregistration using the Nuth-Kaab method, followed by a vertical shift correction, and saves
    the coregistered DEM to the specified output path. Relevant coregistration metadata is stored
    in the output raster.

    Args:
        dem_path (str | Path): Path to the DEM to be coregistered.
        ref_dem_path (str | Path): Path to the reference DEM.
        ref_dem_mask_path (str | Path): Path to the reference DEM mask indicating valid pixels.
        output_dem_path (str | Path): Path where the coregistered DEM will be saved.

    Returns:
        Path: Path to the coregistered DEM file.
    """
    logger.info(f"Start Coregistration of {dem_path}.")

    output_dem_path = Path(output_dem_path)

    # Because ASP's point2dem rounds the bounds, output DEM is not perfectly aligned with the ref DEM
    # so we reproject the source dem with the reference dem
    dem_ref = gu.Raster(ref_dem_path)
    dem_ref_mask = gu.Raster(ref_dem_mask_path)
    dem = gu.Raster(dem_path).reproject(dem_ref, silent=True)

    # check all dems are on the same grid
    assert dem.shape == dem_ref.shape == dem_ref_mask.shape
    assert dem.transform == dem_ref.transform == dem_ref_mask.transform

    # get the dem ref mask
    inlier_mask_vert = dem_ref_mask.data.astype(bool)

    # For horizontal coregistration, also remove very low slopes as they bias the shift estimate
    slope = xdem.terrain.slope(dem_ref)
    inlier_mask_hori = inlier_mask_vert & (slope > 1)

    # Running coregistration
    coreg_hori = xdem.coreg.NuthKaab(vertical_shift=False)
    coreg_vert = xdem.coreg.VerticalShift(vshift_reduc_func=np.median)
    dem_coreg_tmp = coreg_hori.fit_and_apply(dem_ref, dem, inlier_mask=inlier_mask_hori)
    dem_coreg = coreg_vert.fit_and_apply(dem_ref, dem_coreg_tmp, inlier_mask=inlier_mask_vert)

    # save the coregistered dem
    Path(output_dem_path).parent.mkdir(parents=True, exist_ok=True)
    dem_coreg.save(output_dem_path)

    # --- Add metadata tags using rasterio ---
    with rasterio.open(output_dem_path, "r+") as dst:
        dst.update_tags(
            1,
            coreg_method="NuthKaab+VerticalShift",
            coreg_shift_x=coreg_hori.meta["outputs"]["affine"]["shift_x"],
            coreg_shift_y=coreg_hori.meta["outputs"]["affine"]["shift_y"],
            coreg_shift_z=coreg_vert.meta["outputs"]["affine"]["shift_z"],
        )
    return output_dem_path


def generate_ddem(dem_path1: str | Path, dem_path2: str | Path, output_path: str | Path) -> Path:
    """
    Generates a differential DEM (dDEM) by subtracting one DEM from another.

    This function computes the pixel-wise difference between two input DEMs and saves the
    resulting dDEM to the specified output path.

    Args:
        dem_path1 (str | Path): Path to the first DEM (minuend).
        dem_path2 (str | Path): Path to the second DEM (subtrahend).
        output_path (str | Path): Path where the resulting dDEM will be saved.

    Returns:
        Path: Path to the generated dDEM file.
    """
    output_path = Path(output_path)

    dem1 = gu.Raster(dem_path1)
    dem2 = gu.Raster(dem_path2)
    ddem = dem1 - dem2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ddem.save(output_path)
    return output_path


def extract_archive(archive_path: Path | str, output_dir: Path | str, flatten_nested: bool = True) -> None:
    """
    Extract the contents of an archive into a target directory.

    This function supports ZIP, 7z, and common TAR-based formats. If the output
    directory already exists, it is fully removed before extraction. After
    extraction, macOS-specific metadata directories (e.g., ``__MACOSX``) and
    AppleDouble files (``._*``) are automatically cleaned.

    If ``flatten_nested`` is True, the function also removes a redundant
    top-level nested directory **only when** its name matches the output
    directory name. In such cases, the contents of the nested directory are
    moved one level up and the redundant folder is removed.

    Parameters
    ----------
    archive_path : Path or str
        Path to the archive file to extract.
    output_dir : Path or str
        Directory where the archive contents will be extracted.
    flatten_nested : bool, optional
        If True (default), flatten a nested folder with the same name as
        ``output_dir``.

    Raises
    ------
    ValueError
        If the archive format is not supported.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    # Check format before touching the filesystem so failed extractions leave no empty folder
    if archive_path.suffix not in [".zip", ".7z", ".tar", ".tgz", ".gz", ".bz2", ".xz"]:
        raise ValueError(f"Extraction for this type not implemented: {archive_path.suffix}")

    # overwrite if existing
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraction logic
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif archive_path.suffix == ".7z":
        with py7zr.SevenZipFile(archive_path, mode="r") as szf:
            szf.extractall(output_dir)
    elif archive_path.suffix in [".tar", ".tgz", ".gz", ".bz2", ".xz"]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)

    # remove macOS metadata if exists
    macosx_dir = output_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    # Remove AppleDouble files (._xxx)
    for fp in output_dir.rglob("._*"):
        fp.unlink()

    if flatten_nested:
        current = output_dir

        while True:
            children = list(current.iterdir())

            # A SINGLE folder?
            if len(children) == 1 and children[0].is_dir() and children[0].name == current.name:
                current = children[0]
            else:
                break

        if current == output_dir:
            return

        # Now 'current' is the deepest redundant folder
        # Move everything back one time at the top-level
        for item in current.iterdir():
            item.touch()  # Update file time
            shutil.move(str(item), output_dir)

        # Now delete entire chain of empty redundant folders
        # from deepest to upper
        tmp = current
        while tmp != output_dir:
            parent = tmp.parent
            tmp.rmdir()
            tmp = parent


def get_dem_files_from_std_dem(std_dem_file: str | Path, metadata_key: str = "dem_files") -> list[str]:
    """
    Retrieves the list of input DEM file paths stored in the metadata of a standard deviation DEM.

    Args:
        std_dem_file (str | Path): Path to the standard deviation DEM file.
        metadata_key (str, optional): Metadata tag name containing the input DEM file list.
            Defaults to "dem_files".

    Returns:
        list[str] | None: A list of DEM file paths if found in the metadata.
        Returns an empty list if the file does not exist, or None if the metadata key is missing.
    """
    std_dem_file = Path(std_dem_file)
    if not std_dem_file.exists():
        return []

    with rasterio.open(std_dem_file) as src:
        tags = src.tags(1)

        if metadata_key in tags:
            # Already present → return parsed JSON
            return json.loads(tags[metadata_key])
        else:
            return None


def is_existing_std_dem(dem_files: list[str | Path], output_path: str | Path, metadata_key: str = "dem_files") -> bool:
    """
    Check if an existing std DEM already matches the given input DEM list.

    Args:
        dem_files: List of input DEM paths used to compute the std DEM.
        output_path: Path to the supposed std DEM file.
        metadata_key: Metadata key used to store the original DEM file list.

    Returns:
        True if the file exists and its metadata matches the given DEM list, False otherwise.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return False

    try:
        with rasterio.open(output_path) as src:
            tags = src.tags(1)  # or src.tags() if not band-specific

        if metadata_key not in tags:
            return False

        # Normalize paths to absolute str for comparison
        founded_dem_files = [str(Path(p).resolve()) for p in json.loads(tags[metadata_key])]
        expected_dem_files = [str(Path(p).resolve()) for p in dem_files]

        return set(expected_dem_files) == set(founded_dem_files)

    except Exception as e:
        # Defensive: in case of malformed metadata or corrupted file
        logger.warning(f"Could not verify std_dem metadata ({output_path.name}): {e}")
        return False


#######################################################################################################################
##                                                  PLOT FUNCTIONS
#######################################################################################################################


def plot_symlinks(config: Config, submissions_df: pd.DataFrame | None = None) -> None:
    """
    Generate plots summarizing the indexed symlinks directory.

    Computes point-cloud statistics from dense point cloud files and saves
    a bar chart of point counts per submission.

    Parameters
    ----------
    config : Config
    submissions_df : pd.DataFrame, optional
        DataFrame returned by :func:`io.scan_submissions`.  When provided, a
        file-size matrix is saved alongside the presence map.
    """
    pointcloud_files = list((config.proc_dir.symlinks_dir / "dense_pointclouds").iterdir())
    logger.info("Plotting PC count, presence map and file size.")

    output_pc_count = config.plot_dir / "pointcloud_point_count.png"
    if not config.overwrite_plots and is_output_up_to_date(pointcloud_files, output_pc_count):
        logger.info(f"Skip {output_pc_count.name}: output is up to date.")
    else:
        df = stats.compute_pcs_statistics_df(pointcloud_files)
        viz.barplot_var(df, output_pc_count, "point_count", "Point count in dense point-cloud file", overwrite=True)

    directories = [d for d in config.proc_dir.symlinks_dir.iterdir() if d.name != "reports"]
    viz.visualize_files_presence_map(directories, config.plot_dir / "submissions_presence_map.png", overwrite=config.overwrite_plots)

    if submissions_df is not None:
        viz.visualize_files_size_map(submissions_df, config.plot_dir / "submissions_file_sizes.png", overwrite=config.overwrite_plots)

def plot_point2dem(config: Config) -> None:
    """
    Generate plots for the raw DEMs produced by the point-cloud-to-DEM step.

    Saves a bar chart of nodata percentages and per-(site, dataset) DEM mosaics
    for all raw DEMs found in ``raw_dems_dir``.
    """

    df = stats.compute_dems_statistics_df(config.proc_dir.raw_dems_dir.glob("*-DEM.tif"), max_workers=config.max_workers)
    viz.barplot_var(df, config.plot_dir / "raw_dem_voids.png", "percent_nodata", "Raw DEM nodata percent", overwrite=config.overwrite_plots)
    for (site, dataset), group in df.groupby(["site", "dataset"]):
        logger.debug(f"Plotting **** {site} - {dataset} ****")
        output_path = config.plot_dir / f"{site}_{dataset}" / "mosaic" / "mosaic_raw_dem.png"
        vmin, vmax = group["min"].median(), group["max"].median()
        viz.generate_dems_mosaic(group["file"].to_dict(), output_path, vmin, vmax, f"({site} {dataset}) Mosaic Raw DEMs", config.overwrite_plots)


def plot_coregistration(config: Config) -> None:
    """
    Generate plots summarizing the coregistration step.

    Saves per-(site, dataset) DEM mosaics and coregistration-shift scatter plots
    for all coregistered DEMs in ``coreg_dems_dir``. When ``symlinks_dir`` and
    ``raw_dems_dir`` are provided, also saves an updated submissions presence map
    that includes the raw and coregistered DEM directories.
    """
    coreg_dems_dir = config.proc_dir.coreg_dems_dir
    raw_dems_dir = config.proc_dir.raw_dems_dir
    symlinks_dir = config.proc_dir.symlinks_dir

    df = stats.compute_dems_statistics_df(coreg_dems_dir.glob("*-DEM.tif"), max_workers=config.max_workers)
    for (site, dataset), group in df.groupby(["site", "dataset"]):
        logger.debug(f"Plotting **** {site} - {dataset} ****")
        sub_dir = config.plot_dir / f"{site}_{dataset}"
        dem_files_dict = group["file"].to_dict()
        vmin, vmax = group["min"].median(), group["max"].median()

        logger.debug("Plotting coregistered DEMs mosaic")
        viz.generate_dems_mosaic(dem_files_dict, sub_dir / "mosaic" / "mosaic_coreg_dem.png", vmin, vmax, f"({site} {dataset}) Mosaic Coregistered DEMs", config.overwrite_plots)

        logger.debug("Plotting slope mosaic")
        viz.generate_slopes_mosaic(dem_files_dict, sub_dir / "mosaic" / "mosaic_slopes.png", 
                                   title=f"({site} {dataset}) Mosaic slopes of DEMs after coregistration", overwrite=config.overwrite_plots)

        logger.debug("Plotting hillshade mosaic")
        viz.generate_hillshades_mosaic(dem_files_dict, sub_dir / "mosaic" / "mosaic_hillshades.png", 
                                       title=f"({site} {dataset}) Mosaic hillshades of DEMs after coregistration", overwrite=config.overwrite_plots)

    # Build per-(site, dataset) file mapping from df (which has the "file" column)
    coreg_files_by_group = {key: list(g["file"]) for key, g in df.groupby(["site", "dataset"])}

    df_shifts = stats.get_coregistration_statistics_df(coreg_dems_dir.glob("*-DEM.tif"))
    for (site, dataset), group in df_shifts.groupby(["site", "dataset"]):
        output_path = config.plot_dir / f"{site}_{dataset}" / "coregistration_shifts.png"
        viz.generate_plot_coreg_shifts(group, output_path, f"({site} {dataset}) Coregistration shifts",
                                       overwrite=config.overwrite_plots, inputs=coreg_files_by_group.get((site, dataset)))

    if symlinks_dir is not None and raw_dems_dir is not None:
        directories = [d for d in Path(symlinks_dir).iterdir() if d.name != "reports"] + [Path(raw_dems_dir), coreg_dems_dir]
        viz.visualize_files_presence_map(directories, config.plot_dir / "files_presence_map.png", overwrite=config.overwrite_plots)


def plot_ddems(config: Config) -> None:
    """
    Generate plots comparing dDEMs before and after coregistration.

    Saves a global NMAD bar chart, per-(site, dataset) NMAD before-vs-after plots,
    per-submission coregistration plots, and mosaics of dDEMs, slopes, and hillshades.
    """
    before_coreg_ddems_dir = config.proc_dir.before_coreg_ddems_dir
    after_coreg_ddems_dir = config.proc_dir.after_coreg_ddems_dir

    ddem_before_df = stats.compute_dems_statistics_df(before_coreg_ddems_dir.glob("*-DDEM.tif"), "ddem_before_", config.max_workers)
    ddem_after_df = stats.compute_dems_statistics_df(after_coreg_ddems_dir.glob("*-DDEM.tif"), "ddem_after_", config.max_workers)
    df = pd.concat([ddem_before_df, ddem_after_df]).groupby(level=0).first()

    viz.barplot_var(df, config.plot_dir / "nmad_after_coregistration.png", "ddem_after_nmad", "NMAD of Altitude differences with ref DEM after coregistration by code", overwrite=config.overwrite_plots)

    for (site, dataset), group in df.groupby(["site", "dataset"]):
        logger.debug(f"Plotting **** {site} - {dataset} ****")
        sub_dir = config.plot_dir / f"{site}_{dataset}"
        viz.generate_plot_nmad_before_vs_after(group, sub_dir / "nmad_before_vs_after_coregistration.png", f"({site} {dataset}) NMAD of DEM differences before vs after coregistration", overwrite=config.overwrite_plots)
        viz.generate_coregistration_individual_plots(group, sub_dir / "coregistrations", config.overwrite_plots)

        ddem_files_dict = group["ddem_after_file"].dropna().to_dict()
        viz.generate_ddems_mosaic(ddem_files_dict, sub_dir / "mosaic" / "mosaic_ddem.png", 
                                  title=f"({site} {dataset}) Mosaic of DDEMs after coregistration", overwrite=config.overwrite_plots)


def plot_std_dems(config: Config) -> None:
    """
    Generate plots for each STD DEM found in ``std_dems_dir``.
    """
    for file in config.proc_dir.std_dems_dir.glob("*.tif"):
        subdir = file.stem.replace("_std_dem", "")
        output_path = config.plot_dir / subdir / file.with_suffix(".png").name
        viz.generate_std_dem_plots(file, output_path, overwrite=config.overwrite_plots)


def plot_landcover(config: Config) -> None:
    """
    Generate landcover-stratified plots for dDEMs and STD DEMs.

    Computes landcover-stratified statistics on coregistered dDEMs and STD DEMs,
    then saves per-(site, dataset) grouped boxplots and NMAD plots, as well as
    a global boxplot aggregated from all STD DEMs.
    """
    after_coreg_ddems_dir = config.proc_dir.after_coreg_ddems_dir
    std_dems_dir = config.proc_dir.std_dems_dir

    all_ddem_files = list(after_coreg_ddems_dir.glob("*-DDEM.tif"))
    ddem_files_by_group: dict[tuple, list[Path]] = {}
    for f in all_ddem_files:
        try:
            _, meta = parse_filename(f)
            key = (meta["site"], meta["dataset"])
            ddem_files_by_group.setdefault(key, []).append(f)
        except Exception:
            pass

    landcover_df = stats.compute_landcover_statistics(all_ddem_files, config.references_data_mapping, config.max_workers)
    std_lc_df = stats.compute_landcover_statistics_on_std_dems(std_dems_dir.glob("*.tif"), config.references_data_mapping, config.max_workers)

    for (site, dataset), group in landcover_df.groupby(["site", "dataset"]):
        sub_dir = config.plot_dir / f"{site}_{dataset}"
        group_inputs = ddem_files_by_group.get((site, dataset))
        viz.generate_landcover_grouped_boxplot(group, sub_dir / "landcover_grouped_boxplot.png", f"({site} {dataset}) Boxplot of Altitude difference with ref DEM by code/landcover",
                                               overwrite=config.overwrite_plots, inputs=group_inputs)
        viz.generate_landcover_nmad(group, sub_dir / "landcover_nmad.png", f"({site} {dataset}) NMAD of Altitude difference with ref DEM by code/landcover",
                                    overwrite=config.overwrite_plots, inputs=group_inputs)

    std_dem_files = list(std_dems_dir.glob("*.tif"))
    viz.generate_landcover_grouped_boxplot_from_std_dems(std_lc_df, config.plot_dir / "landcover_boxplot_from_std_dems.png",
                                                         overwrite=config.overwrite_plots, inputs=std_dem_files)


#######################################################################################################################
##                                                  PRIVATE FUNCTIONS
#######################################################################################################################


def __estimate_extraction_time(
    archive_files: Iterable[str | Path],
    max_workers: int = 1,
    extraction_speed_per_thread: int = 40 * 1024 * 1024,
) -> None:
    """
    Estimate and display the total extraction time for a set of archive files.
    """
    archive_files: list[Path] = [Path(f) for f in archive_files]

    total_size = sum(f.stat().st_size for f in archive_files)

    estimated_seconds = total_size / (extraction_speed_per_thread * max_workers)

    human_total_size = humanize.naturalsize(total_size)
    human_eta = humanize.naturaldelta(estimated_seconds)
    human_speed = humanize.naturalsize(extraction_speed_per_thread) + "/s"

    logger.info(
        f"Total archives size: {human_total_size} | "
        f"Threads: {max_workers} | "
        f"Speed per thread: {human_speed} | "
        f"Estimated extraction time: ~{human_eta}"
    )
