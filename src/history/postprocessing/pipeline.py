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

import json
import re
import shutil
import subprocess
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import geoutils as gu
import laspy
import numpy as np
import pandas as pd
import py7zr
import rasterio
import xdem
from pyproj import Transformer
from rasterio.windows import Window
from shapely import box, transform
from tqdm import tqdm

from history.postprocessing.io import ReferencesData, parse_filename

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

    supported_extensions = [".zip", ".7z", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz"]

    archives = [
        (fp, output_dir / fp.name[: -len(ext)]) for ext in supported_extensions for fp in input_dir.glob(f"*{ext}")
    ]

    args_list = []
    for input_path, output_path in archives:
        if output_path.exists() and not overwrite:
            print(f"[INFO] Skipping extraction (folder exists): {output_path}")
            continue
        args_list.append((input_path, output_path))

    if not args_list:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_archive, *args): args for args in args_list}

        for fut in tqdm(as_completed(futures), desc="Extraction", total=len(futures)):
            input_path, output_path = futures[fut]
            try:
                fut.result()
                tqdm.write(f"[OK] ✅ {input_path} extract to {output_path}.")
            except Exception as e:
                tqdm.write(f"[ERROR] Extraction error for {input_path}: {e}")
                continue


def index_submissions_and_link_files(input_dir: str | Path, output_dir: str, overwrite: bool = False) -> None:
    """
    Index all submissions contained in the input directory, extract metadata from
    filenames, and create a standardized directory of symbolic links pointing to
    the detected files.

    This function scans each subdirectory of `input_dir`, identifies relevant data
    files based on predefined regex patterns (mandatory and optional), extracts
    metadata using `parse_filename()`, and stores the results in an internal
    DataFrame indexed by submission code. For each detected file, a corresponding
    symbolic link is created in a structured output directory. Missing mandatory
    files are reported but do not stop the processing.

    If `overwrite` is True, all existing symlink directories inside the output
    directory are removed before any new links are created.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing submission subfolders to index. Each subfolder is
        expected to contain files with names that match the configured patterns.
    output_dir : str or Path
        Target directory in which symbolic links will be created, organized by
        file type.
    overwrite : bool, optional
        If True, existing symlink directories within `output_dir` are removed
        before new links are generated. Default is False.

    Notes
    -----
    - Mandatory files are defined by regex patterns such as point cloud, intrinsic,
      and extrinsic calibration files. Their absence is logged as a warning.
    - File metadata (extracted by `parse_filename`) is added to the indexing
      DataFrame prior to link creation.
    - Symlinks are always recreated: existing links or files at the target
      location are removed before new ones are written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # if overwrite remove each symlinks directory
    if overwrite:
        for sub_dir in output_dir.sub_dirs.values():
            if sub_dir.symlinks_dir.base_dir.exists():
                shutil.rmtree(sub_dir.symlinks_dir.base_dir)

    # Define regex patterns mapping to dataframe column names
    mandatory_patterns = {
        "dense_pointcloud_file": r"_dense_pointcloud\.(las|laz)$",
        "sparse_pointcloud_file": r"_sparse_pointcloud\.(las|laz)$",
        "extrinsics_file": r"_extrinsics\.csv$",
        "intrinsics_file": r"_intrinsics\.csv$",
    }
    optional_patterns = {"dem_file": r".*dem.*\.tif$", "orthoimage_file": r".*orthoimage.*\.tif$"}

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
        for colname in patterns:
            if not pd.isna(row[colname]):
                sub_dirname = colname.replace("_file", "")
                if not sub_dirname.endswith("s"):
                    sub_dirname += "s"

                link = output_dir / sub_dirname / Path(row[colname]).name
                link.parent.mkdir(exist_ok=True, parents=True)

                # Remove existing link if it already exists
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(row[colname])


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
            if output_dem_path.exists() and not overwrite:
                print(f"[INFO] Skip point2dem for {code}: output already exists.")
                continue

            args_dict[code] = [file, ref_dem_path, output_dem_path]

        except Exception as e:
            print(f"[ERROR] Error processing {file.name}: {e}")
            continue

    if not args_dict:
        return

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
                tqdm.write(f"[OK] ✅ point2dem complete for {code}")
            except Exception as e:
                tqdm.write(f"[ERROR] Point2dem error for {code}: {e}")
                continue


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
            if output_path.exists() and not overwrite:
                print(f"[INFO] Skip {code} output already exists.")
                continue

            # extract corresponding reference DEM
            ref_dem_path = references_data.get_ref_dem(metadatas["site"], metadatas["dataset"])

            # read the provided DEM and reproject it on the reference DEM
            raster = gu.Raster(file).reproject(gu.Raster(ref_dem_path))

            # save the raster
            raster.save(output_path)
            print(f"[OK] DEM successfully reprojected and saved at {output_path}.")
        except Exception as e:
            print(f"[ERROR] Error while processing {file} : {e}")
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

            # avoid overwriting existing files
            if output_dem_path.exists() and not overwrite:
                print(f"[INFO] Skip coregistration for {code}, output already exists.")
                continue

            # extract corresponding ref dem and mask with site and dataset
            ref_dem_path = references_data.get_ref_dem(metadatas["site"], metadatas["dataset"])
            ref_dem_mask_path = references_data.get_ref_dem_mask(metadatas["site"], metadatas["dataset"])

            args_dict[code] = [file, ref_dem_path, ref_dem_mask_path, output_dem_path]

        except Exception as e:
            print(f"[ERROR] Error processing {file.name}: {e}")
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
                tqdm.write(f"[OK] Coregistration complete for {code}")
            except Exception as e:
                tqdm.write(f"[ERROR] Coregistration error for {code}: {e}")
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

            # avoid overwriting existing files
            if output_path.exists() and not overwrite:
                tqdm.write(f"[INFO] Skip DDEM {code}, output already exists.")
                continue

            # get corresponding reference DEM with site and dataset
            ref_dem_path = references_data.get_ref_dem(metadatas["site"], metadatas["dataset"])

            args_dict[code] = [file, ref_dem_path, output_path, overwrite]
        except Exception as e:
            tqdm.write(f"[ERROR] Error while processing {file} : {e}")

    if not args_dict:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_ddem, *args): code for code, args in args_dict.items()}

        for fut in tqdm(as_completed(futures), desc="ddem", total=len(futures)):
            code = futures[fut]
            try:
                fut.result()
                tqdm.write(f"[OK] DDEM sucessfully generated for {code}")
            except Exception as e:
                tqdm.write(f"[ERROR] DDEM Error for {code}: {e}")
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
        tqdm.write(f"[WARNING] Need at least 2 DEMs for computing the STD DEM : {output_path.name}.")
        return

    if is_existing_std_dem(dem_files, output_path, metadata_key) and not overwrite:
        tqdm.write(f"[INFO] Skip {output_path.name}: output already exists.")
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

    tqdm.write(f"[OK] STD DEM generated at {output_path}")


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
        raise ValueError(f"The reference dem {reference_dem_path} as no CRS.")
    ref_box = box(*ref_dem.bounds)

    # if not crs found in pc_crs test with a list of CRS
    if pc_crs is None:
        test_crs_list = [str(ref_crs), "EPSG:4326"]

        tqdm.write(f"[WARNING] {pointcloud_path.name} : No CRS found, try CRSs : {test_crs_list}")

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
        tqdm.write(f"[INFO] Start Processing {pointcloud_path.name}.")
        cmd = [pdal_exec_path, "pipeline", output_pipeline_path]
        subprocess.run(cmd, check=True)
        tqdm.write(f"[OK] DEM successfully generated for {pointcloud_path.name}")

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
    tqdm.write(f"[INFO] Start Coregistration of {dem_path}.")

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
    dem_coreg.save(output_dem_path, tiled=True)

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
    else:
        raise ValueError(f"Extraction for this type not implemented: {archive_path.suffix}")

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
        print(f"[WARNING] Could not verify std_dem metadata ({output_path.name}): {e}")
        return False
