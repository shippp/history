"""
Utilities for processing DEMs and LiDAR point clouds, including extraction of metadata,
computation of basic statistics, and retrieval of coregistration shifts. Functions parse
filenames for site, dataset, and acquisition info, handle nodata values, and return
results as pandas DataFrames or dictionaries. Supports precomputed statistics when available.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import geoutils as gu
import laspy
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from history.postprocessing.io import parse_filename

#######################################################################################################################
##                                                  MAIN FUNCTIONS
#######################################################################################################################


def compute_dems_statistics_df(
    dem_files: Iterable[str | Path], prefix: str = "", max_workers: int | None = None
) -> pd.DataFrame:
    """
    Compute statistics for a list of DEM files and return a DataFrame containing metadata
    and raster statistics.

    This function performs the following steps for each DEM file:
    1. Parses the filename to extract a unique code and associated metadata.
    2. Adds the metadata to a DataFrame indexed by the code.
    3. Checks if raster statistics are already stored in the file metadata.
       - If present, they are added directly to the DataFrame.
       - If absent, statistics are computed in parallel using `compute_raster_statistics`.
    4. Returns a DataFrame containing both metadata and raster statistics, with optional
       prefix added to column names.

    Parameters
    ----------
    dem_files : Iterable[str | Path]
        Iterable of paths to DEM files to process.
    prefix : str, optional
        Optional string prefix to prepend to all column names for raster statistics.
        Default is an empty string.
    max_workers : int | None, optional
        Maximum number of worker threads to use for parallel computation of raster
        statistics. Default is None, which uses ThreadPoolExecutor's default.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame indexed by DEM code containing metadata and computed raster
        statistics for each DEM. Columns include:
        - Original metadata extracted from filenames
        - File path (`{prefix}file`)
        - Raster statistics (`{prefix}min`, `{prefix}max`, `{prefix}mean`, etc.)

    Notes
    -----
    - Errors encountered while processing individual files are logged via `tqdm.write`
      but do not interrupt processing of other files.
    - Uses `get_raster_statistics` to check for existing statistics before recomputation.
    - Computation of missing statistics is done in parallel to improve performance.
    """
    df = pd.DataFrame()
    df.index.name = "code"

    args_dict = {}

    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            for key, value in metadatas.items():
                df.at[code, key] = value

            df.at[code, f"{prefix}file"] = str(file)

            stats = get_raster_statistics(file)
            if stats is None:
                args_dict[code] = file
            else:
                for key, value in stats.items():
                    df.at[code, prefix + key] = value

        except Exception as e:
            tqdm.write(f"[ERROR] Error while processing {file} : {e}")
            continue

    if not args_dict:
        return df

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_raster_statistics, file): code for code, file in args_dict.items()}

        for fut in tqdm(as_completed(futures), desc="Raster statistics", total=len(futures)):
            code = futures[fut]
            try:
                stats = fut.result()

                for key, value in stats.items():
                    df.at[code, prefix + key] = value
            except Exception as e:
                tqdm.write(f"[ERROR] Error while computing stats for {code}: {e}")
                continue
    return df


def compute_pcs_statistics_df(pointcloud_files: Iterable[str | Path], prefix: str = "") -> pd.DataFrame:
    """
    Build a dataframe containing metadata and statistical attributes extracted
    from a Iterable of point cloud files.

    The function parses each input filename to extract its metadata (e.g., site,
    dataset, acquisition info) and retrieves point-cloud–specific statistics via
    `get_pointcloud_metadatas`. All extracted information is stored in a pandas
    DataFrame indexed by the code derived from the filename. A prefix may be
    optionally added to all point cloud–related columns.

    Parameters
    ----------
    pointcloud_files : Iterable[str | Path]
        Iterable of LAS/LAZ point cloud file paths to process.
    prefix : str, optional
        Optional string added as a prefix to all point cloud attribute columns
        (e.g., `"raw_"`, `"coreg_"`). Default is an empty string.

    Returns
    -------
    pd.DataFrame
        A dataframe where each row corresponds to a point cloud file and includes:
        - Metadata parsed from the filename (site, dataset, date, etc.).
        - A `{prefix}file` column storing the file path.
        - Point cloud metadata and numeric statistics returned by
          `get_pointcloud_metadatas`, prefixed when requested.

    Notes
    -----
    - Files that cannot be parsed or processed produce an error message but do
      not interrupt the global processing.
    - Filenames must be compatible with `parse_filename`.
    """
    df = pd.DataFrame()
    df.index.name = "code"

    for file in pointcloud_files:
        try:
            code, metadatas = parse_filename(file)

            for key, value in metadatas.items():
                df.at[code, key] = value

            df.at[code, f"{prefix}file"] = str(file)

            for key, value in get_pointcloud_metadatas(file).items():
                df.at[code, prefix + key] = value

        except Exception as e:
            print(f"[ERROR] Error while processing {file} : {e}")
            continue

    return df


def get_coregistration_statistics_df(dem_files: Iterable[str | Path]) -> pd.DataFrame:
    """
    Extract coregistration shifts and metadata for a list of DEM files.

    This function parses each DEM filename to extract its metadata (e.g., site,
    dataset, acquisition information) and retrieves coregistration shift values
    stored inside the raster metadata via `get_raster_coregistration_shifts`.
    The extracted information is combined into a pandas DataFrame indexed by the
    DEM code.

    Parameters
    ----------
    dem_files : Iterable[str | Path]
        Iterable of DEM file paths for which coregistration metadata should be
        extracted.

    Returns
    -------
    pd.DataFrame
        A dataframe where each row corresponds to one DEM and includes:
        - All metadata extracted from the filename (e.g., site, dataset, date).
        - Coregistration shift parameters (e.g., dx, dy, dz) retrieved from the
          raster internal metadata.

    Notes
    -----
    - Files that cannot be parsed or processed will generate an error message,
      but will not interrupt processing of the remaining files.
    - The filename format must be compatible with `parse_filename`.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]

    df = pd.DataFrame()
    df.index.name = "code"

    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            for k, v in metadatas.items():
                df.at[code, k] = v

            for k, v in get_raster_coregistration_shifts(file).items():
                df.at[code, k] = v

        except Exception as e:
            print(f"[ERROR] Error while processing {file} : {e}")
    return df


#######################################################################################################################
##                                                  OTHER FUNCTIONS
#######################################################################################################################


def raster_statistics(dem_file: str | Path) -> dict[str, Any]:
    """
    Compute or retrieve raster statistics from metadata.
    If statistics already exist in band metadata, they are returned.
    Otherwise, they are computed, written to metadata, and returned.

    Args:
        dem_file (str | Path): Path to the raster file.

    Returns:
        dict[str, Any]: Raster statistics with optional prefixed keys.
    """

    required_keys = ["min", "max", "mean", "std", "median", "nmad", "q1", "q3", "percent_nodata", "count"]

    # -----------------------------------------------------------
    # 1) Try reading metadata statistics
    # -----------------------------------------------------------
    with rasterio.open(dem_file, "r+") as src:
        tags = src.tags(1)

        # If all required keys exist → return metadata values
        if all(k in tags for k in required_keys):
            stats = {k: float(tags[k]) for k in required_keys}
            stats.update(
                {
                    "crs": src.crs.to_string() if src.crs else None,
                    "resolution": float(src.res[0]),
                }
            )
            return stats

        # -------------------------------------------------------
        # 2) Compute statistics because metadata is missing
        # -------------------------------------------------------
        data = src.read(1, masked=True)
        valid = data.compressed()

        if valid.size == 0:
            # No valid data
            stats = {
                "percent_nodata": 100.0,
                "count": 0,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "nmad": np.nan,
                "q1": np.nan,
                "q3": np.nan,
            }
        else:
            # Compute statistics
            stats = {
                "percent_nodata": float(data.mask.mean() * 100),
                "count": int(valid.size),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "median": float(np.median(valid)),
                "std": float(np.std(valid)),
                "nmad": float(gu.stats.nmad(valid)),
                "q1": float(np.percentile(valid, 25)),
                "q3": float(np.percentile(valid, 75)),
            }

        # Write computed stats to metadata
        src.update_tags(1, **stats)

        # Add CRS + resolution
        stats.update(
            {
                "crs": src.crs.to_string() if src.crs else None,
                "resolution": float(src.res[0]),
            }
        )

    return stats


def get_raster_statistics(dem_file: str | Path) -> dict[str, Any] | None:
    """
    Retrieve raster statistics from band metadata if already computed.

    Args:
        dem_file (str | Path): Path to the raster file.

    Returns:
        dict[str, Any] | None: Statistics dictionary if present in metadata, otherwise None.
    """
    required_keys = ["min", "max", "mean", "std", "median", "nmad", "q1", "q3", "percent_nodata", "count"]

    with rasterio.open(dem_file, "r") as src:
        tags = src.tags(1)

        if all(k in tags for k in required_keys):
            stats = {k: float(tags[k]) for k in required_keys}
            stats.update(
                {
                    "crs": src.crs.to_string() if src.crs else None,
                    "resolution": float(src.res[0]),
                }
            )
            return stats

    return None


def compute_raster_statistics(dem_file: str | Path) -> dict[str, Any]:
    """
    Computes basic statistics for a raster dataset and stores them in its metadata.

    This function calculates statistics such as min, max, mean, median, standard deviation,
    NMAD, quartiles, and the percentage of no-data pixels. The computed statistics are written
    to the raster's metadata and returned as a dictionary.

    Args:
        dem_file (str | Path): Path to the raster file to analyze.

    Returns:
        dict[str, Any]: A dictionary containing raster statistics, CRS, and spatial resolution.
    """

    with rasterio.open(dem_file, "r+") as src:
        # Read masked array
        data = src.read(1, masked=True)
        valid = data.compressed()  # Flattened 1D array without mask

        # Handle empty raster (no valid data)
        if valid.size == 0:
            stats = {
                "percent_nodata": 100.0,
                "count": 0,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "nmad": np.nan,
                "q1": np.nan,
                "q3": np.nan,
            }
        else:
            # Compute statistics for valid data
            stats = {
                "percent_nodata": float(data.mask.mean() * 100),
                "count": int(valid.size),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "median": float(np.median(valid)),
                "std": float(np.std(valid)),
                "nmad": float(gu.stats.nmad(valid)),
                "q1": float(np.percentile(valid, 25)),
                "q3": float(np.percentile(valid, 75)),
            }
        # write stats on tags
        src.update_tags(1, **stats)
        stats.update({"crs": src.crs.to_string() if src.crs else None, "resolution": float(src.res[0])})

    return stats


def get_pointcloud_metadatas(pointcloud_file: str | Path) -> dict[str, Any]:
    """
    Extracts metadata from a LAS/LAZ point cloud file.

    Args:
        pointcloud_file (str | Path): Path to the point cloud file.

    Returns:
        dict[str, Any]: A dictionary containing metadata such as LAS version, CRS, point count,
        and spatial bounds (min/max for X, Y, Z). Returns an empty dictionary if the file
        cannot be processed.
    """
    try:
        with laspy.open(pointcloud_file) as fh:
            header = fh.header
            res = {
                "las_version": f"{header.version.major}.{header.version.minor}",
                "crs": header.parse_crs(),
                "point_count": header.point_count,
                "bounds_x_min": header.mins[0],
                "bounds_x_max": header.maxs[0],
                "bounds_y_min": header.mins[1],
                "bounds_y_max": header.maxs[1],
                "bounds_z_min": header.mins[2],
                "bounds_z_max": header.maxs[2],
            }
        return res
    except Exception as e:
        print(f"Warning: Could not process file '{pointcloud_file}' ({e})")
        return {}


def get_raster_coregistration_shifts(dem_file: str | Path) -> dict[str, float]:
    """
    Extract coregistration shift values from a DEM raster file.

    This function reads the metadata tags of a raster file (band 1) and retrieves
    the coregistration shift values along the X, Y, and Z axes if present.
    These values are expected to be stored under the tags:
    `"coreg_shift_x"`, `"coreg_shift_y"`, and `"coreg_shift_z"`.

    Parameters
    ----------
    dem_file : str or Path
        Path to the DEM raster file.

    Returns
    -------
    dict
        A dictionary containing the available coregistration shift values.
        The keys are `"coreg_shift_x"`, `"coreg_shift_y"`, and/or `"coreg_shift_z"`,
        and the values are floats.

    Examples
    --------
    >>> get_raster_coregistration_shifts("coregistered_dem.tif")
    {'coreg_shift_x': 0.12, 'coreg_shift_y': -0.03, 'coreg_shift_z': 0.01}
    """
    with rasterio.open(dem_file) as src:
        tags = src.tags(1)
        used_keys = ["coreg_shift_x", "coreg_shift_y", "coreg_shift_z"]

        return {k: float(v) for k, v in tags.items() if k in used_keys}
