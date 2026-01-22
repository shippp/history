"""
Utilities for processing DEMs and LiDAR point clouds, including extraction of metadata,
computation of basic and landcover-segmented statistics,
and retrieval of coregistration shifts. Functions parse filenames for site,
dataset, and acquisition info, handle nodata values,
and return results as pandas DataFrames or dictionaries.
Supports precomputed statistics when available and integrates
with reference landcover data for segmented analysis.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import geoutils as gu
import laspy
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from history.postprocessing.io import FILE_CODE_MAPPING, ReferencesData, parse_filename

# code -> label
LANDCOVER_MAPPING = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare / sparse vegetation",
    70: "snow and ice",
    80: "permanent water bodies",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}
# label -> code
LANDCOVER_MAPPING_INV = {v: k for k, v in LANDCOVER_MAPPING.items()}

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


def compute_landcover_statistics(
    dem_files: Iterable[str | Path], references_data: ReferencesData, max_workers: int | None = None
) -> pd.DataFrame:
    """
    Compute landcover-based raster statistics for a collection of DEM files.

    This function processes a list of DEM file paths and ensures that each file has
    corresponding landcover statistics. If statistics already exist (retrieved via
    `get_raster_statistics_by_landcover`), they are reused. Otherwise, the function
    retrieves the appropriate landcover raster from `references_data` and computes
    the statistics in parallel using a thread pool.

    The workflow is as follows:
        1. Parse each DEM filename to extract metadata (code, site, dataset).
        2. Check whether landcover statistics already exist for the DEM.
        3. If not, queue the DEM for computation using the matching landcover file.
        4. Compute missing statistics in parallel.
        5. Aggregate all results into a flat pandas DataFrame.

    Parameters
    ----------
    dem_files : Iterable[str | Path]
        List of DEM file paths to process.
    references_data : ReferencesData
        Object providing access to reference datasets, in particular landcover rasters.
    max_workers : int, optional
        Maximum number of worker threads to use for parallel computation.
        If None, the default number of workers is chosen by `ThreadPoolExecutor`.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains the DEM identifier (code, site, dataset)
        along with the computed landcover statistics.

    Notes
    -----
    - Errors encountered during file parsing or computation are logged and skipped.
    - The resulting DataFrame is in a flattened “records” format for easy analysis.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    args_dict = {}
    stats_dict = {}

    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            # extract site and dataset from the filename
            site, dataset = metadatas["site"], metadatas["dataset"]

            # create a key with this infos
            key = (code, site, dataset)

            # get existing landcover statistics
            stats = get_raster_statistics_by_landcover(file)

            if stats is None:
                # get the corresponding landcover file
                landcover_file = references_data.get_landcover(site, dataset)

                args_dict[key] = (file, landcover_file)
            else:
                stats_dict[key] = stats

        except Exception as e:
            tqdm.write(f"[ERROR] Error while processing {file} : {e}")
            continue

    if args_dict:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_raster_statistics_by_landcover, file, landcover_file): key
                for key, (file, landcover_file) in args_dict.items()
            }

            for fut in tqdm(as_completed(futures), desc="Landcover statistics", total=len(futures)):
                key = futures[fut]
                try:
                    stats_dict[key] = fut.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Error while computing landcover statistics for {key[0]} : {e}")

    # flatten the stats_dict to convert it into a df an return it
    records = []
    for (code, site, dataset), stats in stats_dict.items():
        records += [{"code": code, "site": site, "dataset": dataset, **elem} for elem in stats]
    return pd.DataFrame(records)


def compute_landcover_statistics_on_std_dems(
    dem_files: Iterable[str | Path], references_data: ReferencesData, max_workers: int | None = None
) -> pd.DataFrame:
    """
    Compute landcover-based statistics for a collection of standardized DEM (STD DEM) files.

    This function processes a set of STD DEM file paths and ensures that each file has
    associated landcover statistics. For each DEM file, the function attempts to detect
    the corresponding site and dataset directly from the filename using the
    `FILE_CODE_MAPPING`. If statistics already exist (via `get_raster_statistics_by_landcover`),
    they are reused. Otherwise, the matching landcover raster is retrieved from
    `references_data`, and the missing statistics are computed in parallel.

    The workflow is as follows:
        1. Detect site and dataset identifiers from each DEM filename.
        2. Check if landcover statistics already exist for the DEM.
        3. If missing, schedule computation using the corresponding landcover raster.
        4. Perform computations in parallel using a thread pool.
        5. Aggregate all computed statistics into a flattened pandas DataFrame.

    Parameters
    ----------
    dem_files : Iterable[str | Path]
        Iterable of paths to standardized DEM files to process.
    references_data : ReferencesData
        Object providing access to reference datasets, including landcover rasters.
    max_workers : int, optional
        Maximum number of worker threads to use during parallel computation.
        If None, the default value used by `ThreadPoolExecutor` is applied.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a landcover statistics record,
        including the STD DEM file path, site, dataset, and statistical values.

    Notes
    -----
    - Files for which the site or dataset cannot be identified are skipped with an error message.
    - All errors during computation are logged and do not interrupt the rest of the process.
    - The returned DataFrame is in a flattened “records” format suited for analysis or export.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    stats_dict = {}
    args_dict: dict[tuple[Path, str, str], tuple[Path, Path]] = {}

    for file in dem_files:
        site = next((s for s in FILE_CODE_MAPPING["site"].values() if s in file.name), None)
        dataset = next((d for d in FILE_CODE_MAPPING["dataset"].values() if d in file.name), None)

        if site is None or dataset is None:
            tqdm.write(f"[ERROR] Can't determine the site, dataset of this file : {file}.")
            continue

        key = (file, site, dataset)
        stats = get_raster_statistics_by_landcover(file)
        if stats is None:
            landcover_file = references_data.get_landcover(site, dataset)
            args_dict[key] = (file, landcover_file)
        else:
            stats_dict[key] = stats

    if args_dict:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_raster_statistics_by_landcover, file, landcover_file): key
                for key, (file, landcover_file) in args_dict.items()
            }
            for fut in tqdm(as_completed(futures), desc="Landcover Statistics (STD DEM)", total=len(futures)):
                key = futures[fut]
                try:
                    stats_dict[key] = fut.result()
                except Exception as e:
                    tqdm.write(f"[ERROR] Error while computing landcover statistics for {key[0]} : {e}")
                    continue

    records = []
    for (std_dem_file, site, dataset), stats in stats_dict.items():
        records += [{"std_dem_file": str(std_dem_file), "site": site, "dataset": dataset, **elem} for elem in stats]
    return pd.DataFrame(records)


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


def get_raster_statistics_by_landcover(
    dem_file: str | Path, metadata_key: str = "landcover_stats"
) -> list[dict[str, Any]] | None:
    """
    Retrieves precomputed landcover-based statistics from a raster's metadata.

    Args:
        dem_file (str | Path): Path to the raster file containing the statistics in its metadata.
        metadata_key (str, optional): Metadata tag name storing the statistics. Defaults to "landcover_stats".

    Returns:
        list[dict[str, Any]] | None: A list of dictionaries with statistics for each landcover class
        if the metadata exists; otherwise, returns None.
    """

    with rasterio.open(dem_file, "r") as src:
        tags = src.tags(1)
        if metadata_key in tags:
            # Already present → return parsed JSON
            return json.loads(tags[metadata_key])
        else:
            return None


def compute_raster_statistics_by_landcover(
    raster_file: str | Path,
    landcover_file: str | Path,
    metadata_key: str = "landcover_stats",
) -> list[dict[str, Any]]:
    """
    Computes statistics of a raster dataset grouped by landcover classes.

    This function calculates descriptive statistics (mean, median, quartiles, NMAD, min, max, std)
    for each landcover class present in a landcover raster, considering only valid (unmasked) pixels
    overlapping with the input raster. The results are stored as metadata in the input raster.

    Args:
        raster_file (str | Path): Path to the input raster file to analyze.
        landcover_file (str | Path): Path to the landcover raster file.
        metadata_key (str, optional): Metadata tag name to store the computed statistics.
            Defaults to "landcover_stats".

    Returns:
        list[dict[str, Any]]: A list of dictionaries containing statistics for each landcover class.
    """

    # open the first raster
    raster = gu.Raster(raster_file)

    # open the landcover reprojected on the first raster
    # here we use nearest resampling to preserve class
    landcover = gu.Raster(landcover_file).reproject(raster, resampling="nearest", silent=True)

    # Combine masks to keep only valid pixels in both arrays
    combined_mask = (~raster.data.mask) & (~landcover.data.mask)

    # remove masked values
    raster_valid = raster.data.data[combined_mask]
    landcover_valid = landcover.data.data[combined_mask]

    stats = []
    for c in np.unique(landcover_valid):
        values = raster_valid[landcover_valid == c]
        if len(values) == 0:
            continue
        percent = (len(values) / len(raster_valid)) * 100
        stats.append(
            {
                "landcover_class": int(c),
                "landcover_label": LANDCOVER_MAPPING.get(int(c), "unknown"),
                "count": len(values),
                "percent": percent,
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "q1": float(np.nanpercentile(values, 25)),
                "q3": float(np.nanpercentile(values, 75)),
                "nmad": float(gu.stats.nmad(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "std": float(np.nanstd(values)),
            }
        )
    # Write JSON to raster metadata
    with rasterio.open(raster_file, "r+") as src:
        src.update_tags(1, **{metadata_key: json.dumps(stats)})
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
