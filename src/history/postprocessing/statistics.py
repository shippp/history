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
from pathlib import Path
from typing import Any

import geoutils as gu
import laspy
import numpy as np
import pandas as pd
import rasterio

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


def compute_dems_statistics_df(dem_files: list[str | Path], prefix: str = "") -> pd.DataFrame:
    """
    Build a dataframe containing metadata and statistical attributes extracted
    from a list of DEM (Digital Elevation Model) files.

    The function parses each input filename to extract metadata (e.g., site,
    dataset, acquisition info) and computes raster statistics using
    `get_raster_statistics` or `compute_raster_statistics` if necessary. All
    extracted information is stored in a pandas DataFrame indexed by the code
    derived from the filename. A prefix may be optionally added to all DEM
    attribute columns.

    Parameters
    ----------
    dem_files : list[str | Path]
        List of DEM file paths (GeoTIFF or compatible raster) to process.
    prefix : str, optional
        Optional string added as a prefix to all DEM attribute columns
        (e.g., `"raw_"`, `"coreg_"`). Default is an empty string.

    Returns
    -------
    pd.DataFrame
        A dataframe where each row corresponds to a DEM file and includes:
        - Metadata parsed from the filename (site, dataset, date, etc.).
        - A `{prefix}file` column storing the file path.
        - Raster statistics (min, max, mean, std, etc.) returned by
          `get_raster_statistics` or `compute_raster_statistics`, prefixed
          when requested.

    Notes
    -----
    - Files that cannot be parsed or processed produce an error message but do
      not interrupt the global processing.
    - Filenames must be compatible with `parse_filename`.
    """
    df = pd.DataFrame()
    df.index.name = "code"

    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)

            for key, value in metadatas.items():
                df.at[code, key] = value

            df.at[code, f"{prefix}file"] = str(file)

            stats = get_raster_statistics(file)
            if stats is None:
                stats = compute_raster_statistics(file)

            for key, value in stats.items():
                df.at[code, prefix + key] = value

        except Exception as e:
            print(f"[ERROR] Error while processing {file} : {e}")
            continue

    return df


def compute_pcs_statistics_df(pointcloud_files: list[str | Path], prefix: str = "") -> pd.DataFrame:
    """
    Build a dataframe containing metadata and statistical attributes extracted
    from a list of point cloud files.

    The function parses each input filename to extract its metadata (e.g., site,
    dataset, acquisition info) and retrieves point-cloud–specific statistics via
    `get_pointcloud_metadatas`. All extracted information is stored in a pandas
    DataFrame indexed by the code derived from the filename. A prefix may be
    optionally added to all point cloud–related columns.

    Parameters
    ----------
    pointcloud_files : list[str | Path]
        List of LAS/LAZ point cloud file paths to process.
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


def get_coregistration_statistics_df(dem_files: list[str | Path]) -> pd.DataFrame:
    """
    Extract coregistration shifts and metadata for a list of DEM files.

    This function parses each DEM filename to extract its metadata (e.g., site,
    dataset, acquisition information) and retrieves coregistration shift values
    stored inside the raster metadata via `get_raster_coregistration_shifts`.
    The extracted information is combined into a pandas DataFrame indexed by the
    DEM code.

    Parameters
    ----------
    dem_files : list[str | Path]
        List of DEM file paths for which coregistration metadata should be
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


def compute_landcover_statistics(dem_files: list[str | Path], references_data: ReferencesData) -> pd.DataFrame:
    """
    Compute landcover-based elevation statistics for a list of DEM files.

    This function processes each DEM by:
    1. Parsing its filename to extract metadata such as `code`, `site`, and `dataset`.
    2. Attempting to load precomputed landcover statistics via
       `get_raster_statistics_by_landcover`.
    3. If no precomputed statistics exist, loading the corresponding landcover
       raster from `references_data` and computing new statistics using
       `compute_raster_statistics_by_landcover`.

    All statistics are aggregated into a single pandas DataFrame, with each row
    representing a landcover class for a specific DEM.

    Parameters
    ----------
    dem_files : list[str | Path]
        List of DEM file paths for which landcover-segmented statistics must be
        computed.
    references_data : ReferencesData
        Object that provides access to reference landcover rasters based on the
        site and dataset extracted from DEM filenames.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the aggregated statistics for all DEMs. Each row
        includes:
        - code : Identifier extracted from the filename.
        - site, dataset : Acquisition or project metadata.
        - landcover-specific statistics (e.g., mean, std, count), as produced by
          the underlying statistics functions.

    Notes
    -----
    - Any DEM file that raises an exception during processing produces an error
      message but does not interrupt the computation for other files.
    - Filenames must be compatible with `parse_filename`, otherwise they are
      skipped.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    stats_dict = {}
    for file in dem_files:
        try:
            code, metadatas = parse_filename(file)
            site, dataset = metadatas["site"], metadatas["dataset"]
            key = (code, site, dataset)
            stats = get_raster_statistics_by_landcover(file)
            if stats is None:
                landcover_file = references_data.get_landcover(site, dataset)
                stats_dict[key] = compute_raster_statistics_by_landcover(file, landcover_file)
            else:
                stats_dict[key] = stats
        except Exception as e:
            print(f"[ERROR] Error while processing {file} : {e}")

    records = []
    for (code, site, dataset), stats in stats_dict.items():
        records += [{"code": code, "site": site, "dataset": dataset, **elem} for elem in stats]
    return pd.DataFrame(records)


def compute_landcover_statistics_on_std_dems(
    dem_files: list[str | Path], references_data: ReferencesData
) -> pd.DataFrame:
    """
    Compute landcover-based elevation statistics for a list of STD DEMs.

    This function iterates over all provided STD DEM files, identifies their
    associated site and dataset from the filename, and retrieves or computes
    landcover-segmented elevation statistics. If precomputed statistics are
    available (via `get_raster_statistics_by_landcover`), they are reused;
    otherwise, the function loads the corresponding landcover raster from the
    `references_data` object and computes the statistics using
    `compute_raster_statistics_by_landcover`.

    Parameters
    ----------
    dem_files : list[str | Path]
        List of paths to STD DEM rasters for which statistics must be computed.
    references_data : ReferencesData
        Object providing access to reference auxiliary data (e.g., landcover rasters)
        based on site and dataset identifiers extracted from filenames.

    Returns
    -------
    pd.DataFrame
        A dataframe where each row contains landcover-specific statistics for a
        given STD DEM, including:
        - std_dem_file : Path to the processed STD DEM.
        - site, dataset : Identified from the filename.
        - Additional fields returned by the landcover statistics functions (e.g.
          landcover class, mean elevation, standard deviation, pixel count, etc.).

    Notes
    -----
    - Any DEM that cannot be processed will produce an error message but does not
      stop the full batch execution.
    - STD DEM filenames must contain substrings matching values in
      `FILE_CODE_MAPPING["site"]` and `FILE_CODE_MAPPING["dataset"]`.
    """
    dem_files: list[Path] = [Path(f) for f in dem_files]
    stats_dict = {}
    for file in dem_files:
        try:
            site = next((s for s in FILE_CODE_MAPPING["site"].values() if s in file.name), None)
            dataset = next((d for d in FILE_CODE_MAPPING["dataset"].values() if d in file.name), None)
            key = (file, site, dataset)
            stats = get_raster_statistics_by_landcover(file)
            if stats is None:
                landcover_file = references_data.get_landcover(site, dataset)
                stats_dict[key] = compute_raster_statistics_by_landcover(file, landcover_file)
            else:
                stats_dict[key] = stats
        except Exception as e:
            print(f"[ERROR] Error while processing {file} : {e}")

    records = []
    for (std_dem_file, site, dataset), stats in stats_dict.items():
        records += [{"std_dem_file": std_dem_file, "site": site, "dataset": dataset, **elem} for elem in stats]
    return pd.DataFrame(records)


#######################################################################################################################
##                                                  OTHER FUNCTIONS
#######################################################################################################################


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
