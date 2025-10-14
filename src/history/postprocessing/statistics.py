import warnings
from pathlib import Path

import geoutils as gu
import laspy
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

from history.postprocessing.io import PathsManager

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


def compute_global_statistics(paths_manager: PathsManager, nmad_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute global statistics for all raster and point cloud files managed by a PathsManager instance.

    This function iterates over all entries in the PathsManager file table and extracts
    key descriptive statistics from raster (e.g., DEMs and dDEMs) and point cloud files.
    For rasters, it retrieves or computes pixel-based metrics such as min, max, mean,
    standard deviation, median, and NMAD, while for point clouds it extracts general
    metadata (e.g., point density, extent, etc.).

    Additionally, it estimates and stores horizontal and vertical coregistration shifts
    from the coregistered DEM files. Finally, it applies a robust inlier filter based
    on the NMAD of post-coregistration dDEMs (``ddem_after_nmad``) within each
    (dataset, site) group.

    The inlier mask is determined using a robust threshold defined as:
    ``threshold = median(ddem_after_nmad) + nmad_multiplier * NMAD(ddem_after_nmad)``.
    Entries with ``ddem_after_nmad < threshold`` are flagged as inliers.

    Parameters
    ----------
    paths_manager : PathsManager
        Instance managing the file paths for all datasets and derived products.
        Must provide a `get_filepaths_df()` method returning a DataFrame where
        each row corresponds to a dataset/site pair and columns include:
        - raw_dem_file
        - coreg_dem_file
        - ddem_before_file
        - ddem_after_file
        - dense_pointcloud_file
        - sparse_pointcloud_file
    nmad_multiplier : float, optional
        Multiplicative factor applied to the NMAD to define the upper threshold
        for identifying inliers. Defaults to 3.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame enriched with:
        - Computed raster statistics (e.g., `raw_dem_mean`, `coreg_dem_std`, etc.)
        - Point cloud metadata fields
        - Coregistration shift values (`coreg_shift_x`, `coreg_shift_y`, `coreg_shift_z`)
        - An `inliers` boolean column indicating entries within the robust NMAD-based threshold

    Notes
    -----
    - This function may trigger potentially expensive raster/point cloud reads if
      statistics are not already stored as metadata tags.
    - For large datasets, consider parallelizing or caching results to improve performance.
    - The `gu.stats.nmad()` function from GeoUtils is used for robust dispersion estimation.

    Examples
    --------
    >>> paths_manager = PathsManager("/path/to/project")
    >>> df = compute_global_statistics(paths_manager, nmad_multiplier=3)
    >>> df.head()
         dataset   site   raw_dem_mean  ddem_after_nmad  inliers
    0    glims01  siteA       2543.21             0.12     True
    1    glims01  siteB       2988.45             0.48     True
    2    glims02  siteC       1455.73             1.82    False
    """
    result_df = paths_manager.get_filepaths_df()

    raster_colnames = ["raw_dem_file", "coreg_dem_file", "ddem_before_file", "ddem_after_file"]
    pointcloud_colnames = ["dense_pointcloud_file", "sparse_pointcloud_file"]

    for code, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Computing statistics"):
        # populate the result df with all raster basic statistics
        for colname in raster_colnames:
            if not pd.isna(row[colname]):
                for key, value in get_raster_statistics(row[colname]).items():
                    # exemple : colname for the mean of raw_dem_file -> raw_dem_mean
                    name = colname.removesuffix("file") + key
                    result_df.loc[code, name] = value

        # populate the result df with all pointcloud basic statistics
        for colname in pointcloud_colnames:
            if not pd.isna(row[colname]):
                for key, value in get_pointcloud_metadatas(row[colname]).items():
                    name = colname.removesuffix("file") + key
                    result_df.loc[code, name] = value

        # populate the result df with coregistration shifts
        if not pd.isna(row["coreg_dem_file"]):
            for key, value in get_raster_coregistration_shifts(row["coreg_dem_file"]).items():
                result_df.loc[code, key] = value

    # add a inliers filter based on ddems_after_coreg
    medians = result_df.groupby(["dataset", "site"])["ddem_after_nmad"].transform("median")
    nmads = result_df.groupby(["dataset", "site"])["ddem_after_nmad"].transform(lambda x: gu.stats.nmad(x))

    thresholds = medians + nmad_multiplier * nmads

    result_df["inliers"] = result_df["ddem_after_nmad"] <= thresholds

    return result_df


def compute_landcover_statistics(paths_manger: PathsManager) -> pd.DataFrame:
    """
    Compute landcover-based statistics for all available differential DEMs (ddem_after_file)
    across all (dataset, site) combinations.

    This function:
    - Retrieves the filepaths DataFrame from a `PathsManager` instance.
    - Iterates over each (dataset, site) pair that contains a valid `ddem_after_file`.
    - Computes per-landcover statistics for each raster using `compute_raster_stats_by_landcover()`.
    - Aggregates all computed results into a single combined DataFrame.

    A progress bar (via `tqdm`) tracks the processing progress across all raster–landcover pairs.

    Args:
        paths_manger (PathsManager): Object managing dataset and site paths, providing access to:
            - Differential DEM files (`ddem_after_file`)
            - Corresponding landcover raster paths via `get_landcover(site, dataset)`

    Returns:
        pd.DataFrame: Combined DataFrame containing landcover statistics across all datasets and sites.
        Returns an empty DataFrame if no valid `ddem_after_file` entries are found.

    Output DataFrame Columns (from `compute_raster_stats_by_landcover` and added metadata):
        - 'code': Internal identifier for the processed raster.
        - 'site': Site name.
        - 'dataset': Dataset identifier.
        - 'file_code': Label indicating the processed file type (e.g., 'ddem_after_file').
        - 'landcover_label': Landcover class name or ID.
        - 'nmad' or 'std': Computed statistical metric per class.
        - 'percent': Percentage of pixels per landcover class.
        - 'median', 'q1', 'q3': Distribution statistics per landcover class (if available).

    Notes:
        - Each (dataset, site) pair is processed independently.
        - The function assumes that `compute_raster_stats_by_landcover()` returns a valid DataFrame.
        - If no valid DEM files are found, a warning message is printed instead of raising an exception.
        - The progress bar helps track computation across large datasets.
    """
    filepaths_df = paths_manger.get_filepaths_df()
    df_dropped = filepaths_df.dropna(subset="ddem_after_file")

    all_stats = []  # list to store temporary DataFrames
    with tqdm(desc="landcover computing", total=len(df_dropped)) as pbar:
        for (dataset, site), group in df_dropped.groupby(["dataset", "site"]):
            for code, row in group.iterrows():
                raster_file = row["ddem_after_file"]
                landcover_file = paths_manger.get_landcover(site, dataset)

                # Compute stats for this raster/landcover pair
                tmp_df = compute_raster_stats_by_landcover(raster_file, landcover_file)

                # Add identifier columns
                tmp_df.insert(0, "file_code", "ddem_after_file")  # insert at left
                tmp_df.insert(0, "dataset", dataset)
                tmp_df.insert(0, "site", site)
                tmp_df.insert(0, "code", code)

                # Append to list
                all_stats.append(tmp_df)

                # update the pbar
                pbar.update()

    # Concatenate all temporary DataFrames
    if all_stats:
        final_df = pd.concat(all_stats, ignore_index=True)
        return final_df
    else:
        print("⚠️ No valid ddem_after_file found.")
        return pd.DataFrame()


def compute_landcover_statistics_from_std_dems(
    paths_manager: PathsManager, input_directory: str | Path
) -> pd.DataFrame:
    """
    Compute per-landcover statistics from standard deviation DEMs for each (dataset, site) pair.

    This function iterates through all `std-dem-*.tif` files in the given input directory,
    extracts dataset and site identifiers from their filenames, retrieves the corresponding
    landcover raster via `paths_manager`, and computes statistics for each landcover class
    using `compute_raster_stats_by_landcover()`.

    All resulting per-site DataFrames are concatenated into a single DataFrame summarizing
    landcover statistics across all datasets and sites.

    Args:
        paths_manager (PathsManager): Object used to retrieve file paths (e.g., landcover rasters)
            based on dataset and site identifiers.
        input_directory (str | Path): Directory containing standard deviation DEM files
            named following the pattern `std-dem-{dataset}-{site}.tif`.

    Returns:
        pd.DataFrame: Combined DataFrame containing landcover statistics for all sites and datasets.
        If no valid files are found, an empty DataFrame is returned.

    Expected Output Columns (from `compute_raster_stats_by_landcover`):
        - 'landcover_label': Landcover class name or ID.
        - 'code': Landcover class code.
        - 'nmad' or 'std': Statistical metric per class (depending on the input DEM).
        - 'percent': Percentage of pixels per landcover class.
        - 'median', 'q1', 'q3': Precomputed distribution metrics if available.

    Notes:
        - DEM files must follow the naming pattern `std-dem-{dataset}-{site}.tif`.
        - A warning message is printed if no valid DEMs are found.
        - The function assumes that `compute_raster_stats_by_landcover()` returns a valid DataFrame.
    """
    input_directory = Path(input_directory)
    all_stats = []

    for raster_file in input_directory.glob("std-dem-*.tif"):
        _, _, dataset, site = raster_file.stem.split("-", 3)
        landcover_file = paths_manager.get_landcover(site, dataset)

        tmp_df = compute_raster_stats_by_landcover(raster_file, landcover_file)

        tmp_df.insert(0, "dataset", dataset)
        tmp_df.insert(0, "site", site)

        all_stats.append(tmp_df)

    if all_stats:
        final_df = pd.concat(all_stats, ignore_index=True)
        return final_df
    else:
        print("⚠️ No valid ddem_after_file found.")
        return pd.DataFrame()


def cumulative_mask_filled(raster_files: list[str]) -> gu.Raster:
    """
    Create a cumulative filled mask from multiple raster files.

    This function reads a list of raster files (assumed to be masked arrays),
    combines their valid pixels into a single union mask, fills any internal holes,
    and returns a new `gu.Raster` representing the cumulative mask.

    Parameters
    ----------
    raster_files : list of str
        List of paths to raster files (.tif) to combine. All rasters must have the same
        shape, CRS, and geotransform.

    Returns
    -------
    gu.Raster
        A raster containing a cumulative mask (uint8: 1 for valid pixels, 0 for masked)
        with internal holes filled. The raster uses the CRS and transform of the first
        raster in the list.

    Raises
    ------
    FileNotFoundError
        If the input list `raster_files` is empty.
    AssertionError
        If any raster is not aligned with the first raster (shape, CRS, or transform mismatch).

    Notes
    -----
    - The function assumes that the input rasters are `numpy` masked arrays via `geoutils`.
    - Internal holes are filled using `scipy.ndimage.binary_fill_holes`.
    """
    # read all raster files in the raster_directory
    if not raster_files:
        raise FileNotFoundError("No .tif rasters given")

    # Load first raster as reference
    ref = gu.Raster(raster_files[0])
    union = np.zeros(ref.data.shape, dtype=bool)

    # Accumulate masks
    for f in raster_files:
        r = gu.Raster(f)

        # check if all raster are correctly aligned
        assert r.transform == ref.transform
        assert r.crs == ref.crs
        assert r.shape == ref.shape

        valid = ~r.data.mask  # True if pixel is valid
        union |= valid

    # Fill internal holes
    filled_union = binary_fill_holes(union)

    # Convert to uint8 (0 or 1)
    mask_array = filled_union.astype(np.uint8)

    # Build result as Raster
    mask_raster = gu.Raster.from_array(mask_array, ref.transform, ref.crs, nodata=None)

    return mask_raster


def create_std_dem(dem_files: list[str | Path], output_path: str | Path) -> None:
    """
    Compute a per-pixel standard deviation (STD) digital elevation model (DEM)
    from a list of aligned input DEMs.

    This function reads multiple DEM rasters that are perfectly coregistered
    (same CRS, resolution, extent, and transform). For each pixel position,
    it calculates the standard deviation of elevation values across all input DEMs,
    ignoring NoData values (converted to NaN). The resulting STD DEM is then
    written to disk using the metadata of the first input raster as a reference.

    Parameters
    ----------
    mnt_files : list of str or Path
        List of file paths to input DEM rasters. All rasters must have the same
        CRS, shape, and affine transform.
    output_path : str or Path
        Path to the output GeoTIFF file that will store the computed STD DEM.

    Returns
    -------
    None
        The function writes the output raster directly to disk.

    Notes
    -----
    - Pixels that are NoData in all input DEMs are written as NaN in the output.
    - The function asserts that all input DEMs are spatially aligned; an error is raised otherwise.
    - The output raster inherits the spatial profile (CRS, transform, etc.) of the first DEM.

    Example
    -------
    >>> create_std_ddem(
    ...     ["dem_2020.tif", "dem_2021.tif", "dem_2022.tif"],
    ...     "std_ddem_2020_2022.tif"
    ... )
    """
    # first open the first raster of the list to have a reference profile
    with rasterio.open(dem_files[0]) as src:
        ref_profile = src.profile.copy()
        ref_shape = (src.count, src.height, src.width)

    # loop on every raster to open the masked array with nodata filled with np.nan
    dems = []
    for f in dem_files:
        with rasterio.open(f) as src:
            assert src.transform == ref_profile["transform"]
            assert src.crs == ref_profile["crs"]
            assert (src.count, src.height, src.width) == ref_shape
            dems.append(src.read(1, masked=True).filled(np.nan))

    # stack them
    stack = np.stack(dems, axis=0)  # shape = (N, H, W)

    # calculate std pixel by pixel
    std_array = np.nanstd(stack, axis=0).astype("float32")

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(std_array, 1)


def compute_raster_stats_by_landcover(raster_file: str | Path, landcover_file: str | Path) -> pd.DataFrame:
    """
    Compute descriptive statistics of a raster variable for each landcover class.

    This function opens a continuous raster (e.g., elevation, NDVI, temperature) and a
    categorical landcover raster, reprojects the landcover to match the raster grid
    using nearest-neighbor resampling (to preserve class IDs), and computes per-class
    statistics on all valid pixels.

    The raster mask (nodata values) is automatically applied to both datasets.
    Results are returned as a pandas DataFrame summarizing the distribution of
    raster values within each landcover class.

    Parameters
    ----------
    raster_file : str | Path
        Path to the raster to analyze (e.g., a DEM, NDVI, or temperature raster).
        This raster defines the spatial resolution, extent, and valid data mask.
    landcover_file : str | Path
        Path to the categorical landcover raster. It is reprojected to the same
        CRS, transform, and resolution as the `raster_file` using nearest-neighbor
        resampling to avoid mixing class values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per unique landcover class, including:
        - ``landcover_class`` : integer class ID
        - ``landcover_label`` : class name from the global ``LANDCOVER_MAPPING``
        - ``count`` : number of valid pixels for that class
        - ``percent`` : proportion of valid pixels in percent
        - ``mean`` : mean of raster values for that class
        - ``median`` : median of raster values for that class
        - ``nmad`` : normalized median absolute deviation (robust spread)
        - ``min`` : minimum raster value
        - ``max`` : maximum raster value
        - ``std`` : standard deviation

    Notes
    -----
    - Landcover reprojection uses **nearest-neighbor** resampling to preserve
      discrete classes.
    - Masked or nodata values in the raster are automatically excluded.
    - The global ``LANDCOVER_MAPPING`` dictionary must map class IDs to labels.
    - NMAD is computed as:
      ``1.4826 * median(|x_i - median(x)|)``, providing a robust estimate
      of variability less sensitive to outliers.

    Examples
    --------
    >>> df = compute_raster_stats_by_landcover("elevation.tif", "landcover.tif")
    >>> df.head()
       landcover_class      landcover_label   count  percent    mean  median   nmad    min    max    std
    0               10          tree cover   50213   28.5     412.3   410.2   18.7   100.0  890.0  26.1
    1               20            shrubland  40321   22.9     385.5   382.0   15.1   120.0  710.0  23.7
    """
    # open the first raster
    raster = gu.Raster(raster_file)

    # open the landcover reprojected on the first raster
    # here we use nearest resampling to preserve class
    landcover = gu.Raster(landcover_file).reproject(raster, resampling="nearest")

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

    return pd.DataFrame(stats).sort_values("landcover_class").reset_index(drop=True)


def generate_std_dems_by_site_dataset(
    global_df: pd.DataFrame, output_directory: str | Path, colname: str = "coreg_dem_file"
) -> None:
    """
    Generate standard deviation DEMs (Digital Elevation Models) by (dataset, site) groups.

    For each (dataset, site) pair in the provided DataFrame, this function:
    - Checks that at least two DEMs are available (otherwise emits a warning)
    - Calls `create_std_dem()` to compute the standard deviation DEM
    - Saves the result in the specified output directory

    Args:
        global_df (pd.DataFrame): Input DataFrame containing DEM metadata.
        output_directory (str | Path): Directory where the output files will be saved.
        colname (str, optional): Name of the column containing DEM file paths. Default is "coreg_dem_file".

    Returns:
        None
    """

    # create the output directory if it not exist
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    dropped_df = global_df.dropna(subset=colname)
    for (dataset, site), group in dropped_df.groupby(["dataset", "site"], dropna=False):
        if len(group) < 2:
            warnings.warn(
                f"Skipping STD DEM computation for '{dataset}-{site}': only {len(group)} DEM file(s) available.",
                category=UserWarning,
            )
            continue  # Skip this site/dataset

        dem_files = group[colname].tolist()
        output_file = output_directory / f"std-dem-{dataset}-{site}.tif"

        # Compute and save STD DEM
        create_std_dem(dem_files, output_file)


#######################################################################################################################
##                                                  PRIVATE FUNCTIONS
#######################################################################################################################


def get_raster_coregistration_shifts(dem_file: str | Path) -> dict:
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


def get_raster_statistics(dem_file: str | Path) -> dict:
    """
    Retrieve or compute basic raster statistics for a given DEM file.

    This function first checks if the required statistics are already stored
    in the raster's band metadata (tags). If so, it reads them directly to avoid
    unnecessary computation. Otherwise, it computes them from the raster data,
    stores them as band tags, and returns the results.

    Statistics include:
        - percent_nodata : percentage of masked (nodata) pixels
        - count           : number of valid pixels
        - min, max        : value range
        - mean, median    : central tendencies
        - std, nmad       : dispersion metrics
        - q1, q3          : first and third quartiles
        - crs, resolution : spatial metadata (added, not written to tags)

    Args:
        dem_file (str | Path): Path to the DEM raster file.

    Returns:
        dict: Dictionary containing computed or cached statistics.

    Notes:
        - Statistics are stored as strings in raster tags (for compatibility).
        - On subsequent calls, statistics are read directly from the tags,
          greatly improving performance for large rasters.
    """
    with rasterio.open(dem_file, "r+") as src:
        tags = src.tags(1)
        required_keys = ["min", "max", "mean", "std", "median", "nmad", "q1", "q3", "percent_nodata", "count"]

        # If all required stats exist in metadata, reuse them
        if all(k in tags for k in required_keys):
            stats = {k: float(tags[k]) if k != "crs" else tags[k] for k in required_keys}
        else:
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

            # Save to band metadata (convert to str)
            src.update_tags(1, **{k: str(v) for k, v in stats.items()})

        # Add non-tagged metadata
        stats.update({"crs": src.crs.to_string() if src.crs else None, "resolution": float(src.res[0])})

        return stats


def get_pointcloud_metadatas(pointcloud_file: str | Path) -> dict:
    try:
        with laspy.open(pointcloud_file) as fh:
            header = fh.header
            res = {
                "las_version": f"{header.version.major}.{header.version.minor}",
                "pointcloud_crs": header.parse_crs(),
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
