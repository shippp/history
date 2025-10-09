from pathlib import Path

import geoutils as gu
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_fill_holes

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


def create_std_dem(mnt_files: list[str | Path], output_path: str | Path) -> None:
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
    with rasterio.open(mnt_files[0]) as src:
        ref_profile = src.profile.copy()
        ref_shape = (src.count, src.height, src.width)

    # loop on every raster to open the masked array with nodata filled with np.nan
    dems = []
    for f in mnt_files:
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


def compute_raster_stats_by_landcover(raster_file: str | Path, landcover_file: str | Path) -> None:
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
