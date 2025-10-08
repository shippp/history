from pathlib import Path

import geoutils as gu
import numpy as np
import rasterio
from scipy.ndimage import binary_fill_holes


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
