import os
from glob import glob

import geoutils as gu
import numpy as np
import pandas as pd
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


def get_voids_from_rasters(raster_directory: str, raster_mask: gu.Raster) -> pd.DataFrame:
    raster_files = glob(os.path.join(raster_directory, "*.tif"))
    if not raster_files:
        raise FileNotFoundError(f"No .tif rasters found in {raster_directory}")

    results = []

    # Ensure mask is boolean
    valid_zone = raster_mask.data.squeeze() > 0

    for f in raster_files:
        r = gu.Raster(f)

        # ensure raster are aligned to mask
        assert r.transform == raster_mask.transform
        assert r.shape == raster_mask.shape
        assert r.crs == raster_mask.crs

        nodata_mask = np.ma.getmaskarray(r.data.squeeze())

        # number of nodata in the mask
        n_nodata = np.count_nonzero(nodata_mask & valid_zone)

        # totals pixel in mask
        total_pixels = np.count_nonzero(valid_zone)

        # percent of nodata in total pixels in the mask
        percent_nodata = (n_nodata / total_pixels) * 100 if total_pixels > 0 else np.nan

        results.append({"filename": os.path.basename(f), "void_percentage": percent_nodata})

    return pd.DataFrame(results)


def create_std_ddem(ddem_directory: str, std_ddem_path: str) -> None:
    ddem_files = glob(os.path.join(ddem_directory, "*-DDEM_after.tif"))
    ddems = [gu.Raster(f) for f in ddem_files]

    # check if all ddems are correctly aligned
    for d in ddems[1:]:
        assert ddems[0].transform == d.transform
        assert ddems[0].crs == d.crs
        assert ddems[0].shape == d.shape

    # stack them
    stack = np.stack([d.data for d in ddems], axis=0)  # shape = (N, H, W)

    # calculate std pixel by pixel
    std_array = np.nanstd(stack, axis=0)

    # Create a new raster align on the first one
    std_dem = gu.Raster.from_array(std_array.astype("float32"), transform=ddems[0].transform, crs=ddems[0].crs)

    # save in geotiff
    std_dem.save(std_ddem_path)
