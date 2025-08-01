import geoutils as gu
import xdem
import numpy as np
import pandas as pd
import os

from .file_naming import FileNaming

def iter_coregister_dems(
    input_directory: str,
    output_directory: str,
    output_ddem_before_directory: str | None = None,
    output_ddem_after_directory: str | None = None,
    iceland_ref_dem_zoom: str | None = None,
    iceland_ref_dem_large: str | None = None,
    casagrande_ref_dem_zoom: str | None = None,
    casagrande_ref_dem_large: str | None = None,
    iceland_ref_dem_zoom_mask: str | None = None,
    iceland_ref_dem_large_mask: str | None = None,
    casagrande_ref_dem_zoom_mask: str | None = None,
    casagrande_ref_dem_large_mask: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Coregister multiple DEMs from an input directory to appropriate reference DEMs and return summary statistics.

    This function processes all DEM files in `input_directory` ending with '-DEM.tif'. For each DEM, it:
    - Determines the appropriate reference DEM and mask based on site and dataset parsed via `FileNaming`.
    - Applies the `coregister_dem` function to align the DEM to the reference.
    - Saves coregistered DEMs in `output_directory`.
    - Optionally saves difference DEMs (DDEM) before and after coregistration in specified directories.
    - Skips files already existing if `overwrite` is False.
    - Supports a dry run mode where operations are printed but not executed.

    Parameters
    ----------
    input_directory : str
        Directory containing input DEM files to coregister.
    output_directory : str
        Directory where coregistered DEMs will be saved.
    output_ddem_before_directory : str or None, optional
        Directory where pre-coregistration difference DEMs (original minus reference) will be saved.
        If None, these rasters are not saved. Default is None.
    output_ddem_after_directory : str or None, optional
        Directory where post-coregistration difference DEMs (coregistered minus reference) will be saved.
        If None, these rasters are not saved. Default is None.
    iceland_ref_dem_zoom : str or None, optional
        Iceland zoom reference DEM path for dataset 'AI'.
    iceland_ref_dem_large : str or None, optional
        Iceland large reference DEM path for other datasets.
    casagrande_ref_dem_zoom : str or None, optional
        Casagrande zoom reference DEM path for dataset 'AI'.
    casagrande_ref_dem_large : str or None, optional
        Casagrande large reference DEM path for other datasets.
    iceland_ref_dem_zoom_mask : str or None, optional
        Mask raster path for Iceland zoom reference DEM.
    iceland_ref_dem_large_mask : str or None, optional
        Mask raster path for Iceland large reference DEM.
    casagrande_ref_dem_zoom_mask : str or None, optional
        Mask raster path for Casagrande zoom reference DEM.
    casagrande_ref_dem_large_mask : str or None, optional
        Mask raster path for Casagrande large reference DEM.
    overwrite : bool, optional
        Whether to overwrite existing coregistered DEM outputs. Default is False.
    dry_run : bool, optional
        If True, no processing is performed; planned operations are printed only. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing coregistration results for each processed DEM.
        Each row corresponds to one DEM file, indexed by its filename, containing coregistration shifts
        and residual statistics before and after coregistration.

    Notes
    -----
    - Relies on the `coregister_dem` function for individual DEM coregistration.
    - Assumes presence of a `FileNaming` helper class to parse site and dataset information from filenames.
    - Input and output directories must exist and be writable.
    - Creates output directories if they do not exist.
    """
    series = []

    for filename in os.listdir(input_directory):
        if filename.endswith("-DEM.tif"):
            file_naming = FileNaming(filename)
            dem_path = os.path.join(input_directory, filename)
            output_dem_path = os.path.join(output_directory, filename)
            if output_ddem_before_directory:
                output_ddem_before_path = os.path.join(output_ddem_before_directory, filename.replace("-DEM.tif", "-DDEM_before.tif"))
            else:
                output_ddem_before_path = None

            if output_ddem_after_directory:
                output_ddem_after_path = os.path.join(output_ddem_after_directory, filename.replace("-DEM.tif", "-DDEM_after.tif"))
            else:
                output_ddem_after_path = None

            # not overwrite existing files
            if os.path.exists(output_dem_path) and not overwrite:
                print(f"Skip {filename} : {output_dem_path} already exist.")
                continue

            # select the good reference DEM and mask in terms of the site and the dataset
            if file_naming.site == "CG":
                ref_dem_path = casagrande_ref_dem_zoom if file_naming.dataset == "AI" else casagrande_ref_dem_large
                ref_dem_mask_path = casagrande_ref_dem_zoom_mask if file_naming.dataset == "AI" else casagrande_ref_dem_large_mask
            else:
                ref_dem_path = iceland_ref_dem_zoom if file_naming.dataset == "AI" else iceland_ref_dem_large
                ref_dem_mask_path = iceland_ref_dem_zoom_mask if file_naming.dataset == "AI" else iceland_ref_dem_large_mask

            print(f"coregister_dem({dem_path}, {ref_dem_path}, {ref_dem_mask_path}, {output_dem_path})")
            if not dry_run:
                s = coregister_dem(dem_path, ref_dem_path, ref_dem_mask_path, output_dem_path, output_ddem_before_path, output_ddem_after_path)
                series.append(s)

    return pd.DataFrame(series)




def coregister_dem(
        dem_path: str, 
        ref_dem_path: str, 
        ref_dem_mask_path: str, 
        output_dem_path: str, 
        output_ddem_before_path : str | None = None,
        output_ddem_after_path: str | None = None
) -> pd.Series:
    """
    Coregister a DEM to a reference DEM using horizontal and vertical correction, and evaluate residual errors.

    This function performs DEM coregistration in two steps:
    1. Horizontal correction using the Nuth & Kääb (2011) method.
    2. Vertical bias correction using the median of elevation differences.

    The alignment is constrained using a binary mask (e.g., stable terrain mask) from the reference DEM.
    After alignment, statistics (mean, median, NMAD) of elevation differences before and after coregistration
    are computed, and optionally the difference DEMs (DDEMs) before and after coregistration can be saved to disk.

    Parameters
    ----------
    dem_path : str
        Path to the input DEM to be coregistered.
    ref_dem_path : str
        Path to the reference DEM to align to.
    ref_dem_mask_path : str
        Path to a binary mask (e.g., stable areas) associated with the reference DEM. 
        Only `True` pixels are used for coregistration.
    output_dem_path : str
        Path where the coregistered DEM will be saved.
    output_ddem_after_path : str or None, optional
        If provided, path where the post-coregistration difference DEM (coregistered DEM minus reference) will be saved.
    output_ddem_before_path : str or None, optional
        If provided, path where the pre-coregistration difference DEM (original DEM minus reference) will be saved.

    Returns
    -------
    pd.Series
        A pandas Series containing:
        - coregistration shifts (`coreg_shift_x`, `coreg_shift_y`, `coreg_shift_z`)
        - residual statistics before and after correction (`before_coreg_mean`, `before_coreg_median`, `before_coreg_nmad`,
          `after_coreg_mean`, `after_coreg_median`, `after_coreg_nmad`)
        Named using the basename of the input DEM.

    Notes
    -----
    - The function assumes all input rasters are georeferenced and aligned to the same spatial resolution and extent.
    - The DEM to be corrected is reprojected to match the reference DEM grid using `geoutils`.
    - Requires the `xdem` and `geoutils` libraries.
    - Parent directories for output files are created automatically if they do not exist.
    """
    result = {}

    # Cause to point2dem ASP function which round bounds the dem
    # is not perfectly align with the ref DEM
    # so we reproject align dem with the reference dem
    dem_ref = gu.Raster(ref_dem_path)
    dem_ref_mask = gu.Raster(ref_dem_mask_path)
    dem = gu.Raster(dem_path).reproject(dem_ref)

    # ensure all dem to be aligned
    assert dem.shape == dem_ref.shape == dem_ref_mask.shape
    assert dem.transform == dem_ref.transform == dem_ref_mask.transform

    # get the dem ref mask
    inlier_mask = dem_ref_mask.data.astype(bool)

    # Running coregistration
    coreg_hori = xdem.coreg.NuthKaab(vertical_shift=False)
    coreg_vert = xdem.coreg.VerticalShift(vshift_reduc_func=np.median)
    dem_coreg_tmp = coreg_hori.fit_and_apply(dem_ref, dem, inlier_mask=inlier_mask)
    dem_coreg = coreg_vert.fit_and_apply(dem_ref, dem_coreg_tmp, inlier_mask=inlier_mask)

    # save coregistration shift
    result["coreg_shift_x"] = coreg_hori.meta["outputs"]["affine"]["shift_x"]
    result["coreg_shift_y"] = coreg_hori.meta["outputs"]["affine"]["shift_y"]
    result["coreg_shift_z"] = coreg_vert.meta["outputs"]["affine"]["shift_z"]

    # save the coregister dem
    if os.path.dirname(output_dem_path):
            os.makedirs(os.path.dirname(output_dem_path), exist_ok=True)
    dem_coreg.save(output_dem_path, tiled=True)

    # Print statistics
    ddem_before = dem - dem_ref
    ddem_after = dem_coreg - dem_ref
    ddem_bef_inlier = ddem_before[inlier_mask].compressed()
    ddem_aft_inlier = ddem_after[inlier_mask].compressed()
    result["before_coreg_mean"] = np.mean(ddem_bef_inlier)
    result["before_coreg_median"] = np.median(ddem_bef_inlier)
    result["before_coreg_nmad"] = gu.stats.nmad(ddem_bef_inlier)
    result["after_coreg_mean"] = np.mean(ddem_aft_inlier)
    result["after_coreg_median"] = np.median(ddem_aft_inlier)
    result["after_coreg_nmad"] = gu.stats.nmad(ddem_aft_inlier)

    # if the output_ddem_before_path is set save the ddem before coreg
    if output_ddem_before_path:
        if os.path.dirname(output_ddem_before_path):
            os.makedirs(os.path.dirname(output_ddem_before_path), exist_ok=True)
        ddem_before.save(output_ddem_before_path, tiled=True)

    # if the output_ddem_after_path is set save the ddem after coreg
    if output_ddem_after_path:
        if os.path.dirname(output_ddem_after_path):
            os.makedirs(os.path.dirname(output_ddem_after_path), exist_ok=True)
        ddem_after.save(output_ddem_after_path, tiled=True)

    return pd.Series(result, name=os.path.basename(dem_path))

    
