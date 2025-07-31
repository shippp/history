import geoutils as gu
import xdem
import numpy as np
import os

from .file_naming import FileNaming

def iter_coregister_dems(
    input_directory: str,
    output_directory: str,
    iceland_ref_dem_zoom: str | None = None,
    iceland_ref_dem_large: str | None = None,
    casagrande_ref_dem_zoom: str | None = None,
    casagrande_ref_dem_large: str | None = None,
    iceland_ref_dem_zoom_mask: str | None = None,
    iceland_ref_dem_large_mask: str | None = None,
    casagrande_ref_dem_zoom_mask: str | None = None,
    casagrande_ref_dem_large_mask: str | None = None,
    qc_directory: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
)->None:
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith("-DEM.tif"):
            file_naming = FileNaming(filename)
            dem_path = os.path.join(input_directory, filename)
            output_dem_path = os.path.join(output_directory, filename)

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

            if dry_run:
                print(f"coregister_dem({dem_path}, {ref_dem_path}, {ref_dem_mask_path}, {qc_directory})")
            else:
                coregister_dem(dem_path, ref_dem_path, ref_dem_mask_path, qc_directory)




def coregister_dem(dem_path: str, ref_dem_path: str, ref_dem_mask_path: str, output_dem_path: str, qc_directory: str | None = None):
    # Cause to point2dem ASP function which round bounds the dem
    # is not perfectly align with the ref DEM
    # so we reproject align dem with the reference dem
    dem_ref = gu.Raster(ref_dem_path)
    dem_ref_mask = gu.Raster(ref_dem_mask_path)
    dem = gu.Raster(dem_path).reproject(dem_ref)

    # ensure all dem to be aligned
    assert dem.shape == dem_ref.shape == dem_ref_mask.shape
    assert dem.transform == dem_ref.transform == dem_ref_mask.transform

    # inverse the dem ref mask
    inlier_mask = ~dem_ref_mask.data.astype(bool)

    # Running coregistration
    coreg_hori = xdem.coreg.NuthKaab(vertical_shift=False)
    coreg_vert = xdem.coreg.VerticalShift(vshift_reduc_func=np.median)
    dem_coreg_tmp = coreg_hori.fit_and_apply(dem_ref, dem, inlier_mask=inlier_mask)
    dem_coreg = coreg_vert.fit_and_apply(dem_ref, dem_coreg_tmp, inlier_mask=inlier_mask)

    # Print shifts
    print(coreg_hori.meta["outputs"]["affine"])
    print(coreg_vert.meta["outputs"]["affine"])

    dem_coreg.save(output_dem_path, tiled=True)

    # Print statistics
    ddem_before = dem - dem_ref
    ddem_after = dem_coreg - dem_ref
    ddem_bef_inlier = ddem_before[inlier_mask].compressed()
    ddem_aft_inlier = ddem_after[inlier_mask].compressed()
    print(f"- Before coreg:\n\tmean: {np.mean(ddem_bef_inlier):.3f}\n\tmedian: {np.median(ddem_bef_inlier):.3f}\n\tNMAD: {xdem.spatialstats.nmad(ddem_bef_inlier):.3f}")
    print(f"- After coreg:\n\tmean: {np.mean(ddem_aft_inlier):.3f}\n\tmedian: {np.median(ddem_aft_inlier):.3f}\n\tNMAD: {xdem.spatialstats.nmad(ddem_aft_inlier):.3f}")

    if qc_directory:
        os.makedirs(qc_directory, exist_ok=True)
        filename_before = os.path.basename(dem_path).replace("-DEM.tif", "-DDEM_before.tif")
        filename_after = os.path.basename(dem_path).replace("-DEM.tif", "-DDEM_after.tif")

        ddem_bef_inlier.save(os.path.join(qc_directory, filename_before), tiled=True)
        ddem_aft_inlier.save(os.path.join(qc_directory, filename_after), tiled=True)

    
