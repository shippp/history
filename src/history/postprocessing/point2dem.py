import os
import subprocess
from geoutils import Raster

from .file_naming import FileNaming


def iter_point2dem(
    input_directory: str,
    output_directory: str,
    iceland_ref_dem_zoom: str | None = None,
    iceland_ref_dem_large: str | None = None,
    casagrande_ref_dem_zoom: str | None = None,
    casagrande_ref_dem_large: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
):
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".laz") or filename.endswith(".las"):
            file_naming = FileNaming(filename)
            pointcloud_file = os.path.join(input_directory, filename)
            output_dem_filename = filename.replace("_pointcloud.las", "").replace(
                "_pointcloud.laz", ""
            )
            output_dem = os.path.join(output_directory, output_dem_filename)

            # check the overwrite
            if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
                print(f"Skip {filename} : {output_dem}-DEM.tif already exist.")
                continue

            # Select the appropriate reference DEM based on site and dataset
            if file_naming.site == "CG":
                ref_dem = (
                    casagrande_ref_dem_zoom
                    if file_naming.dataset == "AI"
                    else casagrande_ref_dem_large
                )
            else:
                ref_dem = (
                    iceland_ref_dem_zoom
                    if file_naming.dataset == "AI"
                    else iceland_ref_dem_large
                )

            # Raise an error if no reference DEM is provided
            if ref_dem is None:
                print(
                    f"Skip {filename} : No reference DEM provided (site: {file_naming.site}, dataset: {file_naming.dataset})."
                )
                continue
            point2dem(pointcloud_file, output_dem, ref_dem, dry_run)


def point2dem(
    pointcloud_file: str, output_dem: str, ref_dem: str, dry_run: bool = False
) -> None:
    ref_raster = Raster(ref_dem)

    bounds = ref_raster.bounds
    str_bounds = f"{bounds.left} {bounds.bottom} {bounds.right} {bounds.top}"

    str_crs = ref_raster.crs.to_proj4()

    res = ref_raster.res[0]

    command = f'point2dem --t_srs "{str_crs}" --tr {res} --t_projwin {str_bounds} --datum WGS84  {pointcloud_file} -o {output_dem}'
    print(command)
    if not dry_run:
        subprocess.run(command, shell=True, check=True)
