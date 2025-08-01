import os
import subprocess
from geoutils import Raster
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    max_workers: int = 5
) -> None:
    """
    Batch process point cloud files in a directory to generate DEMs aligned with reference DEMs.

    This function iterates over all point cloud files (*.las or *.laz) in `input_directory`,
    selects the appropriate reference DEM based on site and dataset extracted from filenames,
    and calls `point2dem` to create coregistered DEMs saved in `output_directory`.
    
    Parameters
    ----------
    input_directory : str
        Path to the directory containing input point cloud files.
    output_directory : str
        Directory where output DEM files will be saved.
    iceland_ref_dem_zoom : str or None, optional
        Path to the Iceland zoom reference DEM for the 'AI' dataset.
    iceland_ref_dem_large : str or None, optional
        Path to the Iceland large reference DEM for non-'AI' datasets.
    casagrande_ref_dem_zoom : str or None, optional
        Path to the Casagrande zoom reference DEM for the 'AI' dataset.
    casagrande_ref_dem_large : str or None, optional
        Path to the Casagrande large reference DEM for non-'AI' datasets.
    overwrite : bool, optional
        If True, overwrite existing DEM files. Default is False.
    dry_run : bool, optional
        If True, only print the commands without executing them. Default is False.
    max_workers : int, optional
        max number of process.
    
    Returns
    -------
    None

    Notes
    -----
    - Expects filenames to be parsable by `FileNaming` class to determine site and dataset.
    - Output DEM filenames are derived from input filenames by removing '_pointcloud.las' or '_pointcloud.laz'.
    - Requires `point2dem` function to be defined and accessible.
    - Creates the output directory if it does not exist.
    """
    os.makedirs(output_directory, exist_ok=True)
    ref_dem_mapping = {
        "CGAI": casagrande_ref_dem_zoom,
        "CGMC": casagrande_ref_dem_large,
        "CGPC": casagrande_ref_dem_large,
        "ILAI": iceland_ref_dem_zoom,
        "ILMC": iceland_ref_dem_large,
        "ILPC": iceland_ref_dem_large
    }

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
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
                ref_dem = ref_dem_mapping.get(file_naming.site + file_naming.dataset)
          
                # skip if no reference DEM is provided
                if ref_dem is None:
                    print(
                        f"Skip {filename} : No reference DEM provided (site: {file_naming.site}, dataset: {file_naming.dataset})."
                    )
                    continue

                # start a process of point2dem function
                futures.append(
                    executor.submit(point2dem, pointcloud_file, output_dem, ref_dem, dry_run)
                )
        # Create the pbar and wait for all process to finish
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting into DEM", unit="File"):
            try:
                future.result()
            except Exception as e:
                print(f"[!] Error: {e}")


def point2dem(
    pointcloud_file: str, output_dem: str, ref_dem: str, dry_run: bool = False, max_workers: int = 1
) -> None:
    """
    Generate a DEM raster from a point cloud file using the ASP `point2dem` command,
    aligning output to a reference DEM’s spatial extent, resolution, and coordinate system.

    Parameters
    ----------
    pointcloud_file : str
        Path to the input point cloud file (e.g., .las, .laz) to convert to DEM.
    output_dem : str
        Path where the generated DEM raster will be saved.
    ref_dem : str
        Path to the reference DEM raster used to define the output spatial reference,
        resolution, and bounding box.
    dry_run : bool, optional
        If True, only print the generated command without executing it. Default is False.
    max_workers : int, optional
        Number of threads to run point2dem.

    Returns
    -------
    None

    Notes
    -----
    - This function constructs and runs the `point2dem` command line tool from the Ames Stereo Pipeline (ASP).
    - The output DEM will be projected and clipped to match the reference DEM’s CRS, bounds, and resolution.
    - Requires that `point2dem` is installed and accessible in the system PATH.
    """
    ref_raster = Raster(ref_dem)

    bounds = ref_raster.bounds
    str_bounds = f"{bounds.left} {bounds.bottom} {bounds.right} {bounds.top}"

    str_crs = ref_raster.crs.to_proj4()

    res = ref_raster.res[0]

    command = f'point2dem --t_srs "{str_crs}" --tr {res} --t_projwin {str_bounds} --threads {max_workers} --datum WGS84  {pointcloud_file} -o {output_dem}'
    
    if dry_run:
        print(command)
    else:
        # we don't want the standard output of the command for the multi processing
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL)
