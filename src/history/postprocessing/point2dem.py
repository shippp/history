import json
from math import sqrt
import math
import os
import re
from glob import glob
import subprocess

from pyproj import CRS, Transformer

ICELAND_ZOOM_BOUNDS = [2662800, 124980, 2765310, 173700]
ICELAND_LARGE_BOUNDS = [2484780, 108990, 2808510, 421800]
CASA_GRANDE_ZOOM_BOUNDS = [414000, 3613020, 444000, 3650010]
CASA_GRANDE_LARGE_BOUNDS = [261990, 3405030, 612990, 3880020]


def create_point2dem_makefile(input_directory: str, output_directory: str) -> str:
    """
    Generate a Makefile that creates DEMs from LAS/LAZ point clouds using point2dem.

    The Makefile will:
        - Automatically create the output directory if missing.
        - Create rules for each LAS/LAZ file to generate a corresponding DEM file.
        - Estimate the DEM resolution using PDAL metadata.

    Args:
        input_directory (str): Path to the directory containing LAS/LAZ point clouds.
        output_directory (str): Path where the generated DEM files will be stored.

    Returns:
        str: The path to the generated Makefile.
    """
    makefile_path = os.path.join(input_directory, "Makefile")

    # Compute the output directory path relative to the input directory
    rel_output_dir = os.path.relpath(os.path.abspath(output_directory), input_directory)

    rules = []
    targets = []

    # Rule to ensure the output directory exists
    rules.append(f"{rel_output_dir}/:\n\tmkdir -p {rel_output_dir}\n")

    for filename in sorted(os.listdir(input_directory)):
        if filename.endswith(".las") or filename.endswith(".laz"):

            # Build output DEM filename
            dem_filename = filename.replace("_pointcloud.las", "").replace("_pointcloud.laz", "") + "_dem.tif"
            dem_file = os.path.join(os.path.relpath(os.path.abspath(output_directory), input_directory), dem_filename)

            # Estimate DEM resolution using point density
            dem_resolution = get_pdal_estimated_resolution(os.path.join(input_directory, filename))
            targets.append(dem_file)

            # Create the rule for generating the DEM
            rule = f"""{dem_file}: {filename} | {rel_output_dir}/
\t-{cmd_point2dem(filename, dem_file, dem_resolution)}
"""
            rules.append(rule)

    # Write the Makefile
    with open(makefile_path, "w") as f:
        f.write("all: " + " ".join(targets) + "\n\n")
        for rule in rules:
            f.write(rule + "\n")

    print(f"Makefile written to: {makefile_path}")

    return makefile_path


def cmd_point2dem(pointcloud_file: str, dem_file: str, dem_resolution: int ) -> str:
    """
    Construct a point2dem command string to generate a DEM from a point cloud file.

    The function uses the site and dataset information embedded in the filename
    to determine the appropriate projection (EPSG/Proj4 string) and cropping bounds.

    Args:
        pointcloud_file (str): Input LAS or LAZ file path.
        dem_file (str): Output DEM file path.
        dem_resolution (int): Resolution of the output DEM in meters.

    Returns:
        str: The complete command-line string to run point2dem.
    """
    # Extract metadata from filename
    infos = extract_info_from_file(pointcloud_file)

    # Define the projection string based on the site
    epsg = "+proj=utm +zone=12 +datum=NAD83 +units=m +no_defs +type=crs" if infos["Site"] == "CG" else CRS.from_epsg(8088).to_proj4()

    # Define bounding box depending on site and dataset type
    if infos["Site"] == "CG":
        bounds = CASA_GRANDE_ZOOM_BOUNDS if infos["Dataset"] == "AI" else CASA_GRANDE_LARGE_BOUNDS
    else:
        bounds = ICELAND_ZOOM_BOUNDS if infos["Dataset"] == "AI" else ICELAND_LARGE_BOUNDS
    str_bounds = " ".join(str(x) for x in bounds)

    # Build the point2dem command
    command = f"point2dem --t_srs \"{epsg}\" --tr {dem_resolution} --t_projwin {str_bounds} --datum WGS84 {pointcloud_file} -o {dem_file}"

    return command


def get_pdal_metadata(filepath: str) -> dict:
    """
    Run `pdal info --metadata` on a LAS/LAZ file and return the parsed metadata as a dictionary.

    Args:
        filepath (str): Path to the LAS/LAZ file.

    Returns:
        dict: Parsed PDAL metadata (only the 'metadata' field).

    Raises:
        subprocess.CalledProcessError: If the PDAL command fails.
        json.JSONDecodeError: If the output is not valid JSON.
    """
    try:
        result = subprocess.run(
            ["pdal", "info", "--metadata", filepath],
            check=True,
            capture_output=True,
            text=True 
        )
        metadata = json.loads(result.stdout)
        return metadata.get("metadata")
    except subprocess.CalledProcessError as e:
        print("PDAL command failed:", e)
        print("stderr:", e.stderr)
        raise
    except json.JSONDecodeError as e:
        print("Failed to parse JSON output from PDAL.")
        raise


def get_pdal_estimated_resolution(filepath: str) -> int:
    """
    Estimate the DEM resolution (in meters) from a point cloud LAS/LAZ file using PDAL metadata.

    The function computes the point density over the projected area (in EPSG:3857),
    then estimates the point spacing as the square root of the inverse of this density.
    The result is rounded:
        - to the nearest integer if spacing < 8
        - to the nearest 10 otherwise (e.g., 27 becomes 30)

    Args:
        filepath (str): Path to the LAS/LAZ point cloud file.

    Returns:
        int: Estimated resolution in meters suitable for DEM generation.
    """
    metadata = get_pdal_metadata(filepath)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Reproject the bounding box in a meters crs
    xmin_m, ymin_m = transformer.transform(metadata["minx"], metadata["miny"])
    xmax_m, ymax_m = transformer.transform(metadata["maxx"], metadata["maxy"])

    area_m2 = (xmax_m - xmin_m) * (ymax_m - ymin_m)
    density = metadata["count"] / area_m2
    spacing = math.sqrt(1 / density)

    if spacing < 8:
        return round(spacing)
    else:
        return int(round(spacing, -1))


def extract_info_from_file(file: str) -> dict[str, str|bool]:
    """
    Extract metadata from a structured filename.

    The filename is expected to follow the pattern:
    Name_Site_Dataset_Images_CY/GN_PY/MY/...

    Returns a dictionary with the following keys:
        - 'Name': project or person name (str)
        - 'Site': site identifier (str)
        - 'Dataset': dataset label (str)
        - 'Images': image type or label (str)
        - 'Use of Camera Calibration Information': True if 'CY', else False
        - 'Use of Ground Control Points': True if 'GY', else False
        - 'Point Cloud Coregistration': True if 'PY', else False
        - 'Multi-temporal Bundle Adjustment': True if 'MY', else False
    """
    # Extract just the filename and split it using underscores
    splited_filename = os.path.basename(file).split("_")

    # Map split parts to named fields and convert flags to booleans
    result_info = {
        "Name" : splited_filename[0],
        "Site" : splited_filename[1],
        "Dataset" : splited_filename[2],
        "Images" : splited_filename[3],
        "Use of Camera Calibration Information" : splited_filename[4] == "CY",
        "Use of Ground Control Points" : splited_filename[5] == "GY",
        "Point Cloud Coregistration" : splited_filename[6] == "PY",
        "Multi-temporal Bundle Adjustment" : splited_filename[7] == "MY"
    }
    return result_info