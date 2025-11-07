"""
Core module for DEM and point cloud processing.

This module provides a collection of tasks and utilities for handling, analyzing, and
transforming Digital Elevation Models (DEMs) and point clouds. Functionality includes:

- Extracting metadata from point clouds.
- Generating DEMs from point clouds using reference DEMs.
- Computing standard deviation DEMs from multiple DEMs.
- Calculating raster statistics, both overall and by landcover class.
- Coregistering DEMs horizontally and vertically to a reference DEM.
- Generating differential DEMs (dDEMs) from two DEMs.
- Extracting various types of archives.

The tasks are designed to handle large datasets efficiently, support metadata storage
in raster files, and facilitate reproducible geospatial workflows.
"""

import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import geoutils as gu
import laspy
import numpy as np
import py7zr
import rasterio
import xdem
from prefect import task
from pyproj import Transformer
from rasterio.windows import Window
from shapely import box, transform

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


@task(log_prints=True)
def coregister_dem(
    dem_path: str | Path,
    ref_dem_path: str | Path,
    ref_dem_mask_path: str | Path,
    output_dem_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Coregisters a DEM to a reference DEM using horizontal and vertical adjustment methods.

    This function reprojects the input DEM to match the reference DEM's grid, applies a horizontal
    coregistration using the Nuth-Kaab method, followed by a vertical shift correction, and saves
    the coregistered DEM to the specified output path. Relevant coregistration metadata is stored
    in the output raster.

    Args:
        dem_path (str | Path): Path to the DEM to be coregistered.
        ref_dem_path (str | Path): Path to the reference DEM.
        ref_dem_mask_path (str | Path): Path to the reference DEM mask indicating valid pixels.
        output_dem_path (str | Path): Path where the coregistered DEM will be saved.
        overwrite (bool, optional): Whether to overwrite the output if it already exists. Defaults to False.

    Returns:
        Path: Path to the coregistered DEM file.
    """

    output_dem_path = Path(output_dem_path)

    if output_dem_path.exists() and not overwrite:
        print(f"Skip coregistration of {dem_path.stem}: output already exists.")
        return output_dem_path

    print(f"Start coregistration of {dem_path.stem}")
    # Because ASP's point2dem rounds the bounds, output DEM is not perfectly aligned with the ref DEM
    # so we reproject the source dem with the reference dem
    dem_ref = gu.Raster(ref_dem_path)
    dem_ref_mask = gu.Raster(ref_dem_mask_path)
    dem = gu.Raster(dem_path).reproject(dem_ref, silent=True)

    # check all dems are on the same grid
    assert dem.shape == dem_ref.shape == dem_ref_mask.shape
    assert dem.transform == dem_ref.transform == dem_ref_mask.transform

    # get the dem ref mask
    inlier_mask_vert = dem_ref_mask.data.astype(bool)

    # For horizontal coregistration, also remove very low slopes as they bias the shift estimate
    slope = xdem.terrain.slope(dem_ref)
    inlier_mask_hori = inlier_mask_vert & (slope > 1)

    # Running coregistration
    coreg_hori = xdem.coreg.NuthKaab(vertical_shift=False)
    coreg_vert = xdem.coreg.VerticalShift(vshift_reduc_func=np.median)
    dem_coreg_tmp = coreg_hori.fit_and_apply(dem_ref, dem, inlier_mask=inlier_mask_hori)
    dem_coreg = coreg_vert.fit_and_apply(dem_ref, dem_coreg_tmp, inlier_mask=inlier_mask_vert)

    # save the coregistered dem
    Path(output_dem_path).parent.mkdir(parents=True, exist_ok=True)
    dem_coreg.save(output_dem_path, tiled=True)

    # --- Add metadata tags using rasterio ---
    with rasterio.open(output_dem_path, "r+") as dst:
        dst.update_tags(
            1,
            coreg_method="NuthKaab+VerticalShift",
            coreg_shift_x=coreg_hori.meta["outputs"]["affine"]["shift_x"],
            coreg_shift_y=coreg_hori.meta["outputs"]["affine"]["shift_y"],
            coreg_shift_z=coreg_vert.meta["outputs"]["affine"]["shift_z"],
        )
    return output_dem_path


@task
def generate_ddem(
    dem_path1: str | Path, dem_path2: str | Path, output_path: str | Path, overwrite: bool = False
) -> Path:
    """
    Generates a differential DEM (dDEM) by subtracting one DEM from another.

    This function computes the pixel-wise difference between two input DEMs and saves the
    resulting dDEM to the specified output path. Existing outputs can be optionally overwritten.

    Args:
        dem_path1 (str | Path): Path to the first DEM (minuend).
        dem_path2 (str | Path): Path to the second DEM (subtrahend).
        output_path (str | Path): Path where the resulting dDEM will be saved.
        overwrite (bool, optional): Whether to overwrite the output if it already exists. Defaults to False.

    Returns:
        Path: Path to the generated dDEM file.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"Skip {output_path.stem}: output already exists.")
        return output_path

    dem1 = gu.Raster(dem_path1)
    dem2 = gu.Raster(dem_path2)
    ddem = dem1 - dem2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ddem.save(output_path)
    return output_path


@task
def extract_archive(archive_path: Path | str, output_dir: Path | str, overwrite: bool = False) -> None:
    """
    Extracts the contents of an archive file into a specified directory.

    Supports .zip, .7z, and various tar-based formats (.tar, .tgz, .gz, .bz2, .xz). The function
    can optionally overwrite an existing output directory or skip extraction if the directory exists.

    Args:
        archive_path (Path | str): Path to the archive file to extract.
        output_dir (Path | str): Directory where the archive contents will be extracted.
        overwrite (bool, optional): If True, existing output directories are removed before extraction.
            Defaults to False.

    Raises:
        ValueError: If the archive format is not supported.
    """

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    # Skip or overwrite
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Skipping extraction (folder exists): {output_dir}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraction logic
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif archive_path.suffix == ".7z":
        with py7zr.SevenZipFile(archive_path, mode="r") as szf:
            szf.extractall(output_dir)
    elif archive_path.suffix in [".tar", ".tgz", ".gz", ".bz2", ".xz"]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
    else:
        raise ValueError(f"Extraction for this type not implemented: {archive_path.suffix}")

    print(f"Extraction complete: {output_dir}")


@task
def convert_pointcloud_to_dem(
    pointcloud_path: str | Path,
    reference_dem_path: str | Path,
    output_dem_path: str | Path,
    pdal_exec_path: str = "pdal",
    output_pipeline_path: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    """
    Converts a point cloud file (LAS/LAZ) into a DEM raster using a reference DEM for alignment.

    This function generates a PDAL pipeline to read the point cloud, reproject it to the reference
    DEM's CRS if necessary, and interpolate it into a raster using IDW. The pipeline can be saved
    to a JSON file, and the output DEM is optionally overwritten or skipped if it already exists.

    Args:
        pointcloud_path (str | Path): Path to the input point cloud file.
        reference_dem_path (str | Path): Path to the reference DEM for CRS and resolution.
        output_dem_path (str | Path): Path to the output DEM file.
        pdal_exec_path (str, optional): Path to the PDAL executable. Defaults to "pdal".
        output_pipeline_path (str | Path | None, optional): Path to save the PDAL pipeline JSON.
            Defaults to a "processing_pipelines" folder near the output DEM.
        overwrite (bool, optional): Whether to overwrite the output DEM if it exists. Defaults to False.
        dry_run (bool, optional): If True, only writes the pipeline JSON without executing it. Defaults to False.

    Returns:
        Path: Path to the generated DEM file.
    """

    pointcloud_path = Path(pointcloud_path)
    output_dem_path = Path(output_dem_path)

    # not overwrite existing file
    if output_dem_path.exists() and not overwrite:
        print(f"Skip {output_dem_path.stem} : output already exists.")
        return output_dem_path

    output_pipeline_path = (
        output_dem_path.parent / "processing_pipelines" / f"pdal_pipeline_{output_dem_path.stem}.json"
        if output_pipeline_path is None
        else Path(output_pipeline_path)
    )
    output_pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(pointcloud_path) as las_reader:
        pc_crs = las_reader.header.parse_crs()

    ref_dem = gu.Raster(reference_dem_path)
    ref_crs = ref_dem.crs
    if ref_crs is None:
        raise ValueError(f"The reference dem {reference_dem_path} as no CRS.")
    ref_box = box(*ref_dem.bounds)

    # if not crs found in pc_crs test with a list of CRS
    if pc_crs is None:
        test_crs_list = [str(ref_crs), "EPSG:4326"]
        print(f"{pointcloud_path.name} : No CRS found, try CRSs : {test_crs_list}")

        # open the real bounding box of the pointcloud file
        las = laspy.read(pointcloud_path)
        pc_box = box(float(las.x.min()), float(las.y.min()), float(las.x.max()), float(las.y.max()))

        # buffered of 10% of area the ref_dem bounding box
        ref_box_buffered = ref_box.buffer(np.sqrt(ref_box.area) * 0.1)

        for tested_crs in test_crs_list:
            transformer = Transformer.from_crs(tested_crs, ref_crs, always_xy=True)
            pc_box_reprojected = transform(pc_box, transformer.transform, interleaved=False)
            if pc_box_reprojected.within(ref_box_buffered):
                pc_crs = tested_crs

    if pc_crs is None:
        raise ValueError(f"{pointcloud_path.name} : Can't find a valid CRS")

    # --- PDAL pipeline definition ---
    pipeline_dict = {
        "pipeline": [
            {"type": "readers.las", "filename": str(pointcloud_path)},
            {
                "type": "filters.reprojection",
                "in_srs": str(pc_crs),
                "out_srs": str(ref_crs),
            },
            {
                "type": "writers.gdal",
                "filename": str(output_dem_path),
                "resolution": ref_dem.res[0],
                "output_type": "idw",  # Interpolation like point2dem
                "data_type": "float32",
                "gdaldriver": "GTiff",
                "nodata": -9999,
                "origin_x": ref_dem.bounds.left,
                "origin_y": ref_dem.bounds.bottom,
                "width": ref_dem.shape[1],
                "height": ref_dem.shape[0],
            },
        ]
    }

    # write the pipeline in a json file
    with open(output_pipeline_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_dict, f, ensure_ascii=False, indent=4)

    if not dry_run:
        cmd = [pdal_exec_path, "pipeline", output_pipeline_path]
        subprocess.run(cmd, check=True)
        print(f"DEM successfully generated for {pointcloud_path.name}")

    return output_dem_path


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


@task
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


@task
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


def get_dem_files_from_std_dem(std_dem_file: str | Path, metadata_key: str = "dem_files") -> list[str]:
    """
    Retrieves the list of input DEM file paths stored in the metadata of a standard deviation DEM.

    Args:
        std_dem_file (str | Path): Path to the standard deviation DEM file.
        metadata_key (str, optional): Metadata tag name containing the input DEM file list.
            Defaults to "dem_files".

    Returns:
        list[str] | None: A list of DEM file paths if found in the metadata.
        Returns an empty list if the file does not exist, or None if the metadata key is missing.
    """
    std_dem_file = Path(std_dem_file)
    if not std_dem_file.exists():
        return []

    with rasterio.open(std_dem_file) as src:
        tags = src.tags(1)

        if metadata_key in tags:
            # Already present → return parsed JSON
            return json.loads(tags[metadata_key])
        else:
            return None


def is_existing_std_dem(dem_files: list[str | Path], output_path: str | Path, metadata_key: str = "dem_files") -> bool:
    """
    Check if an existing std DEM already matches the given input DEM list.

    Args:
        dem_files: List of input DEM paths used to compute the std DEM.
        output_path: Path to the supposed std DEM file.
        metadata_key: Metadata key used to store the original DEM file list.

    Returns:
        True if the file exists and its metadata matches the given DEM list, False otherwise.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return False

    try:
        with rasterio.open(output_path) as src:
            tags = src.tags(1)  # or src.tags() if not band-specific

        if metadata_key not in tags:
            return False

        # Normalize paths to absolute str for comparison
        founded_dem_files = [str(Path(p).resolve()) for p in json.loads(tags[metadata_key])]
        expected_dem_files = [str(Path(p).resolve()) for p in dem_files]

        return set(expected_dem_files) == set(founded_dem_files)

    except Exception as e:
        # Defensive: in case of malformed metadata or corrupted file
        print(f"[WARNING] Could not verify std_dem metadata ({output_path.name}): {e}")
        return False


@task
def create_std_dem(
    dem_files: list[str | Path], output_path: str | Path, block_size: int = 256, metadata_key: str = "dem_files"
) -> None:
    """
    Generates a standard deviation Digital Elevation Model (DEM) from a list of input DEM files.

    This function computes the pixel-wise standard deviation across multiple DEMs, processing
    the rasters block by block to efficiently handle large datasets.

    Args:
        dem_files (list[str | Path]): List of paths to input DEM raster files.
        output_path (str | Path): Path to the output standard deviation DEM file.
        block_size (int, optional): Size of the processing block in pixels. Defaults to 256.
        metadata_key (str, optional): Metadata tag name used to store the list of input DEM files
            in the output raster. Defaults to "dem_files".

    Returns:
        None: The function writes the resulting standard deviation DEM to the specified output path.
    """

    # first open the first raster of the list to have a reference profile
    with rasterio.open(dem_files[0]) as src_ref:
        profile = src_ref.profile.copy()
        width, height = src_ref.width, src_ref.height

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    profile.update(dtype="float32", count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        # Loop through the raster by windows
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                win = Window(
                    col_off=x,
                    row_off=y,
                    width=min(block_size, width - x),
                    height=min(block_size, height - y),
                )

                # Read the corresponding window from each DEM
                block_stack = []
                for dem_path in dem_files:
                    with rasterio.open(dem_path) as src:
                        data = src.read(1, window=win, masked=True).filled(np.nan)
                        block_stack.append(data)

                # Compute std for this block
                block_stack = np.stack(block_stack, axis=0)

                # Avoid computing std on empty slices
                if np.all(np.isnan(block_stack)):
                    block_std = np.full(block_stack.shape[1:], np.nan, dtype="float32")
                else:
                    block_std = np.nanstd(block_stack, axis=0).astype("float32")

                # Write the result
                dst.write(block_std, 1, window=win)

        # Add metadata tags
        dem_files_str = [str(p) for p in dem_files]
        dst.update_tags(1, **{metadata_key: json.dumps(dem_files_str)})


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
