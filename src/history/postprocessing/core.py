import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geoutils as gu
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xdem
from tqdm import tqdm

from .file_naming import FileNaming

#######################################################################################################################
##                                                  MAIN
#######################################################################################################################


def iter_point2dem(
    postprocess_df: str,
    output_directory: str,
    overwrite: bool = False,
    dry_run: bool = False,
    asp_path: str = None,
    max_workers: int = 5,
) -> None:
    """
    Run point2dem conversions in parallel for a set of point clouds defined in a DataFrame.

    This function iterates over rows of a DataFrame containing point cloud and reference DEM
    information, and uses the ASP `point2dem` tool to generate DEMs in parallel.
    Existing DEMs are skipped unless `overwrite=True`. If no reference DEM is provided,
    the dataset is skipped.

    Parameters
    ----------
    postprocess_df : pandas.DataFrame
        A DataFrame containing metadata for each dataset, including paths to the
        point cloud (`pointcloud_file`), the reference DEM (`ref_dem_file`), and
        identifiers such as `site` and `dataset`.
    output_directory : str
        Path to the directory where generated DEMs will be stored.
    overwrite : bool, optional
        If True, overwrite existing DEM files. If False (default), skip already existing DEMs.
    dry_run : bool, optional
        If True, do not execute the point2dem command, only simulate the workflow (default: False).
    asp_path : str, optional
        Path to the ASP installation or to the `point2dem` binary. If None, assumes it is in PATH.
    max_workers : int, optional
        Number of parallel worker processes to use for DEM generation (default: 5).

    Returns
    -------
    None
        The function runs processes and writes DEM files to the output directory.
    """
    os.makedirs(output_directory, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for code, row in postprocess_df.iterrows():
            output_dem = os.path.join(output_directory, code)

            # check the overwrite
            if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
                print(f"Skip {code} : {output_dem}-DEM.tif already exist.")
                continue

            # skip if no reference DEM is provided
            if pd.isna(row["ref_dem_file"]) or row["ref_dem_file"] is None:
                print(f"Skip {code} : No reference DEM provided (site: {row['site']}, dataset: {row['dataset']}).")
                continue

            # start a process of point2dem function
            futures.append(
                executor.submit(point2dem, row["pointcloud_file"], output_dem, row["ref_dem_file"], dry_run, asp_path)
            )

        # Create the pbar and wait for all process to finish
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting into DEM", unit="File"):
            try:
                future.result()
            except Exception as e:
                print(f"[!] Error: {e}")


def iter_point2dem_single_cmd(
    postprocess_df: str,
    output_directory: str,
    overwrite: bool = False,
    dry_run: bool = False,
    asp_path: str = None,
    max_workers: int = 5,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    for code, row in postprocess_df.iterrows():
        output_dem = os.path.join(output_directory, code)

        # check the overwrite
        if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
            print(f"Skip {code} : {output_dem}-DEM.tif already exist.")
            continue

        # skip if no reference DEM is provided
        if pd.isna(row["ref_dem_file"]) or row["ref_dem_file"] is None:
            print(f"Skip {code} : No reference DEM provided (site: {row['site']}, dataset: {row['dataset']}).")
            continue

        point2dem(row["pointcloud_file"], output_dem, row["ref_dem_file"], dry_run, asp_path, max_workers, None)


def iter_coregister_dems(
    postprocess_df: pd.DataFrame,
    output_directory: str,
    output_ddem_before_directory: str | None = None,
    output_ddem_after_directory: str | None = None,
    output_plot_directory: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run DEM coregistration for multiple datasets defined in a DataFrame.

    This function iterates through a DataFrame containing raw DEMs and reference DEMs,
    performs DEM coregistration using the `coregister_dem` function, and updates the
    DataFrame with output file paths and metrics. DEMs without the required input files
    are skipped. Results include the coregistered DEM, difference DEMs (before and after),
    and optional diagnostic plots.

    Parameters
    ----------
    postprocess_df : pandas.DataFrame
        A DataFrame with at least the following columns:
        - ``raw_dem_file`` : path to the input DEM to be coregistered.
        - ``ref_dem_file`` : path to the reference DEM.
        - ``ref_dem_mask_file`` : path to the reference DEM mask.
        - ``site`` and ``dataset`` : metadata used for logging.
    output_directory : str
        Path to the directory where the coregistered DEMs will be saved.
    output_ddem_before_directory : str, optional
        Directory to save DDEMs (difference DEMs) before coregistration.
    output_ddem_after_directory : str, optional
        Directory to save DDEMs (difference DEMs) after coregistration.
    output_plot_directory : str, optional
        Directory to save diagnostic plots (PNG files).
    overwrite : bool, optional
        If True, overwrite existing outputs. If False (default), skip files that already exist.
    dry_run : bool, optional
        If True, only print the planned coregistration commands without executing them (default: False).

    Returns
    -------
    pandas.DataFrame
        The input DataFrame updated with new columns:
        - ``coregistered_dem_file`` : path to the coregistered DEM.
        - ``ddem_before_file`` : path to the DDEM before coregistration (if requested).
        - ``ddem_after_file`` : path to the DDEM after coregistration (if requested).
        - ``ddem_plot_file`` : path to the diagnostic plot (if requested).
        Plus any additional metrics returned by `coregister_dem`.

    Notes
    -----
    - Files are skipped if missing or if already existing (unless `overwrite=True`).
    - The function assumes the existence of a `coregister_dem` function that performs the actual coregistration.
    """
    for code, row in postprocess_df.iterrows():
        if pd.isna(row["raw_dem_file"]) or not os.path.exists(row["raw_dem_file"]):
            print(f"Skip {code} : no DEM found")
        else:
            output_dem_path = os.path.join(output_directory, f"{code}-DEM_coreg.tif")

            # if output_ddem_before_directory is set manage the path for the DDEM before coreg
            if output_ddem_before_directory:
                output_ddem_before_path = os.path.join(output_ddem_before_directory, f"{code}-DDEM_before.tif")
            else:
                output_ddem_before_path = None

            # if output_ddem_after_directory is set manage the path for the DDEM after coreg
            if output_ddem_after_directory:
                output_ddem_after_path = os.path.join(output_ddem_after_directory, f"{code}-DDEM_after.tif")
            else:
                output_ddem_after_path = None

            # if output_plot_directory is set manage the path for the plot
            if output_plot_directory:
                output_plot_path = os.path.join(output_plot_directory, f"{code}.png")
            else:
                output_plot_path = None

            # not overwrite existing files
            if os.path.exists(output_dem_path) and not overwrite:
                print(f"Skip {code} : {output_dem_path} already exist.")
                continue

            # skip if no reference DEM is provided
            if pd.isna(row["ref_dem_file"]):
                print(f"Skip {code} : No reference DEM provided (site: {row['site']}, dataset: {row['dataset']}).")
                continue

            # skip if no reference mask DEM is provided
            if pd.isna(row["ref_dem_mask_file"]):
                print(f"Skip {code} : No reference DEM mask provided (site: {row['site']}, dataset: {row['dataset']}).")
                continue

            print(
                f"coregister_dem({row['raw_dem_file']}, {row['ref_dem_file']}, {row['ref_dem_mask_file']}, {output_dem_path})"
            )
            if not dry_run:
                s = coregister_dem(
                    row["raw_dem_file"],
                    row["ref_dem_file"],
                    row["ref_dem_mask_file"],
                    output_dem_path,
                    output_ddem_before_path,
                    output_ddem_after_path,
                    output_plot_path,
                )
                for col, value in s.items():
                    postprocess_df.at[code, col] = value
                postprocess_df.at[code, "coregistered_dem_file"] = output_dem_path
                postprocess_df.at[code, "ddem_before_file"] = output_ddem_before_path
                postprocess_df.at[code, "ddem_after_file"] = output_ddem_after_path
                postprocess_df.at[code, "ddem_plot_file"] = output_plot_path

    return postprocess_df


def init_postprocessed_df(pointcloud_files: list[str]) -> pd.DataFrame:
    """
    Initialize a postprocessing DataFrame from a list of point cloud files.

    This function creates a DataFrame following a predefined schema to store metadata,
    file paths, and processing results related to DEM generation and coregistration.
    For each input point cloud, metadata is extracted from its filename using
    `FileNaming`, and if the file is LAS/LAZ, point cloud statistics are read using `laspy`.

    Parameters
    ----------
    pointcloud_files : list of str
        List of file paths to point cloud files (LAS/LAZ). Other file types can be included,
        but point cloud statistics will only be extracted from LAS/LAZ files.

    Returns
    -------
    pandas.DataFrame
        A DataFrame initialized with the following schema:
        - Metadata: ``author``, ``site``, ``dataset``, ``images``, ``camera_used``,
          ``gcp_used``, ``pointcloud_coregistration``, ``mtp_adjustment``, ``dense``.
        - Point cloud info: ``pointcloud_file``, ``point_count``,
          ``bounds_x_min``, ``bounds_x_max``, ``bounds_y_min``, ``bounds_y_max``,
          ``bounds_z_min``, ``bounds_z_max``.
        - Reference DEMs: ``ref_dem_file``, ``ref_dem_mask_file``.
        - Raw DEMs: ``raw_dem_file``, ``dem_res``.
        - Coregistration outputs: ``coregistered_dem_file``, ``ddem_before_file``,
          ``ddem_after_file``, ``ddem_plot_file``, ``coreg_shift_x``, ``coreg_shift_y``,
          ``coreg_shift_z``, ``before_coreg_mean``, ``before_coreg_median``,
          ``before_coreg_nmad``, ``after_coreg_mean``, ``after_coreg_median``,
          ``after_coreg_nmad``.

    Notes
    -----
    - The DataFrame index is set to the unique code extracted from the filename.
    - LAS/LAZ files are opened with `laspy` to extract bounding box and point count.
    - If a file cannot be processed with `laspy`, a warning is printed and the file is skipped.
    """
    # Load existing CSV if it exists and not overwrite

    schema = {
        "author": "string",
        "site": "string",
        "dataset": "string",
        "images": "string",
        "camera_used": "bool",
        "gcp_used": "bool",
        "pointcloud_coregistration": "bool",
        "mtp_adjustment": "bool",
        "pointcloud_type": "bool",
        # point cloud
        "pointcloud_file": "string",
        "point_count": "Int64",
        "bounds_x_min": "float64",
        "bounds_x_max": "float64",
        "bounds_y_min": "float64",
        "bounds_y_max": "float64",
        "bounds_z_min": "float64",
        "bounds_z_max": "float64",
        # references dems
        "ref_dem_file": "string",
        "ref_dem_mask_file": "string",
        # raw dems
        "raw_dem_file": "string",
        "dem_res": "Int64",
        # coregistrations
        "coregistered_dem_file": "string",
        "ddem_before_file": "string",
        "ddem_after_file": "string",
        "ddem_plot_file": "string",
        "coreg_shift_x": "float64",
        "coreg_shift_y": "float64",
        "coreg_shift_z": "float64",
        "before_coreg_mean": "float64",
        "before_coreg_median": "float64",
        "before_coreg_nmad": "float64",
        "after_coreg_mean": "float64",
        "after_coreg_median": "float64",
        "after_coreg_nmad": "float64",
    }

    df = pd.DataFrame(columns=schema.keys()).astype(schema)

    df.index.name = "code"

    # Process each raster file
    for file in pointcloud_files:
        fn = FileNaming(file)
        code = fn["code"]
        for k in fn:
            if k != "code":
                df.at[code, k] = fn[k]

        if file.lower().endswith((".laz", ".las")):
            try:
                with laspy.open(file) as fh:
                    header = fh.header

                    pointcloud_infos = {
                        "pointcloud_file": file,
                        "bounds_x_min": header.mins[0],
                        "bounds_x_max": header.maxs[0],
                        "bounds_y_min": header.mins[1],
                        "bounds_y_max": header.maxs[1],
                        "bounds_z_min": header.mins[2],
                        "bounds_z_max": header.maxs[2],
                        "point_count": header.point_count,
                    }
                    for col, val in pointcloud_infos.items():
                        df.at[code, col] = val
            except Exception as e:
                print(f"Warning: Could not process file '{file}' ({e})")
                continue

    return df.sort_index()


def set_df_reference_dems(
    df: pd.DataFrame,
    iceland_ref_dem_zoom: str | None = None,
    iceland_ref_dem_large: str | None = None,
    casagrande_ref_dem_zoom: str | None = None,
    casagrande_ref_dem_large: str | None = None,
    iceland_ref_dem_zoom_mask: str | None = None,
    iceland_ref_dem_large_mask: str | None = None,
    casagrande_ref_dem_zoom_mask: str | None = None,
    casagrande_ref_dem_large_mask: str | None = None,
) -> pd.DataFrame:
    """
    Assign reference DEMs and masks to a DataFrame based on site and dataset type.

    This function updates the DataFrame with paths to reference DEMs and their
    corresponding masks depending on the site (`iceland` or `casa_grande`) and
    dataset type (`aerial` vs. non-aerial). Reference DEMs and masks are provided
    as function arguments.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least the columns ``site`` and ``dataset``.
    iceland_ref_dem_zoom : str, optional
        Path to the Iceland aerial reference DEM (zoom).
    iceland_ref_dem_large : str, optional
        Path to the Iceland non-aerial reference DEM (large coverage).
    casagrande_ref_dem_zoom : str, optional
        Path to the Casa Grande aerial reference DEM (zoom).
    casagrande_ref_dem_large : str, optional
        Path to the Casa Grande non-aerial reference DEM (large coverage).
    iceland_ref_dem_zoom_mask : str, optional
        Path to the mask file corresponding to the Iceland aerial reference DEM.
    iceland_ref_dem_large_mask : str, optional
        Path to the mask file corresponding to the Iceland non-aerial reference DEM.
    casagrande_ref_dem_zoom_mask : str, optional
        Path to the mask file corresponding to the Casa Grande aerial reference DEM.
    casagrande_ref_dem_large_mask : str, optional
        Path to the mask file corresponding to the Casa Grande non-aerial reference DEM.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with two new string columns:
        - ``ref_dem_file`` : assigned reference DEM path.
        - ``ref_dem_mask_file`` : assigned mask file path.

    Notes
    -----
    - Rows are selected based on the combination of ``site`` and ``dataset`` values.
    - Aerial datasets receive the zoom DEMs, while non-aerial datasets receive the large DEMs.
    - Both columns are explicitly cast to string type.
    """
    df.loc[(df["site"] == "casa_grande") & (df["dataset"] == "aerial"), "ref_dem_file"] = casagrande_ref_dem_zoom
    df.loc[(df["site"] == "casa_grande") & (df["dataset"] != "aerial"), "ref_dem_file"] = casagrande_ref_dem_large
    df.loc[(df["site"] == "iceland") & (df["dataset"] == "aerial"), "ref_dem_file"] = iceland_ref_dem_zoom
    df.loc[(df["site"] == "iceland") & (df["dataset"] != "aerial"), "ref_dem_file"] = iceland_ref_dem_large

    df.loc[(df["site"] == "casa_grande") & (df["dataset"] == "aerial"), "ref_dem_mask_file"] = (
        casagrande_ref_dem_zoom_mask
    )
    df.loc[(df["site"] == "casa_grande") & (df["dataset"] != "aerial"), "ref_dem_mask_file"] = (
        casagrande_ref_dem_large_mask
    )
    df.loc[(df["site"] == "iceland") & (df["dataset"] == "aerial"), "ref_dem_mask_file"] = iceland_ref_dem_zoom_mask
    df.loc[(df["site"] == "iceland") & (df["dataset"] != "aerial"), "ref_dem_mask_file"] = iceland_ref_dem_large_mask

    for col in ["ref_dem_file", "ref_dem_mask_file"]:
        df[col] = df[col].astype("string")

    return df


def add_dems_to_df(
    postprocess_df: str,
    dems_directory: str,
) -> pd.DataFrame:
    """
    Add DEM file paths and resolutions to a postprocessing DataFrame.

    This function checks for the presence of DEM files in a given directory,
    following the naming convention ``{code}-DEM.tif`` where ``code`` matches
    the DataFrame index. If a DEM is found, its path and spatial resolution
    are added to the corresponding row of the DataFrame.

    Parameters
    ----------
    postprocess_df : pandas.DataFrame
        A DataFrame indexed by unique dataset codes. Must contain the column
        ``raw_dem_file`` (to be updated).
    dems_directory : str
        Path to the directory containing DEM files named ``{code}-DEM.tif``.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with the following columns populated (if DEMs exist):
        - ``raw_dem_file`` : full path to the raw DEM file.
        - ``dem_res`` : DEM spatial resolution (pixel size in X).

    Notes
    -----
    - DEMs are loaded using `geoutils.Raster` to extract their resolution.
    - Only DEMs present in the directory are added; missing DEMs are ignored.
    """
    for code, row in postprocess_df.iterrows():
        dem_file = os.path.join(dems_directory, f"{code}-DEM.tif")
        if os.path.exists(dem_file):
            raster = gu.Raster(dem_file)
            postprocess_df.at[code, "raw_dem_file"] = dem_file
            postprocess_df.at[code, "dem_res"] = raster.res[0]
    return postprocess_df


def find_pointclouds(root_dir: str) -> list[str]:
    """
    Recursively search for point cloud files in a directory.

    This function looks for LAS/LAZ files ending with either
    'sparse_pointcloud.las', 'sparse_pointcloud.laz',
    'dense_pointcloud.las', or 'dense_pointcloud.laz'.

    Parameters
    ----------
    root_dir : str
        Root directory to search in.

    Returns
    -------
    list of str
        List of matching file paths.
    """
    root = Path(root_dir)
    matches = []
    patterns = [
        "*sparse_pointcloud.las",
        "*sparse_pointcloud.laz",
        "*dense_pointcloud.las",
        "*dense_pointcloud.laz",
    ]

    for pattern in patterns:
        matches.extend(root.rglob(pattern))

    return [str(path) for path in matches]


#######################################################################################################################
##                                                  PRIVATE
#######################################################################################################################


def point2dem(
    pointcloud_file: str,
    output_dem: str,
    ref_dem: str,
    dry_run: bool = False,
    asp_path: str = None,
    max_workers: int = 1,
    stdout: int | None = subprocess.DEVNULL,
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
    asp_path : str, optional
        Path to the ASP installation directory. If None, assumes `point2dem` is in system PATH.
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
    ref_raster = gu.Raster(ref_dem)

    bounds = ref_raster.bounds
    str_bounds = f"{bounds.left} {bounds.bottom} {bounds.right} {bounds.top}"

    str_crs = ref_raster.crs.to_proj4()

    res = ref_raster.res[0]

    if asp_path is not None:
        point2dem_exec = os.path.join(asp_path, "point2dem")
    else:
        point2dem_exec = "point2dem"

    command = f'{point2dem_exec} --t_srs "{str_crs}" --tr {res} --t_projwin {str_bounds} --threads {max_workers} --datum WGS84  {pointcloud_file} -o {output_dem}'

    if dry_run:
        print(command)
    else:
        # we don't want the standard output of the command for the multi processing
        subprocess.run(command, shell=True, check=True, stdout=stdout)


def coregister_dem(
    dem_path: str,
    ref_dem_path: str,
    ref_dem_mask_path: str,
    output_dem_path: str,
    output_ddem_before_path: str | None = None,
    output_ddem_after_path: str | None = None,
    output_plot_path: str | None = None,
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
    output_plot_path : str or None, optional
        If provided, path where the plot of difference DEM will be saved.

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

    # save a plot of ddem before transformation and after
    if output_plot_path:
        if os.path.dirname(output_plot_path):
            os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        ddem_before.plot(
            cmap="coolwarm",
            vmin=-10,
            vmax=10,
            cbar_title="Elevation difference (m)",
            ax=axes[0],
        )
        axes[0].set_title(
            f"dDEM before coregistration \n(mean: {result['before_coreg_mean']:.3f}, med: {result['before_coreg_median']:.3f}, nmad: {result['before_coreg_nmad']:.3f})"
        )
        axes[0].axis("off")

        ddem_after.plot(
            cmap="coolwarm",
            vmin=-10,
            vmax=10,
            cbar_title="Elevation difference (m)",
            ax=axes[1],
        )
        axes[1].set_title(
            f"dDEM after coregistration \n(mean: {result['after_coreg_mean']:.3f}, med: {result['after_coreg_median']:.3f}, nmad: {result['after_coreg_nmad']:.3f})"
        )
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=200)
        plt.close()

    return pd.Series(result, name=os.path.basename(dem_path))
