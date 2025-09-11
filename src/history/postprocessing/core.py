"""
This module provides the PostProcessing class, which implements a complete pipeline
for converting point clouds into Digital Elevation Models (DEMs), coregistering DEMs
against reference datasets, and extracting metadata for analysis.

It integrates functions to:
    - Run ASP's point2dem tool to generate DEMs from point clouds.
    - Coregister DEMs to reference DEMs with mask support.
    - Produce difference DEMs (before and after correction) and diagnostic plots.
    - Collect and summarize metadata from point clouds and DEMs.
    - Compile all processing results into unified DataFrames for downstream analysis.

Intended Use:
    This module is designed for workflows in remote sensing, photogrammetry,
    and geomorphological studies where precise DEM generation and evaluation
    are required.
"""

import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from glob import glob
from pathlib import Path

import geoutils as gu
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xdem
from rasterio.enums import Resampling
from tqdm import tqdm


#######################################################################################################################
##                                                  MAIN
#######################################################################################################################
class PostProcessing:
    """
    The PostProcessing class provides tools to manage and analyze DEMs generated from point cloud data.

    This class supports the full workflow from raw point cloud files to processed DEMs, including:
    - Converting point clouds into DEMs using ASP's point2dem (parallel or sequential execution).
    - Coregistering DEMs against reference datasets and generating difference DEMs and plots.
    - Extracting and summarizing metadata from point clouds and DEMs.
    - Building comprehensive DataFrames linking processing outputs with metadata for further analysis.

    Key Features:
        - Parallelized or sequential DEM generation from point cloud files.
        - Automatic handling of overwrite rules, dry runs, and missing references.
        - Coregistration of DEMs with reference DEMs and corresponding masks.
        - Generation of difference DEMs (before and after correction) and diagnostic plots.
        - Extraction of point cloud metadata (e.g., LAS version, CRS, bounding box, point count).
        - Extraction of DEM metadata (e.g., NoData percentage, min/max values, resolution, CRS).
        - Compilation of global DataFrames summarizing file paths, metadata, and coregistration results.

    Typical Workflow:
        1. Generate raw DEMs from point clouds using `iter_point2dem` or `iter_point2dem_single_cmd`.
        2. Coregister DEMs using `iter_coregister_dems`.
        3. Analyze point cloud and DEM metadata with `_get_pointcloud_df`, `_get_dems_df`, and `_get_coreg_df`.
        4. Compile results into a global DataFrame using `compute_global_df`.

    Attributes:
        pointcloud_files (list[str]): List of paths to point cloud files to be processed.

    Returns:
        The class does not directly return values, but its methods produce DEM files,
        coregistered DEMs, difference DEMs, plots, and summary DataFrames.

    Note:
        Reference DEMs and masks must be available in the project directory structure
        for coregistration and DEM generation steps.
    """

    def __init__(
        self,
        pointcloud_files: list[str],
        raw_dems_directory: str | None = None,
        coregistered_dems_directory: str | None = None,
        ddems_before_directory: str | None = None,
        ddems_after_directory: str | None = None,
        plots_directory: str | None = None,
        postproc_csv: str | None = None,
        iceland_ref_dem_zoom: str | None = None,
        iceland_ref_dem_large: str | None = None,
        casagrande_ref_dem_zoom: str | None = None,
        casagrande_ref_dem_large: str | None = None,
        iceland_ref_dem_zoom_mask: str | None = None,
        iceland_ref_dem_large_mask: str | None = None,
        casagrande_ref_dem_zoom_mask: str | None = None,
        casagrande_ref_dem_large_mask: str | None = None,
    ):
        self.pointcloud_files = pointcloud_files
        self._paths = {
            "pointcloud_files": pointcloud_files,
            "raw_dems_directory": raw_dems_directory,
            "coregistered_dems_directory": coregistered_dems_directory,
            "ddems_before_directory": ddems_before_directory,
            "ddems_after_directory": ddems_after_directory,
            "plots_directory": plots_directory,
            "postproc_csv": postproc_csv,
            "iceland_ref_dem_zoom": iceland_ref_dem_zoom,
            "iceland_ref_dem_large": iceland_ref_dem_large,
            "casagrande_ref_dem_zoom": casagrande_ref_dem_zoom,
            "casagrande_ref_dem_large": casagrande_ref_dem_large,
            "iceland_ref_dem_zoom_mask": iceland_ref_dem_zoom_mask,
            "iceland_ref_dem_large_mask": iceland_ref_dem_large_mask,
            "casagrande_ref_dem_zoom_mask": casagrande_ref_dem_zoom_mask,
            "casagrande_ref_dem_large_mask": casagrande_ref_dem_large_mask,
        }

    def get_path(self, key: str) -> str:
        """
        Get a path from the dictionary by key.

        Raises
        ------
        KeyError
            If the key does not exist in the dictionary.
        ValueError
            If the key exists but its value is None.
        """
        if key not in self._paths:
            raise KeyError(f"Key '{key}' not found in PostProcessing paths.")

        path = self._paths[key]
        if path is None:
            raise ValueError(f"Path for key '{key}' is set to None.")

        return path

    def set_path(self, key: str, value: str) -> None:
        """
        Update or add a path in the dictionary.
        """
        self._paths[key] = value

    def iter_point2dem(
        self, overwrite: bool = False, dry_run: bool = False, asp_path: str = None, max_workers: int = 4
    ) -> None:
        """
        Convert all available point cloud files into DEMs using ASP's point2dem tool in parallel.

        This method iterates over all point cloud files, prepares output DEM file paths,
        checks whether processing should be skipped (based on existing files or missing references),
        and submits point2dem tasks to a process pool executor. The execution can be customized
        with overwrite, dry-run, and external ASP binary path options.

        Args:
            overwrite (bool, optional): If True, overwrite existing DEM files. Defaults to False.
            dry_run (bool, optional): If True, simulate processing without executing point2dem. Defaults to False.
            asp_path (str, optional): Path to the ASP binary directory. If None, assumes it is in PATH. Defaults to None.
            max_workers (int, optional): Maximum number of worker processes for parallel execution. Defaults to 4.

        Returns:
            None
        """

        os.makedirs(self.get_path("raw_dems_directory"), exist_ok=True)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in self.pointcloud_files:
                code, metadatas = parse_filename(file)
                output_dem = os.path.join(self.get_path("raw_dems_directory"), code)

                # check the overwrite
                if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
                    print(f"Skip {code} : {output_dem}-DEM.tif already exist.")
                    continue

                # skip if no reference DEM is provided
                try:
                    ref_dem_file = self._get_references_dem_and_mask(metadatas["site"], metadatas["dataset"])[0]
                except Exception as e:
                    print(f"Skip {code} : {e}")
                    continue

                # start a process of point2dem function
                futures.append(executor.submit(point2dem, file, output_dem, ref_dem_file, dry_run, asp_path))

            # Create the pbar and wait for all process to finish
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting into DEM", unit="File"):
                try:
                    future.result()
                except Exception as e:
                    print(f"[!] Error: {e}")

    def iter_point2dem_single_cmd(
        self, overwrite: bool = False, dry_run: bool = False, asp_path: str = None, max_workers: int = 4
    ) -> None:
        """
        Convert point cloud files into DEMs sequentially using ASP's point2dem tool.

        This method iterates over point cloud files, validates output DEM paths,
        checks for overwrite conditions, and ensures that reference DEMs are available.
        Each file is processed with a direct call to point2dem instead of parallel execution.

        Args:
            overwrite (bool, optional): If True, overwrite existing DEM files. Defaults to False.
            dry_run (bool, optional): If True, simulate processing without running point2dem. Defaults to False.
            asp_path (str, optional): Path to the ASP binary directory. If None, assumes it is in PATH. Defaults to None.
            max_workers (int, optional): Number of threads to use within the point2dem call. Defaults to 4.

        Returns:
            None
        """

        os.makedirs(self.get_path("raw_dems_directory"), exist_ok=True)

        for file in self.pointcloud_files:
            code, metadatas = parse_filename(file)
            output_dem = os.path.join(self.get_path("raw_dems_directory"), code)

            # check the overwrite
            if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
                print(f"Skip {code} : {output_dem}-DEM.tif already exist.")
                continue

            # skip if no reference DEM is provided
            try:
                ref_dem_file = self._get_references_dem_and_mask(metadatas["site"], metadatas["dataset"])[0]
            except Exception as e:
                print(f"Skip {code} : {e}")
                continue

            point2dem(file, output_dem, ref_dem_file, dry_run, asp_path, max_workers, None)

    def iter_coregister_dems(
        self, overwrite: bool = False, dry_run: bool = False, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Coregister raw DEMs against reference DEMs and generate outputs including coregistered DEMs,
        difference DEMs (before and after correction), and diagnostic plots.

        This method iterates over DEM files, checks overwrite conditions, retrieves reference DEMs
        and masks, and applies DEM coregistration. Results are stored in multiple output directories,
        and a summary DataFrame of coregistration results is returned and saved as a CSV file.

        Args:
            overwrite (bool, optional): If True, overwrite existing coregistered DEMs. Defaults to False.
            dry_run (bool, optional): If True, simulate coregistration without execution. Defaults to False.
            verbose (bool, optional): If True, print detailed processing information. Defaults to True.

        Returns:
            pd.DataFrame or None: A DataFrame containing coregistration results indexed by DEM code,
            or None if no DEMs were processed.
        """

        data = []
        for file in glob(os.path.join(self.get_path("raw_dems_directory"), "*-DEM.tif")):
            code, metadatas = parse_filename(file)

            output_dem_path = os.path.join(self.get_path("coregistered_dems_directory"), f"{code}-DEM_coreg.tif")

            # not overwrite existing files
            if os.path.exists(output_dem_path) and not overwrite:
                print(f"Skip {code} : {output_dem_path} already exist.")
                continue

            # read the corresponding mask and dem references
            try:
                ref_dem_file, ref_dem_mask_file = self._get_references_dem_and_mask(
                    metadatas["site"], metadatas["dataset"]
                )
            except Exception as e:
                print(f"Skip {code} : {e}")
                continue

            output_ddem_before_path = os.path.join(self.get_path("ddems_before_directory"), f"{code}-DDEM_before.tif")
            output_ddem_after_path = os.path.join(self.get_path("ddems_after_directory"), f"{code}-DDEM_after.tif")
            output_plot_path = os.path.join(self.get_path("plots_directory"), f"{code}.png")

            if verbose:
                print(f"coregister_dem({file}, {ref_dem_file}, {ref_dem_mask_file}, {output_dem_path})")
            if not dry_run:
                try:
                    res = coregister_dem(
                        file,
                        ref_dem_file,
                        ref_dem_mask_file,
                        output_dem_path,
                        output_ddem_before_path,
                        output_ddem_after_path,
                        output_plot_path,
                    )
                    res["code"] = code
                    data.append(res)
                except Exception as e:
                    print(f"Skip {code} : {e}")

        if data:
            df = pd.DataFrame(data).set_index("code")
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            filename = f"coreg_res_{datetime_str}.csv"
            df.to_csv(os.path.join(self.get_path("coregistered_dems_directory"), filename))
            return df
        else:
            return None

    def compute_global_df(self) -> pd.DataFrame:
        """
        Build a global DataFrame summarizing all available processing outputs.

        This method collects file paths for point clouds, raw DEMs, coregistered DEMs,
        and difference DEMs (before and after coregistration), then organizes them by code.
        Metadata extracted from filenames is merged with additional information from
        point clouds, DEMs, and coregistration results.

        Returns:
            pd.DataFrame: A DataFrame indexed by code, containing file paths, metadata,
            and processing-related information from point clouds, DEMs, and coregistration steps.
        """
        mapping = {
            "pointcloud_file": self.pointcloud_files,
            "raw_dem_file": glob(os.path.join(self.get_path("raw_dems_directory"), "*-DEM.tif")),
            "coreg_dem_file": glob(os.path.join(self.get_path("coregistered_dems_directory"), "*-DEM_coreg.tif")),
            "ddem_before_file": glob(os.path.join(self.get_path("ddems_before_directory"), "*-DDEM_before.tif")),
            "ddem_after_file": glob(os.path.join(self.get_path("ddems_after_directory"), "*-DDEM_after.tif")),
        }
        nested_dict = defaultdict(dict)
        for key, files in mapping.items():
            for file in files:
                code, metadatas = parse_filename(file)
                nested_dict[code]["code"] = code
                nested_dict[code].update(metadatas)
                nested_dict[code][key] = file
        df = pd.DataFrame(list(nested_dict.values())).set_index("code")
        df = df.join(self._get_pointcloud_df(df), how="outer")
        df = df.join(self._get_dems_df(df), how="outer")
        df = df.join(self._get_coreg_df(), how="outer")
        return df

    def _get_references_dem_and_mask(self, site: str, dataset: str) -> tuple[str, str]:
        """
        Retrieve the reference DEM and corresponding mask file for a given site and dataset.

        This method uses a predefined mapping of (site, dataset) pairs to locate the
        appropriate reference DEM and its mask within the project directory.

        Args:
            site (str): Name of the study site (e.g., "casa_grande", "iceland").
            dataset (str): Dataset type associated with the site (e.g., "aerial", "kh9mc", "kh9pc").

        Returns:
            tuple[str, str]: A tuple containing the file paths to the reference DEM and its mask.
        """
        mapping = {
            ("casa_grande", "aerial"): "casagrande_ref_dem_zoom",
            ("casa_grande", "kh9mc"): "casagrande_ref_dem_large",
            ("casa_grande", "kh9pc"): "casagrande_ref_dem_large",
            ("iceland", "aerial"): "iceland_ref_dem_zoom",
            ("iceland", "kh9mc"): "iceland_ref_dem_large",
            ("iceland", "kh9pc"): "iceland_ref_dem_large",
        }
        res = mapping[(site, dataset)]
        return self.get_path(res), self.get_path(res + "_mask")

    def _get_pointcloud_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metadata from point cloud files and return as a DataFrame.

        This method iterates over the rows of a given DataFrame, opens each point cloud
        file, and extracts metadata such as LAS version, CRS, point count, and bounding
        box coordinates. Results are compiled into a new DataFrame indexed by code.

        Args:
            df (pd.DataFrame): Input DataFrame containing a "pointcloud_file" column with file paths.

        Returns:
            pd.DataFrame: A DataFrame indexed by code, containing point cloud metadata including:
                - "las_version" (str): LAS file format version.
                - "pointcloud_crs" (CRS): Coordinate reference system of the point cloud.
                - "point_count" (int): Number of points in the cloud.
                - "bounds_x_min", "bounds_x_max" (float): Minimum and maximum X coordinates.
                - "bounds_y_min", "bounds_y_max" (float): Minimum and maximum Y coordinates.
                - "bounds_z_min", "bounds_z_max" (float): Minimum and maximum Z coordinates.
        """

        res = []
        for code, row in df.iterrows():
            if not pd.isna(row["pointcloud_file"]):
                try:
                    with laspy.open(row["pointcloud_file"]) as fh:
                        header = fh.header
                        row_dict = {"code": code}
                        row_dict.update(
                            {
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
                        )
                        res.append(row_dict)
                except Exception as e:
                    print(f"Warning: Could not process file '{row['pointcloud_file']}' ({e})")
        return pd.DataFrame(res).set_index("code")

    def _get_dems_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metadata from raw and coregistered DEM files and return as a DataFrame.

        This method iterates over the rows of a given DataFrame, reads raw and coregistered DEMs
        if available, and computes summary statistics (using `get_dem_informations`).
        Results are aggregated into a new DataFrame indexed by code, with prefixed column names
        to distinguish between raw and coregistered DEMs.

        Args:
            df (pd.DataFrame): Input DataFrame containing "raw_dem_file" and "coreg_dem_file" columns.

        Returns:
            pd.DataFrame: A DataFrame indexed by code, containing DEM metadata including:
                - "raw_dem_percent_nodata", "raw_dem_min", "raw_dem_max", "raw_dem_crs", "raw_dem_resolution"
                - "coreg_dem_percent_nodata", "coreg_dem_min", "coreg_dem_max", "coreg_dem_crs", "coreg_dem_resolution"
        """

        res = []
        for code, row in df.iterrows():
            dict_row = {"code": code}

            if not pd.isna(row["raw_dem_file"]):
                tmp_dict = self.get_dem_informations(row["raw_dem_file"])
                tmp_dict = {f"raw_dem_{k}": v for k, v in tmp_dict.items()}
                dict_row.update(tmp_dict)

            if not pd.isna(row["coreg_dem_file"]):
                tmp_dict = self.get_dem_informations(row["coreg_dem_file"])
                tmp_dict = {f"coreg_dem_{k}": v for k, v in tmp_dict.items()}
                dict_row.update(tmp_dict)

            res.append(dict_row)

        return pd.DataFrame(res).set_index("code")

    def _get_coreg_df(self) -> pd.DataFrame:
        """
        Load and merge coregistration result CSV files into a single DataFrame.

        This method searches for CSV files containing coregistration results,
        loads them, concatenates them into one DataFrame, and removes duplicate
        entries by keeping the first occurrence.

        Returns:
            pd.DataFrame: A DataFrame indexed by DEM code containing coregistration results.

        Raises:
            FileNotFoundError: If no coreg_res_*.csv files are found in the coregistered DEMs directory.
        """
        csv_files = glob(os.path.join(self.get_path("coregistered_dems_directory"), "coreg_res_*.csv"))

        if not csv_files:
            raise FileNotFoundError("No coreg_res_*.csv files found.")

        dfs = []
        for csv_file in csv_files:
            tmp_df = pd.read_csv(csv_file, index_col="code")
            dfs.append(tmp_df)

        df = pd.concat(dfs, axis=0)

        df = df[~df.index.duplicated(keep="first")]

        return df

    @staticmethod
    def get_dem_informations(file, reduction_factor: int = 20) -> dict:
        """
        Extract summary information from a DEM file with optional downsampling.

        This method reads a DEM raster, reduces its resolution by a specified factor,
        and computes basic statistics and metadata such as the percentage of NoData pixels,
        minimum and maximum values, CRS, and resolution.

        Args:
            file (str): Path to the DEM file.
            reduction_factor (int, optional): Factor by which to reduce DEM resolution
                for faster computation. Defaults to 20.

        Returns:
            dict: A dictionary containing DEM information with keys:
                - "percent_nodata" (float): Percentage of NoData pixels.
                - "min" (float or None): Minimum DEM value, or None if empty.
                - "max" (float or None): Maximum DEM value, or None if empty.
                - "crs" (CRS): Coordinate reference system of the DEM.
                - "resolution" (float): DEM spatial resolution.
        """

        res = {}
        with rasterio.open(file) as src:
            new_height = src.height // reduction_factor
            new_width = src.width // reduction_factor

            dem = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.nearest,  # conserve les voids
                masked=True,
            )
            res["percent_nodata"] = (dem.mask.sum() / dem.size) * 100
            res["min"] = float(dem.min()) if dem.count() > 0 else None
            res["max"] = float(dem.max()) if dem.count() > 0 else None
            res["crs"] = src.crs
            res["resolution"] = src.res[0]

        return res


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

    command = f'{point2dem_exec} --t_srs "{str_crs}" --tr {res} --t_projwin {str_bounds} --threads {max_workers} "{pointcloud_file}" -o "{output_dem}"'

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
) -> dict:
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
    result["mean_before_coreg"] = np.mean(ddem_bef_inlier)
    result["median_before_coreg"] = np.median(ddem_bef_inlier)
    result["nmad_before_coreg"] = gu.stats.nmad(ddem_bef_inlier)
    result["mean_after_coreg"] = np.mean(ddem_aft_inlier)
    result["median_after_coreg"] = np.median(ddem_aft_inlier)
    result["nmad_after_coreg"] = gu.stats.nmad(ddem_aft_inlier)

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
            f"dDEM before coregistration \n(mean: {result['mean_before_coreg']:.3f}, med: {result['median_before_coreg']:.3f}, nmad: {result['nmad_before_coreg']:.3f})"
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
            f"dDEM after coregistration \n(mean: {result['mean_after_coreg']:.3f}, med: {result['median_after_coreg']:.3f}, nmad: {result['nmad_after_coreg']:.3f})"
        )
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=200)
        plt.close()

    return result


def parse_filename(file: str) -> tuple[str, dict]:
    """Parse a file name into a code string and metadata dictionary.

    Args:
        file (str): Path or filename to parse.

    Returns:
        tuple[str, dict]: A tuple containing:
            - code (str): Reconstructed code from filename parts.
            - metadatas (dict): Parsed metadata mapping.
    """
    VALID_MAPPING = {
        "site": {"CG": "casa_grande", "IL": "iceland"},
        "dataset": {"AI": "aerial", "MC": "kh9mc", "PC": "kh9pc"},
        "images": {"RA": "raw", "PP": "preprocessed"},
        "camera_used": {"CY": True, "CN": False},
        "gcp_used": {"GY": True, "GN": False},
        "pointcloud_coregistration": {"PY": True, "PN": False},
        "mtp_adjustment": {"MY": True, "MN": False},
    }

    filename = os.path.basename(file)
    parts = filename.split("_")
    code = parts[0]
    metadatas = {"author": parts[0]}

    # Normalize to uppercase for consistency
    parts = [p.upper() for p in parts]

    # Fix special handling of MN/MY
    if parts[7].startswith("MN"):
        parts[7] = "MN"
    elif parts[7].startswith("MY"):
        parts[7] = "MY"

    # Check format length
    expected_parts = len(VALID_MAPPING) + 1  # author + mappings
    if len(parts) < expected_parts:
        raise ValueError(f"File {file} has unexpected format (expected ≥ {expected_parts} parts)")

    for i, (key, mapping) in enumerate(VALID_MAPPING.items()):
        value = parts[i + 1]
        if value not in mapping:
            raise ValueError(f"{value} is not a known code for {key}.")
        metadatas[key] = mapping[value]
        code += "_" + value

    return code, metadatas
