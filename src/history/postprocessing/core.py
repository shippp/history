import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from glob import glob
from pathlib import Path

import geoutils as gu
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xdem
from tqdm import tqdm

from .file_naming import FileNaming

DF_SCHEMA = {
    "author": "string",
    "site": "string",
    "dataset": "string",
    "images": "string",
    "camera_used": "bool",
    "gcp_used": "bool",
    "pointcloud_coregistration": "bool",
    "mtp_adjustment": "bool",
    "pointcloud_type": "bool",
    # pointclouds
    "pointcloud_file": "string",
    "pointcloud_crs": "object",
    "point_count": "Int64",
    "bounds_x_min": "float",
    "bounds_x_max": "float",
    "bounds_y_min": "float",
    "bounds_y_max": "float",
    "bounds_z_min": "float",
    "bounds_z_max": "float",
    # raw dems
    "raw_dem_file": "string",
    "dem_percent_nodata": "float",
    "dem_res": "Int64",
    # coregistration
    "coregistered_dem_file": "string",
    "ddem_before_file": "string",
    "ddem_after_file": "string",
    "coreg_shift_x": "float",
    "coreg_shift_y": "float",
    "coreg_shift_z": "float",
    "mean_before_coreg": "float",
    "median_before_coreg": "float",
    "nmad_before_coreg": "float",
    "mean_after_coreg": "float",
    "median_after_coreg": "float",
    "nmad_after_coreg": "float",
}


#######################################################################################################################
##                                                  MAIN
#######################################################################################################################
class PostProcessing:
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

        # create the base dataframe with all primary information
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
            "pointcloud_file": "string",
        }
        self.base_df = pd.DataFrame(columns=schema.keys()).astype(schema)
        self.base_df.index.name = "code"
        # Process each raster file
        for file in self.pointcloud_files:
            fn = FileNaming(file)
            code = fn["code"]
            for k in fn:
                if k != "code":
                    self.base_df.at[code, k] = fn[k]
            self.base_df.at[code, "pointcloud_file"] = file

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

    def analyze_files(self, verbose: bool = True) -> pd.DataFrame:
        df = self._get_base_df()

        if verbose:
            print(f"Total numbers of pointcloud files : {df.shape[0]}")

        for col in ["raw_dem", "coregistered_dem", "ddem_before", "ddem_after"]:
            df[col] = False
        for col in ["raw_dem_file", "coregistered_dem_file", "ddem_before_file", "ddem_after_file"]:
            df[col] = pd.NA

        for code, row in df.iterrows():
            dem_file = os.path.join(self.get_path("raw_dems_directory"), f"{code}-DEM.tif")
            coregistered_dem_file = os.path.join(self.get_path("coregistered_dems_directory"), f"{code}-DEM_coreg.tif")
            ddem_before_file = os.path.join(self.get_path("ddems_before_directory"), f"{code}-DDEM_before.tif")
            ddem_after_file = os.path.join(self.get_path("ddems_after_directory"), f"{code}-DDEM_after.tif")

            if os.path.exists(dem_file):
                df.at[code, "raw_dem"] = True
                df.at[code, "raw_dem_file"] = dem_file

                if os.path.exists(coregistered_dem_file):
                    df.at[code, "coregistered_dem"] = True
                    df.at[code, "coregistered_dem_file"] = coregistered_dem_file

                    if os.path.exists(ddem_before_file):
                        df.at[code, "ddem_before"] = True
                        df.at[code, "ddem_before_file"] = ddem_before_file

                    if os.path.exists(ddem_after_file):
                        df.at[code, "ddem_after"] = True
                        df.at[code, "ddem_after_file"] = ddem_after_file

        if verbose:
            print(f"Total numbers of raw DEM founds : {df['raw_dem'].sum()}/{df.shape[0]}")
            print(f"Total numbers of coregistered DEM founds : {df['coregistered_dem'].sum()}/{df.shape[0]}")
            print(f"Total numbers of DDEM before coreg founds : {df['ddem_before'].sum()}/{df.shape[0]}")
            print(f"Total numbers of DDEM after coreg founds : {df['ddem_after'].sum()}/{df.shape[0]}")

        return df

    def compute_postproc_df(self, force_compute: bool = False, verbose: bool = False) -> pd.DataFrame:
        # initiate an empty df with the good schema and types
        df = pd.DataFrame(columns=DF_SCHEMA.keys()).astype(DF_SCHEMA)
        df.index.name = "code"

        # fill basic fields with pointcloud file names
        for file in self.pointcloud_files:
            fn = FileNaming(file)
            code = fn["code"]
            for k in fn:
                if k != "code":
                    df.at[code, k] = fn[k]
            df.at[code, "pointcloud_file"] = file

        # open existing df and check is format
        if not force_compute and os.path.exists(self.get_path("postproc_csv")):
            existing_df = pd.read_csv(self.get_path("postproc_csv"), index_col="code")
            if not set(DF_SCHEMA.keys()).issubset(existing_df.columns):
                print(f"Warning : the existing csv file {self.get_path('postproc_csv')} is invalid.")
                existing_df = None
        else:
            existing_df = None

        self._fill_df_with_filepaths(df, verbose)
        self._fill_df_with_pointclouds_info(df, existing_df, verbose)

        df.to_csv(self.get_path("postproc_csv"))
        return df

    def iter_point2dem(
        self, overwrite: bool = False, dry_run: bool = False, asp_path: str = None, max_workers: int = 4
    ) -> None:
        os.makedirs(self.get_path("raw_dems_directory"), exist_ok=True)
        df = self._get_base_df()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for code, row in df.iterrows():
                output_dem = os.path.join(self.get_path("raw_dems_directory"), code)

                # check the overwrite
                if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
                    print(f"Skip {code} : {output_dem}-DEM.tif already exist.")
                    continue

                # skip if no reference DEM is provided
                try:
                    ref_dem_file = self._get_references_dem_and_mask(row["site"], row["dataset"])[0]
                except Exception as e:
                    print(f"Skip {code} : {e}")
                    continue

                # start a process of point2dem function
                futures.append(
                    executor.submit(point2dem, row["pointcloud_file"], output_dem, ref_dem_file, dry_run, asp_path)
                )

            # Create the pbar and wait for all process to finish
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting into DEM", unit="File"):
                try:
                    future.result()
                except Exception as e:
                    print(f"[!] Error: {e}")

    def iter_point2dem_single_cmd(
        self, overwrite: bool = False, dry_run: bool = False, asp_path: str = None, max_workers: int = 4
    ) -> None:
        os.makedirs(self.get_path("raw_dems_directory"), exist_ok=True)
        df = self._get_base_df()

        for code, row in df.iterrows():
            output_dem = os.path.join(self.get_path("raw_dems_directory"), code)

            # check the overwrite
            if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
                print(f"Skip {code} : {output_dem}-DEM.tif already exist.")
                continue

            # skip if no reference DEM is provided
            try:
                ref_dem_file = self._get_references_dem_and_mask(row["site"], row["dataset"])[0]
            except Exception as e:
                print(f"Skip {code} : {e}")
                continue

            point2dem(row["pointcloud_file"], output_dem, ref_dem_file, dry_run, asp_path, max_workers, None)

    def iter_coregister_dems(
        self, overwrite: bool = False, dry_run: bool = False, verbose: bool = True
    ) -> pd.DataFrame:
        df = self._get_base_df()

        data = []
        for code, row in df.iterrows():
            raw_dem_file = os.path.join(self.get_path("raw_dems_directory"), f"{code}-DEM.tif")
            if not os.path.exists(raw_dem_file):
                print(f"Skip {code} : no DEM found")
            else:
                output_dem_path = os.path.join(self.get_path("coregistered_dems_directory"), f"{code}-DEM_coreg.tif")

                # not overwrite existing files
                if os.path.exists(output_dem_path) and not overwrite:
                    print(f"Skip {code} : {output_dem_path} already exist.")
                    continue

                # read the corresponding mask and dem references
                try:
                    ref_dem_file, ref_dem_mask_file = self._get_references_dem_and_mask(row["site"], row["dataset"])
                except Exception as e:
                    print(f"Skip {code} : {e}")
                    continue

                output_ddem_before_path = os.path.join(
                    self.get_path("ddems_before_directory"), f"{code}-DDEM_before.tif"
                )
                output_ddem_after_path = os.path.join(self.get_path("ddems_after_directory"), f"{code}-DDEM_after.tif")
                output_plot_path = os.path.join(self.get_path("plots_directory"), f"{code}.png")

                if verbose:
                    print(f"coregister_dem({raw_dem_file}, {ref_dem_file}, {ref_dem_mask_file}, {output_dem_path})")
                if not dry_run:
                    try:
                        res = coregister_dem(
                            raw_dem_file,
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

    def get_statistics(self, verbose: bool = True) -> pd.DataFrame:
        df = self.analyze_files(verbose)
        int_col = ["point_count", "dem_res"]
        float_col = [
            "bounds_x_min",
            "bounds_x_max",
            "bounds_y_min",
            "bounds_y_max",
            "bounds_z_min",
            "bounds_z_max",
            "coreg_shift_x",
            "coreg_shift_y",
            "coreg_shift_z",
            "mean_before_coreg",
            "median_before_coreg",
            "nmad_before_coreg",
            "mean_after_coreg",
            "median_after_coreg",
            "nmad_after_coreg",
        ]
        for col in int_col + float_col:
            df[col] = np.nan
        df["pointcloud_crs"] = pd.NA

        for code, row in df.iterrows():
            # get the statistics of the pointcloud file
            try:
                with laspy.open(row["pointcloud_file"]) as fh:
                    header = fh.header
                    df.at[code, "pointcloud_crs"] = header.parse_crs()
                    pointcloud_infos = {
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
                print(f"Warning: Could not process file '{row['pointcloud_file']}' ({e})")

            # get the resolution of the dem
            if row["raw_dem"]:
                try:
                    raster = gu.Raster(row["raw_dem_file"])
                    df.at[code, "dem_res"] = raster.res[0]

                    # number of nodata in the mask
                    n_nodata = np.count_nonzero(np.ma.getmaskarray(raster.data.squeeze()))
                    total_pixels = raster.shape[0] * raster.shape[1]
                    df.at[code, "percent_nodata"] = (n_nodata / total_pixels) * 100

                except Exception as e:
                    print(f"Warning: Could not process file '{row['raw_dem_file']}' ({e})")

        # populate coregistration statistics with all csv saved in the coreg dems directory
        csv_files = glob(os.path.join(self.get_path("coregistered_dems_directory"), "coreg_res_*.csv"))
        for csv_file in csv_files:
            tmp_df = pd.read_csv(csv_file, index_col="code")
            for code, row in tmp_df.iterrows():
                for key, value in row.items():
                    df.at[code, key] = value
        return df

    def get_adv_statistics(self, verbose: bool = True) -> pd.DataFrame:
        df = self.get_statistics(verbose)
        cols = ["percent_nodata"]

        for col in cols:
            df[col] = pd.NA

        for code, row in df.iterrows():
            # get the nodata percent
            if row["raw_dem"]:
                try:
                    raster = gu.Raster(row["raw_dem_file"])

                    # number of nodata in the mask
                    n_nodata = np.count_nonzero(np.ma.getmaskarray(raster.data.squeeze()))
                    total_pixels = raster.shape[0] * raster.shape[1]
                    df.at[code, "percent_nodata"] = (n_nodata / total_pixels) * 100

                except Exception as e:
                    print(f"Warning: Could not process file '{row['raw_dem_file']}' ({e})")
        return df

    def _get_references_dem_and_mask(self, site: str, dataset: str) -> tuple[str, str]:
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

    def _get_base_df(self) -> pd.DataFrame:
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
            "pointcloud_file": "string",
        }
        df = pd.DataFrame(columns=schema.keys()).astype(schema)
        df.index.name = "code"
        # Process each raster file
        for file in self.pointcloud_files:
            fn = FileNaming(file)
            code = fn["code"]
            for k in fn:
                if k != "code":
                    df.at[code, k] = fn[k]
            df.at[code, "pointcloud_file"] = file
        return df.sort_index()

    def _fill_df_with_filepaths(self, df: pd.DataFrame, verbose: bool = False) -> None:
        if verbose:
            print(f"Total numbers of pointcloud files : {df.shape[0]}")

        for code, row in df.iterrows():
            dem_file = os.path.join(self.get_path("raw_dems_directory"), f"{code}-DEM.tif")
            coregistered_dem_file = os.path.join(self.get_path("coregistered_dems_directory"), f"{code}-DEM_coreg.tif")
            ddem_before_file = os.path.join(self.get_path("ddems_before_directory"), f"{code}-DDEM_before.tif")
            ddem_after_file = os.path.join(self.get_path("ddems_after_directory"), f"{code}-DDEM_after.tif")

            if os.path.exists(dem_file):
                df.at[code, "raw_dem_file"] = dem_file

            if os.path.exists(coregistered_dem_file):
                df.at[code, "coregistered_dem_file"] = coregistered_dem_file

            if os.path.exists(ddem_before_file):
                df.at[code, "ddem_before_file"] = ddem_before_file

            if os.path.exists(ddem_after_file):
                df.at[code, "ddem_after_file"] = ddem_after_file

        if verbose:
            print(f"Total numbers of raw DEM founds : {df['raw_dem_file'].count()}/{df.shape[0]}")
            print(f"Total numbers of coregistered DEM founds : {df['coregistered_dem_file'].count()}/{df.shape[0]}")
            print(f"Total numbers of DDEM before coreg founds : {df['ddem_before_file'].count()}/{df.shape[0]}")
            print(f"Total numbers of DDEM after coreg founds : {df['ddem_after_file'].count()}/{df.shape[0]}")

    def _fill_df_with_pointclouds_info(
        self, df: pd.DataFrame, existing_df: pd.DataFrame | None, verbose: bool = False
    ) -> None:
        cols = [
            "pointcloud_crs",
            "bounds_x_min",
            "bounds_x_max",
            "bounds_y_min",
            "bounds_y_max",
            "bounds_z_min",
            "bounds_z_max",
            "point_count",
        ]
        for code, row in df.iterrows():
            try:
                if existing_df.loc[code, cols].notna().all():
                    df.loc[code, cols] = existing_df.loc[code, cols]
            except Exception:
                print("acces file")
                try:
                    with laspy.open(row["pointcloud_file"]) as fh:
                        header = fh.header
                        df.at[code, "pointcloud_crs"] = header.parse_crs()
                        pointcloud_infos = {
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
                    print(f"Warning: Could not process file '{row['pointcloud_file']}' ({e})")

    def _fill_df_with_dems_info(self, df: pd.DataFrame, force_compute: bool = False, verbose: bool = False) -> None:
        pass


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

    command = f'{point2dem_exec} --t_srs "{str_crs}" --tr {res} --t_projwin {str_bounds} --threads {max_workers} --datum WGS84  "{pointcloud_file}" -o "{output_dem}"'

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

    return result
