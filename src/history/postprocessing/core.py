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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import geoutils as gu
import numpy as np
import pandas as pd
import xdem
from tqdm import tqdm

from history.postprocessing.io import PathsManager, parse_filename, uncompress_all_submissions
from history.postprocessing.utils import get_dems_df, get_pointcloud_df, load_coreg_results
from history.postprocessing.visualization import plot_files_recap


#######################################################################################################################
##                                                  MAIN
#######################################################################################################################
class PostProcessing:
    """
    Post-processing pipeline for photogrammetry and DEM generation.

    This class centralizes all operations required after initial photogrammetry
    processing, including:
        - Extracting and organizing submission archives
        - Generating DEMs from dense point clouds using ASP `point2dem`
        - Coregistering raw DEMs against reference DEMs
        - Computing metadata for point clouds and DEMs
        - Building consolidated DataFrames for analysis and reporting
        - Producing visual summaries and mosaics of DEM outputs

    It relies on a `PathsManager` instance to manage input and output directories,
    and interacts with external tools (e.g., ASP `point2dem`) as well as custom
    coregistration utilities.

    Attributes:
        paths_manager (PathsManager): Manager for locating input and output files
            within the processing workspace.

    Typical workflow:
        1. Uncompress submissions into working directories
        2. Convert dense point clouds into raw DEMs
        3. Coregister DEMs to reference datasets
        4. Build global DataFrames with filepaths and metadata
        5. Generate visual reports (recap plots, DEM mosaics)

    Example:
        >>> pp = PostProcessing(paths_manager)
        >>> pp.uncompress_all_submissions()
        >>> pp.iter_point2dem(overwrite=False)
        >>> coreg_df = pp.iter_coregister_dems()
        >>> pp.compute_global_df()
        >>> pp.plot_files_recap()
    """

    def __init__(self, paths_manager: PathsManager):
        """
        Initialize the post-processing pipeline.

        Args:
            paths_manager (PathsManager): Manager providing access to all input and
                output directories required for processing.
        """
        self.paths_manager = paths_manager

    def iter_point2dem(
        self,
        overwrite: bool = False,
        dry_run: bool = False,
        asp_path: str = None,
        max_concurrent_commands: int = 1,
        max_threads_per_command: int = 4,
    ) -> None:
        """
        Convert dense point cloud files into DEMs using the ASP `point2dem` tool.

        This method iterates over all available dense point cloud files, selects the
        appropriate reference DEM for each dataset and site, and generates aligned
        DEMs. Processing is executed in parallel using a process pool, with optional
        control over concurrency and threading per command.

        Args:
            overwrite (bool, optional): If True, overwrite existing DEMs. Defaults to False.
            dry_run (bool, optional): If True, simulate execution without running `point2dem`. Defaults to False.
            asp_path (str, optional): Path to the ASP `point2dem` executable. If None, assumes it is in PATH.
            max_concurrent_commands (int, optional): Maximum number of `point2dem` processes to run in parallel. Defaults to 1.
            max_threads_per_command (int, optional): Number of threads allocated per `point2dem` process. Defaults to 4.

        Returns:
            None
        """
        output_dir = self.paths_manager.get_path("raw_dems_dir")
        output_dir.mkdir(parents=True, exist_ok=True)

        # here hide the output of each command multiple command are running
        stdout = None if max_concurrent_commands == 1 else subprocess.DEVNULL

        with ProcessPoolExecutor(max_workers=max_concurrent_commands) as executor:
            futures = []
            for file in self.paths_manager.dense_pointcloud_files:
                code, metadatas = parse_filename(file)
                output_dem = output_dir / code

                # check the overwrite
                if os.path.exists(f"{output_dem}-DEM.tif") and not overwrite:
                    print(f"Skip {code} : {output_dem}-DEM.tif already exist.")
                    continue

                # skip if no reference DEM is provided
                try:
                    ref_dem_file, _ = self.paths_manager.get_ref_dem_and_mask(metadatas["site"], metadatas["dataset"])
                except Exception as e:
                    print(f"Error {code} : {e}")
                    continue

                # start a process of point2dem function
                futures.append(
                    executor.submit(
                        point2dem, file, output_dem, ref_dem_file, dry_run, asp_path, max_threads_per_command, stdout
                    )
                )

            # create a pbar if max_concurrent_commands > 1
            iterand = (
                as_completed(futures)
                if max_concurrent_commands == 1
                else tqdm(as_completed(futures), total=len(futures), desc="Converting into DEM", unit="File")
            )
            for future in iterand:
                try:
                    future.result()
                except Exception as e:
                    print(f"[!] Error: {e}")

    def iter_coregister_dems(
        self, overwrite: bool = False, dry_run: bool = False, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Coregister raw DEMs against reference DEMs and generate diagnostic outputs.

        This method processes all available raw DEM files by aligning them to
        reference DEMs using the `coregister_dem` function. It produces:
            - Coregistered DEMs
            - Difference DEMs (before and after correction)
            - Diagnostic plots

        For each DEM, overwrite conditions are checked, reference DEMs and masks are
        retrieved, and coregistration is applied unless `dry_run` is enabled. A
        summary of coregistration results is saved as a timestamped CSV file in
        `coreg_dems_dir`.

        Args:
            overwrite (bool, optional): If True, overwrite existing coregistered DEMs.
                Defaults to False.
            dry_run (bool, optional): If True, simulate coregistration without
                performing computations. Defaults to False.
            verbose (bool, optional): If True, print detailed progress information.
                Defaults to True.

        Returns:
            pd.DataFrame or None: DataFrame of coregistration results indexed by code,
            or None if no DEMs were processed.
        """

        data = []
        for file in self.paths_manager.raw_dem_files:
            code, metadatas = parse_filename(file)

            output_dem_path = self.paths_manager.get_path("coreg_dems_dir") / f"{code}-DEM_coreg.tif"

            # not overwrite existing files
            if output_dem_path.exists() and not overwrite:
                print(f"Skip {code} : {output_dem_path} already exist.")
                continue

            # read the corresponding mask and dem references
            try:
                ref_dem_file, ref_dem_mask_file = self.paths_manager.get_ref_dem_and_mask(
                    metadatas["site"], metadatas["dataset"]
                )
            except Exception as e:
                print(f"Error {code} : {e}")
                continue

            output_ddem_before_path = self.paths_manager.get_path("ddems_dir") / f"{code}-DDEM_before.tif"
            output_ddem_after_path = self.paths_manager.get_path("ddems_dir") / f"{code}-DDEM_after.tif"

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
                    )
                    res["code"] = code
                    data.append(res)
                except Exception as e:
                    print(f"Skip {code} : {e}")

        if data:
            df = pd.DataFrame(data).set_index("code")
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            filename = f"coreg_res_{datetime_str}.csv"
            df.to_csv(self.paths_manager.get_path("coreg_dems_dir") / filename)
            return df
        else:
            return None

    def compute_global_df(self) -> None:
        """
        Build and save a global DataFrame combining metadata from point clouds, DEMs,
        and coregistration results.

        This method collects filepaths and metadata from multiple sources, merges them
        into a single DataFrame, and exports the result as a CSV file for further
        analysis.

        Steps:
            1. Load base filepaths DataFrame from the paths manager.
            2. Join point cloud metadata.
            3. Join DEM metadata.
            4. Join coregistration results.
            5. Save the consolidated DataFrame to disk.

        Returns:
            None
        """
        df = self.paths_manager.get_filepaths_df()
        df = df.join(get_pointcloud_df(df), how="outer")
        df = df.join(get_dems_df(df), how="outer")
        df = df.join(load_coreg_results(self.paths_manager.get_path("coreg_dems_dir")), how="outer")
        df.to_csv(self.paths_manager.get_path("postproc_csv"))

    def uncompress_all_submissions(
        self, overwrite: bool = False, dry_run: bool = False, verbose: bool = False
    ) -> dict[str, str]:
        """
        Uncompress all raw submissions into the extracted submissions directory.

        This method takes archived submission files from the `raw_submissions_dir` and
        extracts their contents into the `extracted_submissions_dir`. It supports
        optional overwrite, dry-run mode, and verbose logging.

        Args:
            overwrite (bool, optional): If True, overwrite existing extracted files. Defaults to False.
            dry_run (bool, optional): If True, simulate extraction without writing to disk. Defaults to False.
            verbose (bool, optional): If True, print detailed processing information. Defaults to False.

        Returns:
            dict[str, str]: A mapping from original archive filenames to their extraction directories.
        """
        input_dir = self.paths_manager.get_path("raw_submissions_dir")
        output_dir = self.paths_manager.get_path("extracted_submissions_dir")
        return uncompress_all_submissions(input_dir, output_dir, overwrite, dry_run, verbose)

    def plot_files_recap(self, output_path: str | None = None, show: bool = True) -> None:
        """
        Generate a visual recap of available files for each code.

        This method creates a matrix-style plot showing the presence or absence of
        different file types (point clouds, camera parameters, DEMs, etc.) for each
        dataset entry. The plot can either be displayed, saved to disk, or both.

        Args:
            output_path (str | None, optional): Path to save the plot image. If None,
                the plot is not saved. Defaults to None.
            show (bool, optional): If True, display the plot interactively. Defaults to True.

        Returns:
            None
        """
        plot_files_recap(self.paths_manager.get_filepaths_df(), output_path, show)

    def get_global_df(self) -> pd.DataFrame:
        """
        Load the consolidated post-processing DataFrame from disk.

        This method reads the global CSV file previously generated by
        `compute_global_df` and returns it as a pandas DataFrame, indexed by code.

        Returns:
            pd.DataFrame: The consolidated DataFrame containing filepaths, metadata,
            and coregistration results for all processed datasets.
        """
        return pd.read_csv(self.paths_manager.get_path("postproc_csv"), index_col="code")


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

    return result
