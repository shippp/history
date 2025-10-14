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
from pathlib import Path

import geoutils as gu
import numpy as np
import rasterio
import xdem
from tqdm import tqdm

from history.postprocessing.io import PathsManager, uncompress_all_submissions
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

        # open the filepaths df with only valid dense pointcloud files
        filepaths_df = self.paths_manager.get_filepaths_df().dropna(subset="dense_pointcloud_file")
        print(f"{len(filepaths_df)} dense point cloud file(s) found.")

        if not overwrite:
            mask = filepaths_df["raw_dem_file"].isna()
            filepaths_df = filepaths_df[mask]
            if sum(~mask) > 0:
                print(f"Skipping {sum(~mask)} existing Poind2Dem result(s) (overwrite disabled).")

        with ProcessPoolExecutor(max_workers=max_concurrent_commands) as executor:
            futures = []
            for code, row in filepaths_df.iterrows():
                # check the overwrite
                output_dem = output_dir / code

                # skip if the referecne DEM doesn't exists
                ref_dem_file = row["ref_dem_file"]
                if not os.path.exists(ref_dem_file):
                    print(f"Error {code} : Reference dem not found at {ref_dem_file}")
                    continue

                # start a process of point2dem function
                futures.append(
                    executor.submit(
                        point2dem,
                        row["dense_pointcloud_file"],
                        output_dem,
                        ref_dem_file,
                        dry_run,
                        asp_path,
                        max_threads_per_command,
                        stdout,
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

    def iter_coregister_dems(self, overwrite: bool = False, dry_run: bool = False, verbose: bool = True) -> None:
        """
        Iterate over all DEMs listed in the filepaths table and perform coregistration
        with their corresponding reference DEMs and masks.

        This function automates batch DEM alignment by calling `coregister_dem()` for each entry.
        It manages overwriting behavior, supports dry-run mode, and prints progress
        information when verbose mode is enabled.

        Args:
            overwrite (bool, optional):
                If True, overwrite existing coregistered DEMs.
                If False, skip DEMs that already have a coregistered output.
                Default is False.
            dry_run (bool, optional):
                If True, simulate the process without performing any coregistration
                (useful for debugging or checking setup). Default is False.
            verbose (bool, optional):
                If True, print detailed progress and executed commands. Default is True.

        Returns:
            None

        Notes:
            - The method expects valid paths for each DEM entry in the filepaths table:
              `raw_dem_file`, `ref_dem_file`, and `ref_dem_mask_file`.
            - Output files are written to the `coreg_dems_dir` defined by `self.paths_manager`.
            - Coregistration errors are caught per DEM to ensure uninterrupted iteration.
            - The function does not modify the input DataFrame or return any results.

        Example:
            >>> pipeline.iter_coregister_dems(
            ...     overwrite=False,
            ...     dry_run=False,
            ...     verbose=True
            ... )
        """
        filepaths_df = self.paths_manager.get_filepaths_df().dropna(subset="raw_dem_file")
        print(f"{len(filepaths_df)} raw DEM file(s) found.")

        # manage to not overwrite existing coregistered files
        if not overwrite:
            mask = filepaths_df["coreg_dem_file"].isna()
            filepaths_df = filepaths_df[mask]
            if sum(~mask) > 0:
                print(f"Skipping {sum(~mask)} existing coregistered result(s) (overwrite disabled).")

        for code, row in filepaths_df.iterrows():
            input_file = row["raw_dem_file"]
            output_dem_path = self.paths_manager.get_path("coreg_dems_dir") / f"{code}-DEM_coreg.tif"
            output_ddem_before_path = self.paths_manager.get_path("ddems_before_dir") / f"{code}-DDEM.tif"
            output_ddem_after_path = self.paths_manager.get_path("ddems_after_dir") / f"{code}-DDEM.tif"

            # read the corresponding mask and dem references
            ref_dem_file = row["ref_dem_file"]
            if not os.path.exists(ref_dem_file):
                print(f"Error {code} : Reference dem not found at {ref_dem_file}")
                continue
            ref_dem_mask_file = row["ref_dem_mask_file"]
            if not os.path.exists(ref_dem_mask_file):
                print(f"Error {code} : Reference mask dem not found at {ref_dem_mask_file}")
                continue

            if verbose:
                print(f"coregister_dem({input_file}, {ref_dem_file}, {ref_dem_mask_file}, {output_dem_path})")
            if not dry_run:
                try:
                    coregister_dem(
                        input_file,
                        ref_dem_file,
                        ref_dem_mask_file,
                        output_dem_path,
                        output_ddem_before_path,
                        output_ddem_after_path,
                    )
                except Exception as e:
                    print(f"Skip {code} : {e}")

    def generate_ddems(self, overwrite: bool = False, verbose: bool = True) -> None:
        filepaths_df = self.paths_manager.get_filepaths_df()

        # generate ddem before coregistration
        for input_colname, output_colname, output_dir_key in [
            ("raw_dem_file", "ddem_before_file", "ddems_before_dir"),
            ("coreg_dem_file", "ddem_after_file", "ddems_after_dir"),
        ]:
            droped_df = filepaths_df.dropna(subset=input_colname)

            if verbose:
                print(f"{len(droped_df)} {input_colname} found(s)")

            # manage to not overwrite existing coregistered files
            if not overwrite:
                mask = droped_df[output_colname].isna()
                droped_df = droped_df[mask]
                if sum(~mask) > 0 and verbose:
                    print(f"Skipping {sum(~mask)} existing dDEMs result(s) (overwrite disabled).")

            for code, row in droped_df.iterrows():
                output_dem_path = self.paths_manager.get_path(output_dir_key) / f"{code}-DDEM.tif"
                dem1_path = row[input_colname]
                dem2_path = row["ref_dem_file"]
                mask_path = row["ref_dem_mask_file"]

                generate_ddem(dem1_path, dem2_path, output_dem_path, mask_path)

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
    output_ddem_before_path: str | None,
    output_ddem_after_path: str | None,
) -> None:
    """
    Coregister a DEM to a reference DEM using combined horizontal and vertical corrections.

    This function performs a two-step DEM coregistration:
        1. Horizontal correction using the Nuth & Kääb (2011) algorithm.
        2. Vertical correction using a median-based vertical shift model.

    Both corrections are applied sequentially to minimize systematic elevation differences
    between the input DEM and a reference DEM. The function also computes differential DEMs
    (dDEMs) before and after coregistration if requested.

    A reference mask is used to exclude invalid or unreliable pixels (e.g., nodata, clouds, water).
    For the horizontal coregistration, an additional slope-based filter is applied to remove
    nearly flat terrain areas that can bias the shift estimation.

    The resulting coregistered DEM is saved to disk and annotated with metadata tags
    describing the applied coregistration method and estimated translation parameters.

    Args:
        dem_path (str): Path to the input DEM to be coregistered.
        ref_dem_path (str): Path to the reference DEM used as spatial and vertical reference.
        ref_dem_mask_path (str): Path to the binary mask indicating valid reference DEM pixels.
        output_dem_path (str): Path where the coregistered DEM will be saved.
        output_ddem_before_path (str | None): Optional path to save the differential DEM
            before coregistration (input DEM - reference DEM).
        output_ddem_after_path (str | None): Optional path to save the differential DEM
            after coregistration (coregistered DEM - reference DEM).

    Returns:
        None

    Notes:
        - The function assumes all DEMs and the mask share the same spatial resolution,
          projection, and extent. DEMs are reprojected to the reference grid if necessary.
        - The horizontal correction uses the Nuth & Kääb (2011) algorithm implemented in xDEM.
        - The vertical correction is applied via a median-based shift to minimize vertical bias.
        - Coregistration shifts are stored as raster tags (`coreg_shift_x`, `coreg_shift_y`, `coreg_shift_z`).
        - If dDEMs are saved, they are masked using the same valid-pixel mask as the reference DEM.
    """
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

    if output_ddem_before_path is not None:
        ddem_before = dem - dem_ref
        Path(output_ddem_before_path).parent.mkdir(exist_ok=True, parents=True)
        ddem_before.save(output_ddem_before_path)

    if output_ddem_after_path is not None:
        ddem_after = dem_coreg - dem_ref
        Path(output_ddem_after_path).parent.mkdir(exist_ok=True, parents=True)
        ddem_after.save(output_ddem_after_path)


def generate_ddem(dem1_path: str, dem2_path: str, output_dem_path: str, mask_path: str | None = None) -> None:
    """
    Generate a differential DEM (dDEM = DEM1 - DEM2), optionally applying a mask.

    This function loads two DEM rasters, reprojects DEM1 onto the grid of DEM2,
    computes their difference, and saves the result as a new GeoTIFF.
    If a binary mask is provided, pixels outside the mask are set to NaN in the output.

    Parameters
    ----------
    dem1_path : str
        Path to the first DEM (typically the "before" DEM).
    dem2_path : str
        Path to the second DEM (typically the "after" or reference DEM).
    output_dem_path : str
        Path where the differential DEM (dDEM) will be saved.
    mask_path : str or None, optional
        Optional path to a binary mask raster. If provided, only pixels where the
        mask is True (nonzero) will be retained in the output; others will be set to NaN.

    Returns
    -------
    None
        The resulting differential DEM is written to disk.

    Notes
    -----
    - The function ensures that DEM1 is reprojected to match DEM2 (extent, CRS, resolution).
    - The mask, if provided, is also reprojected to DEM2’s grid for consistency.
    - Output is saved as a tiled GeoTIFF using GeoUtils' `Raster.save()` method.

    Examples
    --------
    >>> generate_ddem(
    ...     dem1_path="before_dem.tif",
    ...     dem2_path="after_dem.tif",
    ...     output_dem_path="ddem.tif",
    ...     mask_path="inlier_mask.tif"
    ... )
    """
    # Load reference DEM
    dem2 = gu.Raster(dem2_path)

    # Load and reproject the first DEM to match dem2
    dem1 = gu.Raster(dem1_path).reproject(dem2, silent=True)

    # Compute the differential DEM
    ddem = dem1 - dem2

    # Apply mask if provided
    if mask_path is not None:
        mask_raster = gu.Raster(mask_path).reproject(dem2, silent=True)
        mask = mask_raster.data.astype(bool)
        # Apply mask: keep valid pixels only
        ddem.data[~mask] = np.nan

    # Ensure output directory exists
    output_path = Path(output_dem_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the dDEM
    ddem.save(output_path, tiled=True)
