from pathlib import Path

import laspy
import pandas as pd
import rasterio


def get_pointcloud_df(df: pd.DataFrame) -> pd.DataFrame:
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
        dict_row = {"code": code}

        if not pd.isna(row["dense_pointcloud_file"]):
            tmp_dict = get_pointcloud_informations(row["dense_pointcloud_file"])
            tmp_dict = {f"dense_{k}": v for k, v in tmp_dict.items()}
            dict_row.update(tmp_dict)

        if not pd.isna(row["sparse_pointcloud_file"]):
            tmp_dict = get_pointcloud_informations(row["sparse_pointcloud_file"])
            tmp_dict = {f"sparse_{k}": v for k, v in tmp_dict.items()}
            dict_row.update(tmp_dict)

        res.append(dict_row)

    return pd.DataFrame(res).set_index("code")


def get_dems_df(df: pd.DataFrame) -> pd.DataFrame:
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
            tmp_dict = get_dem_informations(row["raw_dem_file"])
            tmp_dict = {f"raw_dem_{k}": v for k, v in tmp_dict.items()}
            dict_row.update(tmp_dict)

        if not pd.isna(row["coreg_dem_file"]):
            tmp_dict = get_dem_informations(row["coreg_dem_file"])
            tmp_dict = {f"coreg_dem_{k}": v for k, v in tmp_dict.items()}
            dict_row.update(tmp_dict)

        res.append(dict_row)

    return pd.DataFrame(res).set_index("code")


def load_coreg_results(coreg_dir: Path | str) -> pd.DataFrame:
    """
    Load and merge coregistration result CSV files into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with 'code' as index.
        If no CSV files are found, returns an empty DataFrame.
    """
    csv_files = list(Path(coreg_dir).glob("coreg_res_*.csv"))

    if not csv_files:
        return pd.DataFrame()

    dfs = [pd.read_csv(csv_file, index_col="code") for csv_file in csv_files]

    df = pd.concat(dfs, axis=0)

    df = df[~df.index.duplicated(keep="first")]

    return df


def get_dem_informations(file) -> dict:
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
        dem = src.read(
            1,
            masked=True,
        )
        res["percent_nodata"] = dem.mask.mean() * 100
        res["min"] = float(dem.min()) if dem.count() > 0 else None
        res["max"] = float(dem.max()) if dem.count() > 0 else None
        res["crs"] = src.crs
        res["resolution"] = src.res[0]

    return res


def get_pointcloud_informations(file: str) -> dict:
    try:
        with laspy.open(file) as fh:
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
        print(f"Warning: Could not process file '{file}' ({e})")
        return {}
