import os
import sys
from pathlib import Path
import requests

import xdem
import matplotlib.pyplot as plt
import subprocess
import geoutils as gu
import geopandas as gpd
from tqdm.auto import tqdm
from shapely.geometry import Polygon
import numpy as np


def retrieve_lat_lon_codes(lat, lon):
    """
    Given latitude and longitude in decimals, returns a code:
    - for latitude N/S leading characters and 2 digits
    - for longitude W/E leading characters and 3 digits
    Longitude must be between -180 and 180, latitude between -90 and 90.
    """
    if lon > 180 or lon < -180:
        raise ValueError("`lon` must be in the range [-180, 180]")
    if lat > 90 or lat < -90:
        raise ValueError("`lat` must be in the range [-90, 90]")

    if lon > 0:
        lon_code = f"E{lon:03d}"
    else:
        lon_code = f"W{-lon:03d}"

    if lat > 0:
        lat_code = f"N{lat:02d}"
    else:
        lat_code = f"S{-lat:02d}"

    return lat_code, lon_code

def download_cop30_tiles(bounding_box, outfolder, overwrite=False):
    """
    Download all Copernicus DEM tiles in the bounding box (xmin, ymin, xmax, ymax) specified in degrees.
    The GTiff files are saved in outfolder and the list of files is returned.
    """
    base_url = 'https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/'

    all_tiles = []

    for lon in range(bounding_box[0], bounding_box[2]):
        for lat in range(bounding_box[1], bounding_box[3]):

            lat_code, lon_code = retrieve_lat_lon_codes(lat, lon)
            tile_name = f'Copernicus_DSM_10_{lat_code}_00_{lon_code}_00_DEM.tif'
            tile_url = "/".join([base_url.rstrip("/"), tile_name]) 
            outfile = os.path.join(outfolder, tile_name)

            cmd = f"wget {tile_url} -O {outfile}"

            try:
                all_tiles.append(outfile)
                if overwrite or (not os.path.exists(outfile)):
                    print(cmd); subprocess.run(cmd, shell=True, check=True)

            # If error is returned, tile probably does not exist
            except subprocess.CalledProcessError:
                print(f"Tile {tile_url} does not exist")

    return all_tiles


def download_cop30_tiles_reproject(bounding_box, outfolder, epsg_code, res=[30, 30], overwrite=False):
    """
    Download all Copernicus DEM tiles in the bounding box (xmin, ymin, xmax, ymax) specified in degrees and \
    reproject to the given EPSG code projection and resolution.

    The GTiff files are saved in outfolder and the list of files is returned.
    """
    base_url = 'https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/'

    all_tiles = []

    for lon in range(bounding_box[0], bounding_box[2]):
        for lat in range(bounding_box[1], bounding_box[3]):

            lat_code, lon_code = retrieve_lat_lon_codes(lat, lon)
            tile_name = f'Copernicus_DSM_10_{lat_code}_00_{lon_code}_00_DEM.tif'
            tile_url = "/".join([base_url.rstrip("/"), tile_name]) 
            outfile = os.path.join(outfolder, tile_name)

            gdal_cmd = f"gdalwarp -r cubic -t_srs 'EPSG:{epsg_code}' -tr {res[0]} {res[1]} -tap /vsicurl/{tile_url} {outfile} -co COMPRESS=LZW -co TILED=yes -dstnodata -9999 -overwrite"
            all_tiles.append(outfile)

            if overwrite or (not os.path.exists(outfile)):
                try:
                    print(gdal_cmd)
                    subprocess.run(gdal_cmd, shell=True, check=True)
                except subprocess.CalledProcessError:
                    print(f"Tile {tile_url} does not exist")

    return all_tiles


def geodiff(dem1_fn, dem2_fn, vmax=10, src_vcrs=None, dst_vcrs=None):
    dem1, dem2 = gu.raster.load_multiple_rasters([dem1_fn, dem2_fn], crop=True, ref_grid=0)
    if src_vcrs is not None:
        dem2 = xdem.DEM(dem2)
        dem2.set_vcrs(src_vcrs)
        dem2 = dem2.to_vcrs(dst_vcrs)
    ddem = dem2 - dem1
    ddem.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax)
    plt.show()


def reproject_gdal(src_fn, ref_fn, out_fn, resampling_algo="bilinear", overwrite=True):
    """
    Reproject file src_fn on the same CRS and grid as ref_fn using gdalwarp.
    """

    ref_rst = gu.Raster(ref_fn)
    te_str = " ".join(np.array(ref_rst.bounds).astype("str"))
    tr_str = " ".join(np.array(ref_rst.res).astype("str"))
    crs_str = ref_rst.crs.to_proj4()
    cmd = f"gdalwarp -r {resampling_algo} -te {te_str} -tr {tr_str} -t_srs '{crs_str}' {src_fn} {out_fn} -co compress=DEFLATE"
    if overwrite:
        cmd += " -overwrite"
    print(cmd); subprocess.run(cmd, shell=True, check=True)


def coregister(dem_ref_fn, dem_tbc_fn, gl_outlines_fn=None, vmax=10):

    dem_ref, dem_tbc = gu.raster.load_multiple_rasters([dem_ref_fn, dem_tbc_fn], crop=True, ref_grid=0)
    
    ddem_before = dem_tbc - dem_ref

    coreg = xdem.coreg.NuthKaab()
    gl_outlines = gu.Vector(gl_outlines_fn)
    gl_mask = gl_outlines.create_mask(ddem_before)
    dem_coreg = coreg.fit_and_apply(dem_ref, dem_tbc, inlier_mask=~gl_mask)
    coreg.info()

    ddem_after = dem_coreg - dem_ref

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    plt.show()


def download_esa_worldcover(output_folder, bounds, year=2021, overwrite=False, dryrun=False):
    """
    Function to download the ESA worldcover tiles for a given area of interest.

    Largely inspired from https://github.com/ESA-WorldCover/esa-worldcover-datasets/blob/main/scripts/download.py.

    Args:
        output_folder (str or Path): Folder to save the downloaded files.
        bounds (list of float): List of 4 floats representing the bounding box in the format [xmin, ymin, xmax, ymax].
        year (int): Map year (2020 or 2021), defaults to the most recent 2021 map.
        overwrite (bool): If True, overwrite existing files.
        dryrun (bool): If True, only print the download commands without executing them.
    """
    # Base URL
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

    # algo version (depends on the year)
    version = {2020: 'v100',
               2021: 'v200'}[year]

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # load worldcover grid
    url = f'{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson'
    grid = gpd.read_file(url)

    # Find tiles intersecting with bounds
    geom = Polygon.from_bounds(*bounds)
    tiles = grid[grid.intersects(geom)]

    if tiles.shape[0] == 0:
        print(f"No tiles in the selected area {geom.bounds}")
        sys.exit()

    out_files = []
    for tile in tqdm(tiles.ll_tile):
        url = f"{s3_url_prefix}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        out_fn = os.path.join(output_folder, f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")
        out_files.append(out_fn)
        
        if Path(out_fn).is_file() and not overwrite:
            print(f"{out_fn} already exists.")
            continue

        if not dryrun:
            r = requests.get(url, allow_redirects=True)
            with open(out_fn, 'wb') as f:
                f.write(r.content)
        else:
            print(f"Downloading {url} to {out_fn}")

    return out_files
