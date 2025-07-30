"""
Script to prepare the auxiliary data for the Casa Grande site for the History project. The following data are prepared:
- reference lidar DEM over Maricopa county, provided by the USGS: https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/1m/Projects/AZ_MaricopaPinal_2020_B20/
- Copernicus 30 m DEM: 1x1 degree tiles are downloaded, merged and reprojected on the same horizontal and vertical reference system as lidar DEM. It is then coregistered horizontally with a Nuth & Kaan (2011) algorithm and vertically by calculating a median shift in stable areas. Stable areas are defined as bare land, grassland and shrubland in the ESA worldcover (see below) and excluding sibsiding land (see below).
- ESA worldcover dataset: each 1x1 degree tile are downloaded from the S3 bucket and merged
- Land subsidence vector file downloaded from https://azgeo-open-data-agic.hub.arcgis.com/datasets/azwater::land-subsidence/explore

Author: Amaury Dehecq
Last modified: June 2025 
"""

import os
import subprocess
from glob import glob

import src.history.aux_data.download_tools as dt
import xdem
import matplotlib.pyplot as plt
import geoutils as gu
from skimage import filters, morphology
import numpy as np
import pandas as pd
import geopandas as gpd

# Create output folder
outfolder = "./casa_grande/aux/"
os.makedirs(outfolder, exist_ok=True)

overwrite = False

# Bounding box of the two area of interest
zoom_bounds = [414000, 3613020, 444000, 3650010]
large_bounds = [261990, 3405030, 612990, 3880020]
epsg_str = "EPSG:26912"

# Save to gjson files
zoom_gdf = gpd.GeoDataFrame(geometry=[gu.projtools.bounds2poly(zoom_bounds),], crs=epsg_str)
zoom_gdf.to_file(os.path.join(outfolder, "zoom_aoi.geojson"))

large_gdf = gpd.GeoDataFrame(geometry=[gu.projtools.bounds2poly(large_bounds),], crs=epsg_str)
large_gdf.to_file(os.path.join(outfolder, "large_aoi.geojson"))

# Convert bounding boxes to latlon
zoom_bounds_latlon = gu.projtools.bounds2poly(zoom_bounds, in_crs=epsg_str, out_crs="EPSG:4326").bounds
large_bounds_latlon = gu.projtools.bounds2poly(large_bounds, in_crs=epsg_str, out_crs="EPSG:4326").bounds

# and swap axes for lon/lat
zoom_bounds_lonlat = [zoom_bounds_latlon[k] for k in [1, 0, 3, 2]]
large_bounds_lonlat = [large_bounds_latlon[k] for k in [1, 0, 3, 2]]


# -- Download ESA worldcover classification --

landcover_folder = os.path.join(outfolder, "landcover")
os.makedirs(landcover_folder, exist_ok=True)

landcover_vrt_fn = os.path.join(landcover_folder, "tmp.vrt")

if overwrite or (not os.path.exists(landcover_vrt_fn)):

    print("\nProcessing ESA world_landcover")
    landcover_tiles = dt.download_esa_worldcover(landcover_folder, large_bounds_lonlat, year=2021, overwrite=overwrite, dryrun=False)

    # Create mosaic
    list_fn = os.path.join(landcover_folder, "list_tiles.txt")
    pd.Series(landcover_tiles).to_csv(list_fn, header=False, index=False)

    cmd = f"gdalbuildvrt -r nearest {landcover_vrt_fn} -input_file_list {list_fn} -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    print("\nDone with ESA world_landcover\n")

# -- Download land subsidence mask --
# For now, file has to be downloaded manually from https://azgeo-open-data-agic.hub.arcgis.com/datasets/azwater::land-subsidence/explore

subsid_poly_fn = os.path.join(outfolder, "Land_Subsidence.geojson")


# -- Create lidar DEM mosaic --

lidar_folder = os.path.join(outfolder, "lidardem")
os.makedirs(lidar_folder, exist_ok=True)
lidar_dem_fn = os.path.join(outfolder, "casagrande_refdem.tif")

if overwrite or (not os.path.exists(lidar_dem_fn)):

    print("\nProcessing Lidar DEM")
        
    # Download tiles
    # Note: the list of files has been manually restricted to only cover the zoom area
    cmd = f"wget -c -r -np -nd -A '*x4[1-4]*y36[1-5]*.tif' -P {lidar_folder} https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/1m/Projects/AZ_MaricopaPinal_2020_B20/TIFF/"
    print(cmd); subprocess.run(cmd, shell=True)

    # Create mosaic
    list_fn = os.path.join(lidar_folder, "list_tiles.txt")
    tiles_downloaded = pd.Series(glob(os.path.join(lidar_folder, "USGS_1M*tif")))
    tiles_downloaded.to_csv(list_fn, header=False, index=False)

    tmp_vrt_fn = os.path.join(lidar_folder, "tmp.vrt")
    cmd = f"gdalbuildvrt -r cubic {tmp_vrt_fn} -input_file_list {list_fn} -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    # Load and crop to zoom extent
    lidar_dem = xdem.DEM(tmp_vrt_fn).crop(zoom_bounds)

    # Convert from NAVD88 geoid to ellipsoid - very memory intensive !
    # The file was selected from https://cdn.proj.org/ and compared to ASP's dem_geoid (script test_datum.py)
    # The mean diff with ASP is 0.4 mm and std of 5 mm, reason?
    # According to report https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/AZ_MaricopaPinal_2020_B20/USGS_AZ_MaricopaPinal_2020_B20_Project_Report.pdf, the geoid should be GEOID18, while this is geoid09.
    print("\nChanging vertical datum - this takes a few minutes\n")
    lidar_dem.set_vcrs("us_noaa_geoid09_conus.tif")
    lidar_dem = lidar_dem.to_vcrs("WGS84")

    # Save
    lidar_dem.save(lidar_dem_fn, tiled=True)

    print("\nDone with Lidar DEM\n")
else:
    print(f"Using existing lidar dem {lidar_dem_fn}")


# --- Create COP30 mosaic ---

cop30_folder = os.path.join(outfolder, "cop30")
os.makedirs(cop30_folder, exist_ok=True)

cop30_large_uncoreg_fn = os.path.join(cop30_folder, "cop30_large_uncoreg.tif")

if overwrite or (not os.path.exists(cop30_large_uncoreg_fn)):

    print("\nProcessing COP30 DEM")
    # - Download tiles -
    # Round bbox to nearest degree, outward.
    bbox = np.array([
        np.floor(large_bounds_lonlat[0]),
        np.floor(large_bounds_lonlat[1]),
        np.ceil(large_bounds_lonlat[2]),
        np.ceil(large_bounds_lonlat[3]),
    ],
                    dtype="int"
                    )

    cop30_tiles = dt.download_cop30_tiles(bbox, cop30_folder, overwrite=overwrite)

    # earlier attempt with direct reprojecting -> causes issues at tiles edges
    # cop30_tiles = dt.download_cop30_tiles_reproject(bbox, cop30_folder, "6341", overwrite=overwrite)

    # - Create mosaic -
    cop30_tiles_avail = pd.Series([tile for tile in cop30_tiles if os.path.exists(tile)])
    list_fn = os.path.join(cop30_folder, "list_tiles.txt")
    cop30_tiles_avail.to_csv(list_fn, header=False, index=False)

    # Note - options "-r cubic" and "-resolution highest" are very important otherwise shifts/artifacts can occur
    tmp_vrt_fn = os.path.join(cop30_folder, "tmp.vrt")
    cmd = f"gdalbuildvrt -r cubic {tmp_vrt_fn} -input_file_list {list_fn} -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    # Sanity check - the mosaic should be equal to the original tiles
    dt.geodiff(tmp_vrt_fn, cop30_tiles_avail[1], vmax=1)

    # Reproject and crop
    vrt_mosaic = xdem.DEM(tmp_vrt_fn)
    # cop30_large_crop = vrt_mosaic.crop(large_bounds)
    # cop30_zoom_crop = vrt_mosaic.crop(zoom_bounds)

    bounds = dict(zip(["left", "bottom", "right", "top"], large_bounds))
    cop30_large = vrt_mosaic.reproject(crs=epsg_str, bounds=bounds, res=30)

    # bounds = dict(zip(["left", "bottom", "right", "top"], zoom_bounds))
    # cop30_zoom = vrt_mosaic.reproject(crs=epsg_str, bounds=bounds, res=30)

    # sanity check - the mosaic should roughly equal to the original tiles
    for tile_fn in cop30_tiles_avail:
        print(f"Checking vs tile {tile_fn}")
        tile = xdem.DEM(tile_fn)
        ddem = tile - cop30_large.reproject(tile)
        assert np.std(ddem) < 1.5
        assert np.max(np.abs(ddem)) < 30
        assert abs(np.mean(ddem)) < 1e-2

    # earlier attempt with gdal_translate. Note: bounds are UL corner to LR
    # bbox_gdal = " ".join([str(zoom_bounds[k]) for k in [0, 3, 2, 1]])

    # final_cop30_fn = os.path.join(outfolder, "casagrande_cop30_dem.tif")
    # cmd = f"gdal_translate -a_ullr {bbox_gdal} {tmp_vrt_fn} {final_cop30_fn} -co COMPRESS=LZW -co TILED=yes"
    # print(cmd); subprocess.run(cmd, shell=True, check=True)

    # -- Convert geoid to ellipsoid
    cop30_large.set_vcrs("EGM08")
    cop30_large = cop30_large.to_vcrs("WGS84")

    # Save
    cop30_large.save(cop30_large_uncoreg_fn, tiled=True)
    print("\nDone with COP30 DEM\n")

else:
    print(f"Using existing COP30 dem {cop30_large_uncoreg_fn}")

# Alternative from David Shean?
# First download opentopo mosaic, then reproject and change vertical datum
# ./opentopo_dem/download_global_DEM.py -demtype COP30 -extent '-111.92002 32.65145 -111.59714 32.94645' -out_fn test.tif
# gdalwarp -s_srs EPSG:4326+3855 -t_srs EPSG:32612+4979 -tr 30.0 30.0 -r bilinear test.tif test_utm12_wgs84.tif -dstnodata 0 -overwrite


# -- Coregister both DEMs --

cop30_large_coreg_fn = os.path.join(outfolder, "casagrande_cop30_dem.tif")

if overwrite or (not os.path.exists(cop30_large_coreg_fn)):

    print("\nRunning coregistration\n")

    # Load DEMs
    dem_tbc, dem_ref = gu.raster.load_multiple_rasters([cop30_large_uncoreg_fn, lidar_dem_fn], crop=True, ref_grid=0)
    ddem_before = dem_tbc - dem_ref

    # Reprojecting landcover
    landcover = gu.Raster(landcover_vrt_fn)
    landcover = landcover.reproject(dem_tbc, resampling="nearest")

    # Creating subsidence mask
    subsid_poly = gu.Vector(subsid_poly_fn)
    subsid_poly_clip = subsid_poly.reproject(crs=dem_ref.crs).crop(dem_ref, clip=True)
    subsid_mask = subsid_poly.create_mask(dem_tbc)

    # DEPRECATED
    # For coreg, apply median filter with 120 m radius to remove isolated patches of classification
    # then keep only shrubland and grassland and remove subsiding areas
    # Finally, for horizontal coregistration, remove very low slopes as they bias the shift estimate
    # landcover_med = filters.median(landcover.data, footprint=morphology.disk(4))
    # inlier_mask_vert = np.isin(landcover_med, [20, 30]) & ~subsid_mask

    # New approach, consistent with provided stable mask
    # For coreg, keep only shrubland, grassland, bareland and remove subsiding areas
    # For horizontal coregistration, also remove very low slopes as they bias the shift estimate
    slope = xdem.terrain.slope(dem_ref)

    inlier_mask_vert = np.isin(landcover.data, [20, 30, 60]) & ~subsid_mask
    inlier_mask_hori = inlier_mask_vert & (slope > 1)

    # Running coregistration
    coreg_hori = xdem.coreg.NuthKaab(vertical_shift=False)
    coreg_vert = xdem.coreg.VerticalShift(vshift_reduc_func=np.median)
    dem_coreg_tmp = coreg_hori.fit_and_apply(dem_ref, dem_tbc, inlier_mask=inlier_mask_hori)
    dem_coreg = coreg_vert.fit_and_apply(dem_ref, dem_coreg_tmp, inlier_mask=inlier_mask_vert)
        
    # Print shifts
    print(coreg_hori.meta["outputs"]["affine"])
    print(coreg_vert.meta["outputs"]["affine"])

    # Print statistics
    ddem_after = dem_coreg - dem_ref
    ddem_bef_inlier = ddem_before[inlier_mask_vert].compressed()
    ddem_aft_inlier = ddem_after[inlier_mask_vert].compressed()
    print(f"- Before coreg:\n\tmean: {np.mean(ddem_bef_inlier):.3f}\n\tmedian: {np.median(ddem_bef_inlier):.3f}\n\tNMAD: {xdem.spatialstats.nmad(ddem_bef_inlier):.3f}")
    print(f"- After coreg:\n\tmean: {np.mean(ddem_aft_inlier):.3f}\n\tmedian: {np.median(ddem_aft_inlier):.3f}\n\tNMAD: {xdem.spatialstats.nmad(ddem_aft_inlier):.3f}")

    # Save DEM diff
    ddem_after.save(os.path.join(outfolder, "casagrande_dem-diff.tif"), tiled=True)

    # -- Plots --

    # General map
    vmax=5
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[0], alpha=0.2, add_cbar=False)
    subsid_poly_clip.plot(ax=axes[0], fc="none", ec="k")

    ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[1], alpha=0.2, add_cbar=False)
    subsid_poly_clip.plot(ax=axes[1], fc="none", ec="k")

    plt.show()

    # Zoom on mine pit
    xlim = [422500, 424500]
    ylim = [3645500, 3647500]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[0], alpha=0.2, add_cbar=False)
    subsid_poly_clip.plot(ax=axes[0], fc="none", ec="k")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)

    ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[1], alpha=0.2, add_cbar=False)
    subsid_poly_clip.plot(ax=axes[1], fc="none", ec="k")

    plt.show()

    # Zoom on mountain range
    xlim = [431000, 437000]
    ylim = [3627300, 3633900]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[0], alpha=0.2, add_cbar=False)
    subsid_poly_clip.plot(ax=axes[0], fc="none", ec="k")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)

    ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[1], alpha=0.2, add_cbar=False)
    subsid_poly_clip.plot(ax=axes[1], fc="none", ec="k")

    plt.show()

    # Loading full extent COP30, applying transformation and saving
    cop30_uncoreg = xdem.DEM(cop30_large_uncoreg_fn)
    cop30_coreg = coreg_vert.apply(coreg_hori.apply(cop30_uncoreg))
    cop30_coreg.save(cop30_large_coreg_fn, tiled=True)

    # Note: using coreg pipeline does not work...
    # coreg = coreg_hori + coreg_vert
    # coreg.apply(dem_tbc)

else:
    print(f"Using existing coregistered COP30 DEM {cop30_large_coreg_fn}")


# -- Reprojecting the landcover maps and creating masked DEMs --

print("Creating DEM masks")

# Load input data
landcover = gu.Raster(landcover_vrt_fn)
subsid_poly = gu.Vector(subsid_poly_fn)

lidar_dem = xdem.DEM(lidar_dem_fn)
cop30_coreg = xdem.DEM(cop30_large_coreg_fn)

def create_stable_mask(landcover, classes_to_keep = [20, 30, 60, 100], blur_size_m=150, subsid_poly=None):
    """
    An attempt to smooth a bit the mask. Not to be used at this stage.
    """
    # Apply some "smoothing to the landcover to avoid isolated pixels
    # First urban and forest pizels are dilated to remain
    # Then a median filter is run to keep the mode in a disk of radius blur_size_m
    blur_size_pix = int(blur_size_m / landcover.res[0])
    kernel = morphology.disk(blur_size_pix)
    tree_mask = morphology.binary_dilation(landcover_large.data == 10, footprint=kernel)
    urban_mask = morphology.binary_dilation(landcover_large.data == 60, footprint=kernel)
    landcover_med = filters.median(landcover.data, footprint=kernel)

    # Create a raster mask of the land subsidence
    if subsid_poly is not None:
        subsid_mask = subsid_poly.create_mask(landcover)

    # Creating combined mask
    mask = np.isin(landcover_med, classes_to_keep) & tree_mask & urban_mask & ~subsid_mask

    return mask


# -  Creating mask for large area

landcover_large = landcover.reproject(cop30_coreg, resampling="nearest")
landcover_large.save(os.path.join(outfolder, "landcover_large.tif"), tiled=True)
subsid_mask = subsid_poly.create_mask(cop30_coreg)

# Masking anything but shrubland, grassland, bare/sparse vegetation, moss/lichen. + subsidence
mask_large = np.isin(landcover_large.data, [20, 30, 60, 100]) & ~subsid_mask
mask_large.save(os.path.join(outfolder, "casagrande_cop30_dem_mask.tif"), tiled=True)

# -  Creating mask for zoom area

landcover_zoom = landcover.reproject(lidar_dem, resampling="nearest")
landcover_zoom.save(os.path.join(outfolder, "landcover_zoom.tif"), tiled=True)
subsid_mask = subsid_poly.create_mask(lidar_dem)
mask_zoom = np.isin(landcover_zoom.data, [20, 30, 60, 100]) & ~subsid_mask

# mask_zoom = mask.reproject(lidar_dem, resampling="nearest")
mask_zoom.save(os.path.join(outfolder, "casagrande_refdem_mask.tif"), tiled=True)
