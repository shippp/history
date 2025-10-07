"""
Script to prepare the auxiliary data for the Iceland site for the History project. The following data are prepared:
- reference DEM, seamless mosaic of ArcticDEM strips and lidar DEM over glaciers, provided by Joaquin. It is reprojected to UTM 27N.
- Copernicus 30 m DEM: 1x1 degree tiles are downloaded, merged and reprojected on the same horizontal and vertical reference system as reference DEM. It is then coregistered horizontally with a Nuth & Kaan (2011) algorithm and vertically by calculating a median shift in stable areas. Stable areas are defined as bare land, grassland and shrubland in the ESA worldcover (see below) and excluding glaciers (see below).
- ESA worldcover dataset: each 1x1 degree tile are downloaded from the S3 bucket and merged.

Author: Amaury Dehecq
Last modified: October 2025 
"""

import os
import subprocess
from glob import glob

import xdem
import matplotlib.pyplot as plt
import geoutils as gu
from skimage import filters, morphology
import numpy as np
import pandas as pd
import geopandas as gpd
import history

# TODO
# check glacier outlines

# Create output folder
outfolder = "/mnt/summer/USERS/DEHECQA/history/data_prep/iceland/aux_data/"
os.makedirs(outfolder, exist_ok=True)

# Temporary folder for intermediate files
tmp_folder = os.path.join(outfolder, "tmp")
os.makedirs(tmp_folder, exist_ok=True)

overwrite = False

# Bounding box of the two area of interest, in ISN2016 (EPSG:8088)
# Chosen to be multiple of 30 m (check with % 30)
# zoom_bounds = [2662800,  124980, 2765310, 173700]
# large_bounds = [2484780, 108990, 2808510, 421800]
# epsg_str = "EPSG:8088"

# Bounding box of the two area of interest, in UTM 27N (EPSG:32627)
# Chosen to be multiple of 30 m (check with % 30)
zoom_bounds = [561120,  7033920, 665040,  7085760]
large_bounds = [375420, 7012260, 708690, 7335120]
epsg_str = "EPSG:32627"

print("Creating GJSON files of AOI")

# Save to gjson files
zoom_gdf = gpd.GeoDataFrame(geometry=[gu.projtools.bounds2poly(zoom_bounds),], crs=epsg_str)
zoom_gdf.to_file(os.path.join(tmp_folder, "zoom_aoi.geojson"))

large_gdf = gpd.GeoDataFrame(geometry=[gu.projtools.bounds2poly(large_bounds),], crs=epsg_str)
large_gdf.to_file(os.path.join(tmp_folder, "large_aoi.geojson"))

# Convert bounding boxes to latlon
zoom_bounds_latlon = gu.projtools.bounds2poly(zoom_bounds, in_crs=epsg_str, out_crs="EPSG:4326").bounds
large_bounds_latlon = gu.projtools.bounds2poly(large_bounds, in_crs=epsg_str, out_crs="EPSG:4326").bounds

# and swap axes for lon/lat
zoom_bounds_lonlat = [zoom_bounds_latlon[k] for k in [1, 0, 3, 2]]
large_bounds_lonlat = [large_bounds_latlon[k] for k in [1, 0, 3, 2]]

print("Done with AOI")

# -- Download ESA worldcover classification --

landcover_folder = os.path.join(tmp_folder, "landcover")
os.makedirs(landcover_folder, exist_ok=True)

landcover_vrt_fn = os.path.join(landcover_folder, "tmp.vrt")

if overwrite or (not os.path.exists(landcover_vrt_fn)):

    print("\nProcessing ESA world_landcover")
    landcover_tiles = history.aux_data.download_tools.download_esa_worldcover(landcover_folder, large_bounds_lonlat, year=2021, overwrite=overwrite, dryrun=False)

    # Create mosaic
    list_fn = os.path.join(landcover_folder, "list_tiles.txt")
    pd.Series(landcover_tiles).to_csv(list_fn, header=False, index=False)

    cmd = f"gdalbuildvrt -r nearest {landcover_vrt_fn} -input_file_list {list_fn} -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    print("\nDone with ESA world_landcover\n")

# -- Download RGI outlines --
# For now, file has to be downloaded manually from https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/RGI2000-v7.0-G-06_iceland.zip

glacier_poly_fn = os.path.join(tmp_folder, "RGI2000-v7.0-G-06_iceland.zip")
assert os.path.exists(glacier_poly_fn)


# -- Create lidar DEM mosaic --

lidar_folder = os.path.join(tmp_folder, "lidardem")
os.makedirs(lidar_folder, exist_ok=True)
lidar_dem_fn = os.path.join(outfolder, "reference_dem_zoom.tif")

if overwrite or (not os.path.exists(lidar_dem_fn)):

    print("\nProcessing Lidar DEM")
        
    # Download tiles
    # Note: the list of files has been manually restricted to only cover the zoom area
    cmd = f"wget -c -r -np -nd -A '*isn2016*.tif' -P {lidar_folder} https://ftp.lmi.is/.stm/joaquin/history/iceland/refdem_original/"
    print(cmd); # TMP! subprocess.run(cmd, shell=True)

    # Create mosaic
    # Use nearest interpolation as no resampling is needed
    # Overwrite the CRS definition as WKT is not complete
    list_fn = os.path.join(lidar_folder, "list_tiles.txt")
    tiles_downloaded = pd.Series(glob(os.path.join(lidar_folder, "IslandsDEM*isn2016*tif"))).sort_values()
    tiles_downloaded.to_csv(list_fn, header=False, index=False)

    tmp_vrt_fn = os.path.join(lidar_folder, "tmp.vrt")
    cmd = f"gdalbuildvrt -r nearest {tmp_vrt_fn} -input_file_list {list_fn} -a_srs EPSG:8088 -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    # Crop and reproject
    bbox_gdal = " ".join([str(bound) for bound in zoom_bounds])
    cmd = f"gdalwarp {tmp_vrt_fn} {lidar_dem_fn} -te {bbox_gdal} -tr 2 2 -t_srs {epsg_str} -co COMPRESS=DEFLATE -co tiled=yes -co bigtiff=if_safer -r cubic"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    # Sanity check - the mosaic should be close to equal to the original tiles
    tile = xdem.DEM(tiles_downloaded.iloc[6])
    lidar_dem = xdem.DEM(lidar_dem_fn)
    ddem = tile - lidar_dem.reproject(tile)
    # assert np.max(np.abs(ddem)) == 0
    assert np.std(ddem) < 0.2

    # # Load and crop to zoom extent - Takes ~8 min
    # lidar_dem = xdem.DEM(tmp_vrt_fn).crop(zoom_bounds)

    # # Sanity check - the mosaic should be equal to the original tiles
    # tile = xdem.DEM(tiles_downloaded[0])
    # ddem = tile - lidar_dem.reproject(tile)
    # assert np.max(np.abs(ddem)) == 0

    # # No need to convert to ellipsoid
    # # print("/nChanging vertical datum - this takes a few minutes/n")
    # # lidar_dem.set_vcrs("us_noaa_geoid09_conus.tif")
    # # lidar_dem = lidar_dem.to_vcrs("WGS84")

    # # Save - Takes ~2 min
    # lidar_dem.save(lidar_dem_fn, tiled=True)

    print("\nDone with Lidar DEM\n")
else:
    print(f"Using existing lidar dem {lidar_dem_fn}")


# --- Create COP30 mosaic ---

cop30_folder = os.path.join(tmp_folder, "cop30")
os.makedirs(cop30_folder, exist_ok=True)

cop30_uncoreg_fn = os.path.join(cop30_folder, "cop30_uncoreg.tif")

if overwrite or (not os.path.exists(cop30_uncoreg_fn)):

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

    cop30_tiles = history.aux_data.download_tools.download_cop30_tiles(bbox, cop30_folder, overwrite=overwrite)

    # - Create mosaic -
    cop30_tiles_avail = pd.Series([tile for tile in cop30_tiles if os.path.exists(tile)])
    list_fn = os.path.join(cop30_folder, "list_tiles.txt")
    cop30_tiles_avail.to_csv(list_fn, header=False, index=False)

    # Note - options "-r cubic" and "-resolution highest" are very important otherwise shifts/artifacts can occur
    tmp_vrt_fn = os.path.join(cop30_folder, "tmp.vrt")
    cmd = f"gdalbuildvrt -r cubic {tmp_vrt_fn} -input_file_list {list_fn} -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    # Sanity check - the mosaic should be equal to the original tiles
    history.aux_data.download_tools.geodiff(tmp_vrt_fn, cop30_tiles_avail[1], vmax=1)

    # Reproject and crop
    vrt_mosaic = xdem.DEM(tmp_vrt_fn)
    bounds = dict(zip(["left", "bottom", "right", "top"], large_bounds))
    cop30 = vrt_mosaic.reproject(crs=epsg_str, bounds=bounds, res=30)

    # sanity check - the mosaic should roughly equal to the original tiles
    # for tile_fn in cop30_tiles_avail:
    #     print(f"Checking vs tile {tile_fn}")
    #     tile = xdem.DEM(tile_fn)
    #     ddem = tile - cop30.reproject(tile)
    #     assert np.std(ddem) < 1.5
    #     assert np.max(np.abs(ddem)) < 30
    #     assert abs(np.mean(ddem)) < 1e-2

    # earlier attempt with gdal_translate. Note: bounds are UL corner to LR
    # bbox_gdal = " ".join([str(zoom_bounds[k]) for k in [0, 3, 2, 1]])

    # final_cop30_fn = os.path.join(tmp_folder, "iceland_cop30_dem.tif")
    # cmd = f"gdal_translate -a_ullr {bbox_gdal} {tmp_vrt_fn} {final_cop30_fn} -co COMPRESS=LZW -co TILED=yes"
    # print(cmd); subprocess.run(cmd, shell=True, check=True)

    # -- Convert geoid to ellipsoid
    cop30.set_vcrs("EGM08")
    cop30 = cop30.to_vcrs("WGS84")

    # Save
    cop30.save(cop30_uncoreg_fn, tiled=True)
    print("\nDone with COP30 DEM\n")

else:
    print(f"Using existing COP30 dem {cop30_uncoreg_fn}")


# -- Coregister both DEMs --

# TODO
# very memory consuming - see if can be improved?

cop30_coreg_fn = os.path.join(outfolder, "reference_dem_large.tif")

if overwrite or (not os.path.exists(cop30_coreg_fn)):

    print("\nRunning coregistration\n")
    
    # Load DEMs (time and memory consuming... ~1.5 min)
    from time import time
    t0 = time()
    dem_tbc, dem_ref = gu.raster.load_multiple_rasters([cop30_uncoreg_fn, lidar_dem_fn], crop=True, ref_grid=0)
    ddem_before = dem_tbc - dem_ref
    print(f"Took {(time() - t0)/60.} min")

    # Reprojecting landcover (time consuming... ~5 min > 40 GB RAM)
    t0 = time()
    landcover = gu.Raster(landcover_vrt_fn)
    landcover = landcover.reproject(dem_tbc, resampling="nearest")
    print(f"Took {(time() - t0)/60.} min")

    # Creating glacier mask
    glacier_poly = gu.Vector(glacier_poly_fn)
    glacier_poly_clip = glacier_poly.reproject(crs=dem_ref.crs).crop(dem_ref, clip=True)
    glacier_mask = glacier_poly.create_mask(dem_tbc)

    # For coreg, keep only shrubland, grassland, bareland, moss/lichens and remove glacier areas
    # For horizontal coregistration, also remove very low slopes as they bias the shift estimate
    slope = xdem.terrain.slope(dem_ref)

    inlier_mask_vert = np.isin(landcover.data, [20, 30, 60, 100]) & ~glacier_mask
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

    # Save DEM diff and inlier mask
    ddem_after.save(os.path.join(tmp_folder, "iceland_dem-diff.tif"), tiled=True)
    inlier_mask_vert.save(os.path.join(tmp_folder, "coreg_mask.tif"), tiled=True)

    # -- Plots --

    # General map
    vmax=5
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[0], alpha=0.2, add_cbar=False)
    glacier_poly_clip.plot(ax=axes[0], fc="none", ec="k")

    ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[1], alpha=0.2, add_cbar=False)
    glacier_poly_clip.plot(ax=axes[1], fc="none", ec="k")

    plt.show()

    # # Zoom on relief
    xlim = [606570, 610600]
    ylim = [7057400, 7061720]
    # xlim = [2707600, 2711500]
    # ylim = [147100, 151300]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[0], alpha=0.2, add_cbar=False)
    glacier_poly_clip.plot(ax=axes[0], fc="none", ec="k")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)

    ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[1], alpha=0.2, add_cbar=False)
    glacier_poly_clip.plot(ax=axes[1], fc="none", ec="k")

    plt.show()

    # Zoom on relief 2
    xlim = [606850, 611870]
    ylim = [7070900, 7075000]
    # xlim = [2708300, 2713200]
    # ylim = [160600, 164525]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[0], alpha=0.2, add_cbar=False)
    glacier_poly_clip.plot(ax=axes[0], fc="none", ec="k")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)

    ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    inlier_mask_vert.plot(cmap="gray", ax=axes[1], alpha=0.2, add_cbar=False)
    glacier_poly_clip.plot(ax=axes[1], fc="none", ec="k")

    plt.show()

    # # Zoom on mountain range
    # xlim = [431000, 437000]
    # ylim = [3627300, 3633900]

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    # ddem_before.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[0], title="Before coreg")
    # inlier_mask_vert.plot(cmap="gray", ax=axes[0], alpha=0.2, add_cbar=False)
    # glacier_poly_clip.plot(ax=axes[0], fc="none", ec="k")
    # axes[0].set_xlim(xlim)
    # axes[0].set_ylim(ylim)

    # ddem_after.plot(cmap="RdYlBu", vmin=-vmax, vmax=vmax, ax=axes[1], title="After coreg")
    # inlier_mask_vert.plot(cmap="gray", ax=axes[1], alpha=0.2, add_cbar=False)
    # glacier_poly_clip.plot(ax=axes[1], fc="none", ec="k")

    # plt.show()

    # Loading full extent COP30, applying transformation and saving
    cop30_uncoreg = xdem.DEM(cop30_uncoreg_fn)
    cop30_coreg = coreg_vert.apply(coreg_hori.apply(cop30_uncoreg))
    cop30_coreg.save(cop30_coreg_fn, tiled=True)

    # Note: using coreg pipeline does not work...
    # coreg = coreg_hori + coreg_vert
    # coreg.apply(dem_tbc)

else:
    print(f"Using existing coregistered COP30 DEM {cop30_coreg_fn}")


# -- Reprojecting the landcover maps and creating masked DEMs --

print("## Creating DEM masks ##")

mask_large_fn = os.path.join(outfolder, "reference_dem_large_mask.tif")
mask_zoom_fn = os.path.join(outfolder, "reference_dem_zoom_mask.tif")

if overwrite or (not os.path.exists(mask_large_fn)) or (not os.path.exists(mask_zoom_fn)):
    
    # Load input data
    landcover = gu.Raster(landcover_vrt_fn)
    glacier_poly = gu.Vector(glacier_poly_fn)

    lidar_dem = xdem.DEM(lidar_dem_fn)
    cop30_coreg = xdem.DEM(cop30_coreg_fn)

    # -  Creating mask for large area

    landcover_large_fn = os.path.join(tmp_folder, "landcover_large.tif")
    history.aux_data.download_tools.reproject_gdal(landcover_vrt_fn, cop30_coreg_fn, landcover_large_fn, resampling_algo="nearest")
    landcover_large = gu.Raster(landcover_large_fn)

    #landcover_large = landcover.reproject(cop30_coreg, resampling="nearest")
    #landcover_large.save(os.path.join(tmp_folder, "landcover_large.tif"), tiled=True)
    glacier_mask = glacier_poly.create_mask(cop30_coreg)

    # Masking anything but shrubland, grassland, bare/sparse vegetation, moss/lichen. + glacier
    mask_large = np.isin(landcover_large.data, [20, 30, 60, 100]) & ~glacier_mask

    mask_large.save(mask_large_fn, tiled=True)

    # -  Creating mask for zoom area

    landcover_zoom = landcover.reproject(lidar_dem, resampling="nearest")
    landcover_zoom.save(os.path.join(tmp_folder, "landcover_zoom.tif"), tiled=True)
    glacier_mask = glacier_poly.create_mask(lidar_dem)
    mask_zoom = np.isin(landcover_zoom.data, [20, 30, 60, 100]) & ~glacier_mask

    # mask_zoom = mask.reproject(lidar_dem, resampling="nearest")
    mask_zoom.save(mask_zoom_fn, tiled=True)

print("Done with DEM masks")
