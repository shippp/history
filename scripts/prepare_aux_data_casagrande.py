"""
Script to prepare the auxiliary data for the Casa Grande site for the History project. The following data are prepared:
- reference lidar DEM over Maricopa county, provided by the USGS: https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/1m/Projects/AZ_MaricopaPinal_2020_B20/. Update: the lidar DEM is processed using UW lidar_tools.
- Copernicus 30 m DEM: 1x1 degree tiles are downloaded, merged and reprojected on the same horizontal and vertical reference system as lidar DEM. It is then coregistered horizontally with a Nuth & Kaan (2011) algorithm and vertically by calculating a median shift in stable areas. Stable areas are defined as bare land, grassland and shrubland in the ESA worldcover (see below) and excluding sibsiding land (see below).
- ESA worldcover dataset: each 1x1 degree tile are downloaded from the S3 bucket and merged
- Land subsidence vector file downloaded from https://azgeo-open-data-agic.hub.arcgis.com/datasets/azwater::land-subsidence/explore

Author: Amaury Dehecq
Last modified: November 2025 
"""

import os
import subprocess
from glob import glob
import folium

import xdem
import matplotlib.pyplot as plt
import geoutils as gu
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import rasterio
import history.aux_data as aux


# # Global settings

VISUALIZATION = True
OVERWRITE = False

# PATH SETTINGS

OUTPUT_DIRECTORY = "/mnt/summer/USERS/DEHECQA/history/data_prep/casa_grande/aux_dems/"
VISUALIZATION_DIRECTORY = "/mnt/summer/USERS/DEHECQA/history/data_prep/casa_grande/aux_dems/visualizations"

# final generated files
LARGE_DEM_FILE = os.path.join(OUTPUT_DIRECTORY, "CG_reference_dem_large.tif")
ZOOM_DEM_FILE = os.path.join(OUTPUT_DIRECTORY, "CG_reference_dem_zoom.tif")
LARGE_DEM_MASK_FILE = os.path.join(OUTPUT_DIRECTORY, "CG_reference_dem_large_mask.tif")
ZOOM_DEM_MASK_FILE = os.path.join(OUTPUT_DIRECTORY, "CG_reference_dem_zoom_mask.tif")

# temporary directory to download tiles and process
TMP_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "tmp")

LANDCOVER_DIRECTORY = os.path.join(TMP_DIRECTORY, "landcover")
LIDARDEM_DIRECTORY = os.path.join(TMP_DIRECTORY, "lidar_dem")
COP30DEM_DIRECTORY = os.path.join(TMP_DIRECTORY, "cop30_dem")

os.makedirs(TMP_DIRECTORY, exist_ok=True)
os.makedirs(VISUALIZATION_DIRECTORY, exist_ok=True)


# Bounding box of the two area of interest
# epsg_str = "EPSG:26912"
# zoom_bounds = [414000, 3613020, 444000, 3650010]
# large_bounds = [261990, 3405030, 612990, 3880020]

epsg_str = "EPSG:6341"
zoom_bounds = [414000, 3613020, 444000, 3650010]
large_bounds = [261990, 3405030, 612990, 3880020]


# Save to gjson files
zoom_gdf = gpd.GeoDataFrame(geometry=[gu.projtools.bounds2poly(zoom_bounds),], crs=epsg_str)
zoom_gdf.to_file(os.path.join(TMP_DIRECTORY, "CG_zoom_aoi.geojson"))

large_gdf = gpd.GeoDataFrame(geometry=[gu.projtools.bounds2poly(large_bounds),], crs=epsg_str)
large_gdf.to_file(os.path.join(TMP_DIRECTORY, "CG_large_aoi.geojson"))

zoom_bounds_latlon = gu.projtools.bounds2poly(zoom_bounds, in_crs=epsg_str, out_crs="EPSG:4326").bounds
large_bounds_latlon = gu.projtools.bounds2poly(large_bounds, in_crs=epsg_str, out_crs="EPSG:4326").bounds

# and swap axes for lon/lat
zoom_bounds_lonlat = [zoom_bounds_latlon[k] for k in [1, 0, 3, 2]]
large_bounds_lonlat = [large_bounds_latlon[k] for k in [1, 0, 3, 2]]


# -- Download ESA worldcover classification --

os.makedirs(LANDCOVER_DIRECTORY, exist_ok=True)

landcover_vrt_fn = os.path.join(LANDCOVER_DIRECTORY, "tmp.vrt")

if OVERWRITE or (not os.path.exists(landcover_vrt_fn)):

    print("\nProcessing ESA world_landcover")
    landcover_tiles = aux.download_tools.download_esa_worldcover(LANDCOVER_DIRECTORY, large_bounds_lonlat, year=2021, overwrite=OVERWRITE, dryrun=False)

    # Create mosaic
    list_fn = os.path.join(LANDCOVER_DIRECTORY, "list_tiles.txt")
    pd.Series(landcover_tiles).to_csv(list_fn, header=False, index=False)

    cmd = f"gdalbuildvrt -r nearest {landcover_vrt_fn} -input_file_list {list_fn} -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    print("\nDone with ESA world_landcover\n")

# -- Download land subsidence mask --
# For now, file has to be downloaded manually from https://azgeo-open-data-agic.hub.arcgis.com/datasets/azwater::land-subsidence/explore

land_subsidence_file = os.path.join(TMP_DIRECTORY, "Land_Subsidence.geojson")
assert os.path.exists(land_subsidence_file)


# -- Create lidar DEM mosaic --

# The lidar DSM mosaic has been created by David Shean with lidar_tools (https://github.com/uw-cryo/lidar_tools).
# with the following command:
# lidar-tools rasterize --geometry reference_dem_zoom.geojson --output 1m_NAD83_2011_UTM12N --dst-crs NAD83_2011_UTM12N_3D.wkt --num-process 7 --overwrite

# The file was briefly postprocessed to exactly match the zoom grid and include the CRS directly in metadata rather than the .aux.xml (used for defining 3d CRS).
# gdalwarp -tr 1 1 -te 414000 3613020 444000 3650010 reference_dem_zoom-DSM_mos.tif reference_dem_zoom.tif -co compress=lzw -co tiled=yes -overwrite
# gdal_edit -a_srs EPSG:6341 reference_dem_zoom.tif

# if OVERWRITE or (not os.path.exists(ZOOM_DEM_FILE)):

#     print("\nProcessing Lidar DEM")
        
#     # Download tiles
#     # Note: the list of files has been manually restricted to only cover the zoom area
#     cmd = f"wget -c -r -np -nd -A '*x4[1-4]*y36[1-5]*.tif' -P {LIDARDEM_DIRECTORY} https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/1m/Projects/AZ_MaricopaPinal_2020_B20/TIFF/"
#     print(cmd); #subprocess.run(cmd, shell=True)

#     # Create mosaic
#     list_fn = os.path.join(LIDARDEM_DIRECTORY, "list_tiles.txt")
#     tiles_downloaded = pd.Series(glob(os.path.join(LIDARDEM_DIRECTORY, "USGS_1M*tif")))
#     tiles_downloaded.to_csv(list_fn, header=False, index=False)

#     tmp_vrt_fn = os.path.join(LIDARDEM_DIRECTORY, "tmp.vrt")
#     cmd = f"gdalbuildvrt -r cubic {tmp_vrt_fn} -input_file_list {list_fn} -resolution highest"
#     print(cmd); subprocess.run(cmd, shell=True, check=True)

#     # Crop and reproject
#     bbox_gdal = " ".join([str(bound) for bound in zoom_bounds])
#     cmd = f"gdalwarp {tmp_vrt_fn} {ZOOM_DEM_FILE} -te {bbox_gdal} -tr 1 1 -t_srs {epsg_str} -co COMPRESS=DEFLATE -co tiled=yes -co bigtiff=if_safer -r cubic"
#     print(cmd); subprocess.run(cmd, shell=True, check=True)

#     # Convert from NAVD88 geoid to ellipsoid - very memory intensive !
#     # According to report https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/AZ_MaricopaPinal_2020_B20/USGS_AZ_MaricopaPinal_2020_B20_Project_Report.pdf, the geoid should be GEOID18.
#     # The file was selected from https://cdn.proj.org/
#     print("\nChanging vertical datum - this takes a few minutes\n")
#     lidar_dem = xdem.DEM(ZOOM_DEM_FILE)#.crop(zoom_bounds)
#     lidar_dem.set_vcrs("us_noaa_g2018u0.tif") 
#     lidar_dem = lidar_dem.to_vcrs("WGS84")
    
#     # Save
#     lidar_dem.save(ZOOM_DEM_FILE, tiled=True)

#     print("\nDone with Lidar DEM\n")
# else:
print(f"Using existing lidar dem {ZOOM_DEM_FILE}")


# --- Create COP30 mosaic ---

cop30_large_uncoreg_fn = os.path.join(COP30DEM_DIRECTORY, "cop30_uncoreg.tif")

if OVERWRITE or (not os.path.exists(cop30_large_uncoreg_fn)):

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

    cop30_tiles = aux.download_tools.download_cop30_tiles(bbox, COP30DEM_DIRECTORY, overwrite=OVERWRITE)

    # earlier attempt with direct reprojecting -> causes issues at tiles edges
    # cop30_tiles = aux.download_tools.download_cop30_tiles_reproject(bbox, cop30_folder, "6341", overwrite=OVERWRITE)

    # - Create mosaic -
    cop30_tiles_avail = pd.Series([tile for tile in cop30_tiles if os.path.exists(tile)])
    list_fn = os.path.join(COP30DEM_DIRECTORY, "list_tiles.txt")
    cop30_tiles_avail.to_csv(list_fn, header=False, index=False)

    # Note - options "-r cubic" and "-resolution highest" are very important otherwise shifts/artifacts can occur
    tmp_vrt_fn = os.path.join(COP30DEM_DIRECTORY, "tmp.vrt")
    cmd = f"gdalbuildvrt -r cubic {tmp_vrt_fn} -input_file_list {list_fn} -resolution highest"
    print(cmd); subprocess.run(cmd, shell=True, check=True)

    # Sanity check - the mosaic should be equal to the original tiles
    aux.download_tools.geodiff(tmp_vrt_fn, cop30_tiles_avail[1], vmax=1)

    # Reproject and crop
    vrt_mosaic = xdem.DEM(tmp_vrt_fn)
    # cop30_large_crop = vrt_mosaic.crop(large_bounds)
    # cop30_zoom_crop = vrt_mosaic.crop(zoom_bounds)

    bounds = dict(zip(["left", "bottom", "right", "top"], large_bounds))
    cop30_large = vrt_mosaic.reproject(crs=epsg_str, bounds=bounds, res=30)

    # bounds = dict(zip(["left", "bottom", "right", "top"], zoom_bounds))
    # cop30_zoom = vrt_mosaic.reproject(crs=epsg_str, bounds=bounds, res=30)

    # sanity check - the mosaic should be roughly equal to the original tiles
    for tile_fn in cop30_tiles_avail:
        print(f"Checking vs tile {tile_fn}")
        tile = xdem.DEM(tile_fn)
        ddem = tile - cop30_large.reproject(tile)
        assert np.std(ddem) < 1.5
        assert np.max(np.abs(ddem)) < 30
        assert abs(np.mean(ddem)) < 1e-1

    # earlier attempt with gdal_translate. Note: bounds are UL corner to LR
    # bbox_gdal = " ".join([str(zoom_bounds[k]) for k in [0, 3, 2, 1]])

    # final_cop30_fn = os.path.join(OUTPUT_DIRECTORY, "casagrande_cop30_dem.tif")
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

if OVERWRITE or (not os.path.exists(LARGE_DEM_FILE)):

    print("\nRunning coregistration\n")

    # Load DEMs
    from time import time
    t0 = time()
    dem_tbc, dem_ref = gu.raster.load_multiple_rasters([cop30_large_uncoreg_fn, ZOOM_DEM_FILE], crop=True, ref_grid=0)
    ddem_before = dem_tbc - dem_ref
    print(f"Took {(time() - t0)/60.} min")

    # Reprojecting landcover
    t0 = time()
    landcover = gu.Raster(landcover_vrt_fn)
    landcover = landcover.reproject(dem_tbc, resampling="nearest")
    print(f"Took {(time() - t0)/60.} min")

    # Creating subsidence mask
    subsid_poly = gu.Vector(land_subsidence_file)
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
    ddem_after.save(os.path.join(OUTPUT_DIRECTORY, "casagrande_dem-diff.tif"), tiled=True)

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
    cop30_coreg.save(LARGE_DEM_FILE, tiled=True)

    # Note: using coreg pipeline does not work...
    # coreg = coreg_hori + coreg_vert
    # coreg.apply(dem_tbc)

else:
    print(f"Using existing coregistered COP30 DEM {LARGE_DEM_FILE}")


# -- Reprojecting the landcover maps and creating masked DEMs --

print("Creating DEM masks")

if OVERWRITE or (not os.path.exists(LARGE_DEM_MASK_FILE)) or (not os.path.exists(ZOOM_DEM_MASK_FILE)):

    # Load input data
    landcover = gu.Raster(landcover_vrt_fn)
    subsid_poly = gu.Vector(land_subsidence_file)

    lidar_dem = xdem.DEM(ZOOM_DEM_FILE)
    cop30_coreg = xdem.DEM(LARGE_DEM_FILE)

    # -  Creating mask for large area

    landcover_large = landcover.reproject(cop30_coreg, resampling="nearest")
    landcover_large.save(os.path.join(TMP_DIRECTORY, "CG_landcover_large.tif"), tiled=True)
    subsid_mask = subsid_poly.create_mask(cop30_coreg)

    # Masking anything but shrubland, grassland, bare/sparse vegetation, moss/lichen. + subsidence
    mask_large = np.isin(landcover_large.data, [20, 30, 60, 100]) & ~subsid_mask
    mask_large.save(LARGE_DEM_MASK_FILE, tiled=True)

    # -  Creating mask for zoom area

    landcover_zoom = landcover.reproject(lidar_dem, resampling="nearest")
    landcover_zoom.save(os.path.join(TMP_DIRECTORY, "CG_landcover_zoom.tif"), tiled=True)
    subsid_mask = subsid_poly.create_mask(lidar_dem)
    mask_zoom = np.isin(landcover_zoom.data, [20, 30, 60, 100]) & ~subsid_mask

    # mask_zoom = mask.reproject(lidar_dem, resampling="nearest")
    mask_zoom.save(ZOOM_DEM_MASK_FILE, tiled=True)

print("Done with DEM masks")
