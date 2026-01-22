#!/bin/bash

# Bash script to generate the reference orthomosaics for Casa Grande and Iceland
# using gdal to download and reproject the WMS tiles.
# The images are converted fro RGB to B/W using this formula:
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert

#SBATCH -J mosaics

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=cryodyn

#SBATCH --mem=30G

#SBATCH --time=5:00:00
#SBATCH --output mosaics.%j.out
#SBATCH --error  mosaics.%j.out


# Note on data compression
# Several algorithm have been used, following https://kokoalberti.com/articles/geotiff-compression-optimization-guide/
# JPEG with 80% quality seem a good compromise with little visual differences for Byte images
# Deflate with predictor 2 is best choice for UInt16 images (JPEG does not work)
# example on Iceland
# gdal_translate -co COMPRESS=JPEG -co JPEG_QUALITY=80 -co TILED=YES iceland/aux_orthos/tmp/IL_esri_mosaic_0.5m_bw.tif iceland/aux_orthos/tmp/IL_esri_mosaic_0.5m_bw_jpeg80.tif -co NUM_THREADS=2
# gdaladdo -ro --config COMPRESS_OVERVIEW JPEG --config JPEG_QUALITY_OVERVIEW 80 --config INTERLEAVE_OVERVIEW PIXEL -r average iceland/aux_orthos/tmp/IL_esri_mosaic_0.5m_bw_jpeg80.tif 2 4 8 16


# Setup environment
. "/workdir2/cryodyn/dehecqa/bin/miniforge3/etc/profile.d/conda.sh" 
mamba activate history
gdal_calc --version  # check

# Base directory
ROOT_DIR=/mnt/summer/USERS/DEHECQA/history/data_prep

# -- Prepare WMS description files --

# ESRI mosaic
ESRI_WMS=${ROOT_DIR}/wms_esri_satellite.xml
gdal_translate "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer?f=json" ${ESRI_WMS} -of WMS

# Google mosaic, downloaded from
# https://github.com/OSGeo/gdal/blob/release/3.11/frmts/wms/frmt_wms_googlemaps_tms.xml
GG_WMS=${ROOT_DIR}/wms_google_satellite.xml

# --- Iceland - zoom ---
res=0.5
RGB_FN=${ROOT_DIR}/iceland/aux_orthos/tmp/IL_esri_mosaic_${res}m_zoom.tif
BW_FN=${ROOT_DIR}/iceland/aux_orthos/tmp/IL_esri_mosaic_${res}m_bw_zoom.tif

# Create ESRI mosaic
time gdalwarp ${ESRI_WMS} ${RGB_FN} -t_srs "EPSG:32627" -r bilinear -te 561120 7033920 665040 7085760 -tr ${res} ${res} -co bigtiff=if_safer -overwrite

# Convert to B/W image
gdal_calc -A ${RGB_FN} --A_band=1 -B ${RGB_FN} --B_band=2 -C ${RGB_FN} --C_band=3 --outfile ${BW_FN} --calc "0.299*A + 0.587*B + 0.114*C" --co compress=JPEG --co JPEG_QUALITY=80 --co tiled=yes --co bigtiff=if_safer --overwrite

mv $BW_FN ${ROOT_DIR}/iceland/IL_orthomosaic_${res}m_zoom.tif

# --- Iceland - large ---
res=5
RGB_FN=${ROOT_DIR}/iceland/aux_orthos/tmp/IL_esri_mosaic_${res}m_large.tif
BW_FN=${ROOT_DIR}/iceland/aux_orthos/tmp/IL_esri_mosaic_${res}m_bw_large.tif

# Create ESRI mosaic
time gdalwarp ${ESRI_WMS} ${RGB_FN} -t_srs "EPSG:32627" -r bilinear -te 375420 7012260 708690 7335120 -tr ${res} ${res} -co bigtiff=if_safer -overwrite

# Convert to B/W image
gdal_calc -A ${RGB_FN} --A_band=1 -B ${RGB_FN} --B_band=2 -C ${RGB_FN} --C_band=3 --outfile ${BW_FN} --calc "0.299*A + 0.587*B + 0.114*C" --co compress=JPEG --co JPEG_QUALITY=80 --co tiled=yes --co bigtiff=if_safer --overwrite

mv $BW_FN ${ROOT_DIR}/iceland/IL_orthomosaic_${res}m_large.tif


# -- Casa Grande - zoom --

res=0.25
RGB_FN=${ROOT_DIR}/casa_grande/aux_orthos/tmp/CG_google_mosaic_${res}m_zoom.tif
BW_FN=${ROOT_DIR}/casa_grande/aux_orthos/tmp/CG_google_mosaic_${res}m_bw_zoom.tif

# # Create Google mosaic
time gdalwarp ${GG_WMS} ${RGB_FN} -t_srs "EPSG:6341" -r bilinear -te 414000 3613020 444000 3650010 -tr ${res} ${res} -co BIGTIFF=if_safer -overwrite

# Convert to B/W image
gdal_calc -A ${RGB_FN} --A_band=1 -B ${RGB_FN} --B_band=2 -C ${RGB_FN} --C_band=3 --outfile ${BW_FN} --calc "0.299*A + 0.587*B + 0.114*C" --co compress=JPEG --co JPEG_QUALITY=80 --co tiled=yes --co bigtiff=if_safer --overwrite

mv $BW_FN ${ROOT_DIR}/casa_grande/CG_orthomosaic_${res}m_zoom.tif

# -- Casa Grande - large --
res=5
RGB_FN=${ROOT_DIR}/casa_grande/aux_orthos/tmp/CG_google_mosaic_${res}m_large.tif
BW_FN=${ROOT_DIR}/casa_grande/aux_orthos/tmp/CG_google_mosaic_${res}m_bw_large.tif

# # Create Google mosaic
time gdalwarp ${GG_WMS} ${RGB_FN} -t_srs "EPSG:6341" -r bilinear -te 261990 3405030 612990 3880020 -tr ${res} ${res} -co BIGTIFF=if_safer -overwrite

# Convert to B/W image
gdal_calc -A ${RGB_FN} --A_band=1 -B ${RGB_FN} --B_band=2 -C ${RGB_FN} --C_band=3 --outfile ${BW_FN} --calc "0.299*A + 0.587*B + 0.114*C" --co compress=JPEG --co JPEG_QUALITY=80 --co tiled=yes --co bigtiff=if_safer --overwrite

mv $BW_FN ${ROOT_DIR}/casa_grande/CG_orthomosaic_${res}m_large.tif
