# history

**Historical Images for Surface Topography Reconstruction over the Last 50 Years**

This repository aims to download and preprocess a dataset of historical images for use in stereo photogrammetry.  
The dataset is divided into two sites: **Casa Grande** and **Iceland**, and for each site, it includes three types of historical imagery:

- Aerial images  
- KH-9 Mapping Camera images  
- KH-9 Panoramic Camera images  

For each image subset, a dedicated Python notebook is provided to download the raw images and preprocess them.  
Some of the raw images are retrieved using [`usgsxplore`](https://github.com/adehecq/usgs_explorer/), which requires [user credentials](https://github.com/adehecq/usgs_explorer?tab=readme-ov-file#credentials).

**Note:** The preprocessing workflows differ significantly between image types. Each preprocessing pipeline is built using the [`hipp`](https://github.com/shippp/hipp) Python package. Currently, the KH-9 Mapping Camera preprocessing is in a private repository but will be integrated into `hipp` in the near future.

## Images Preprocessing

here is a short explanation of each preprocessing methods:

### Aerial Images

The preprocessing method mainly following [Knuth et al. (2023)](https://www.sciencedirect.com/science/article/pii/S0034425722004850?via%3Dihub):

1. If RGB convert images to grayscale by using the luminance formula.
2. Detect fiducials markers at subpixel level by performing template matching
3. Outliers are rejected by removing markers that have a too low matching score.
4. Compute affine transformations to align the detected fiducials markers with calibrated fiducials markers.
5. Apply the transformation and crop the images around the center of fiducials markers and apply CLAHE.

Data preprocessing notebooks are available here:

- [casa_grande_aerial.ipynb](https://github.com/shippp/history/blob/main/notebooks/casa_grande_aerial.ipynb)
- [iceland_aerial.ipynb](https://github.com/shippp/history/blob/main/notebooks/iceland_aerial.ipynb)

### KH-9 Panoramic Camera Images

The preprocessing method mainly following [Ghuffar et al. (2023)](https://tc.copernicus.org/articles/17/1299/2023/):

1. The image pieces are joined into a single composite image using ASP’s [`image_mosaic`](https://stereopipeline.readthedocs.io/en/latest/tools/image_mosaic.html) command. It uses IP matching and affine transformation to align the pieces together.
2. The film corners are selected manually.
3. The image is rotated and cropped so that the film edge is horizontal.

Data preprocessing notebooks are available here:

- [casa_grande_kh9pc.ipynb](https://github.com/shippp/history/blob/main/notebooks/casa_grande_kh9pc.ipynb)
- [iceland_kh9pc.ipynb](https://github.com/shippp/history/blob/main/notebooks/iceland_kh9pc.ipynb)

### KH-9 Mapping Camera Images

The preprocessing method follows the method of [Dehecq et al. (2020)](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.566802/full). In brief:

1. the position of the reseau markers (dark crosses) is detected at subpixel resolution using a convolution with a cross-like kernel.
2. outliers are rejected by removing markers that are not regularly spaced (1 cm)
3. a degree 3 polynomial transformation is first applied to remove rotation and scaling, then a Thin Plate Spline interpolation is used to calculate the distortion at each pixel.
4. the two pieces are stitched together by matching the areas of overlap using NCC.
5. the image is cropped to a fix distance from the outermost markers to keep a constant image dimension among the images
6. the reseau markers are filled with inpainting.

## Dataset

| Dataset Name | Date | Images Count | Preprocess Notebook | Images Provider| Preprocessed images size | Raw images size| Bounding Box |
|--|--|--|--|--|--|--|--|
| Casa Grande Aerial | 1978/09/06 | 37 | [casa_grande_aerial.ipynb](notebooks/casa_grande_aerial.ipynb)| [USGS EE](https://earthexplorer.usgs.gov/), [single frame dataset](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-aerial-photo-single-frames?qt-science_center_objects=0#qt-science_center_objects)| 2.6 Go | 7.2 Go |-111.913862   32.699015 -111.685857   32.942928|
| Casa Grande KH-9 PC | 1978/03/25 | 6 |[casa_grande_kh9pc.ipynb](notebooks/casa_grande_kh9pc.ipynb) | [USGS EE](https://earthexplorer.usgs.gov/), [Declass 3](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-3) | 43 Go | 47 Go |-113.535   32.28  -109.812   33.25|
| Casa Grande KH-9 MC | 1978/03/25 | 4 | [casa_grande_kh9mc.ipynb](https://github.com/shippp/history/blob/main/notebooks/casa_grande_kh9mc.ipynb)| [USGS EE](https://earthexplorer.usgs.gov/), [Declass 2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-2?qt-science_center_objects=0#qt-science_center_objects) | 8.2 Go | 8.9 Go |-112.645   30.782 -110.574   35.053|
| Iceland Aerial | 1980/08/22 | 125 | [iceland_aerial.ipynb](notebooks/iceland_aerial.ipynb)| FTP | 14 Go | 12 Go |-19.74345739  63.42948958 -17.68651768  63.86200756|
| Iceland KH-9 PC | 1980/08/22 | 6 | [iceland_kh9pc.ipynb](notebooks/iceland_kh9pc.ipynb) | [USGS EE](https://earthexplorer.usgs.gov/), [Declass 3](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-3) | 33 Go | 30 Go |-23.362  62.923 -15.674  64.405|
| Iceland KH-9 MC | 1980/08/22 | 4 |[iceland_kh9mc.ipynb](https://github.com/shippp/history/blob/main/notebooks/iceland_kh9mc.ipynb) | [USGS EE](https://earthexplorer.usgs.gov/), [Declass 2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-declassified-data-declassified-satellite-imagery-2?qt-science_center_objects=0#qt-science_center_objects) | 8.1 Go | 6.4 Go |-21.946  61.43  -16.633  66.089|

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/shippp/history.git
cd history
conda env create -f environment.yml
```

Once the environment is created, launch Jupyter and select the history kernel to run the notebooks.

## Auxiliary data

### `images_footprint.geojson` & `camera_model_extrinsics.csv`

The image footprints and metadata were downloaded from USGS EarthExplorer for the satellite images and Casa Grande aerial images. For the Iceland aerial images, the footprints were manually generated by Joaquin Belart, using a pre-existing, approximate location of the camera centers. In QGIS (version 3.4), using the digitizing tools, a generic, approximate footprint of one of the images, was drawn, which was then copied successively over all the camera centers. The camera extrinsics were pulled from the footprint files and converted into a CSV file.

### `camera_model_intrinsics.csv`

The camera intrinsics CSV files were filled manually based on the calibration report or information from the data provider.

### Reference DEMs

The high resolution DEMs were downloaded as raw GTiff tiles from the provider, mosaicked without resampling and the vertical datum updated (for Casa Grande only, from NAVD88 geoid to ellipsoid). The low resolution Global Copernicus 30m DEM was downloaded as raw GTiff tiles from OpenTopography, reprojected on the same horizontal CRS as the high-res DEM and converted from original EGM2008 heights to ellipsoid heights. It was then coregistered to the high-res DEM by applying a horizontal and vertical shift, calculated using xDEM’s implementation of the Nuth & Kääb (2011) algorithm. Pixels outside the “stable mask” (see below) are excluded during coregistration.

### Stable mask

The ESA worldcover raw GTiff tiles were downloaded and mosaicked, and reprojected on the same CRS and grid as the two reference DEMs. Anything but shrubland, grassland, bare/sparse vegetation, moss/lichen (respective values of 20, 30, 60 and 100) are masked as unstable. For Iceland, the Randolph glacier inventory (RGI) v7 outlines were rasterized to the DEM grids and masked. For Casa Grande, the subsidence mask was rasterized to the DEM grids and masked. The DEM and stable mask processing scripts are located at https://github.com/shippp/history/tree/main/src/history/aux_data (yet to be added to the repo as of 3 July 2025).

### Ground Control Points - `gcp.csv`

GCPs are provided only for the aerial dataset. They have been manually picked by Joaquin Belart using the data provided in this experiment, as well as the orthomosaic from Bing Maps in QGIS (©Microsoft, data accessible in QGIS via XYZ tiles using http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1). For picking of GCPs, the QGIS plugin “Coordinate capturer” was used. This allows extracting pixel coordinates of the aerial photographs (rows and columns), as well as geographical coordinates of the reference data (longitude and latitude). The measurements were gathered into an ASCII list. Then, the elevation values were extracted from the reference DEM using the command geodiff in the Ames StereoPipeline software (version 3.6, Beyer et al., 2018).
