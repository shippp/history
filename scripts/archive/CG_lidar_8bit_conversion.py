"""
An attempt to convert the lidar intensity mosaic from 16 bits to 8 bit, using a Gamma strecth.
Original implementation from Camillo in Matlab.
The code works but memory requirements are very large due to the file size.
Made an attempt with geoutils multiprocessing tools, but fail saving a BIGTIFF file.

Author: Amaury Dehecq
Date: Dec 2025
"""

import numpy as np
import matplotlib.pyplot as plt


def gamma_stretch(I, Gmin, Gmax, Gmed):
    
    # 1. [Gmin  Gmax] --> [0 1]
    k1 = 1/(Gmax-Gmin)
    I = k1*(I.astype("float32") - Gmin)
    Gmed1 = k1*(Gmed - Gmin)

    # casting
    I[I<0] = 0
    I[I>1] = 1

    # 2. [0 1] --> [0 1] using gamma such that Gmed1 becomes 0.5
    gamma = np.log(0.5)/np.log(Gmed1)
    I = I**gamma

    # 3. unit8
    J = np.uint8(255*I)
    # plt.hist(J, bins=255)
    # plt.show()

    return J


import geoutils as gu

# Test on a cropped version -> works
# lidar_16bit = gu.Raster("zoom_aoi-intensity_mos_25cm.tif")
# test = lidar_16bit.icrop([0, 0, 10000, 10000])
# test_8bit = gamma_stretch(test.data, 7316, 42163, 35788)

# test_8bit_rst = gu.Raster.from_array(test_8bit, transform=test.transform, crs=test.crs)
# test_8bit_rst.save("tmp.tif", tiled=True)


# Test multiproc -> as of Dec 2025, fails as map_overlap_multiproc_save does not handle BIGTIFFs
from geoutils.raster import map_overlap_multiproc_save, MultiprocConfig, ClusterGenerator
config_basic = MultiprocConfig(chunk_size=10000, outfile="lidar_mos_crop_8bit.tif", cluster=None)

lidar_16bit = gu.Raster("zoom_aoi-intensity_mos_25cm.tif")
test = lidar_16bit.icrop([0, 0, 40000, 40000])
test.save("lidar_mos_crop.tif")

def gamma_stretch_custom(rst):
    output = gamma_stretch(rst.data, 7316, 42163, 35788)
    rst.data = output
    return rst

raster_filtered = map_overlap_multiproc_save(gamma_stretch_custom, "lidar_mos_crop.tif", config_basic)

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(test.data, cmap="gray")
# axes[1].imshow(test_8bit, cmap="gray")
# plt.show()


# lidar_8bit = gamma_stretch(lidar, 7316, 42163, 35788)
