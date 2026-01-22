"""
Same as CG_lidar_8bit_conversion.py but with a Dask implementation to reduce memory usage.
However, the worker config is not very reliable. The culprit might be on the saving, that put all memory into one worker.
Also some values seem to be masked in the process.

Author: Fridrich Knut, Amaury Dehecq
Date: Dec 2025
"""

import psutil
import numpy as np
import rioxarray as rxr
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster

fn = "casa_grande/aux_data/tmp/zoom_aoi-intensity_mos_25cm.tif"
out_fn = "casa_grande/aux_data/lidar_intensity_mosaic_0.25m_8bit.tif"
workers = 4 #psutil.cpu_count(logical=True)-1

def gamma_stretch_block(block, Gmin, Gmax, Gmed):
    k1 = 1 / (Gmax - Gmin)
    I = k1 * (block.astype("float32") - Gmin)
    Gmed1 = k1 * (Gmed - Gmin)
    
    I = np.clip(I, 0, 1)
    
    gamma = np.log(0.5) / np.log(Gmed1)
    I = I ** gamma
    
    return (254 * I + 1).astype(np.uint8)

def main():
    print('dask version', dask.__version__)
    print('xarray version', xr.__version__)
    print('rioxarray version', rxr.__version__)
    
    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=1,
    )
    client = Client(cluster)
    print(client)

    da = rxr.open_rasterio(fn, chunks='auto')
    stretched = xr.map_blocks(
        gamma_stretch_block,
        da,
        kwargs={'Gmin': 7316, 'Gmax': 42163, 'Gmed': 35788},
        template=da.astype(np.uint8)
    )

    stretched.rio.to_raster(out_fn, nodata=0, tiled=True, compress="JPEG", jpeg_quality=80)
    

if __name__ == "__main__":
    main()
