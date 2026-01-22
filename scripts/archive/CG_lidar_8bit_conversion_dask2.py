"""
Another attempt at dask implementation.

Author: Friedrich Knuth, Amaury Dehecq
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
# memory_limit='auto'
memory_limit='64GB'
# chunksize = 'auto'
chunksize = 4096
workers = 2  #psutil.cpu_count(logical=True)-1

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
    
    if memory_limit=='auto':
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=1,
        )
    else:
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
        )
    client = Client(cluster)
    print('Dashboard at:', cluster.dashboard_link)

    worker_info = client.scheduler_info()['workers']
    if worker_info:
        first_worker_addr = list(worker_info.keys())[0]
        mem_limit_bytes = worker_info[first_worker_addr].get('memory_limit', 'Not set')
        if mem_limit_bytes != 'Not set':
            mem_limit_gb = mem_limit_bytes / (1024 ** 3)
            print(f"Memory limit per worker: {mem_limit_gb:.2f} GB")
        else:
            print(f"Memory limit per worker: {mem_limit_bytes}")
    else:
        print("No workers found")

    if chunksize == 'auto':
        da = rxr.open_rasterio(fn, chunks='auto')
    else:
        da = rxr.open_rasterio(fn, chunks=(1, chunksize, chunksize))
    print('chunk size:', 
          da.data.chunks[0][0], 
          da.data.chunks[1][0], 
          da.data.chunks[2][0])
    stretched = xr.map_blocks(
        gamma_stretch_block,
        da,
        kwargs={'Gmin': 7316, 'Gmax': 42163, 'Gmed': 35788},
        template=da.astype(np.uint8)
    )

    stretched.rio.to_raster(out_fn, 
                            nodata=0, 
                            tiled=True, 
                            compress="JPEG", 
                            jpeg_quality=80, 
                            bigtiff='if_safer',
                           )
    

if __name__ == "__main__":
    main()
