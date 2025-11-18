from prefect import task

import history.postprocessing.core as core
import history.postprocessing.visualization as viz

extract_archive = task(core.extract_archive)
convert_pointcloud_to_dem = task(core.convert_pointcloud_to_dem)
coregister_dem = task(core.coregister_dem)
generate_ddem = task(core.generate_ddem)
compute_raster_statistics = task(core.compute_raster_statistics)
compute_raster_statistics_by_landcover = task(core.compute_raster_statistics_by_landcover)
create_std_dem = task(core.create_std_dem)

generate_std_dem_plots = task(viz.generate_std_dem_plots)
generate_coregistration_individual_plots = task(viz.generate_coregistration_individual_plots)
generate_dems_mosaic = task(viz.generate_dems_mosaic)
generate_ddems_mosaic = task(viz.generate_ddems_mosaic)
generate_slopes_mosaic = task(viz.generate_slopes_mosaic)
generate_hillshades_mosaic = task(viz.generate_hillshades_mosaic)
