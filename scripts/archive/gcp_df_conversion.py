"""
Script to convert GCP files in Adina/Camillo's format into the Historix format.

Author: Friedrich Knuth
Date: Jan 2026
"""

import pandas as pd
from pyproj import Transformer

# # For first format (GCP_ID, Xref, Yref, Zref, image, Ximage, Yimage)
# df1 = pd.read_csv('KH9_MC_CasaGrande_GCPs_merged_TU_Adina_converted.csv')
# mapping1 = {
#     'gcp_id': 'GCP_ID',
#     'x_ref': 'Xref',
#     'y_ref': 'Yref',
#     'z_ref': 'Zref',
#     'image': 'image',
#     'x_image': 'Ximage',
#     'y_image': 'Yimage'
# }
# df_output = convert_gcp_dataframe_format(df1, mapping1, source_epsg=32612)

# # For second format (GCPid, Xutm, Yutm, H, imgFname, x, y)
# df2 = pd.read_csv('KH9MC_Iceland_TUWGCPs_converted.csv')
# mapping2 = {
#     'gcp_id': 'GCPid',
#     'x_ref': 'Xutm',
#     'y_ref': 'Yutm',
#     'z_ref': 'H',
#     'image': 'imgFname',
#     'x_image': 'x',
#     'y_image': 'y'
# }
# df_output = convert_gcp_dataframe_format(df2, mapping2, source_epsg=32627)

def convert_gcp_dataframe_format(
    df_input: pd.DataFrame,
    column_mapping: dict,
    source_epsg: int = 32612,  # UTM Zone 
    target_epsg: int = 4326,   # WGS84
    default_accuracy: dict = None
) -> pd.DataFrame:
    """
    Convert GCP dataframe to HISTORIX format.
    
    Parameters:
    - df_input: Input dataframe with GCP data
    - column_mapping: Dictionary mapping input columns to standard names:
        {
            'gcp_id': 'GCP_ID',      # or 'GCPid'
            'x_ref': 'Xref',         # or 'Xutm'
            'y_ref': 'Yref',         # or 'Yutm'
            'z_ref': 'Zref',         # or 'H'
            'image': 'image',        # or 'imgFname'
            'x_image': 'Ximage',     # or 'x'
            'y_image': 'Yimage'      # or 'y'
        }
    - source_epsg: Source EPSG code for UTM coordinates
    - target_epsg: Target EPSG code (default: 4326 for WGS84)
    - default_accuracy: Dictionary with accuracy values (lon_acc, lat_acc, elev_acc)
    
    Returns:
    - DataFrame in HISTORIX format with columns:
      [gcp_label, image_file_name, x, y, x_map, y_map, lon, lat, elev, lon_acc, lat_acc, elev_acc]
    """
    if default_accuracy is None:
        default_accuracy = {
            "lon_acc": 2,
            "lat_acc": 2,
            "elev_acc": 1
        }
    
    # Extract data using provided mapping
    gcp_id = df_input[column_mapping['gcp_id']]
    x_ref = df_input[column_mapping['x_ref']]
    y_ref = df_input[column_mapping['y_ref']]
    z_ref = df_input[column_mapping['z_ref']]
    image_name = df_input[column_mapping['image']]
    x_image = df_input[column_mapping['x_image']]
    y_image = df_input[column_mapping['y_image']]
    
    # Create transformer for coordinate conversion
    transformer = Transformer.from_crs(source_epsg, target_epsg, always_xy=True)
    
    # Convert coordinates to lon/lat
    lon, lat = transformer.transform(x_ref.values, y_ref.values)
    
    # Create output dataframe
    df_output = pd.DataFrame({
        'gcp_label': gcp_id,
        'image_file_name': image_name,
        'x': x_image,
        'y': y_image,
        'x_map': x_ref,
        'y_map': y_ref,
        'lon': lon,
        'lat': lat,
        'elev': z_ref,
        'lon_acc': default_accuracy['lon_acc'],
        'lat_acc': default_accuracy['lat_acc'],
        'elev_acc': default_accuracy['elev_acc']
    })
    
    return df_output
