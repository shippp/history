import warnings
from pathlib import Path
from typing import Union, List, Tuple, Dict

import xdem
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
import contextily as ctx
import math
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors
import matplotlib.colorbar
from matplotlib.ticker import ScalarFormatter
from matplotlib.legend_handler import HandlerPatch
from pyproj import Transformer

def plot_extrinsics_comparison(
    initial_df : pd.DataFrame,
    final_df : pd.DataFrame,
    plot_ovals: bool = False,
    oval_scale_factor: float = None,
    oval_alpha: float = 0.6,
    basemap_alpha: float = 0.8,
    basemap_xyz: str = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    colormap: str = "managua",
    cbar_title: str = "Altitude Shift (m)",
    verbose: bool = True,
    show_ticks: bool = True,
    plot_title: str = None,
) -> None:
    """
    Compare initial and final extrinsics positions and plot them on a map.

    Parameters
    ----------
    initial_gdf : pd.DataFrame
        DataFrame with initial camera positions
    final_gdf : pd.DataFrame
        DataFrame with final camera positions
    plot_ovals : bool, default False
        If True, plot error ellipses (ovals) representing position shifts instead of separate points
    oval_scale_factor : float, optional
        Scale factor for oval size. If None, automatically calculated based on 95th percentile of shifts and map scale
    oval_alpha : float, default 0.6
        Transparency for ovals (0=fully transparent, 1=opaque)
    basemap_alpha : float, default 0.8
        Transparency for basemap tiles (0=fully transparent, 1=opaque)
    basemap_xyz : str or None, default Google Satellite
        XYZ tile provider URL for basemap. If None, no basemap is plotted.
    colormap : str, default "managua"
        Colormap for altitude shift visualization. Must be a valid matplotlib colormap name.
    """

    warnings.filterwarnings('ignore', category=UserWarning)

    # Check required columns
    required_cols = ['lat', 'lon', 'alt']
    for df_name, df in [('initial', initial_df), ('final', final_df)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns in {df_name} data: {missing_cols}")
            return
    
    # GeoDataFrames (WGS84)
    initial_gdf = gpd.GeoDataFrame(
        initial_df, 
        geometry=[Point(xy) for xy in zip(initial_df['lon'], initial_df['lat'])],
        crs='EPSG:4326'
    )
    final_gdf = gpd.GeoDataFrame(
        final_df,
        geometry=[Point(xy) for xy in zip(final_df['lon'], final_df['lat'])],
        crs='EPSG:4326'
    )

    # Shifts (for stats)
    if 'image_file_name' in initial_df.columns and 'image_file_name' in final_df.columns:
        merged_df = pd.merge(
            initial_df[['image_file_name', 'lon', 'lat', 'alt']].add_suffix('_initial'),
            final_df[['image_file_name', 'lon', 'lat', 'alt']].add_suffix('_final'),
            left_on='image_file_name_initial',
            right_on='image_file_name_final',
            how='inner'
        )
        if merged_df.empty:
            if verbose:
                print("No matching images found between initial and final datasets")
            return
        x_shift = merged_df['lon_final'] - merged_df['lon_initial']
        y_shift = merged_df['lat_final'] - merged_df['lat_initial'] 
        z_shift = merged_df['alt_final'] - merged_df['alt_initial']
        if verbose:
            print(f"  X (longitude) shift: mean={x_shift.mean():.6f}°, std={x_shift.std():.6f}°")
            print(f"  Y (latitude) shift: mean={y_shift.mean():.6f}°, std={y_shift.std():.6f}°")
            print(f"  Z (altitude) shift: mean={z_shift.mean():.3f}m, std={z_shift.std():.3f}m")
    else:
        print("Warning: Cannot match images by filename, showing all positions")
        initial_centroid = [initial_df['lon'].mean(), initial_df['lat'].mean(), initial_df['alt'].mean()]
        final_centroid = [final_df['lon'].mean(), final_df['lat'].mean(), final_df['alt'].mean()]
        x_shift = final_centroid[0] - initial_centroid[0]
        y_shift = final_centroid[1] - initial_centroid[1]
        z_shift = final_centroid[2] - initial_centroid[2]
        print(f"  X (longitude) shift: {x_shift:.6f}°")
        print(f"  Y (latitude) shift: {y_shift:.6f}°") 
        print(f"  Z (altitude) shift: {z_shift:.3f}m")
    
    # Choose UTM zone by centroid and reproject
    all_gdf = pd.concat([initial_gdf, final_gdf], ignore_index=True)
    lon_c = float(all_gdf['lon'].mean())
    lat_c = float(all_gdf['lat'].mean())
    utm_zone = int(np.floor((lon_c + 180) / 6) + 1)
    utm_epsg = 32600 + utm_zone if lat_c >= 0 else 32700 + utm_zone
    utm_crs = f"EPSG:{utm_epsg}"
    initial_gdf_utm = initial_gdf.to_crs(utm_crs)
    final_gdf_utm   = final_gdf.to_crs(utm_crs)
    all_gdf_utm     = all_gdf.to_crs(utm_crs)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    bounds = all_gdf_utm.total_bounds
    map_extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    
    if plot_ovals and 'image_file_name' in initial_df.columns and 'image_file_name' in final_df.columns:
        merged_df_utm = pd.merge(
            initial_gdf_utm[['image_file_name', 'geometry']].rename(columns={'geometry': 'geom_initial'}),
            final_gdf_utm[['image_file_name', 'geometry']].rename(columns={'geometry': 'geom_final'}),
            on='image_file_name',
            how='inner'
        )
        if merged_df_utm.empty:
            print("No matching images found between initial and final datasets for oval plotting")
            return
        
        initial_x = merged_df_utm['geom_initial'].x.values
        initial_y = merged_df_utm['geom_initial'].y.values
        final_x   = merged_df_utm['geom_final'].x.values
        final_y   = merged_df_utm['geom_final'].y.values
        
        # Use altitude shifts from the original merged_df (not projected)
        z_shift = merged_df['alt_final'] - merged_df['alt_initial']
        
        if oval_scale_factor is None:
            # scale from 95th percentile of displacement magnitude
            disp = np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2)
            p95 = np.percentile(disp, 95)
            oval_scale_factor = map_extent / (20 * p95) if p95 > 0 else 1000
            if verbose:
                print(f"Auto-calculated oval scale factor: {oval_scale_factor:.2f} m/m")
        
        max_abs_z = max(abs(z_shift.min()), abs(z_shift.max()))
        norm = plt.Normalize(vmin=-max_abs_z, vmax=max_abs_z)
        try:
            cmap = plt.get_cmap(colormap)
        except Exception:
            cmap = plt.cm.managua if hasattr(plt.cm, "managua") else plt.cm.RdBu_r

        # ellipse placement to guarantee final on far end
        for i in range(len(merged_df_utm)):
            dx = final_x[i] - initial_x[i]
            dy = final_y[i] - initial_y[i]
            d  = np.hypot(dx, dy)

            # unit direction from initial -> final (handle zero shift)
            if d > 0:
                ux, uy = dx / d, dy / d
            else:
                ux, uy = 1.0, 0.0  # arbitrary

            # semi-major axis a scales with displacement magnitude
            a = max((d * oval_scale_factor) / 2.0, map_extent / 150.0)  # ensure min size
            # semi-minor axis b (visual choice; elongated along shift)
            b = max(a / 4.0, map_extent / 150.0)

            # center so final lies at the "far corner" (+a along û)
            center_x = final_x[i] - a * ux
            center_y = final_y[i] - a * uy

            angle_deg = np.degrees(np.arctan2(dy, dx)) if d > 0 else 0.0
            color = cmap(norm(z_shift.iloc[i]))

            oval = Ellipse((center_x, center_y), 2*a, 2*b, angle=angle_deg,
                           facecolor=color, alpha=oval_alpha, edgecolor='black', linewidth=0.5)
            ax.add_patch(oval)

            # Plot final position (inside ellipse, at far end)
            ax.scatter(final_x[i], final_y[i], c='blue', marker='.', s=40, alpha=0.7, label=None)

        # Legend proxies
        legend_ellipse = Ellipse((0, 0), 1.0, 0.6, facecolor='gray', edgecolor='black', alpha=oval_alpha)
        final_proxy = matplotlib.lines.Line2D([], [], marker='.', linestyle='None',
                                              color='blue', markersize=8, label='Final positions')
        ax.legend(
            [legend_ellipse, final_proxy],
            [f'Oval scale factor ×{oval_scale_factor:.2f}', 'Final positions'],
            handler_map={Ellipse: HandlerPatch()},
            loc='lower left',
            frameon=True
        )

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="2%")
        norm_cbar = matplotlib.colors.Normalize(vmin=-max_abs_z, vmax=max_abs_z)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm_cbar)
        cbar.solids.set_alpha(oval_alpha)
        if cbar_title is not None:
            cbar.set_label(cbar_title)
        
    else:
        # Points only (UTM)
        initial_gdf_utm.plot(ax=ax, color='red', marker='o', markersize=40, alpha=0.7, label='Initial positions')
        final_gdf_utm.plot(ax=ax, color='blue', marker='.', markersize=40, alpha=0.7, label='Final positions')
        ax.legend(loc='upper right')

    
    # Basemap (reproject tiles to UTM if provided)
    if basemap_xyz is not None:
        try:
            ctx.add_basemap(ax, source=basemap_xyz, alpha=basemap_alpha, crs=utm_crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Extent + padding (UTM meters)
    bounds = all_gdf_utm.total_bounds
    padding = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
    ax.set_xlim(bounds[0] - padding, bounds[2] + padding)
    ax.set_ylim(bounds[1] - padding, bounds[3] + padding)

    # Keep shapes true in meters
    ax.set_aspect('equal', adjustable='box')
    
    # Labels (UTM)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Stats
    if verbose:
        print(f"\nDataset statistics:")
        print(f"  Initial positions: {len(initial_gdf)} points")
        print(f"  Final positions: {len(final_gdf)} points")
        if 'image_file_name' in initial_df.columns and 'image_file_name' in final_df.columns:
            print(f"  Matched positions: {len(merged_df)} points")
            if plot_ovals:
                x_shift_m = final_x - initial_x
                y_shift_m = final_y - initial_y
                shift_magnitudes_m = np.sqrt(x_shift_m**2 + y_shift_m**2)
                print(f"\nPosition shift statistics (projected coordinates):")
                print(f"  X shift: mean={x_shift_m.mean():.2f}m, std={x_shift_m.std():.2f}m")
                print(f"  Y shift: mean={y_shift_m.mean():.2f}m, std={y_shift_m.std():.2f}m")
                print(f"  Altitude shift: mean={z_shift.mean():.2f}m, std={z_shift.std():.2f}m")
                print(f"  Oval scale factor used: {oval_scale_factor:.2f}")


def get_epsg_code(dem_file_name):
    """
    Extract the EPSG code from a raster file.
    """
    return str(rasterio.open(dem_file_name).crs.to_epsg())

def df_coords_to_gdf(
    df, 
    lon="lon", 
    lat="lat", 
    z=None, 
    epsg_code="4326"
):
    """
    Convert a pandas DataFrame with coordinate columns into a GeoDataFrame.
    """
    if z is not None and z in df:
        geometry = [Point(xyz) for xyz in zip(df[lon], df[lat], df[z])]
    else:
        geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=f"EPSG:{epsg_code}")
    return gdf

def extract_gpd_geometry(point_gdf):
    """
    Function to extract x, y, z coordinates and add as columns to input geopandas data frame.
    """
    point_gdf = point_gdf.copy()
    x = []
    y = []
    z = []
    for i in range(len(point_gdf)):
        x.append(point_gdf['geometry'].iloc[i].coords[:][0][0])
        y.append(point_gdf['geometry'].iloc[i].coords[:][0][1])
        if len(point_gdf['geometry'].iloc[i].coords[:][0]) == 3:
            z.append(point_gdf['geometry'].iloc[i].coords[:][0][2])

    point_gdf['x'] = x
    point_gdf['y'] = y
    if len(point_gdf['geometry'].iloc[0].coords[:][0]) == 3:
        point_gdf['z'] = z
    return point_gdf

def _as_nan_ndarray(a) -> np.ndarray:
    """Return plain ndarray with mask -> NaN."""
    return np.ma.filled(a, np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
    
def compute_stats(array: np.ndarray) -> Dict[str, float]:
    """
    Compute robust and classical summary stats for an array of elevation differences.

    Returns
    -------
    dict with keys:
        - 'nmad'   : xdem.spatialstats.nmad (NaN-safe)
        - 'median' : np.nanmedian
        - 'mean'   : np.nanmean
        - 'std'    : np.nanstd
    """
    arr = _as_nan_ndarray(array)
    
    nmad   = xdem.spatialstats.nmad(arr)
    median = np.nanmedian(arr)
    mean   = np.nanmean(arr)
    std    = np.nanstd(arr)
    return {'nmad': nmad, 'median': median, 'mean': mean, 'std': std}


def plot_diff_histogram(
    diff_arrays: List[np.ndarray],
    array_keys: List[str],
    error_metrics: List[str] = ['nmad', 'median'],
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[int, int] = (5, 5),
    alpha: float = 0.5,
    bins: int = 256
):
    """
    Plot histograms of one or more elevation-difference arrays and annotate with selected metrics.

    Parameters
    ----------
    diff_arrays : list of np.ndarray
        Arrays of dh values (NaNs allowed).
    array_keys : list of str
        Labels corresponding to each array (e.g., ["Before", "After"]).
    error_metrics : list of {"nmad","median","mean","std"}
        Which metrics to display in the annotations. If none of the provided
        metrics are supported, defaults to ['nmad','median'].
    xlim, ylim : tuple or None
        Axis limits for the histogram.
    figsize : tuple
        Figure size (inches).
    alpha : float
        Histogram alpha.
    bins : int
        Number of histogram bins.

    Returns
    -------
    fig, ax, stats_list
        Matplotlib figure and axis, and a list of per-array stats dicts.
    """
    if len(diff_arrays) != len(array_keys):
        raise ValueError("diff_arrays and array_keys must have the same length.")

    arrays = [_as_nan_ndarray(a) for a in diff_arrays]
    
    supported_error_metrics = ['nmad', 'median', 'mean', 'std']
    # Validate selected error metrics; default if none valid
    em_valid = [m for m in error_metrics if m in supported_error_metrics]
    if len(em_valid) == 0:
        em_valid = ['nmad', 'median']

    # Compute stats for each array
    stats_list = [compute_stats(a) for a in arrays]

    fig, ax = plt.subplots(1, figsize=figsize)

    # Define color cycler
    cmap = cm.get_cmap("brg_r")
    color_cycle = [cmap(i) for i in np.linspace(0.15, 1, len(arrays))]

    # Plot histograms
    for i, (arr, key) in enumerate(zip(arrays, array_keys)):
        finite = np.isfinite(arr)
        color = color_cycle[i % len(color_cycle)]
        ax.hist(arr[finite].ravel(), bins=bins, edgecolor='none', alpha=alpha, label=key, color=color)

    # Axis limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        # flatten all arrays, drop NaNs
        vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
        low, up = np.percentile(vals, [1, 99])
        lim = max(abs(low), abs(up))     # symmetric about zero
        ax.set_xlim(-lim, lim)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        # compute histogram counts manually to find 95th percentile of counts
        all_counts = []
        for arr in arrays:
            finite = arr[np.isfinite(arr)].ravel()
            counts, _ = np.histogram(finite, bins=bins)
            all_counts.append(counts)
        counts_concat = np.concatenate(all_counts)
        up = np.percentile(counts_concat, 99)
        ax.set_ylim(0, up)   # add a small headroom
        
    ax.set_xlabel('Elevation differences (m)')
    ax.set_ylabel('Count')

    # Build and place annotation strings
    # Each label gets its own block: "<Name>\nmetric: value\n..."
    # Place them across the top inside the axes, spaced evenly.
    n = len(arrays)
    if n > 0:
        xs = np.linspace(0.02, 0.68, n)  # normalized x positions
        y = 0.98
        for i, (key, stats) in enumerate(zip(array_keys, stats_list)):
            lines = [f"{key}"]
            for m in em_valid:
                val = stats[m]
                lines.append(f"{m}: {val:.3f}")
            text = "\n".join(lines)
            color = color_cycle[i % len(color_cycle)]
            ax.text(xs[i], y, text, transform=ax.transAxes, va='top', ha='left',
                    fontsize=11, color=color, bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.6, ec=color, lw=1))

    ax.axvline(0, color='black', alpha=0.5, linestyle=':')
    fig.tight_layout()
    return fig, ax, stats_list

def write_asp_transform(output_filename, x_shift=0.0, y_shift=0.0, z_shift=0.0,
                        rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0):
    """
    Write a 4x4 ASP transform file with rotations (deg) and translations (m).
    Rotation order is Z (yaw), then Y (pitch), then X (roll).
    """
    
    # Convert degrees to radians
    rx, ry, rz = np.deg2rad([rot_x_deg, rot_y_deg, rot_z_deg])

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])

    # Compose R (Z * Y * X)
    R = Rz @ Ry @ Rx

    # Build homogeneous 4x4 transform
    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = [x_shift, y_shift, z_shift]

    # Write to disk (space-separated, one row per line)
    np.savetxt(output_filename, M, fmt="%.17g")

    return M, output_filename

def write_asp_pivoted_rotation(
    output_filename,
    pivot_center,           # (x, y, z) in DEM CRS
    rot_x_deg=0.0,          # roll about ECEF X
    rot_y_deg=0.0,          # pitch about ECEF Y
    rot_z_deg=0.0,          # yaw about ECEF Z
    crs_epsg=26912          # source CRS of pivot (e.g., UTM zone 12N)
):
    """
    Write an ASP 4x4 transform that rotates about a local pivot (DEM center).
    """
    # 1) pivot to ECEF
    x, y, z = pivot_center
    to_ecef = Transformer.from_crs(
        f"EPSG:{crs_epsg}", "EPSG:4978", always_xy=True
    )
    Xp, Yp, Zp = to_ecef.transform(x, y, z)

    # 2) rotation matrix R = Rz * Ry * Rx (degrees -> radians)
    rx, ry, rz = np.deg2rad([rot_x_deg, rot_y_deg, rot_z_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])
    R = Rz @ Ry @ Rx

    # 3) build 4x4s
    def T(tx, ty, tz):
        M = np.eye(4)
        M[:3, 3] = [tx, ty, tz]
        return M

    M = np.eye(4)
    M[:3, :3] = R

    M_pivot = T(Xp, Yp, Zp) @ M @ T(-Xp, -Yp, -Zp)

    # 4) write file for ASP (space-separated)
    np.savetxt(output_filename, M_pivot, fmt="%.17g")
    return M_pivot

def analyze_transform(R):
    """
    Analyze a 3x3 transform matrix for rotation, scale, and shear.

    Parameters
    ----------
    R : array-like, shape (3,3)
        Linear transform matrix (e.g. from an ASP 4x4).

    Returns
    -------
    dict
        - scale : (sx, sy, sz)
            Lengths of the column vectors, i.e. scale factors along x, y, z.
        - shear : (shear_xy, shear_xz, shear_yz)
            Dot products between normalized axes; zero means perfectly orthogonal.
        - rotation_deg : (roll, pitch, yaw)
            Euler angles (degrees) recovered in ZYX order
            (yaw about Z, pitch about Y, roll about X).

    Notes
    -----
    - The matrix is decomposed into scale, shear, and rotation parts.
    - If scale ≈ 1 and shear ≈ 0, the transform is a pure rotation.
    - Euler angle extraction uses the ZYX convention and handles gimbal lock.
    - Assumes right-handed coordinate system and column-vector convention.
    """
    R = np.asarray(R, dtype=float)
    
    # --- scale factors (norms of columns) ---
    sx = np.linalg.norm(R[:,0])
    sy = np.linalg.norm(R[:,1])
    sz = np.linalg.norm(R[:,2])
    
    # --- normalize columns to get pure rotation + shear ---
    Rn = np.zeros_like(R)
    Rn[:,0] = R[:,0] / sx
    Rn[:,1] = R[:,1] / sy
    Rn[:,2] = R[:,2] / sz
    
    # --- shear: dot products between axes (should be 0 if orthogonal) ---
    shear_xy = np.dot(Rn[:,0], Rn[:,1])
    shear_xz = np.dot(Rn[:,0], Rn[:,2])
    shear_yz = np.dot(Rn[:,1], Rn[:,2])
    
    # --- recover Euler angles (ZYX order: roll=X, pitch=Y, yaw=Z) ---
    if abs(Rn[2,0]) < 1.0:  # not gimbal locked
        ry = -np.arcsin(Rn[2,0])
        rx = np.arctan2(Rn[2,1], Rn[2,2])
        rz = np.arctan2(Rn[1,0], Rn[0,0])
    else:  # gimbal lock
        ry = np.pi/2 if Rn[2,0] < 0 else -np.pi/2
        rx = np.arctan2(-Rn[0,1], -Rn[0,2])
        rz = 0.0

    print("=== Transform analysis ===")
    print(f"Scale: sx={sx:.6f}, sy={sy:.6f}, sz={sz:.6f}")
    print(f"Shear (dot products): xy={shear_xy:.6e}, xz={shear_xz:.6e}, yz={shear_yz:.6e}")
    print(f"Rotation (deg): rx={np.degrees(rx):.6f}, ry={np.degrees(ry):.6f}, rz={np.degrees(rz):.6f}")

    return dict(scale=(sx,sy,sz),
                shear=(shear_xy,shear_xz,shear_yz),
                rotation_deg=(np.degrees(rx), np.degrees(ry), np.degrees(rz)))