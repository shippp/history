import warnings
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors
import matplotlib.colorbar
from matplotlib.ticker import ScalarFormatter
from matplotlib.legend_handler import HandlerPatch

from .io import *


def plot_distortion_grids(
    intrinsics_df: pd.DataFrame,
    site: str = None,
    extent_mm: float = 114.3,
    verbose: bool = False,
    units: str = "mm",  # "mm" or "px" (uses pixel_pitch for "px")
    viz_gain: float = 1.0,
) -> None:
    """
    Plot a gallery of distortion grids (Brown–Conrady, radial only), one panel per experiment.

    Parameters
    ----------
    intrinsics_df : pd.DataFrame
        Combined intrinsics DataFrame from combine_intrinsics_files()
    site : str, optional
        Site code to filter by (e.g., 'CG', 'IL'). If None, all sites will be included.
    extent_mm : float, default 114.3
        Half-size of the square plotting domain, in millimeters.
        For 9" × 9" film (228.6 mm side), the half-size is 114.3 mm.
        The diagonal distance from the center to a corner is √2 * extent_mm ≈ 162 mm.
    verbose : bool, default False
        Whether to print details for each experiment.
    units : {"mm", "px"}, default "mm"
        Units for grid axes and plotted lines. "px" needs a valid `pixel_pitch` (mm/px).
    viz_gain : float, default 1.0
        Visual-only multiplier applied to the radial distortion series
        (1 + viz_gain*(k1 r^2 + k2 r^4 + k3 r^6 + k4 r^8)).
        Increase (e.g., 20–50) to make small distortions visible.
    """
    # Filter by site if specified
    if site is not None:
        filtered_df = intrinsics_df[intrinsics_df['site'] == site].copy()
        if filtered_df.empty:
            print(f"No data found for site={site}")
            return
    else:
        filtered_df = intrinsics_df.copy()

    # Ensure required columns exist
    for col in ['k1', 'k2', 'k3', 'k4', 'focal_length', 'pixel_pitch']:
        if col not in filtered_df.columns:
            filtered_df[col] = np.nan

    # Need at least one radial coefficient and a focal length (mm)
    has_distortion = filtered_df[['k1', 'k2', 'k3', 'k4']].notna().any(axis=1)
    df = filtered_df[has_distortion].copy()
    df = df[df['focal_length'].notna()]
    if df.empty:
        print("No rows with distortion coefficients and focal_length (mm).")
        return

    # Unique experiments
    experiments = df['experiment_code'].astype(str).unique()
    n = len(experiments)
    if n == 0:
        print("No experiments to plot.")
        return

    # Auto layout (square-ish)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Figure size
    figsize = (cols * 4.2, rows * 4.2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    # Grid definition
    t = np.linspace(-1.0, 1.0, 600)
    n_lines = 11
    undist_color = (0.7, 0.7, 0.7)
    dist_color = 'k'

    exp_idx = 0
    for r_i in range(rows):
        for c_i in range(cols):
            ax = axes[r_i, c_i]
            ax.axis('off')

            if exp_idx >= n:
                continue

            exp_code = experiments[exp_idx]
            row = df[df['experiment_code'] == exp_code].iloc[0].fillna(0)

            k1 = float(row.get('k1', 0))
            k2 = float(row.get('k2', 0))
            k3 = float(row.get('k3', 0))
            k4 = float(row.get('k4', 0))
            f_mm = float(row.get('focal_length', 0))
            if f_mm <= 0:
                if verbose:
                    print(f"Skipping {exp_code}: invalid focal_length")
                exp_idx += 1
                continue

            pixel_pitch = row.get('pixel_pitch', 0)  # mm/px
            if units == "px":
                if pixel_pitch is None or pixel_pitch == 0:
                    if verbose:
                        print(f"Skipping {exp_code}: missing/invalid pixel_pitch for px units")
                    exp_idx += 1
                    continue

            # Half-extent in chosen units
            R_plot = extent_mm if units == "mm" else (extent_mm / pixel_pitch)

            line_coords = np.linspace(-R_plot, R_plot, n_lines)

            def distort_xy(X_plot, Y_plot):
                # Convert to mm for normalization
                if units == "mm":
                    X_mm, Y_mm = X_plot, Y_plot
                else:
                    X_mm, Y_mm = X_plot * pixel_pitch, Y_plot * pixel_pitch

                r2_norm = (X_mm**2 + Y_mm**2) / (f_mm**2)
                # Apply viz_gain to the distortion series ONLY
                s = 1 + viz_gain * (k1*r2_norm + k2*(r2_norm**2) + k3*(r2_norm**3) + k4*(r2_norm**4))

                Xd_mm, Yd_mm = X_mm * s, Y_mm * s
                return (Xd_mm, Yd_mm) if units == "mm" else (Xd_mm / pixel_pitch, Yd_mm / pixel_pitch)

            # Draw grid
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-R_plot, R_plot)
            ax.set_ylim(-R_plot, R_plot)
            ax.set_title(exp_code, fontsize=10)
            ax.set_xlabel(f"x ({units})", fontsize=9)
            ax.set_ylabel(f"y ({units})", fontsize=9)

            # Vertical lines
            for x0 in line_coords:
                X = np.full_like(t, x0)
                Y = t * R_plot
                # ax.plot(X, Y, color=undist_color, linewidth=1)
                Xd, Yd = distort_xy(X, Y)
                ax.plot(Xd, Yd, color=dist_color, linewidth=1)

            # Horizontal lines
            for y0 in line_coords:
                X = t * R_plot
                Y = np.full_like(t, y0)
                # ax.plot(X, Y, color=undist_color, linewidth=1)
                Xd, Yd = distort_xy(X, Y)
                ax.plot(Xd, Yd, color=dist_color, linewidth=1)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(labelsize=8)
            ax.set_xticks(np.linspace(-R_plot, R_plot, 5))
            ax.set_yticks(np.linspace(-R_plot, R_plot, 5))
            ax.axis('on')

            if verbose:
                print(f"{exp_code}: k=[{k1:.3e},{k2:.3e},{k3:.3e},{k4:.3e}], "
                      f"{units}, extent={R_plot:.2f}, f_mm={f_mm:.3f}, viz_gain={viz_gain}")

            exp_idx += 1

    # Hide any unused axes
    while exp_idx < rows * cols:
        r_i = exp_idx // cols
        c_i = exp_idx % cols
        axes[r_i, c_i].axis('off')
        exp_idx += 1

    plt.tight_layout()
    plt.show()



def plot_focal_length_spread(intrinsics_df: pd.DataFrame, site: str = 'CG', images: str = 'PP', dataset: str = None) -> None:
    """
    Plot the spread of focal length values for specified experimental conditions.
    
    Parameters
    ----------
    intrinsics_df : pd.DataFrame
        Combined intrinsics DataFrame from combine_intrinsics_files()
    site : str, default 'CG'
        Site code to filter by (e.g., 'CG' for Casa Grande, 'IL' for Iceland)
    images : str, default 'PP'
        Images code to filter by (e.g., 'PP' for Pre-processed, 'RA' for Raw)
    dataset : str, optional
        Dataset code to filter by (e.g., 'AI' for Aerial, 'MC' for KH-9 MC, 'PC' for KH-9 PC)
        If None, all datasets will be included
    """
    # Filter the data using the reusable filtering function
    filtered_df = filter_experiment_data(intrinsics_df, site=site, images=images, dataset=dataset)
    
    if filtered_df.empty:
        filter_text = f"site={site} and images={images}"
        if dataset is not None:
            filter_text += f" and dataset={dataset}"
        print(f"No data found for {filter_text}")
        return
    
    # Check if focal_length column exists
    if 'focal_length' not in filtered_df.columns:
        print("No 'focal_length' column found in the data")
        print("Available columns:", list(filtered_df.columns))
        return
    
    # Remove any NaN values
    focal_lengths = filtered_df['focal_length'].dropna()
    
    if focal_lengths.empty:
        filter_text = f"site={site} and images={images}"
        if dataset is not None:
            filter_text += f" and dataset={dataset}"
        print(f"No valid focal length data found for {filter_text}")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Get unique experiment codes and assign colors
    unique_experiments = filtered_df['experiment_code'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_experiments)))
    
    # Box plot without outliers (to avoid duplication with scatter points)
    ax.boxplot(focal_lengths, vert=True, showfliers=False)
    
    # Plot scatter points colored by experiment code
    for i, exp_code in enumerate(unique_experiments):
        exp_data = filtered_df[filtered_df['experiment_code'] == exp_code]['focal_length'].dropna()
        if not exp_data.empty:
            ax.scatter(np.ones(len(exp_data)), exp_data, alpha=0.7, s=40, color=colors[i], label=exp_code)
    
    ax.set_ylabel('Focal Length')
    
    # Format y-axis to avoid scientific notation - use multiple approaches
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    
    # Build title with dataset info if specified
    title_parts = [f'Site: {site}', f'Images: {images}']
    if dataset is not None:
        title_parts.append(f'Dataset: {dataset}')
    ax.set_title(f'Focal Length Spread\n({", ".join(title_parts)})')
    
    # Build x-axis label
    xlabel_parts = [site, images]
    if dataset is not None:
        xlabel_parts.append(dataset)
    ax.set_xticklabels(['_'.join(xlabel_parts)])
    
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)


    plt.tight_layout()
    plt.show()



def plot_distortion_profiles(
    intrinsics_df: pd.DataFrame,
    site: str = None,
    max_radius_mm: float = 162.0,
    log_scale: bool = False,
    verbose: bool = False,
    units: str = "mm",   # <-- NEW: choose "mm" or "px"
) -> None:
    """
    Plot radial distortion profiles for experiments using Brown-Conrady camera model.

    Parameters
    ----------
    intrinsics_df : pd.DataFrame
        Combined intrinsics DataFrame from combine_intrinsics_files()
    site : str, optional
        Site code to filter by (e.g., 'CG' for Casa Grande, 'IL' for Iceland)
        If None, all sites will be included
    max_radius_mm : float, default 162.0
        Maximum radial distance to plot (in mm). Default covers corner-to-center
        distance for 228.6mm × 228.6mm square images (9" × 9")
    log_scale : bool, default False
        Whether to use logarithmic scale for y-axis to better show small distortion values
    verbose : bool, default False
        Whether to print distortion coefficients for each experiment
    units : {"mm","px"}, default "mm"
        Whether to plot radial distance and displacement in millimeters or pixels.
    """
    # Filter by site if specified
    if site is not None:
        filtered_df = intrinsics_df[intrinsics_df['site'] == site].copy()
        if filtered_df.empty:
            print(f"No data found for site={site}")
            return
    else:
        filtered_df = intrinsics_df.copy()

    # Ensure needed columns exist
    for col in ['k1', 'k2', 'k3', 'k4', 'focal_length', 'pixel_pitch']:
        if col not in filtered_df.columns:
            filtered_df[col] = np.nan

    has_distortion = filtered_df[['k1', 'k2', 'k3']].notna().any(axis=1)
    distortion_df = filtered_df[has_distortion].copy()

    if distortion_df.empty:
        site_text = f" for site={site}" if site else ""
        print(f"No experiments with radial distortion parameters found{site_text}")
        return

    # Radial distances in mm
    r_mm = np.linspace(0, max_radius_mm, 1000)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get unique experiment codes and assign colors
    unique_experiments = distortion_df['experiment_code'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_experiments)))
    line_styles = ['-', '--', '-.', ':']

    # Plot distortion profile for each experiment
    for i, exp_code in enumerate(unique_experiments):
        exp_data = distortion_df[distortion_df['experiment_code'] == exp_code].iloc[0].fillna(0)

        # Coefficients
        k1, k2, k3, k4 = exp_data.get('k1', 0), exp_data.get('k2', 0), exp_data.get('k3', 0), exp_data.get('k4', 0)

        # Need focal length in mm
        f_mm = exp_data.get('focal_length', 0)
        if f_mm is None or f_mm == 0:
            if verbose:
                print(f"Skipping {exp_code}: missing/invalid focal_length (mm)")
            continue

        # Pixel pitch for px conversion
        pixel_pitch = exp_data.get('pixel_pitch', None)  # mm/px

        if units == "px":
            if pixel_pitch is None or pixel_pitch == 0:
                if verbose:
                    print(f"Skipping {exp_code}: missing/invalid pixel_pitch for px conversion")
                continue
            # Convert to pixel units
            r = r_mm / pixel_pitch
            f = f_mm / pixel_pitch
        else:  # mm
            r = r_mm
            f = f_mm

        if verbose:
            print(exp_code, f"f={f} {units}", k1, k2, k3, k4)

        # Evaluate Brown–Conrady on normalized radius
        r_norm = r_mm / f_mm  # always normalize with mm
        r2, r4, r6, r8 = r_norm**2, (r_norm**2)**2, (r_norm**2)**2 * r_norm**2, (r_norm**2)**4

        dr_mm = r_mm * (k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8)

        # Convert displacement to pixels if needed
        dr = dr_mm / pixel_pitch if units == "px" else dr_mm

        # Plot
        if log_scale:
            abs_dr = np.abs(dr)
            if np.any(abs_dr > 0):
                ax.plot(r, abs_dr, color=colors[i], linewidth=2,
                        linestyle=line_styles[i % len(line_styles)], label=exp_code)
        else:
            ax.plot(r, dr, color=colors[i], linewidth=2,
                    linestyle=line_styles[i % len(line_styles)], label=exp_code)

    # Labels and titles
    ax.set_xlabel(f'Radial Distance from Principal Point ({units})')
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel(f'Absolute Radial Distortion ({units}) [Log Scale]')
        ax.set_title(f'Radial Distortion Profiles - Brown-Conrady Model (Log Scale){f" (Site: {site})" if site else ""}')
    else:
        ax.set_ylabel(f'Radial Distortion ({units})')
        ax.set_title(f'Radial Distortion Profiles - Brown-Conrady Model{f" (Site: {site})" if site else ""}')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_extrinsics_comparison(
    extrinsics_df: pd.DataFrame,
    experiment_code: str,
    initial_extrinsics_path: Union[str, Path],
    plot_ovals: bool = False,
    oval_scale_factor: float = None,
    oval_alpha: float = 0.6,
    basemap_alpha: float = 0.8,
    basemap_xyz: str = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    colormap: str = "managua",
    cbar_title: str = "Altitude Shift (m)",
    verbose: bool = True,
    show_ticks: bool = True,
) -> None:
    """
    Compare initial and final extrinsics positions and plot them on a map.

    Parameters
    ----------
    extrinsics_df : pd.DataFrame
        Combined extrinsics DataFrame from combine_extrinsics_files()
    experiment_code : str
        Experiment code to filter by (e.g., 'ESM_CG_AI_PP_CN_GY_PN_MN')
    initial_extrinsics_path : str or Path
        Path to the initial extrinsics CSV file
    plot_ovals : bool, default False
        If True, plot error ellipses (ovals) representing position shifts instead of separate points
    oval_scale_factor : float, optional
        Scale factor for oval size. If None, automatically calculated based on 95th percentile of shifts
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
    
    # Read initial extrinsics file
    try:
        initial_df = pd.read_csv(initial_extrinsics_path)
    except Exception as e:
        print(f"Error reading initial extrinsics file: {e}")
        return
    
    # Filter extrinsics_df to specified experiment code
    final_df = extrinsics_df[extrinsics_df['experiment_code'] == experiment_code].copy()
    
    if final_df.empty:
        print(f"No data found for experiment code: {experiment_code}")
        return
    
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
            print(f"Position shifts for {experiment_code}:")
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
        print(f"Centroid shifts for {experiment_code}:")
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

        # ---- ellipse placement to guarantee final on far end ----
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
            [f'Scale factor ×{oval_scale_factor:.2f}', 'Final positions'],
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

        plot_title = f'Extrinsics Comparison: {experiment_code}'
        
    else:
        # Points only (UTM)
        initial_gdf_utm.plot(ax=ax, color='red', marker='o', markersize=40, alpha=0.7, label='Initial positions')
        final_gdf_utm.plot(ax=ax, color='blue', marker='.', markersize=40, alpha=0.7, label='Final positions')
        ax.legend(loc='upper right')
        plot_title = f'Extrinsics Comparison: {experiment_code}'
    
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
                print(f"  Horizontal shift: mean={shift_magnitudes_m.mean():.2f}m, max={shift_magnitudes_m.max():.2f}m")
                print(f"  Altitude shift: mean={z_shift.mean():.2f}m, std={z_shift.std():.2f}m")
                print(f"  Oval scale factor used: {oval_scale_factor:.2f}")