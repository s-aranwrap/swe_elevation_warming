"""
Plotting and analysis functions for the storm-SWE figures.

Paths, basins and elevation bins come from config.py. Figure-specific choices
(which storm to show, colour limits, label positions, bootstrap size, ...) are
keyword arguments whose defaults reproduce the current figures.
"""
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import piecewise_regression
import pwlf
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from matplotlib.colors import BoundaryNorm, LightSource, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.interpolate import PchipInterpolator
from scipy.stats import linregress, t
from shapely.geometry import box
from shapely.ops import unary_union
from sklearn.metrics import r2_score

from config import data_folder, Basins, basin_names, basin_names_list, shapes, bins, elev_bins
from utils import calculate_water_year


# =============================================================================
# Paths and shared settings
# =============================================================================
CLEANED = os.path.join(data_folder, 'Data/Cleaned')
ERA5_PATH = os.path.join(CLEANED, 'ERA5/era5_complete_ivt.nc4')
DEM_PATH = os.path.join(data_folder, 'Data/DEM/coarsened_15.nc4') # if this does not run, change to 'Data/DEM/coarsened.nc4'

FIG_DIR = 'Figures'

K_TO_C = 273.15
WY_RANGE = (1985, 2021)                    # SWE-period water years (inclusive)
MAP_EXTENT = (-124.5, -117.5, 34, 42)      # lon_min, lon_max, lat_min, lat_max
HEIGHT_TARGET = np.linspace(0.0, 5000.0, num=1000)  # m, for the cross-section

ELEV_COLORS = dict(zip(bins, ['#ff7f00', '#4daf4a', '#984ea3']))
MAX_SIERRA_ELEV_M = 4500  # anything higher means elevations are in feet

LOW_ELEV_COLOR = 'darkgray'  # basin area outside the study area
LOW_ELEV_ALPHA = 0.45       # ...drawn semi-transparent over the slopes
HILLSHADE = dict(azdeg=315, altdeg=30, vert_exag=0.3, dx=5, dy=5, strength=0.2, gamma=2.2)

# Basin name labels on the DEM map: basin -> (lon, lat, text)
BASIN_LABELS = {
    'LowerSacramento': (-121.9, 39.3, 'Lower\nSacramento'),
    'SanJoaquin': (-120.7, 37.5, 'San\nJoaquin'),
    'TulareBuenaVistaLakes': (-119.65, 36.1, 'Tulare-Buena\nVista Lakes'),
    'MonoOwensLakes': (-118.25, 38.4, 'Mono-Owens\nLakes'),
}

# SNOTEL HUC6 basins moved to the neighbouring western basin, so they get a storm temperature
SNOTEL_BASIN_REMAP = {
    'Walker': 'San Joaquin',
    'Carson': 'San Joaquin',
    'Truckee': 'Lower Sacramento',
    'Oregon Closed Basins': 'Klamath',
}


# =============================================================================
# Small helpers
# =============================================================================
def basin_polygons(basins=None):
    """Shapes for `basins` (default: every basin in config.basin_names)."""
    names = [basin_names[b] for b in (basins if basins is not None else basin_names)]
    return shapes[shapes['name'].isin(names)]


def add_panel_labels(axes, labels, x=0, y=1, fontsize=13, pad=0.3, ha='left'):
    """Bold boxed panel letters (A, B, ...) in axes coordinates."""
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight='bold', va='top', ha=ha,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle=f'square,pad={pad}'))


def add_inset_colorbar(ax, mappable, title, anchor=(-0.05, -0.07, 1, 1), title_y=1.07, title_size=8,
                       tick_size=7, tick_width=None, linespacing=None,
                       subtitle=None, subtitle_y=-0.08, subtitle_size=7):
    """Small vertical colorbar in the upper right of `ax`, ticks on the left, title above."""
    cax = inset_axes(ax, width='4%', height='20%', loc='upper right', bbox_to_anchor=anchor,
                     bbox_transform=ax.transAxes, borderpad=0.0)
    cb = plt.colorbar(mappable, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=tick_size, **({} if tick_width is None else {'width': tick_width}))
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.text(1.3, title_y, title, transform=cb.ax.transAxes, ha='right', va='bottom',
               fontsize=title_size, linespacing=linespacing)
    if subtitle:
        cb.ax.text(1.3, subtitle_y, subtitle, transform=cb.ax.transAxes, ha='right', va='top',
                   fontsize=subtitle_size)
    return cb


def legend_without_scatter(ax, **kwargs):
    """Legend that skips artists labelled 'scatter'."""
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l != 'scatter']
    return ax.legend(*zip(*keep), **kwargs)


def truncate_colormap(cmap, vmin=0.0, vmax=1.0, n=256):
    """Return a new colormap that maps [vmin, vmax] -> [0, 1]."""
    cmap = plt.get_cmap(cmap)
    return mcolors.LinearSegmentedColormap.from_list(f'{cmap.name}_trunc', cmap(np.linspace(vmin, vmax, n)))


TERRAIN_CMAP = truncate_colormap('terrain', 0.43, 1)


def check_elevation_m(elevation, source):
    """Raise if station elevations look like feet rather than metres."""
    top = np.nanmax(np.asarray(elevation, dtype=float))
    if top > MAX_SIERRA_ELEV_M:
        raise ValueError(f'{source}: max elevation {top:.0f} looks like feet, expected metres.')


def lon_formatter(x, pos):
    """Tick label like 121°W."""
    if x < 0:
        return f'{abs(x):.0f}°W'
    return f'{x:.0f}°E' if x > 0 else '0°'


# =============================================================================
# Maps
# =============================================================================
def add_features(ax, facecolor='lightgray'):
    ax.add_feature(cfeature.LAND, facecolor=facecolor)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2)
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.3)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='gray', linewidth=0.2)
    ax.add_feature(cfeature.RIVERS, edgecolor='deepskyblue', linewidth=0.2)


def format_map(ax, polygons, extent=MAP_EXTENT):
    """Basin outlines, map extent and left/bottom gridline labels. Returns the gridliner."""
    polygons.plot(edgecolor='black', facecolor='none', linewidth=0.5, ax=ax)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    ax.set_aspect('equal')
    return gl


def shaded_relief(z, cmap, norm, azdeg=315, altdeg=30, vert_exag=0.3, dx=5, dy=5, strength=0.2, gamma=2.2):
    """RGBA image of `z` coloured by `cmap`, gently modulated by a hillshade; NaN -> transparent."""
    rgb = cmap(norm(z))[..., :3]
    hs = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(z, vert_exag=vert_exag, dx=dx, dy=dy)
    gain = 1 + strength * (hs - 0.5) * 2                               # ~[1-strength, 1+strength]
    shaded = np.clip(np.clip(rgb, 0, 1) ** gamma * gain[..., None], 0, 1) ** (1 / gamma)  # in linear light
    return np.dstack([shaded, np.isfinite(z).astype(float)])


def plot_dem_and_elev_bins(basins=Basins, dem_path=DEM_PATH, vlim=(0, 4500), hillshade=HILLSHADE,
                           basin_labels=BASIN_LABELS, extent=MAP_EXTENT, figsize=(8, 8), panel_labels=('A', 'B')):
    """(A) shaded-relief DEM of the basins, (B) elevation bins 2–4."""
    polygons = basin_polygons(basins)
    dem = (xr.open_dataset(dem_path)['Elevation']
             .rio.write_crs('EPSG:4326')
             .rio.set_spatial_dims(x_dim='Longitude', y_dim='Latitude')
             .rio.clip([unary_union(polygons.geometry)], crs='EPSG:4326'))
    img_extent = [float(dem.Longitude.min()), float(dem.Longitude.max()),
                  float(dem.Latitude.min()), float(dem.Latitude.max())]

    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()},
                             constrained_layout=True, figsize=figsize)
    for ax in axes:
        add_features(ax)

    # (A) shaded relief
    ax = axes[0]
    norm = Normalize(*vlim)
    ax.set_facecolor('1')
    ax.imshow(shaded_relief(dem.values, TERRAIN_CMAP, norm, **hillshade), extent=img_extent)
    ax.set_title('')
    add_inset_colorbar(ax, plt.cm.ScalarMappable(norm=norm, cmap=TERRAIN_CMAP), 'Elevation [m]',
                       anchor=(-0.05, -0.06, 1, 1), title_y=1.05, tick_width=0.5)
    for basin in basins:
        if basin in basin_labels:
            x, y, text = basin_labels[basin]
            ax.text(x, y, text, ha='center', va='center', fontsize=6, transform=ccrs.PlateCarree())

    # (B) elevation bins (top edge = basin max, for the legend)
    ax = axes[1]
    edges = list(elev_bins[:4]) + [dem.max().item()]
    elev_bin = xr.apply_ufunc(np.digitize, dem, edges)
    elev_bin = elev_bin.where((elev_bin > 1) & (elev_bin < len(edges)))
    bin_colors = [ELEV_COLORS[b] for b in bins]
    elev_bin.plot(ax=ax, cmap=ListedColormap(bin_colors), norm=BoundaryNorm(np.arange(1.5, 5, 1), ncolors=3),
                  add_colorbar=False)
    ax.set_title('')

    handles = [mpatches.Patch(color=c, label=f'{edges[i + 1]:.0f}m-{edges[i + 2]:.0f}m')
               for i, c in enumerate(bin_colors)][::-1]
    leg = ax.legend(handles=handles, loc='upper right', fontsize=7, borderpad=0.2)
    leg.get_frame().set_alpha(0)
    leg.set_title('Elevation Bins')
    leg.get_title().set_fontsize(8)

    for ax in axes:
        gl = format_map(ax, polygons, extent)
    gl.left_labels = False  # right-hand panel

    add_panel_labels(axes, panel_labels, x=0.04, y=0.96, fontsize=13, pad=0.25)
    return fig, axes


def plot_elev_cdfs(basins=Basins, elev_range=(1500, 4315), breaks=tuple(elev_bins[2:4]), figsize=(3, 6),
                   panel_labels=('C', 'D', 'E', 'F')):
    """Cumulative distribution of basin area above elev_range[0], shaded by elevation bin."""
    lo, hi = elev_range
    edges = [lo, *breaks, hi]
    fig, axes = plt.subplots(len(basins), 1, constrained_layout=True, figsize=figsize, sharex=True)

    for ax, basin in zip(axes, basins):
        elev = xr.open_dataset(os.path.join(data_folder, 'Data/DEM', f'{basin}_FixedElevBands.nc4'))['Elevation']
        elev = np.append(elev.values.ravel(), hi)  # so every curve reaches the same maximum
        elev = np.sort(elev[(elev > lo) & ~np.isnan(elev)])
        cdf = np.arange(1, elev.size + 1) / elev.size * 100

        ax.plot(elev, cdf, color='black')
        ax.fill_between(elev, cdf, 0, facecolor='none', hatch='\\\\\\', edgecolor='black',
                        linewidth=0.0, alpha=0.9, zorder=0)
        ax.set_title(basin_names[basin])
        ax.grid(True, alpha=0.3)
        ax.set_xlim([lo, hi])
        ax.set_ylim([0, 100])
        for b in breaks:
            ax.axvline(b, color='black', linestyle='--', linewidth=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:.0f}%'))
        for (e0, e1), b in zip(zip(edges[:-1], edges[1:]), bins):
            ax.axvspan(e0, e1, color=ELEV_COLORS[b], alpha=0.3, zorder=0)

    axes[-1].set_xlabel('Elevation [m]')
    add_panel_labels(axes, panel_labels, x=0.95, y=0.87, fontsize=15, ha='right')
    return fig, axes


def _keep_eastern_run(row, run_len=2, n_west=4):
    """Map-figure mask rule for one latitude row (west -> east): walking west from the east edge,
    find the first run of `run_len` consecutive True cells (or a lone True at the west edge),
    and keep from its eastern end to (run length + n_west - 1) cells west of it."""
    keep = np.zeros(row.size, dtype=bool)
    start, count = None, 0
    for i in range(row.size - 1, -1, -1):
        if row[i]:
            start = i if start is None else start
            count += 1
            if count == run_len:
                break
        elif count:
            start, count = None, 0
    if count:
        keep[max(0, start - count - n_west + 1):start + 1] = True
    return keep


IVT_BOX = dict(xy=(-122.3, 38.2), width=2.25, height=0.7, angle=15)


def plot_upstream_ivt_map(ax, date='1997-01-01', era5_path=ERA5_PATH, dem_path=DEM_PATH,
                          basin_name_list=basin_names_list, low_elev=1000, exclude_ne=(-120, 38.5),
                          max_lat=41.5, run_len=2, n_west=4, dem_vlim=(-5500, 4000), box=IVT_BOX,
                          extent=MAP_EXTENT):
    """ERA5 IVT on `date` over the low-elevation cells upstream of the basins, on a grey DEM."""
    era5 = xr.open_dataset(era5_path).sortby('latitude').sortby('longitude')
    polygons = shapes[shapes['name'].isin(basin_name_list)]

    dem = xr.open_dataset(dem_path)['Elevation'].rename({'Latitude': 'latitude', 'Longitude': 'longitude'})
    dem_clipped = (dem.rio.write_crs('EPSG:4326')
                      .rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
                      .rio.clip([unary_union(polygons.geometry)], crs='EPSG:4326'))

    # Low-elevation ERA5 cells (note: cells outside the DEM come out True, NaN -> True)
    mask = (dem < low_elev).astype(int).interp(latitude=era5.latitude, longitude=era5.longitude,
                                                method='nearest').astype(bool)
    lon_ne, lat_ne = exclude_ne
    mask = mask.where(~((mask.longitude > lon_ne) & (mask.latitude > lat_ne)), False)
    mask = xr.apply_ufunc(_keep_eastern_run, mask, input_core_dims=[['longitude']],
                          output_core_dims=[['longitude']], vectorize=True,
                          kwargs={'run_len': run_len, 'n_west': n_west})

    add_features(ax)
    dem_clipped.plot(ax=ax, cmap='gist_gray', vmin=dem_vlim[0], vmax=dem_vlim[1], add_colorbar=False)
    ivt = era5['ivt'].sel(valid_time=date, latitude=slice(None, max_lat)).where(mask)
    p = ivt.plot(ax=ax, alpha=0.6, add_colorbar=False)
    add_inset_colorbar(ax, p, r'ERA5 IVT [kg m$^{-1}$ s$^{-1}$]', anchor=(-0.05, -0.07, 1, 1),
                       title_y=1.07, tick_width=0.6)

    ax.set_title('ERA5 Integrated Vapor Transport, 900-600 hPa')
    ax.add_patch(Rectangle(box['xy'], box['width'], box['height'], angle=box['angle'],
                           linewidth=3, edgecolor='darkred', facecolor='none'))
    format_map(ax, polygons, extent)  # last, so outlines and extent sit on top of the plots
    return ax



def plot_outside_study_area(ax, polygons, min_elev=elev_bins[1], color=LOW_ELEV_COLOR, alpha=LOW_ELEV_ALPHA,
                            dem_path=DEM_PATH, extent=MAP_EXTENT, legend=True, zorder=2):
    """Shade land in the map extent that is outside the study area — outside `polygons`, or inside them
    but below min_elev (or without DEM data) — in one semi-transparent colour, as a single image at the
    DEM's resolution. Ocean and lakes are left unshaded (same coastline/lakes as add_features draws)."""
    dem = xr.open_dataset(dem_path)['Elevation']
    res = abs(float(dem['Latitude'].diff('Latitude').median()))
    lon = np.arange(extent[0] + res / 2, extent[1], res)
    lat = np.arange(extent[3] - res / 2, extent[2], -res)  # north -> south
    elev = dem.interp(Latitude=lat, Longitude=lon, method='nearest').values  # NaN outside the DEM

    shape = (lat.size, lon.size)
    transform = from_origin(extent[0], extent[3], res, res)
    bbox = box(extent[0], extent[2], extent[1], extent[3])

    def covered(geoms):
        """True for grid cells whose centre falls inside any of `geoms`."""
        geoms = [g.intersection(bbox) for g in geoms if g.intersects(bbox)]
        return ~geometry_mask(geoms, out_shape=shape, transform=transform) if geoms else np.zeros(shape, bool)

    basins_geoms = polygons.to_crs('EPSG:4326').geometry if polygons.crs else polygons.geometry
    in_study = covered(basins_geoms) & (elev >= min_elev)
    on_land = covered(cfeature.LAND.geometries()) & ~covered(cfeature.LAKES.geometries())

    shade = xr.DataArray(np.where(on_land & ~in_study, 1.0, np.nan), dims=('Latitude', 'Longitude'),
                         coords={'Latitude': lat, 'Longitude': lon})
    shade.plot.imshow(ax=ax, cmap=ListedColormap([color]), alpha=alpha, interpolation='nearest',
                      add_colorbar=False, zorder=zorder)
    if legend:
        leg = ax.legend(handles=[mpatches.Patch(color=color, alpha=alpha, label='Outside study area')],
                        loc='lower left', fontsize=7, frameon=False)
        leg.set_zorder(20)
 
 
def plot_slope_map(ax, slope_das, vlim, title, subtitle, subtitle_y=-0.08, subtitle_size=7,
                   basins=Basins, extent=MAP_EXTENT, shade_low_elev=False, low_elev_color=LOW_ELEV_COLOR,
                   low_elev_alpha=LOW_ELEV_ALPHA):
    """Map of per-pixel ∆var/∆T slopes, diverging colours centred on 0.
    With shade_low_elev, basin area below 1500 m (outside the study area) is shaded in low_elev_color."""
    add_features(ax, facecolor='whitesmoke')
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    for basin in basins:
        p = slope_das[basin].plot(cmap='RdBu', norm=norm, add_colorbar=False, ax=ax)
    polygons = basin_polygons()
    if shade_low_elev:
        plot_outside_study_area(ax, basin_polygons(basins), color=low_elev_color, alpha=low_elev_alpha,
                                extent=extent)
    add_inset_colorbar(ax, p, title, anchor=(-0.038, -0.16, 1, 1), title_y=1.15, tick_width=0.6,
                       linespacing=1.4, subtitle=subtitle, subtitle_y=subtitle_y, subtitle_size=subtitle_size)
    format_map(ax, polygons, extent)
    ax.set_title('')  # xarray adds a title from leftover scalar coordinates
    return ax


def plot_prec_map(ax, prec_slope_das, vlim=200, **kwargs):
    return plot_slope_map(ax, prec_slope_das, vlim, 'Change in Storm-\nDriven Precipitation\nper Degree Warming',
                          r'$\Delta P_\text{s}$/$\Delta T_\text{s}$ [mm/˚C]', subtitle_y=-0.15, subtitle_size=7.5, **kwargs)


def plot_swe_map(ax, slope_das, vlim=120, **kwargs):
    return plot_slope_map(ax, slope_das, vlim, 'Change in Peak\nSWE per Degree\nWarming',
                          r'$\Delta SWE_\text{p}$/$\Delta T_\text{s}$ [mm/˚C]', subtitle_y=-0.08, subtitle_size=7,
                          **kwargs)


def plot_cdec_map(ax, df, vlim=120, legend_pos=None):
    """Station slopes on a map (SNOTEL diamonds, CDEC circles), with a small marker legend."""
    legend_pos = legend_pos or {'SNOTEL': (-118.52, 34.62), 'CDEC': (-118.42, 34.32)}
    styles = {'SNOTEL': dict(marker='D', s=12, zorder=10), 'CDEC': dict(marker='o', s=14, zorder=9)}
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    for network, style in styles.items():
        sub = df[df['Type'] == network]
        ax.scatter(sub['Longitude'], sub['Latitude'], c=sub['Slope'], cmap='RdBu', norm=norm,
                   edgecolors='black', linewidths=0.3, transform=ccrs.PlateCarree(), alpha=0.8, **style)

    for network, (x, y) in legend_pos.items():
        ax.scatter([x], [y], c='white', s=styles[network]['s'], edgecolors='black', linewidths=0.3,
                   marker=styles[network]['marker'])
        ax.text(x + 0.12, y - 0.02, network, ha='left', va='center', color='black', fontsize=7)


# =============================================================================
# Cross-section
# =============================================================================
def _snap_pchip(z_col, y_col, target):
    """One vertical column -> values on `target` heights.
    Drops NaNs, snaps each level's height to the nearest target height, averages levels that
    land on the same height, then PCHIP-interpolates between them (no extrapolation)."""
    z, y = np.asarray(z_col, dtype=float), np.asarray(y_col, dtype=float)
    ok = np.isfinite(z) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(target, np.nan, dtype=float)
    z, y = z[ok], y[ok]

    snapped = target[np.abs(target[:, None] - z[None, :]).argmin(axis=0)]
    anchors, inv = np.unique(snapped, return_inverse=True)
    y_anchor = np.bincount(inv, weights=y) / np.bincount(inv)

    if anchors.size < 2:  # one anchor: can't interpolate
        out = np.full_like(target, np.nan, dtype=float)
        out[np.searchsorted(target, anchors[0])] = y_anchor[0]
        return out
    return PchipInterpolator(anchors, y_anchor, extrapolate=False)(target)


def plot_storm_cross_section(ax, storm_ds, group=13, lat=38.8, dem_lat_offset=0.1,
                             era5_path=ERA5_PATH, dem_path=DEM_PATH, height_target=HEIGHT_TARGET,
                             temp_lim=(-15, 5), ylim=(1000, 4200), xlim=(-122, -119.4),
                             pressure_ticks=range(600, 950, 50), band_colors=None,
                             profile_box_x=(-121.99, -121.0),
                             profile_text='Vertical\ntemperature\nprofile of\nupstream area',
                             bin_label_x=None):
    """West–east section at `lat`: storm-mean ERA5 temperature on height, the terrain profile
    coloured by elevation bin, and a pressure-level axis."""
    band_colors = band_colors or ['black', *ELEV_COLORS.values()]
    bin_label_x = bin_label_x or {1500: -120.86, 1800: -120.71, 2500: -120.44}

    storm = storm_ds.sel(Group=group)
    start = storm['Start_Date'].values
    end = start + np.timedelta64(int(storm['Duration'].max()) - 1, 'D')
    print(pd.Timestamp(start).strftime('%m-%d-%Y'))

    # Storm-mean temperature interpolated from pressure levels onto heights
    era5 = xr.open_dataset(era5_path).sortby('latitude').sel(valid_time=slice(start, end)).mean('valid_time')
    temp_c = xr.apply_ufunc(
        _snap_pchip, era5['z'], era5['t'] - K_TO_C,
        input_core_dims=[['pressure_level'], ['pressure_level']], output_core_dims=[['height']],
        vectorize=True, kwargs={'target': height_target}, output_dtypes=[np.float64],
    ).assign_coords(height=('height', height_target))
    temp_c['height'].attrs.update({'units': 'm', 'long_name': 'geometric height'})

    temp_plot = temp_c.sel(latitude=lat, method='nearest').plot(
        y='height', yincrease=True, robust=True, ax=ax, vmin=temp_lim[0], vmax=temp_lim[1],
        cmap='coolwarm', add_colorbar=False, zorder=0)

    # Terrain profile, filled with elevation-bin colours
    dem = xr.open_dataset(dem_path)['Elevation'].rename({'Latitude': 'latitude', 'Longitude': 'longitude'})
    profile = dem.sel(latitude=lat - dem_lat_offset, method='nearest')
    profile.plot(yincrease=True, ax=ax, color='black')

    x, y = profile['longitude'].values, profile.values
    dx = np.diff(x)
    x_edges = np.concatenate([[x[0] - dx[0] / 2], (x[:-1] + x[1:]) / 2, [x[-1] + dx[-1] / 2]])
    y_center = np.interp(0.5 * (x_edges[:-1] + x_edges[1:]), x, y)

    band_edges = np.array(elev_bins)
    bands = np.tile(np.arange(len(band_edges) - 1, dtype=float)[:, None], (1, y_center.size))
    bands = np.where(y_center[None, :] < band_edges[:-1][:, None], np.nan, bands)  # bands wholly above ground
    X, Y = np.meshgrid(x_edges, band_edges)
    pc = ax.pcolormesh(X, Y, bands, shading='flat', linewidth=0, alpha=1, vmin=0, vmax=len(band_edges) - 2,
                       cmap=LinearSegmentedColormap.from_list('elev_bands', band_colors, N=256))

    mountain = Polygon(np.column_stack([np.concatenate([x, x[::-1]]),
                                        np.concatenate([y, np.full_like(x, band_edges[0])])]),
                       closed=True, facecolor='none', transform=ax.transData)
    ax.add_patch(mountain)
    pc.set_clip_path(mountain)  # clip the bands to the terrain

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel('')
    ax.get_yaxis().set_visible(False)
    ax.set_title('ERA5 Atmospheric Temperature, 900-600 hPa')
    add_inset_colorbar(ax, temp_plot, 'ERA5 Temperature [˚C]', anchor=(-0.05, -0.07, 1, 1), title_y=1.1)

    # Pressure-level axis (left) and elevation axis (right)
    z_axis = (era5['z'].sel(latitude=lat - dem_lat_offset, method='nearest').mean('longitude')
                        .reindex(pressure_level=pressure_ticks)
                        .interpolate_na(dim='pressure_level', method='linear'))
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(z_axis.values)
    ax2.set_yticklabels(z_axis.pressure_level.values)
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('left')
    ax2.spines['left'].set_position(('axes', 0))
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(right=False, labelsize=8)
    ax2.set_ylabel('Pressure Level [hPa]')

    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.tick_params(labelsize=8)
    ax3.set_ylabel('Elevation [m]')

    # Box marking the upstream profile, with arrows
    x1, x2 = profile_box_x
    y1, y2 = ylim[0] + 10, ylim[1] - 10
    mid_x, mid_y, text_height = (x1 + x2) / 2, (y1 + y2) / 2, 700
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor='darkred', facecolor='none'))
    ax.text(mid_x, mid_y, profile_text, ha='center', va='center', fontsize=10)
    arrow = dict(arrowstyle='->', color='black', lw=1)
    ax.annotate('', xy=(mid_x, y1 + 200), xytext=(mid_x, mid_y - text_height / 2), arrowprops=arrow)
    ax.annotate('', xy=(mid_x, y2 - 200), xytext=(mid_x, mid_y + text_height / 2), arrowprops=arrow)

    for edge in elev_bins[1:4]:
        ax.text(bin_label_x[edge], edge - 20, f'{edge}m', transform=ax.transData, fontsize=10)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.tick_params(labelsize=8)
    ax.tick_params(axis='x', length=0)





def plot_fig2(storm_ds, figsize=(10.5, 5.5), width_ratios=(1, 0.9), wspace=0.25,
              arrow_start=(0.345, 0.6), arrow_end=(0.5, 0.6), panel_labels=('A', 'B'), **cross_section_kwargs):
    """Fig 2: (A) upstream IVT map and (B) storm temperature cross-section, joined by an arrow.
    `storm_ds` and any extra keyword arguments go to plot_storm_cross_section.
    Arrow ends are in figure coordinates. Returns (fig, (ax_map, ax_section))."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=width_ratios, wspace=wspace)
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_section = fig.add_subplot(gs[1])
 
    plot_upstream_ivt_map(ax_map)
    plot_storm_cross_section(ax_section, storm_ds, **cross_section_kwargs)
 
    add_panel_labels([ax_map, ax_section], panel_labels, x=-0.14, y=1.08, fontsize=15)
    fig.patches.append(FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', mutation_scale=40,
                                       color='darkred', lw=3, transform=fig.transFigure, clip_on=False))
    return fig, (ax_map, ax_section)






def load_gridmet_storms(basins=Basins, max_temp=280):
    """All GridMET-period storms (<basin>_storms_long.nc4) in one DataFrame. Keeps storms colder
    than max_temp [K] (warmer ones are odd outliers, probably not winter storms) with no missing values."""
    frames = []
    for basin in basins:
        ds = xr.open_dataset(os.path.join(CLEANED, f'{basin}_storms_long.nc4'))
        frames.append(ds.where(ds['Temperature'] < max_temp).dropna('Group').to_dataframe())
    return pd.concat(frames, ignore_index=True)
 
 
def storm_precip_das(basins=Basins, wy_range=WY_RANGE, chunk_days=100,
                     prec_path=os.path.join(CLEANED, 'GridMET/prec_da.nc4'),
                     labels_path=os.path.join(CLEANED, 'GridMET/storm_labels.nc4')):
    """Yearly storm precipitation [mm] per GridMET cell, clipped to each basin: GridMET precipitation
    summed over the days with a storm at that cell, per water year in wy_range.
    Both files are streamed `chunk_days` days at a time, so memory stays small."""
    days = slice(f'{wy_range[0] - 1}-10-01', f'{wy_range[1]}-09-30')
    prec = xr.open_dataarray(prec_path, chunks={'Day': chunk_days}).sel(Day=days)
    is_storm = (xr.open_dataset(labels_path, chunks={'Day': chunk_days})['StormLabel']
                  .reindex(Day=prec['Day']).notnull())  # label file only has storm days
 
    storm_prec = (prec * is_storm).assign_coords(WY=('Day', np.asarray(calculate_water_year(prec['Day']))))
    yearly = (storm_prec.groupby('WY').sum('Day', min_count=1)  # 0 in storm-free years, NaN off-grid
                        .compute()
                        .rio.write_crs('EPSG:4326')
                        .rio.set_spatial_dims(x_dim='Longitude', y_dim='Latitude'))
 
    das = {}
    for basin in basins:
        polygon = shapes.loc[shapes['name'] == basin_names[basin], 'geometry'].iloc[0]
        das[basin] = yearly.rio.clip([polygon], crs='EPSG:4326').rename('Precipitation')
    return das
 
 
 


def plot_fig3(storms, prec_slope_das, prec_vlim=200, shade_low_elev=True, low_elev_color=LOW_ELEV_COLOR,
              low_elev_alpha=LOW_ELEV_ALPHA, figsize=(10, 5), panel_labels=('A', 'B')):
    """(A) storm IVT vs. storm temperature with exponential fit, (B) map of ∆P/∆T.
    `storms` is a DataFrame with Temperature [K] and IVT (load_gridmet_storms()).
    prec_vlim sets the map's colour limits to ±prec_vlim [mm/˚C]; shade_low_elev shades basin area
    below 1500 m (outside the study area) in low_elev_color at low_elev_alpha.
    Returns (fig, (ax_fit, ax_map), (a, b)) where IVT = a·exp(b·T)."""
    fig = plt.figure(figsize=figsize)
    ax_fit = fig.add_subplot(1, 2, 1)
    ax_map = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

    fit = fit_and_plot_exp_with_ci(ax_fit, storms['Temperature'], storms['IVT'],
                                   'Storm Temperature [˚C]', r'Storm IVT [kg m$^{-1}$ s$^{-1}$]')
    plot_prec_map(ax_map, prec_slope_das, vlim=prec_vlim, shade_low_elev=shade_low_elev,
                  low_elev_color=low_elev_color, low_elev_alpha=low_elev_alpha)
    add_panel_labels([ax_fit, ax_map], panel_labels, x=-0.12, y=1.05, fontsize=14)
    return fig, (ax_fit, ax_map), fit





def load_peak_swe(basins=Basins):
    """Peak SWE [m] per basin, (WY, Latitude, Longitude), from <basin>_peak_swe.nc4."""
    return {b: xr.open_dataarray(os.path.join(CLEANED, f'{b}_peak_swe.nc4')) for b in basins}
 
 
def slopes_by_elevation(slope_das, band_edges=range(1000, 4100, 200), dem_path=DEM_PATH):
    """Pool per-pixel slopes from all basins into elevation bands (keyed by each band's lower edge).
    Pixel elevation comes from the DEM interpolated onto the slope grid."""
    dem = xr.open_dataset(dem_path)['Elevation']
    edges = list(band_edges)
    binned = {lo: [] for lo in edges[:-1]}
    for slope_da in slope_das.values():
        elev = dem.interp_like(slope_da)
        for lo, hi in zip(edges[:-1], edges[1:]):
            vals = slope_da.where((elev > lo) & (elev < hi)).values
            binned[lo] += vals[~np.isnan(vals)].tolist()
    return binned
 
 

def plot_fig4(slope_das, binned_slopes, band_edges=range(1000, 4100, 200), swe_vlim=120,
              shade_low_elev=True, low_elev_color=LOW_ELEV_COLOR, low_elev_alpha=LOW_ELEV_ALPHA,
              figsize=(10, 5), wspace=0.25, panel_labels=('A', 'B')):
    """(A) map of ∆SWE_p/∆T, (B) the same slopes as boxplots by elevation band.
    swe_vlim sets the map's colour limits to ±swe_vlim [mm/˚C]; shade_low_elev shades basin area
    below 1500 m (outside the study area) in low_elev_color at low_elev_alpha.
    Returns (fig, (ax_map, ax_box))."""
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=wspace)
    ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax_box = fig.add_subplot(1, 2, 2)

    plot_swe_map(ax_map, slope_das, vlim=swe_vlim, shade_low_elev=shade_low_elev,
                 low_elev_color=low_elev_color, low_elev_alpha=low_elev_alpha)
    plot_swe_boxplot(ax_box, binned_slopes, list(band_edges))
    add_panel_labels([ax_map, ax_box], panel_labels, x=-0.12, y=1.08, fontsize=14)
    return fig, (ax_map, ax_box)
 
 
 




# =============================================================================
# Trends and fits
# =============================================================================
def trend_slope(x, y, min_n=4, min_positive_fraction=None, min_positive_count=None, require_finite=False):
    """Slope of a linear fit of y on x, or NaN if the series fails any check.
    The positive-value checks count NaN as not positive."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < min_n:
        return np.nan
    if min_positive_fraction is not None and np.mean(y > 0) < min_positive_fraction:
        return np.nan
    if min_positive_count is not None and np.sum(y > 0) < min_positive_count:
        return np.nan
    if require_finite:
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size < min_n or np.std(x) == 0:
            return np.nan
    return np.polyfit(x, y, 1)[0]


def yearly_storm_temperature(storm_ds, wy_range=WY_RANGE):
    """Mean storm Temperature [K] per water year, reindexed to wy_range (inclusive)."""
    return (storm_ds['Temperature'].groupby(storm_ds['WY']).mean()
            .reindex(WY=range(wy_range[0], wy_range[1] + 1)))


def calculate_slope_das(da_dict, var_name, var_label, storm_ds_dict, basins=Basins, wy_range=WY_RANGE,
                        min_years=4, min_positive_fraction=None, min_positive_count=None, scale=1.0):
    """Per-pixel slope of `var_name` against the basin's yearly mean storm temperature.
    Peak SWE: min_positive_fraction=0.5, scale=1000 (m -> mm). Precipitation: min_positive_count=1."""
    slope_das = {}
    for basin in basins:
        ds = da_dict[basin].rename(var_name).to_dataset()
        ds['Temperature'] = yearly_storm_temperature(storm_ds_dict[basin], wy_range).broadcast_like(ds)
        df = ds[['Temperature', var_name]].to_dataframe().reset_index().dropna(subset=['Temperature', var_name])

        slopes = df.groupby(['Latitude', 'Longitude'])[['Temperature', var_name]].apply(
            lambda g: trend_slope(g['Temperature'], g[var_name], min_years,
                                  min_positive_fraction=min_positive_fraction,
                                  min_positive_count=min_positive_count))
        slope_da = slopes.rename('Slope').to_xarray() * scale
        slope_da.name = f'∆{var_label}/∆T [mm/K]'
        slope_da.attrs['long_name'] = slope_da.name
        slope_das[basin] = slope_da
    return slope_das




def bootstrap_breakpoint_ci(x, y, n_boot=1000, seed=None):
    """2.5–97.5% range of the breakpoint over bootstrap refits that keep an up-then-down shape.
    seed=None uses numpy's global random state (as before)."""
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed) if seed is not None else np.random
    samples = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        fit = pwlf.PiecewiseLinFit(x[idx], y[idx])
        try:
            bp = fit.fit(2)
        except Exception:
            continue
        if fit.slopes[0] > 0 and fit.slopes[1] < 0:
            samples.append(bp[1])
    return np.percentile(samples, [2.5, 97.5])


def fit_and_plot_exp_with_ci(ax, x, y, xlabel, ylabel, alpha=0.05, x_offset=K_TO_C,
                             legend_title='Change in IVT per Degree Warming', color='#377eb8', fontsize=8):
    """Fit y = a·exp(b·x) by regressing ln(y) on x (y > 0 only); plot data, fit and (1-alpha) CI band.
    x is shifted by -x_offset first (K -> ˚C by default). Returns (a, b)."""
    x = np.asarray(x) - x_offset
    y = np.asarray(y)
    pos = y > 0
    x_fit, y_fit = x[pos], y[pos]
    if x_fit.size < 3:
        raise ValueError('Need at least 3 positive-y points to fit and get CI.')

    logy = np.log(y_fit)
    b, ln_a = np.polyfit(x_fit, logy, 1)
    a = np.exp(ln_a)

    n, dof = x_fit.size, x_fit.size - 2
    s_err = np.sqrt(np.sum((logy - (ln_a + b * x_fit)) ** 2) / dof)  # residual SE in log space
    x_mean = x_fit.mean()
    sxx = np.sum((x_fit - x_mean) ** 2)
    t_val = t.ppf(1 - alpha / 2, dof)

    x_line = np.linspace(x_fit.min(), x_fit.max(), 400)
    logy_line = ln_a + b * x_line
    half_width = t_val * s_err * np.sqrt(1 / n + (x_line - x_mean) ** 2 / sxx)

    se_b = s_err / np.sqrt(sxx)
    se_ln_a = s_err * np.sqrt(1 / n + x_mean ** 2 / sxx)
    b_ci = (b - t_val * se_b, b + t_val * se_b)
    a_ci = np.exp([ln_a - t_val * se_ln_a, ln_a + t_val * se_ln_a])

    ci = f'{(1 - alpha) * 100:.0f}%'
    print('Exponential model: y = a * exp(b x)')
    print(f'a = {a:.4g}')
    print(f'{ci} CI for a: [{a_ci[0]:.4g}, {a_ci[1]:.4g}]')
    print(f'b = {b:.4g}')
    print(f'{ci} CI for b: [{b_ci[0]:.4g}, {b_ci[1]:.4g}]')

    pct = (np.exp(b) - 1) * 100
    pct_lo, pct_hi = (np.exp(b_ci) - 1) * 100
    ax.scatter(x, y, s=5, color=color, alpha=0.7)
    ax.plot(x_line, np.exp(logy_line), color='black', lw=2, label=fr'{pct:+.2f}% (˚C)$^{{-1}}$')
    ax.fill_between(x_line, np.exp(logy_line - half_width), np.exp(logy_line + half_width), color='black',
                    alpha=0.2, label=fr'{pct_lo:+.2f}% – {pct_hi:+.2f}% (˚C)$^{{-1}}$')

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(direction='in', top=True, right=True, labelsize=fontsize)
    ax.legend(title=legend_title, fontsize=fontsize, title_fontsize=fontsize + 0.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)

    print('p-value:', linregress(x_fit, logy).pvalue)
    return a, b


def create_basin_dfs(storm_ds_dict, basins=Basins, max_temp=280):
    """Per basin: one row per storm with Temperature [K], Start_Date, Month and SWE per elevation bin
    (columns 2, 3, 4). Storms at or above max_temp [K] are dropped."""
    basin_dfs = {}
    for basin in basins:
        ds = storm_ds_dict[basin]
        df = pd.DataFrame({'Temperature': ds['Temperature'].values, 'Start_Date': ds['Start_Date'].values})
        df['Month'] = df['Start_Date'].dt.month
        for b in bins:
            df[b] = ds['SWE'].sel(ElevationBin=b).values
        basin_dfs[basin] = df[df['Temperature'] < max_temp]
    return basin_dfs



def fit_storm_swe(x, y, force_piecewise=False, p_threshold=0.05):
    """The storm-SWE vs. temperature fit used in plot_piecewise_fit: one-breakpoint piecewise fit if
    the Davies test p <= p_threshold and the first slope is positive (or if forced), otherwise linear.
    x in ˚C. Returns (predict, kind, p_value, pwlf_model); predict(t) gives SWE [mm] at temperature t."""
    p_value = piecewise_regression.Fit(list(x), list(y), n_breakpoints=1).get_results()['davies']
    model = pwlf.PiecewiseLinFit(x, y)
    model.fit(2)
    if (p_value > p_threshold or model.slopes[0] < 0) and not force_piecewise:
        fit = linregress(x, y)
        return (lambda t: fit.slope * np.asarray(t, dtype=float) + fit.intercept), 'linear', fit.pvalue, model
    return model.predict, 'piecewise', p_value, model
 
 
def plot_piecewise_fit(df_dict, basins, bin_lims, size=(8, 6), force_piecewise=(('SanJoaquin', 4),),
                       n_boot=1000, seed=None, text_x=(-6, 5), text_y=None, p_text_y=350,
                       panel_labels=('A', 'B', 'C')):
    """Storm SWE vs. storm temperature per basin and elevation bin. A bin gets a piecewise fit
    (breakpoint, bootstrap CI, slopes, p) if the Davies test is significant and the first slope
    is positive, or if (basin, bin) is in force_piecewise; otherwise a linear fit."""
    text_y = text_y or {2: 270, 3: 200, 4: 270}
    lims = [f'{lim}m' for lim in bin_lims]
    bin_text = {b: f'{lims[i]}-{lims[i + 1]}' for i, b in enumerate(bins)}
 
    fig, axes = plt.subplots(len(basins), 1, figsize=size, sharex=True, sharey=True)
    for ax, basin in zip(axes, basins):
        data = df_dict[basin]
        x_pred = np.linspace(data['Temperature'].min() - K_TO_C, data['Temperature'].max() - K_TO_C, 100)
 
        for b in reversed(bins):
            df_bin = data[['Temperature', b]].rename(columns={b: 'SWE'}).dropna()
            x, y = df_bin['Temperature'] - K_TO_C, df_bin['SWE']
            c = ELEV_COLORS[b]
 
            predict, kind, p_value, model = fit_storm_swe(x, y, force_piecewise=(basin, b) in force_piecewise)
            y_pred, y_fit = predict(x_pred), predict(x)
 
            if kind == 'piecewise':
                bp = model.fit_breaks[1]
                ax.axvline(bp, color=c, linestyle='--', alpha=0.6)
                ax.text(text_x[0], text_y[b], f'slope\n+{model.slopes[0]:.2f}mm/˚C',
                        ha='center', va='center', color=c, fontsize=9)
                ax.text(text_x[1], text_y[b], f'slope\n–{-model.slopes[1]:.2f}mm/˚C',
                        ha='center', va='center', color=c, fontsize=9)
                ax.text(bp + 0.2, p_text_y, f'p={p_value:.3f}', ha='left', va='center', color=c, fontsize=9)
                ax.axvspan(*bootstrap_breakpoint_ci(x, y, n_boot, seed), color=c, alpha=0.1)
 
            ax.scatter(x, y, color=c, s=4, alpha=0.5, label='scatter')
            r2 = r2_score(y, y_fit)  # r2 of the line actually drawn
            ax.plot(x_pred, y_pred, color=c, linewidth=2.5 if p_value <= 0.05 else 0.5,
                    label=f'Elevation {bin_text[b]} (r2={r2:.2f})')
 
        ax.set_ylabel('∆SWE [mm]')
        ax.set_ylim(bottom=0)
        ax.set_title('')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
        leg = legend_without_scatter(ax, loc='upper left', title=f'{basin_names[basin]} Basin', fontsize=9.5)
        leg.get_title().set_fontsize(10)
        ax.tick_params(direction='in', top=True, right=True)
 
    axes[-1].set_xlabel('Storm Temperature [˚C]')
    add_panel_labels(axes, panel_labels, x=-0.12, y=1, fontsize=13)
    plt.tight_layout()
    return fig
 


def plot_swe_boxplot(ax, binned_slopes, plot_bins, bin_step=200, top=3800):
    """Horizontal boxplots of slopes per elevation band, shaded by elevation bin."""
    ax.boxplot(binned_slopes.values(), tick_labels=binned_slopes.keys(), orientation='horizontal',
               showfliers=False)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, zorder=0, alpha=0.6)
    ax.set_ylabel('Elevation [m]', fontsize=8.5)
    ax.set_xlabel(r'∆SWE$_\text{p}$/∆T [mm/˚C]', fontsize=8.5)
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)

    for b in bins:
        lo = np.floor(elev_bins[b - 1] / bin_step) * bin_step
        hi = min(np.floor(elev_bins[b] / bin_step) * bin_step, top)
        ax.axhspan(plot_bins.index(lo) + 1.5, plot_bins.index(hi) + 1.5, color=ELEV_COLORS[b], alpha=0.06)
    return ax


# =============================================================================
# Observations
# =============================================================================
def cdec_basin_by_box(lat, lon):
    """Rough CDEC basin assignment by latitude/longitude boxes (4 southern basins only)."""
    if lat > 38.5:
        return 'Lower Sacramento'
    if lat > 37 and lon < -119:
        return 'San Joaquin'
    if lat <= 37 and lon < -118.5:
        return 'Tulare-Buena Vista Lakes'
    return 'Mono-Owens Lakes'


def _add_temperature(peak, basin_temp):
    """Join yearly basin storm temperature onto a (Station_ID, WY) peak-SWE table by HUC6 Name."""
    return (peak.reset_index()
                .merge(basin_temp.reset_index(), on=['WY', 'HUC6 Name'], how='left')
                .set_index(['Station_ID', 'WY']))


def _station_slopes(peak, wy_range, min_years, min_snow_fraction):
    """Slope of peak SWE vs. storm temperature for each station, over wy_range."""
    wy = peak.index.get_level_values('WY')
    peak = peak[(wy >= wy_range[0]) & (wy <= wy_range[1])]
    return (peak.groupby('Station_ID')[['Temperature', 'SWE']]
                .apply(lambda g: trend_slope(g['Temperature'], g['SWE'], min_years,
                                             min_positive_fraction=min_snow_fraction, require_finite=True))
                .rename('Slope').to_frame())


def clean_obs_data(storm_ds_dict, basins=Basins, wy_range=WY_RANGE, max_lat=41, min_years=4,
                   min_snow_fraction=0.5, snotel_basin_remap=SNOTEL_BASIN_REMAP, cdec_basin_fn=cdec_basin_by_box):
    """Per-station slope of observed peak SWE [mm] vs. yearly mean storm temperature of the
    station's basin, for SNOTEL and CDEC. Returns one row per station with a valid slope:
    Slope, Elevation [m], Latitude, Longitude, HUC6 Name, Type."""
    station_cols = ['Elevation', 'Latitude', 'Longitude', 'HUC6 Name']
    basin_temp = pd.concat({basin_names[b]: yearly_storm_temperature(storm_ds_dict[b], wy_range).to_pandas()
                            for b in basins}, names=['HUC6 Name']).rename('Temperature')

    # SNOTEL: basin from the station list's HUC6 name (some moved west); elevation in ft
    s_stations = pd.read_csv(os.path.join(data_folder, 'Data/SNOTEL/NRCS_SNOTEL_Locations_noAlaska.csv'),
                             index_col=0).rename_axis('Station_ID')
    s_peak = xr.open_dataset(os.path.join(CLEANED, 'SNOTEL/snotel_peak_swe.nc4')).to_dataframe()  # mm
    huc = s_peak.index.get_level_values('Station_ID').map(s_stations['HUC6 Name'])
    s_peak['HUC6 Name'] = huc.map(lambda name: snotel_basin_remap.get(name, name))
    s_slopes = _station_slopes(_add_temperature(s_peak, basin_temp), wy_range, min_years, min_snow_fraction)
    s_slopes = s_slopes.join(s_stations[station_cols], on='Station_ID', how='left')
    s_slopes['Elevation'] = s_slopes['Elevation'] * 0.3048  # ft -> m

    # CDEC: basin from lat/lon; elevation already in m
    c_stations = (xr.open_dataset(os.path.join(CLEANED, 'CDEC/cdec_ds.nc4'))[['Elevation', 'Latitude', 'Longitude']]
                    .to_dataframe().rename_axis('Station_ID'))
    c_stations['HUC6 Name'] = [cdec_basin_fn(la, lo) for la, lo in zip(c_stations['Latitude'], c_stations['Longitude'])]
    c_peak = xr.open_dataset(os.path.join(CLEANED, 'CDEC/cdec_peak_swe.nc4')).to_dataframe()
    c_peak['HUC6 Name'] = c_peak.index.get_level_values('Station_ID').map(c_stations['HUC6 Name'])
    c_slopes = _station_slopes(_add_temperature(c_peak, basin_temp), wy_range, min_years, min_snow_fraction)
    c_slopes = c_slopes.join(c_stations[station_cols], on='Station_ID', how='left')

    df = pd.concat([s_slopes.assign(Type='SNOTEL'), c_slopes.assign(Type='CDEC')])
    check_elevation_m(df['Elevation'], 'clean_obs_data')
    df = df[df['Latitude'] < max_lat]
    return df[df['Slope'].notna()]


def plot_cdec_elev_boxplot(df, ax, elev_range=(1500, 3500), step=200,
                           class_spans=((0.5, 2.5), (2.5, 5.5), (5.5, 9.5))):
    """Station slopes per elevation band (boxplots) with station counts alongside.
    class_spans: boxplot positions covered by each elevation bin's shading (low, mid, high)."""
    edges = list(range(elev_range[0], elev_range[1], step))
    binned = {lo: df.loc[(df['Elevation'] > lo) & (df['Elevation'] < hi), 'Slope'].values
              for lo, hi in zip(edges[:-1], edges[1:])}

    ax.boxplot(binned.values(), tick_labels=binned.keys(), orientation='horizontal', showfliers=False)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, zorder=0, alpha=0.6)
    ax.set_ylabel('Elevation [m]', fontsize=9)
    ax.set_xlabel(r'Observed $\Delta SWE_\text{p}$/∆$T_\text{s}$ [mm/˚C]', fontsize=9)
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    ax.margins(y=0)
    for (lo, hi), b in zip(class_spans, bins):
        ax.axhspan(lo, hi, color=ELEV_COLORS[b], alpha=0.06)

    ax2 = make_axes_locatable(ax).append_axes('right', size='30%', pad=0.1)
    ax2.barh(range(1, len(binned) + 1), [len(v) for v in binned.values()], alpha=0.3, color='gray')
    ax2.set_yticks([])
    ax2.set_ylim(ax.get_ylim())
    ax2.tick_params(direction='in', top=True)
    ax2.set_xlabel('Number of Stations', fontsize=9)


def plot_piecewise_obs(ax, snotel_path=os.path.join(CLEANED, 'SNOTEL/snotel_storm_ds.nc4'),
                       cdec_path=os.path.join(CLEANED, 'CDEC/cdec_storm_ds.nc4'), linear_bins=(2,),
                       n_boot=1000, seed=None, text_x=(-4, 5), text_y=None, p_text_y=550):
    """Observed storm SWE gain vs. ERA5 storm temperature for SNOTEL + CDEC stations, by elevation bin.
    Bins in linear_bins get a linear fit; the others a piecewise fit with breakpoint CI."""
    text_y = text_y or {4: 450, 3: 350}
    obs = xr.concat([xr.open_dataset(snotel_path), xr.open_dataset(cdec_path)], dim='Storm_Label')
    obs = obs.assign_coords(Storm_Label=np.arange(obs.sizes['Storm_Label']))
    check_elevation_m(obs['Elevation'], 'station storm files')  # SNOTEL must be converted in the obs notebook

    for b in reversed(bins):  # high, mid, low
        lo = elev_bins[b - 1]
        hi = elev_bins[b] if b < max(bins) else np.inf
        ds = obs.sel(Storm_Label=(obs['Elevation'] >= lo) & (obs['Elevation'] < hi))
        x, y = ds['Temperature'].values - K_TO_C, ds['Delta_SWE'].values
        c = ELEV_COLORS[b]
        ax.scatter(x, y, color=c, s=4, alpha=0.5, label='scatter')

        ok = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[ok], y[ok]
        model = pwlf.PiecewiseLinFit(x, y)
        model.fit(2)
        p_value = piecewise_regression.Fit(list(x), list(y), n_breakpoints=1).get_results()['davies']
        x_pred = np.linspace(x.min(), x.max(), 100)
        y_pred, y_fit = model.predict(x_pred), model.predict(x)

        if b in linear_bins:
            fit = linregress(x, y)
            p_value = fit.pvalue
            y_pred = fit.slope * x_pred + fit.intercept
            y_fit = fit.slope * x + fit.intercept
        else:
            bp = model.fit_breaks[1]
            ax.axvline(bp, color=c, linestyle='--', alpha=0.6)
            ax.axvspan(*bootstrap_breakpoint_ci(x, y, n_boot, seed), color=c, alpha=0.1)
            ax.text(text_x[0], text_y[b], f'slope\n+{model.slopes[0]:.2f}mm/˚C',
                    ha='center', va='center', color=c, fontsize=8)
            ax.text(text_x[1], text_y[b], f'slope\n–{-model.slopes[1]:.2f}mm/˚C',
                    ha='center', va='center', color=c, fontsize=8)
            ax.text(bp + 0.2, p_text_y, f'p={p_value:.3f}', ha='left', va='center', color=c, fontsize=9)

        r2 = r2_score(y, y_fit)  # r2 of the line actually drawn
        ax.plot(x_pred, y_pred, linewidth=2, color=c,
                label=f'Elevation {elev_bins[b - 1]}m-{elev_bins[b]}m (r2={r2:.2f})')

    ax.set_xlabel('Storm Temperature [˚C]')
    ax.set_ylabel('Observed ∆SWE [mm]')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_ylim(bottom=0)
    leg = legend_without_scatter(ax, loc='upper left', title='Observed Storm SWE vs. Temperature')
    leg.get_title().set_fontsize(11)
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_title('')
    ax.figure.tight_layout()






def plot_figS1(slope_das, station_df, swe_vlim=120, figsize=(10, 5), dpi=400):
    """(Left) map of reanalysis ∆SWE_p/∆T with SNOTEL/CDEC station slopes on top (one shared colour
    scale, ±swe_vlim [mm/˚C]); (right) station slopes by elevation band.
    `station_df` comes from clean_obs_data(). Returns (fig, (ax_map, ax_box))."""
    fig = plt.figure(figsize=figsize, dpi=dpi, layout='constrained')
    ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax_box = fig.add_subplot(1, 2, 2)
 
    plot_swe_map(ax_map, slope_das, vlim=swe_vlim)
    plot_cdec_map(ax_map, station_df, vlim=swe_vlim)
    plot_cdec_elev_boxplot(station_df, ax_box)
    return fig, (ax_map, ax_box)




# =============================================================================
# Summary table
# =============================================================================
def _basin_mean_precip(prec, grid, polygon, high):
    """Mean annual precipitation [mm/yr] over the basin's GridMET cells where `high` is True.
    `prec` is read lazily in chunks of days, so only a few chunks are in memory at a time;
    `grid` is one day of it (in memory, with a CRS) used to find the cells inside the basin."""
    inside = grid.rio.clip([polygon], crs='EPSG:4326', drop=False).notnull()
    daily = prec.where(inside & high, drop=True).mean(['Latitude', 'Longitude']).compute()
    wy = np.asarray(calculate_water_year(daily['Day']))
    return daily.assign_coords(WY=('Day', wy)).groupby('WY').sum().mean().item()
 
 
def _basin_mean_peak_swe(basin, wy_range, min_elev):
    """Mean peak SWE [mm], averaged over the basin's SWE pixels above min_elev and over wy_range."""
    peak = xr.open_dataarray(os.path.join(CLEANED, f'{basin}_peak_swe.nc4')).sel(WY=slice(*wy_range))
    elev = xr.open_dataset(os.path.join(data_folder, 'Data/DEM', f'{basin}_FixedElevBands.nc4'))['Elevation']
    elev = elev.interp_like(peak, method='nearest')
    return peak.where(elev > min_elev).mean(['Latitude', 'Longitude']).mean('WY').item() * 1000  # m -> mm
 
 
def basin_summary(storm_ds_dict, basins=Basins, wy_range=WY_RANGE, include_climate=True,
                  area_path=os.path.join(data_folder, 'Data/DEM/elev_bin_areas.csv'),
                  prec_path=os.path.join(CLEANED, 'GridMET/prec_da.nc4'), chunk_days=100, decimals=1):
    """One row per basin plus an 'All basins' row: area of each elevation bin and in total [km²],
    max elevation, mean elevation of the binned area, mean storm temperature [°C] and storms per year.
    With include_climate, also mean annual precipitation and mean peak SWE over the binned area
    (above 1500 m) for wy_range. Precipitation is streamed from disk `chunk_days` days at a time
    (lower it if memory is tight); this reads the GridMET file once per basin and takes a few minutes.
    All-basin means are pooled over all pixels / storms (area-weighted for precipitation and SWE)."""
    n_years = wy_range[1] - wy_range[0] + 1
    low = elev_bins[1]
    bin_names = {b: f'{elev_bins[b - 1]}–{elev_bins[b]} m' for b in bins}
    bin_names[max(bins)] = f'>{elev_bins[max(bins) - 1]} m'
 
    areas = pd.read_csv(area_path, index_col=0).loc[list(bins), list(basins)] / 1e6  # m² -> km²
 
    if include_climate:
        prec = (xr.open_dataarray(prec_path, chunks={'Day': chunk_days})
                  .sel(Day=slice(f'{wy_range[0] - 1}-10-01', f'{wy_range[1]}-09-30')))
        grid = (prec.isel(Day=0, drop=True).load()
                    .rio.write_crs('EPSG:4326')
                    .rio.set_spatial_dims(x_dim='Longitude', y_dim='Latitude'))
        high = xr.open_dataset(DEM_PATH)['Elevation'].interp_like(grid) > low
 
    rows, all_elev, all_temps = {}, [], []
    for basin in basins:
        elev = xr.open_dataset(os.path.join(data_folder, 'Data/DEM', f'{basin}_FixedElevBands.nc4'))['Elevation']
        elev = elev.values.ravel()
        elev_binned = elev[elev > low]
        storms = storm_ds_dict[basin]
        in_range = (storms['WY'] >= wy_range[0]) & (storms['WY'] <= wy_range[1])
        temps = storms['Temperature'].where(in_range).values  # only storms in wy_range
        temps = temps[~np.isnan(temps)]
        all_elev.append(elev_binned)
        all_temps.append(temps)
 
        row = {
            **{f'Area {name} [km²]': areas.loc[b, basin] for b, name in bin_names.items()},
            f'Area >{low} m [km²]': areas[basin].sum(),
            'Max elevation [m]': np.nanmax(elev),
            f'Mean elevation >{low} m [m]': elev_binned.mean(),
        }
        if include_climate:
            polygon = shapes.loc[shapes['name'] == basin_names[basin], 'geometry'].iloc[0]
            row['Mean precipitation [mm/yr]'] = _basin_mean_precip(prec, grid, polygon, high)
            row['Mean peak SWE [mm]'] = _basin_mean_peak_swe(basin, wy_range, low)
        row['Mean storm temperature [°C]'] = temps.mean() - K_TO_C
        row['Storms per year'] = temps.size / n_years
        rows[basin_names[basin]] = row
 
    summary = pd.DataFrame(rows).T
    total = {
        **summary.filter(like='Area').sum(),
        'Max elevation [m]': summary['Max elevation [m]'].max(),
        f'Mean elevation >{low} m [m]': np.concatenate(all_elev).mean(),
        'Mean storm temperature [°C]': np.concatenate(all_temps).mean() - K_TO_C,
        'Storms per year': summary['Storms per year'].sum(),
    }
    if include_climate:
        weights = summary[f'Area >{low} m [km²]']
        for col in ['Mean precipitation [mm/yr]', 'Mean peak SWE [mm]']:
            total[col] = np.average(summary[col], weights=weights)
    summary.loc['All basins'] = total
    return summary if decimals is None else summary.round(decimals)
 
 
def basin_summary_latex(storm_ds_dict=None, basins=Basins, path=None, storm_threshold=200, summary=None,
                        **summary_kwargs):
    """LaTeX tabular of basin_summary() for `basins` (one column each). Returns the LaTeX string,
    and also writes it to `path` if given. Pass an unrounded `summary` (basin_summary(..., decimals=None))
    to reuse one you already computed; otherwise it is computed here (extra kwargs go to basin_summary)."""
    low = elev_bins[1]
    if summary is None:
        summary = basin_summary(storm_ds_dict, basins=basins, decimals=None, **summary_kwargs)
    s = summary.loc[[basin_names[b] for b in basins]]
    total_area = s[f'Area >{low} m [km²]']
    top_elev = s['Max elevation [m]'].max()
 
    lines = [r'\begin{tabular}{l' + 'c' * len(s) + '}',
             ' & ' + ' & '.join(s.index) + r'  \\',
             r'\hline']
 
    def add_row(label, cells):
        lines.append(f'{label} \n& ' + ' & '.join(cells) + '\n\\\\')
 
    def num(values, decimals=0):
        return [f'${v:.{decimals}f}$' for v in values]
 
    add_row(r'Area [km$^2$]', num(total_area))
    for b in bins:
        hi = elev_bins[b] if b < max(bins) else top_elev
        col = [c for c in s.columns if c.startswith('Area ') and c != f'Area >{low} m [km²]'][b - min(bins)]
        cells = [f'${a:.0f} \\ ({a / t * 100:.0f}\\%)$' for a, t in zip(s[col], total_area)]
        add_row(f'Area, Elev. {elev_bins[b - 1]}--{hi:.0f}~m [km$^2$]', cells)
    add_row('Mean Elevation [m]', num(s[f'Mean elevation >{low} m [m]']))
    add_row('Max. Elevation [m]', num(s['Max elevation [m]']))
    if 'Mean precipitation [mm/yr]' in s:
        add_row('Mean Precipitation [mm/yr]', num(s['Mean precipitation [mm/yr]']))
        add_row(r'Mean Peak SWE (SWE$_\text{p}$) [mm/yr]', num(s['Mean peak SWE [mm]']))
    add_row(rf'Mean $\geq${storm_threshold} mm Storm Temperature ($T_\text{{storm}}$) [$^\circ$C]',
            num(s['Mean storm temperature [°C]'], 2))
    add_row(rf'Mean $\geq${storm_threshold} mm Storm Count [storms/yr]', num(s['Storms per year'], 2))
    lines += [r'\hline', r'\end{tabular}']
 
    latex = '\n'.join(lines)
    if path:
        with open(path, 'w') as f:
            f.write(latex + '\n')
    return latex



# =============================================================================
# Time series 
# =============================================================================
def load_time_series_data(yearly_path=os.path.join(CLEANED, 'yearly_data.nc4'),
                          area_path=os.path.join(data_folder, 'Data/DEM/elev_bin_areas.csv'), basins=Basins):
    """
    Annual series for the time-series figure, as 1-D DataArrays indexed by WY (NaN years dropped):
      'winter_temp' – area-weighted winter temperature across basins and elevation bins [˚C]
      'storm_count' – total storm count, summed over basins
      'max_ivt'     – maximum IVT across basins [kg m-1 s-1]
    """
    yearly = xr.open_dataset(yearly_path).sel(Basin=basins)

    # Area of each elevation bin in each basin, used to weight winter temperature
    areas = pd.read_csv(area_path, index_col='ElevationBin')
    area = xr.DataArray(areas.values, dims=['ElevationBin', 'Basin'],
                        coords={'ElevationBin': areas.index, 'Basin': areas.columns})

    series = {
        'winter_temp': yearly['Winter_Temp'].weighted(area).mean(['Basin', 'ElevationBin']) - K_TO_C,
        'storm_count': yearly['Count'].sum('Basin'),
        'max_ivt': yearly['Max_IVT'].max('Basin'),
    }
    return {k: da.dropna('WY') for k, da in series.items()}


def plot_time_series(data, xlim=(1979, 2026), panel_labels=('A', 'B', 'C')):
    """Three stacked panels from load_time_series_data(), each an annual series with its linear
    trend, slope and p-value. Returns (fig, axes)."""
    ivt_unit = r'kg m$^{-1}$ s$^{-1}$'
    panels = [  # (key in data, y-axis label, unit in legend)
        ('winter_temp', 'Winter Temperature [˚C]', '˚C'),
        ('storm_count', 'Storm Count', ' storms'),
        ('max_ivt', f'Max IVT [{ivt_unit}]', f' {ivt_unit}'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'hspace': 0.05})
    for ax, (key, name, unit) in zip(axes, panels):
        x, y = data[key].WY.values, data[key].values
        fit = linregress(x, y)
        ax.plot(x, y, color='black')
        ax.plot(x, fit.slope * x + fit.intercept, color='black',
                label=f'{fit.slope:.2f}{unit}/yr, p= {fit.pvalue:.2f}')
        ax.set_ylabel(name)
        ax.legend(loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_xlim(*xlim)
        ax.tick_params(direction='in', top=True, right=True)

    add_panel_labels(axes, panel_labels, x=-0.09, y=1, fontsize=14)
    return fig, axes











def storm_swe_warming(storm_ds_dict, basin, warmings=(1.5, 3.0), wy_range=WY_RANGE,
                      force_piecewise=(('SanJoaquin', 4),), max_temp=280,
                      area_path=os.path.join(data_folder, 'Data/DEM/elev_bin_areas.csv')):
    """Back-of-envelope change in storm SWE if the average storm were w ˚C warmer.
    Per elevation bin: fitted storm SWE (as in Fig 5) at mean storm T + w minus at mean storm T,
    times storms per year, spread over the bin's whole area. Prints one line per warming."""
    n_years = wy_range[1] - wy_range[0] + 1
    df = create_basin_dfs(storm_ds_dict, [basin], max_temp)[basin]
    area = pd.read_csv(area_path, index_col=0)[basin]  # m²
    groups = {f'low/mid ({elev_bins[1]}–{elev_bins[3]} m)': (2, 3), f'high (>{elev_bins[3]} m)': (4,)}
    t_mean = df['Temperature'].mean() - K_TO_C
 
    change = {w: {} for w in warmings}  # m³/yr per bin
    for b in bins:
        d = df[['Temperature', b]].dropna()
        predict, *_ = fit_storm_swe(d['Temperature'] - K_TO_C, d[b], force_piecewise=(basin, b) in force_piecewise)
        for w in warmings:
            dswe = float(predict([t_mean + w])[0] - predict([t_mean])[0])  # mm per storm
            change[w][b] = dswe / 1000 * area[b] * len(d) / n_years
 
    print(f'{basin_names[basin]}: mean storm temperature {t_mean:.1f} ˚C')
    for w in warmings:
        parts = []
        for name, gbins in groups.items():
            vol = sum(change[w][b] for b in gbins)
            depth = vol / sum(area[b] for b in gbins) * 1000
            parts.append(f'{name} {depth:+.0f} mm/yr ({vol / 1e9:+.3f} km³/yr)')
        print(f'  +{w:g} ˚C: ' + '; '.join(parts))
 