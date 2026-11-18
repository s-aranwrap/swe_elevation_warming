import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

M_PER_ARCSEC = 1852 / 60  # 1 arcmin of latitude = 1 nautical mile


def calculate_water_year(date_array):
    year = date_array.dt.year
    month = date_array.dt.month
    water_year = xr.where(month >= 10, year +1, year)
    water_year = water_year.fillna(0) # NaT returns 0
    return water_year.astype("int16")


def calculate_area(dem, pixel_arcsec):
    """
    Area [m²] of elevation bins 1–4 in `dem` (needs an 'ElevationBin' variable).
    `pixel_arcsec` is the pixel size of `dem` in arcseconds.
    Adds an 'Area' variable (per-pixel area, shrinks with cos(latitude)) if missing.
    Returns [0, bin1, bin2, bin3, bin4] so the list index matches the bin number.
    """
    if 'Area' not in dem:
        pixel_area = np.cos(np.deg2rad(dem['Latitude'])) * (M_PER_ARCSEC * pixel_arcsec) ** 2
        dem['Area'] = pixel_area.broadcast_like(dem)

    by_bin = dem['Area'].groupby(dem['ElevationBin']).sum()
    return [0] + [by_bin.sel(ElevationBin=i).item() for i in range(1, 5)]



def visualize_basin_dem(dem, name):
    """Elevation and elevation bins for one basin, side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    dem['Elevation'].plot(ax=ax1, cmap='Spectral', cbar_kwargs={'label': 'Elevation [m]'})
    dem['ElevationBin'].plot(ax=ax2, levels=np.arange(0.5, 5), colors=BIN_COLORS,
                             cbar_kwargs={'label': 'Elevation bin', 'ticks': [1, 2, 3, 4]})
    ax1.set_title(f'{name} DEM')
    ax2.set_title(f'{name} elevation bins')
    plt.show()




# for SNOTEL and CDEC observations

def remove_swe_jumps(swe, max_jump=1000):
    """Drop SWE values that jump by max_jump [mm] or more from the previous day.
    A missing previous day is not treated as a jump."""
    jump = swe.diff('Day', label='upper').reindex(Day=swe['Day'])
    return swe.where((jump.abs() < max_jump) | jump.isnull())


def swe_gains(swe):
    """Daily SWE gain [mm]: losses -> 0, NaN where SWE is missing."""
    delta = swe.diff('Day', label='upper')
    return delta.where((delta >= 0) | swe.isnull(), 0)