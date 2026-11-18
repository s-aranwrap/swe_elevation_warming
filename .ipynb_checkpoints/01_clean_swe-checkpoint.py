"""
Clean the UCLA Western US snow reanalysis (WUS_UCLA_SR) SWE data for each basin.

For each basin and water year, this script:
  1. merges the downloaded tiles for that year,
  2. keeps only the ensemble median, 25th and 75th percentiles,
  3. adds the daily change in SWE (Delta_SWE),
  4. clips to the basin outline and saves the result to Data/Cleaned/<basin>/.
"""
import gc
import glob
import os

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from config import data_folder, Basins, basin_names, years_swe

# Stats dimension, in file order: mean, std, median, 25th pct, 75th pct
KEEP_STATS = [2, 3, 4]  # median, 25th, 75th

shapes = gpd.read_file('Data/WBD_18_HU2_Shape/Shape/WBDHU6.shp')


def make_swe_ds(file, year):
    """Open one SWE tile for water year `year` (Oct 1 of `year` to Sep 30 of `year`+1)."""
    ds = xr.open_dataset(file, chunks='auto', cache=False, engine='netcdf4')
    ds = (
        ds.assign_coords(
            Day=pd.date_range(f'{year}-10-01', f'{year + 1}-09-30'),
            Stats=range(5),
        )
        .transpose('Latitude', 'Longitude', 'Day', 'Stats')
        .drop_vars('SCA_Post', errors='ignore')
        .sel(Stats=KEEP_STATS)
    )
    # Day-to-day change in SWE; NaNs (including the first day) become 0
    ds['Delta_SWE'] = ds['SWE_Post'].diff('Day', label='upper')
    ds['Delta_SWE'] = ds['Delta_SWE'].fillna(0)
    return ds


def clean_basin_year(basin, polygon, year, out_dir):
    """Merge all tiles for one basin and water year, clip to the basin, and save."""
    pattern = os.path.join(
        data_folder, f'Data/{basin}',
        f'WUS_UCLA_SR_v01_*_agg_16_WY{year}*_SWE_SCA_POST.nc',
    )
    tiles = [make_swe_ds(f, year) for f in glob.glob(pattern)]

    swe = (
        xr.combine_by_coords(tiles)
        .rio.write_crs('EPSG:4326')
        .rio.set_spatial_dims(x_dim='Longitude', y_dim='Latitude')
        .rio.clip([polygon], crs='EPSG:4326')
    )
    # e.g. WY1998_99, WY1999_00, WY2008_09
    swe.to_netcdf(os.path.join(out_dir, f'{basin}_SWE_WY{year}_{(year + 1) % 100:02d}.nc4'))


for basin, name in zip(Basins, basin_names):
    print(basin)
    polygon = shapes.loc[shapes['name'] == name, 'geometry'].iloc[0]
    out_dir = os.path.join(data_folder, f'Data/Cleaned/{basin}')
    os.makedirs(out_dir, exist_ok=True)

    for year in years_swe:
        clean_basin_year(basin, polygon, year, out_dir)
        print(year)
        gc.collect()