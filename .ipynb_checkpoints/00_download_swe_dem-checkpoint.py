"""
Download the input data for each study basin from NASA Earthdata:
  1. ASTER Global DEM (ASTGTM) tiles covering each basin
  2. Western US UCLA snow reanalysis SWE (WUS_UCLA_SR), water years 1985–2021

Basin names, labels, outlines and the data root come from config.py.
"""

import os
import earthaccess
from config import data_folder, Basins, basin_names, shapes

earthaccess.login(persist=True)

# (min_lon, min_lat, max_lon, max_lat) for each basin, in the same order as Basins
bounding_boxes = [tuple(shapes[shapes['name'] == basin_name].total_bounds) for basin_name in basin_names]

# DEM tiles (the *_num files that come with them are quality flags)
dem_dir = os.path.join(data_folder, 'Data/DEM/')
for box in bounding_boxes:
    results = earthaccess.search_data(short_name='ASTGTM', bounding_box=box)
    earthaccess.download(results, dem_dir)

# SWE reanalysis, one folder per basin
for label, box in zip(Basins, bounding_boxes):
    results = earthaccess.search_data(
        short_name='WUS_UCLA_SR',
        bounding_box=box,
        temporal=('1984-10', '2021-09'),
        granule_name='*SWE*',
    )
    earthaccess.download(results, os.path.join(data_folder, f'Data/{label}/'))