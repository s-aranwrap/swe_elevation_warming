import geopandas as gpd
import numpy as np


data_folder = '/global/scratch/users/sarahryu' # Folder to store all downloaded data

shapefile = '../SWE/Data/WBD_18_HU2_Shape/Shape/WBDHU6.shp' # shapefile of basin boundaries

# Basin names in basin boundary shapefile
basin_names_list = [
    'Lower Sacramento',
    'San Joaquin',
    'Tulare-Buena Vista Lakes'
]

# Basin labels for naming files
Basins = [
    'LowerSacramento',
    'SanJoaquin',
    'TulareBuenaVistaLakes'
]

basin_names = dict(zip(Basins, basin_names_list))


shapes = gpd.read_file(shapefile)


# years to analyze SWE reanalysis
years_swe = np.array(range(1984, 2022))
water_years_swe = years_swe[1:] # removes 1984

# years to analyze precipitation and storm data, longer than years_swe based on data availability
water_years_prec = np.arange(1980, 2026)


elev_bins = [0, 1500, 1800, 2500, 4315]
bins = range(2, len(elev_bins))