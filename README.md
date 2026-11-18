# Warming Drives Opposing Snow Accumulation Trends Across Elevations

**Sarah Ryu, Stefano Casirati, Manuela Girotto**

This repository holds the code for our analysis of how snow water equivalent (SWE) accumulation in California's Sierra Nevada responds to storm temperature at different elevations. Running it end to end downloads the input datasets, cleans and combines them, and reproduces every figure and table in the paper.

## Abstract

Water resources depend on mountain snowpacks. While it is commonly assumed that warming decreases snow volume, the effect of warming on mountain snow accumulation is uncertain due to two competing processes: an upslope shift in the rain-snow transition and increasingly intense precipitation events. Using a high-resolution snow reanalysis of California's Sierra Nevada (1985–2021), we find an elevation-dependent response of snow water equivalent (SWE) accumulation to warming. Warmer storms deliver more precipitation to all elevations but significantly decrease SWE at low elevations (<2500 m). Contrary to expectations, high-elevation SWE accumulation responds nonmonotonically to storm temperature, first increasing due to enhanced atmospheric moisture transport, but eventually declining as precipitation phase shifts outpace enhanced moisture. Thus, warming may temporarily increase high-elevation snowpack, but the warmest storms already reduce SWE accumulation even at the coldest, highest elevations. Water management should prepare for substantial mid-elevation snow loss, and observational networks should prioritize sampling high elevations where the most consequential, nonlinear changes are emerging.

## Contents

- [Study design in brief](#study-design-in-brief)
- [Repository layout](#repository-layout)
- [Data sources](#data-sources)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Script-by-script reference](#script-by-script-reference)
- [Figures and tables](#figures-and-tables)
- [Key definitions](#key-definitions)
- [Citation](#citation)

## Study design in brief

The analysis covers three HUC6 basins on the western slope of the Sierra Nevada: **Lower Sacramento**, **San Joaquin** and **Tulare-Buena Vista Lakes**. Each basin is split into elevation bins, and only terrain above 1500 m (bins 2–4) is analyzed for snow.

| Bin | Elevation | Used in the snow analysis |
|-----|-----------|---------------------------|
| 1 | < 1500 m | No |
| 2 | 1500–1800 m | Yes |
| 3 | 1800–2500 m | Yes |
| 4 | 2500–4315 m | Yes |

The workflow, in order:

1. **Identify large storms** from GridMET daily precipitation: runs of consecutive wet days at a pixel that total at least 200 mm and fall inside a 3-day window with more than 200 mm.
2. **Measure SWE accumulation per storm** by summing daily SWE gains from the UCLA Western US Snow Reanalysis over each storm's days, then averaging (area-weighted) across each elevation bin.
3. **Assign each storm a temperature and moisture transport** from ERA5, using low-elevation (< 1000 m) cells just upwind of each basin: the 600–900 hPa mean temperature where specific humidity is in the storm's top 20%, and the mean integrated vapor transport (IVT) where IVT is in the storm's top 20%.
4. **Relate storm SWE and precipitation to storm temperature**, using per-pixel linear trends against yearly mean storm temperature and storm-level piecewise (one-breakpoint) regressions per elevation bin.
5. **Check against station observations** from SNOTEL and CDEC snow pillows.

## Repository layout

```
.
├── config.py                       # Paths, basins, study years and elevation bins (edit this first)
├── utils.py                        # Small helpers shared across scripts (water year, bin areas, SWE QC)
├── figures.py                      # All analysis and plotting functions used by notebook 10
├── environment.yml                 # Conda environment
├── metdata_wget.sh                 # wget commands for the raw GridMET files
│
├── 00_download_swe_dem.py          # Download ASTER DEM tiles and UCLA SWE reanalysis
├── 01_clean_swe.py                 # Merge, trim and clip SWE to each basin
├── 02_clean_dem.ipynb              # Coarsen the DEM, assign elevation bins, compute bin areas
├── 03_clean_gridmet.ipynb          # GridMET temperature and precipitation, storm detection
├── 04_clean_storm_data.ipynb       # Combine SWE, temperature and storms; per-storm SWE
├── 05_clean_era5.ipynb             # Download ERA5 for storm months and compute IVT
├── 06_add_storm_temps.ipynb        # Add ERA5 storm temperature and IVT to each storm
├── 07_aggregate_annual_data.ipynb  # Yearly peak SWE, storm counts, max IVT, winter temperature
├── 08_download_observations.ipynb  # SNOTEL and CDEC station SWE, quality control, peak SWE
├── 09_clean_storm_data_obs.ipynb   # Per-storm SWE gain at stations, with ERA5 storm temperature
├── 10_data_analysis_and_figures.ipynb  # Makes every figure and table in the paper
│
├── Figures/                        # Output PDFs (Fig1–Fig5, FigS1–FigS3)
└── Tables/                         # Output LaTeX table (basin_summary.tex)
```

The numbered files are meant to be run in order. Each notebook opens with a markdown cell listing its steps, inputs and outputs, so those cells are the best place to look for detail beyond this README.

## Data sources

| Dataset | What we use | Resolution | How it is obtained |
|---------|-------------|------------|--------------------|
| [UCLA Western US Snow Reanalysis](https://nsidc.org/data/wus_ucla_sr) (`WUS_UCLA_SR`, v01) | Daily SWE, ensemble median and 25th/75th percentiles, WY 1985–2021 | 16 arcsec (~500 m) | `00_download_swe_dem.py` via `earthaccess` |
| [ASTER Global DEM](https://lpdaac.usgs.gov/products/astgtmv003/) (`ASTGTM`) | Elevation | 1 arcsec, coarsened to 15 arcsec | `00_download_swe_dem.py` via `earthaccess` |
| [GridMET](https://www.climatologylab.org/gridmet.html) | Daily precipitation (`pr`), min/max temperature (`tmmn`, `tmmx`), 1979–2025 | ~4 km | `metdata_wget.sh` |
| [ERA5 daily pressure-level statistics](https://cds.climate.copernicus.eu/datasets/derived-era5-pressure-levels-daily-statistics) | Geopotential, temperature, specific humidity, u/v wind at 600–900 hPa, storm months only | 0.25° | `05_clean_era5.ipynb` via `cdsapi` |
| [USGS Watershed Boundary Dataset](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset), region 18 | HUC6 basin outlines (`WBDHU6.shp`) | Vector | Download manually |
| [CDEC](https://cdec.water.ca.gov/) | Daily revised SWE (sensor 82) at snow stations | Point | `08_download_observations.ipynb` downloads it, given a station list CSV |
| [NRCS SNOTEL](https://www.nrcs.usda.gov/resources/data-and-reports/snow-and-climate-monitoring-predictive-tools) | Daily SWE at SNOTEL stations | Point | Per-station CSVs and a station location table, prepared beforehand |

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate swe-elevation-warming
```

### 2. Set up data accounts

- **NASA Earthdata** (for the SWE reanalysis and DEM). Create a free account at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/). The first time `00_download_swe_dem.py` runs, `earthaccess.login(persist=True)` asks for your credentials and saves them to `~/.netrc`.
- **Copernicus Climate Data Store** (for ERA5). Create an account at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/), accept the ERA5 licence, and put your API key in `~/.cdsapirc` as described in the [CDS API instructions](https://cds.climate.copernicus.eu/how-to-api).

### 3. Edit `config.py`

| Setting | Meaning |
|---------|---------|
| `data_folder` | Root folder where all downloaded and processed data live. Everything is read from and written to `<data_folder>/Data/...`. |
| `shapefile` | Path to `WBDHU6.shp` from the Watershed Boundary Dataset. |
| `basin_names_list`, `Basins` | Basin names as they appear in the shapefile, and the short labels used in file names. |
| `years_swe`, `water_years_swe` | Years covered by the SWE reanalysis. |
| `water_years_prec` | Water years covered by GridMET and ERA5 (longer than the SWE record). |
| `elev_bins` | Elevation bin edges in metres: `[0, 1500, 1800, 2500, 4315]`. |

### 4. Expected data folder

After all the steps run, `data_folder` will look roughly like this. Folders marked *(manual)* need to be filled in by hand before the scripts that read them.

```
<data_folder>/Data/
├── DEM/                      # ASTER tiles, coarsened.nc4, <basin>_FixedElevBands.nc4, elev_bin_areas.csv
├── LowerSacramento/          # Raw SWE reanalysis tiles, one folder per basin
├── SanJoaquin/
├── TulareBuenaVistaLakes/
├── GridMET/                  # Raw pr_*.nc, tmmn_*.nc, tmmx_*.nc (move them here after running metdata_wget.sh)
├── ERA5/                     # Raw monthly ERA5 downloads
├── SNOTEL/                   # (manual) df_<station_id>.csv files and NRCS_SNOTEL_Locations_noAlaska.csv
├── CDEC/                     # (manual) cdec_snow_stations.csv; Revised/ holds the downloads
└── Cleaned/                  # Every processed output (see below)
    ├── <basin>/              # Cleaned yearly SWE files
    ├── GridMET/
    ├── ERA5/
    ├── SNOTEL/
    └── CDEC/
```

## Running the pipeline

Run the steps in numerical order. Most were run on an HPC cluster, and a few need a lot of memory, so the table below notes the heavier ones.

| Step | File | Depends on | Notes |
|------|------|-----------|-------|
| 0a | `00_download_swe_dem.py` | Earthdata login | Large download |
| 0b | `metdata_wget.sh` | — | Run it, then move the `.nc` files into `Data/GridMET/` |
| 1 | `01_clean_swe.py` | 0a | |
| 2 | `02_clean_dem.ipynb` | 0a | ~3 GB memory for the tile merge |
| 3 | `03_clean_gridmet.ipynb` | 0b | Per-basin clip at the end needs a larger node |
| 4 | `04_clean_storm_data.ipynb` | 1, 2, 3 | Step 1 needs >128 GB memory and about an hour per basin; uses a local Dask cluster (set `n_workers` and `total_memory_gb`) |
| 5 | `05_clean_era5.ipynb` | 3 | Download speed depends on the CDS queue; restartable. The IVT step needs a lot of memory |
| 6 | `06_add_storm_temps.ipynb` | 2, 4, 5 | |
| 7 | `07_aggregate_annual_data.ipynb` | 1, 2, 3, 6 | |
| 8 | `08_download_observations.ipynb` | SNOTEL and CDEC inputs | |
| 9 | `09_clean_storm_data_obs.ipynb` | 3, 6, 8 | |
| 10 | `10_data_analysis_and_figures.ipynb` | All of the above | Writes to `Figures/` and `Tables/` |

```bash
python 00_download_swe_dem.py
bash metdata_wget.sh && mv pr_*.nc tmmn_*.nc tmmx_*.nc <data_folder>/Data/GridMET/
python 01_clean_swe.py
# then open and run notebooks 02 to 10 in order, e.g.
jupyter lab
```

## Script-by-script reference

### Shared modules

**`config.py`** loads the basin shapefile once (`shapes`) and defines the basins, years and elevation bins that every other file imports.

**`utils.py`**
- `calculate_water_year(dates)`: water year for each date (October–September, labelled by the ending year).
- `calculate_area(dem, pixel_arcsec)`: area in m² of elevation bins 1–4, accounting for pixel area shrinking with latitude.
- `visualize_basin_dem(dem, name)`: quick side-by-side plot of a basin's elevation and elevation bins.
- `remove_swe_jumps(swe, max_jump=1000)`: drops station SWE values that jump by 1000 mm or more from the previous day.
- `swe_gains(swe)`: daily SWE gain, with losses set to zero.

**`figures.py`** contains every function used to make the paper's figures and table, plus the statistical fitting code. The main pieces are:
- *Trend maps:* `calculate_slope_das` fits, per pixel, a linear trend of a yearly variable (peak SWE or storm precipitation) against the basin's yearly mean storm temperature.
- *Storm-level fits:* `fit_storm_swe` uses a one-breakpoint piecewise linear fit when the Davies test is significant (p ≤ 0.05) and the first segment slopes upward, and a linear fit otherwise. `bootstrap_breakpoint_ci` gives a 95% bootstrap interval for the breakpoint.
- *IVT fit:* `fit_and_plot_exp_with_ci` fits IVT = a·exp(b·T) with a confidence band.
- *Observations:* `clean_obs_data` computes station-level peak SWE trends, and `plot_piecewise_obs` repeats the storm-level fit using station data.
- *Summary numbers:* `basin_summary` and `basin_summary_latex` build Table 1, and `storm_swe_warming` gives a back-of-envelope estimate of the change in storm SWE under +1.5 °C and +3 °C of storm warming.

### Pipeline steps

**`00_download_swe_dem.py`** searches NASA Earthdata within each basin's bounding box and downloads the ASTER GDEM tiles and the UCLA SWE reanalysis (SWE files only).

**`01_clean_swe.py`** merges each basin's SWE tiles for each water year, keeps the ensemble median and 25th/75th percentiles, adds the daily SWE change (`Delta_SWE`), clips to the basin outline and saves `Data/Cleaned/<basin>/<basin>_SWE_WY<yyyy>_<yy>.nc4`.

**`02_clean_dem.ipynb`** merges the ASTER tiles, coarsens them to 15 arcsec, saves the full study-area DEM (`coarsened.nc4`), clips it to each basin, assigns elevation bins (`<basin>_FixedElevBands.nc4`) and saves bin areas (`elev_bin_areas.csv`).

**`03_clean_gridmet.ipynb`** builds daily mean temperature and precipitation over the study area, labels storms at every pixel (see [Key definitions](#key-definitions)), and saves the list of storm days for the whole area and for each basin.

**`04_clean_storm_data.ipynb`** puts SWE gains, GridMET temperature and storm labels on the SWE grid, then:
1. saves yearly climate summaries and a storm-days-only daily file per basin,
2. records each pixel-storm's start date, duration and total SWE gain in bins 2–4,
3. merges pixel-storms that overlap in time into basin-wide storm groups and computes each group's area-weighted mean SWE gain per elevation bin,
4. summarizes storms per water year.

**`05_clean_era5.ipynb`** downloads ERA5 daily means for every month containing a storm (days aligned to UTC−7 to match GridMET), merges the files, converts geopotential to height and computes IVT over 600–900 hPa.

**`06_add_storm_temps.ipynb`** builds, for each basin, a mask of the five easternmost ERA5 cells below 1000 m in each latitude row (the valley edge just upwind of the mountains), and uses it to add storm temperature to the SWE-period storms (`<basin>_grouped_storm_ds_final.nc4`) and storm temperature plus IVT to the full-record GridMET storms (`<basin>_storms_long.nc4`). An optional cell tests how sensitive storm temperature is to the 80th-percentile humidity cutoff.

**`07_aggregate_annual_data.ipynb`** computes per-pixel peak SWE for each water year and a yearly table of storm counts, maximum storm IVT and winter (December–March) temperature by elevation bin.

**`08_download_observations.ipynb`** combines SNOTEL CSVs, downloads CDEC daily SWE, converts everything to mm, applies quality control (see [Key definitions](#key-definitions)) and saves daily and peak SWE for both networks.

**`09_clean_storm_data_obs.ipynb`** finds the GridMET storms at each station's grid cell, sums the observed SWE gain during each one, and gives each a storm temperature from the overlapping basin-wide storm.

**`10_data_analysis_and_figures.ipynb`** loads the processed data and calls `figures.py` to produce every figure and table.

## Figures and tables

All are produced by `10_data_analysis_and_figures.ipynb`.

| Output | Content | Main functions |
|--------|---------|----------------|
| `Figures/Fig1.pdf` | (A) Shaded-relief DEM of the study basins, (B) elevation bins, (C–F) cumulative area by elevation per basin. Assembled from `Fig1ab.pdf` and `Fig1cdef.pdf` | `plot_dem_and_elev_bins`, `plot_elev_cdfs` |
| `Tables/basin_summary.tex` | Table 1: area, elevation, precipitation, peak SWE, storm temperature and storm counts per basin | `basin_summary`, `basin_summary_latex` |
| `Figures/Fig2.pdf` | (A) Upwind IVT map for an example storm, (B) temperature cross-section of that storm over the San Joaquin basin | `plot_fig2` |
| `Figures/Fig3.pdf` | (A) Storm IVT against storm temperature with an exponential fit, (B) map of the change in storm precipitation per °C of storm warming | `load_gridmet_storms`, `storm_precip_das`, `plot_fig3` |
| `Figures/Fig4.pdf` | (A) Map of the change in peak SWE per °C, (B) the same slopes grouped by 200 m elevation band | `load_peak_swe`, `calculate_slope_das`, `plot_fig4` |
| `Figures/Fig5.pdf` | Storm SWE accumulation against storm temperature for each basin and elevation bin, with piecewise or linear fits | `create_basin_dfs`, `plot_piecewise_fit` |
| `Figures/FigS1.pdf` | Station peak-SWE trends (SNOTEL, CDEC) on top of the reanalysis trend map, and by elevation | `clean_obs_data`, `plot_figS1` |
| `Figures/FigS2.pdf` | Observed storm SWE gain against storm temperature at stations, by elevation bin | `plot_piecewise_obs` |
| `Figures/FigS3.pdf` | Yearly time series of storm count, maximum storm IVT and winter temperature, with trends | `load_time_series_data`, `plot_time_series` |

The last cell of the notebook prints the warming scenario estimate (`storm_swe_warming`) for the Tulare-Buena Vista Lakes basin.

## Key definitions

- **Water year (WY):** October 1 to September 30, labelled by the year it ends in.
- **Storm (at a pixel):** a run of consecutive days with precipitation > 0 that lies within a 3-day window totalling more than 200 mm, and whose own total is at least 200 mm. Storm IDs are numbered separately at each pixel.
- **Storm group:** pixel-storms within a basin whose date ranges overlap, directly or through a chain of overlaps, merged into one basin-wide storm. This is the unit of analysis for storm SWE (Fig. 5).
- **Storm SWE:** total daily SWE gain (losses set to zero) over a storm's days, averaged across the pixels of an elevation bin weighted by pixel area. Uses the reanalysis ensemble median.
- **Storm temperature:** 600–900 hPa mean ERA5 temperature over the storm's days, averaged over the masked upwind cells where 600–900 hPa mean specific humidity is in the top 20% for that storm. Stored in K; shown in °C in the figures.
- **Storm IVT:** IVT = |(1/g) ∫ q **V** dp| over 600–900 hPa from daily means, averaged over masked cells in the storm's top 20% of IVT. Because it covers fewer levels and uses daily means, it is lower than the conventional 1000–300 hPa IVT from sub-daily data.
- **Peak SWE:** the maximum daily SWE in a water year.
- **Station quality control:** remove negative values and day-to-day jumps of 1000 mm or more, keep January–May only, and drop station-years with fewer than 7 observations or none in April. Known faulty CDEC values (stations SPS and EBB above 2500 mm) are also removed.

## Citation

If you use this code, please cite:

> Ryu, S., Casirati, S., & Girotto, M. *Warming Drives Opposing Snow Accumulation Trends Across Elevations.* [Journal, year, DOI to be added]
