# Input data

The data used in the paper are available on Zenodo:

> Karim, S. (2026). *Multisource remote sensing dataset for leached metal hazard mapping at four coal combustion residual sites in Virginia, USA (2019–2025)* (Version 1.0) [Dataset]. Zenodo. [https://doi.org/10.5281/zenodo.23823223](https://doi.org/10.5281/zenodo.23823223)

The data are not stored in this repository. Download the dataset and place the files as below. Both folders are searched in the
order of `SEARCH_DIRS` in `src/hazard_ml.py`, so a file may sit in either.

```
data/
├── features_labels/          # Earth Engine exports of src/hazard_ml_gee.py
│   ├── <stem>_Features.tif   # predictor stack, one band per feature (band names in the band descriptions)
│   ├── <stem>_Label.tif      # optional: class 1 = normalized HI >= 25th percentile, class 0 = the rest
│   └── <stem>_HI.tif         # optional: kriged HI (without it, HI is kriged locally with the site variogram)
└── well_maps/
    ├── <Site>_wells_HI_composite_named.csv   # one row per well (see below)
    ├── <Site>_samples.csv                     # optional: raw samples (well gradient, extra metals)
    └── ccr-units.shp (+ .dbf, .prj, .shx)     # optional: outlines of the CCR units
```

`<stem>` and the file names of each site are set in the `SITES` dictionary of `src/hazard_ml.py`.

## Well HI composite (`*_wells_HI_composite_named.csv`)

| Column | Required | Meaning |
|---|---|---|
| `latitude`, `longitude` | yes | well position (WGS 84) |
| `HI` | yes | Hazard Index: sum of the constituent HQs at the well |
| `HQ_<metal>` | for the HQ steps | HQ of one constituent, e.g. `HQ_arsenic`, `HQ_cobalt` (concentration / reference threshold) |
| `well_name` | recommended | well identifier |
| `gradient` | recommended | `Upgradient` or `Downgradient` (otherwise taken from the sample table, else `Unknown`) |
| `n_HQ_in_HI` | optional | number of constituents summed into HI; wells below `MIN_HQ_IN_HI` are left out |

## Sample table (`*_samples.csv`, optional)

This table supplies the well gradient position and the HQ of metals missing from the composite (`HQ_FROM_SAMPLES`).
It uses these columns:
- `well_name`, `well_designation` (upgradient / downgradient), `well_latitude`, `well_longitude`;
- `contaminant`, `concentration`, `measurement_unit`;
- `standard_threshold`, `standard_unit`.

Concentrations in µg/L are converted to mg/L.

## CCR units (`ccr-units.shp`, optional)

These are polygon outlines of the CCR impoundments and landfills. The unit name is read from the field
`CCR_NAME_FIELD`, and the units are drawn on every map. The distance from each well to the nearest unit is used in
the partial correlations.

## Earth Engine inputs (`src/hazard_ml_gee.py`)

The script uses public Earth Engine datasets:
- Sentinel-2 surface reflectance with Cloud Score+;
- SRTM;
- MODIS land surface temperature, NLDAS, GLDAS and CHIRPS;
- SoilGrids;
- MERIT Hydro;
- Dynamic World.

The well HI composite of each site is read either from an Earth Engine asset (`asset` in `SITES`) or from a CSV
(`WELLS_FROM_CSV`).

Note: SoilGrids stores pH as pH × 10 and sand, silt and clay as g/kg. Pixels without a SoilGrids prediction (water,
built-up areas) are exported as 0.
