# Data Directory Structure

Place your CSV files in the following locations:

## data/raw/
- farms.csv: Farm locations and characteristics
- weather.csv: Weather/climate data  
- satellite.csv: Vegetation indices from satellite data
- labels.csv: Disease labels for training

## Expected Data Formats

### farms.csv
Required columns: farm_id, latitude, longitude, crop_type
Optional columns: farm_size, soil_type, irrigation_type

### weather.csv  
Required columns: farm_id, date, temperature, humidity, precipitation
Optional columns: wind_speed, wind_direction, solar_radiation

### satellite.csv
Required columns: farm_id, date, ndvi
Optional columns: evi, savi, ndwi

### labels.csv
Required columns: farm_id, date, disease_type
Optional columns: severity, confidence

## Processed Data
The following directories will be populated automatically:
- data/processed/: Cleaned and preprocessed data
- data/graphs/: Graph structures for GNN models
- data/labels/: Processed labels and encodings
