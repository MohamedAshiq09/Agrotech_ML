# test_project

AgroGraphNet project for crop disease prediction using Graph Neural Networks.

## Data Requirements

Place your CSV files in the `data/raw/` directory:

1. **farms.csv** - Farm locations and characteristics
   - Required columns: farm_id, latitude, longitude, crop_type
   - Optional columns: farm_size, soil_type, irrigation_type

2. **weather.csv** - Weather/climate data
   - Required columns: farm_id, date, temperature, humidity, precipitation
   - Optional columns: wind_speed, wind_direction, solar_radiation

3. **satellite.csv** - Vegetation indices from satellite data
   - Required columns: farm_id, date, ndvi
   - Optional columns: evi, savi, ndwi

4. **labels.csv** - Disease labels for training
   - Required columns: farm_id, date, disease_type
   - Optional columns: severity, confidence

## Usage

1. Validate your data:
   ```bash
   agrographnet validate
   ```

2. Train models:
   ```bash
   agrographnet train --model graphsage --epochs 100
   ```

3. Make predictions:
   ```bash
   agrographnet predict --input data/new_data.csv --output results/predictions.csv
   ```

4. Generate analysis:
   ```bash
   agrographnet analyze --output results/analysis_report.html
   ```
