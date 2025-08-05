# AgroGraphNet: Crop Disease Prediction using Graph Neural Networks

## Project Overview
AgroGraphNet uses satellite imagery, environmental data, and Graph Neural Networks (GNNs) to predict crop disease spread across agricultural regions. The project combines geospatial analysis with deep learning to provide early warning systems for farmers.

## Project Structure
```
AgroGraphNet/
├── notebooks/                          # Jupyter notebooks for step-by-step development
├── data/                              # Dataset storage
├── src/                               # Utility functions and modules
├── models/                            # Saved model checkpoints
├── results/                           # Output visualizations and reports
└── requirements.txt                   # Python dependencies
```

## Setup Instructions

### 1. Environment Setup
```bash
# Activate your virtual environment
# On Windows:
agro_env\Scripts\activate
# On Linux/Mac:
source agro_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup
Place your datasets in the following structure:
```
data/
├── raw/
│   ├── satellite/                     # Sentinel-2 GeoTIFF files
│   ├── weather/                       # Weather CSV files
│   ├── disease_labels/                # Disease occurrence CSV
│   └── farm_locations/                # Farm coordinates CSV
├── processed/                         # Will be created by notebooks
├── graphs/                           # Will be created by notebooks
└── labels/                           # Will be created by notebooks
```

### 3. Running the Project
Execute notebooks in order:
1. `01_data_collection.ipynb` - Data acquisition and exploration
2. `02_data_preprocessing.ipynb` - Data cleaning and processing
3. `03_graph_construction.ipynb` - Build farm network graphs
4. `04_feature_engineering.ipynb` - Create features for ML
5. `05_model_development.ipynb` - Train GNN models
6. `06_evaluation_visualization.ipynb` - Analyze results
7. `07_prediction_pipeline.ipynb` - End-to-end predictions

## Dataset Requirements

### Required Data Files:
1. **Satellite Imagery**: Sentinel-2 GeoTIFF files (place in `data/raw/satellite/`)
2. **Weather Data**: CSV with columns: date, lat, lon, temperature, humidity, precipitation
3. **Disease Labels**: CSV with columns: date, lat, lon, farm_id, disease_type, severity
4. **Farm Locations**: CSV with columns: farm_id, lat, lon, crop_type, area_hectares

### Sample Data Sources:
- Sentinel-2: https://scihub.copernicus.eu/
- Weather: OpenWeatherMap API
- Disease data: Agricultural survey data or PlantVillage dataset
- Farm locations: Regional agricultural databases

## Key Features
- Interactive data exploration and visualization
- Graph-based disease spread modeling
- Multi-temporal satellite image analysis
- Geospatial prediction mapping
- Early warning system dashboard

## Expected Results
- Disease classification accuracy >80%
- Spatial prediction maps
- Temporal disease spread analysis
- Early warning capabilities (2-4 weeks ahead)