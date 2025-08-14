"""
Configuration file for AgroGraphNet project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data subdirectories
SATELLITE_DIR = RAW_DATA_DIR / "satellite"
WEATHER_DIR = RAW_DATA_DIR / "weather"
DISEASE_LABELS_DIR = RAW_DATA_DIR / "disease_labels"
FARM_LOCATIONS_DIR = RAW_DATA_DIR / "farm_locations"
GRAPHS_DIR = DATA_DIR / "graphs"
LABELS_DIR = DATA_DIR / "labels"

# Create data subdirectories
for dir_path in [SATELLITE_DIR, WEATHER_DIR, DISEASE_LABELS_DIR, FARM_LOCATIONS_DIR, GRAPHS_DIR, LABELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100,
    'early_stopping_patience': 10
}

# Graph construction parameters
GRAPH_CONFIG = {
    'distance_threshold_km': 5.0,  # Maximum distance for farm connections
    'min_neighbors': 2,            # Minimum neighbors per node
    'max_neighbors': 10,           # Maximum neighbors per node
    'edge_features': ['distance', 'elevation_diff', 'weather_similarity']
}

# Disease classes
DISEASE_CLASSES = {
    0: 'Healthy',
    1: 'Blight',
    2: 'Rust',
    3: 'Mosaic',
    4: 'Bacterial'
}

# Satellite bands (Sentinel-2)
SATELLITE_BANDS = {
    'B02': 'Blue',
    'B03': 'Green', 
    'B04': 'Red',
    'B08': 'NIR',
    'B11': 'SWIR1',
    'B12': 'SWIR2'
}

# Vegetation indices
VEGETATION_INDICES = ['NDVI', 'EVI', 'SAVI', 'NDWI']

# Random seed for reproducibility
RANDOM_SEED = 42