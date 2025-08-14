#!/usr/bin/env python3
"""
Simple test to verify the CLI training and prediction workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create simple test data
def create_simple_test_data():
    """Create minimal test data for training"""
    
    # Simple farm data
    farms = pd.DataFrame({
        'farm_id': [1, 2, 3, 4, 5],
        'latitude': [40.7128, 40.7589, 40.6892, 40.7831, 40.7282],
        'longitude': [-74.0060, -73.9851, -74.0445, -73.9712, -73.7949],
        'crop_type': ['corn', 'wheat', 'soybean', 'corn', 'wheat'],
        'area_hectares': [10.5, 15.2, 8.7, 12.3, 9.8]
    })
    
    # Simple weather data (aggregated)
    weather = pd.DataFrame({
        'farm_id': [1, 2, 3, 4, 5],
        'date': ['2023-01-01'] * 5,
        'temperature': [15.2, 14.8, 16.1, 15.7, 15.3],
        'humidity': [65.3, 63.9, 67.2, 64.8, 66.5],
        'precipitation': [0.0, 0.0, 0.0, 0.0, 0.0],
        'wind_speed': [12.1, 11.7, 10.5, 12.9, 11.3]
    })
    
    # Simple satellite data
    satellite = pd.DataFrame({
        'farm_id': [1, 2, 3, 4, 5],
        'date': ['2023-01-01'] * 5,
        'ndvi': [0.65, 0.71, 0.58, 0.68, 0.74],
        'evi': [0.42, 0.48, 0.38, 0.45, 0.51],
        'savi': [0.58, 0.64, 0.51, 0.61, 0.67],
        'ndwi': [0.25, 0.31, 0.21, 0.27, 0.33]
    })
    
    # Simple labels
    labels = pd.DataFrame({
        'farm_id': [1, 2, 3, 4, 5],
        'date': ['2023-01-01'] * 5,
        'disease_type': ['healthy', 'blight', 'rust', 'healthy', 'mildew'],
        'severity': [0, 2, 1, 0, 3],
        'confidence': [0.95, 0.87, 0.92, 0.89, 0.94]
    })
    
    return farms, weather, satellite, labels

def save_test_data():
    """Save test data to CSV files"""
    farms, weather, satellite, labels = create_simple_test_data()
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    farms.to_csv(data_dir / 'farms.csv', index=False)
    weather.to_csv(data_dir / 'weather.csv', index=False)
    satellite.to_csv(data_dir / 'satellite.csv', index=False)
    labels.to_csv(data_dir / 'labels.csv', index=False)
    
    print("✅ Test data saved successfully!")

if __name__ == "__main__":
    save_test_data()