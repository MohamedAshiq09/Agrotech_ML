#!/usr/bin/env python3
"""
Create a larger test dataset with more samples to ensure proper train/val/test splits
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_larger_test_data():
    """Create larger test data for training"""
    
    np.random.seed(42)  # For reproducibility
    
    n_farms = 20
    
    # Create farm data
    farms = pd.DataFrame({
        'farm_id': range(1, n_farms + 1),
        'latitude': np.random.uniform(40.6, 40.8, n_farms),
        'longitude': np.random.uniform(-74.1, -73.8, n_farms),
        'crop_type': np.random.choice(['corn', 'wheat', 'soybean'], n_farms),
        'area_hectares': np.random.uniform(8, 16, n_farms)
    })
    
    # Create weather data (one record per farm)
    weather = pd.DataFrame({
        'farm_id': range(1, n_farms + 1),
        'date': ['2023-01-01'] * n_farms,
        'temperature': np.random.uniform(14, 18, n_farms),
        'humidity': np.random.uniform(60, 75, n_farms),
        'precipitation': np.random.uniform(0, 5, n_farms),
        'wind_speed': np.random.uniform(8, 16, n_farms)
    })
    
    # Create satellite data
    satellite = pd.DataFrame({
        'farm_id': range(1, n_farms + 1),
        'date': ['2023-01-01'] * n_farms,
        'ndvi': np.random.uniform(0.5, 0.8, n_farms),
        'evi': np.random.uniform(0.3, 0.6, n_farms),
        'savi': np.random.uniform(0.4, 0.7, n_farms),
        'ndwi': np.random.uniform(0.1, 0.4, n_farms)
    })
    
    # Create labels with balanced distribution
    diseases = ['healthy', 'blight', 'rust', 'mildew']
    disease_types = []
    severities = []
    confidences = []
    
    # Ensure each disease type appears at least 3 times
    for disease in diseases:
        disease_types.extend([disease] * 3)
        severities.extend([0, 1, 2] if disease != 'healthy' else [0, 0, 0])
        confidences.extend(np.random.uniform(0.85, 0.98, 3))
    
    # Fill remaining slots randomly
    remaining = n_farms - len(disease_types)
    if remaining > 0:
        disease_types.extend(np.random.choice(diseases, remaining))
        severities.extend(np.random.choice([0, 1, 2, 3], remaining))
        confidences.extend(np.random.uniform(0.85, 0.98, remaining))
    
    # Shuffle to randomize order
    indices = np.random.permutation(n_farms)
    
    labels = pd.DataFrame({
        'farm_id': range(1, n_farms + 1),
        'date': ['2023-01-01'] * n_farms,
        'disease_type': [disease_types[i] for i in indices],
        'severity': [severities[i] for i in indices],
        'confidence': [confidences[i] for i in indices]
    })
    
    return farms, weather, satellite, labels

def save_larger_test_data():
    """Save larger test data to CSV files"""
    farms, weather, satellite, labels = create_larger_test_data()
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    farms.to_csv(data_dir / 'farms.csv', index=False)
    weather.to_csv(data_dir / 'weather.csv', index=False)
    satellite.to_csv(data_dir / 'satellite.csv', index=False)
    labels.to_csv(data_dir / 'labels.csv', index=False)
    
    print("✅ Larger test data saved successfully!")
    print(f"Created {len(farms)} farms with balanced disease distribution:")
    print(labels['disease_type'].value_counts())

if __name__ == "__main__":
    save_larger_test_data()