"""
Data loading and preprocessing utilities for AgroGraphNet
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pathlib import Path
import os
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

def load_satellite_image(file_path: str, bands: List[str] = None) -> Tuple[np.ndarray, dict]:
    """
    Load satellite imagery from GeoTIFF files
    
    Args:
        file_path: Path to GeoTIFF file
        bands: List of band names to load
    
    Returns:
        Tuple of (image_array, metadata)
    """
    with rasterio.open(file_path) as src:
        if bands:
            # Load specific bands
            band_indices = [i for i, band in enumerate(src.descriptions, 1) if band in bands]
            image = src.read(band_indices)
        else:
            # Load all bands
            image = src.read()
        
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'count': src.count
        }
    
    return image, metadata

def calculate_vegetation_indices(image: np.ndarray, band_mapping: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Calculate vegetation indices from satellite bands
    
    Args:
        image: Satellite image array (bands, height, width)
        band_mapping: Mapping of band names to array indices
    
    Returns:
        Dictionary of vegetation indices
    """
    indices = {}
    
    # Get band arrays
    red = image[band_mapping['Red']].astype(float)
    nir = image[band_mapping['NIR']].astype(float)
    blue = image[band_mapping['Blue']].astype(float)
    swir1 = image[band_mapping['SWIR1']].astype(float)
    
    # NDVI (Normalized Difference Vegetation Index)
    indices['NDVI'] = np.divide(nir - red, nir + red, 
                               out=np.zeros_like(nir), where=(nir + red) != 0)
    
    # EVI (Enhanced Vegetation Index)
    indices['EVI'] = 2.5 * np.divide(nir - red, nir + 6 * red - 7.5 * blue + 1,
                                    out=np.zeros_like(nir), 
                                    where=(nir + 6 * red - 7.5 * blue + 1) != 0)
    
    # SAVI (Soil Adjusted Vegetation Index)
    L = 0.5  # Soil brightness correction factor
    indices['SAVI'] = (1 + L) * np.divide(nir - red, nir + red + L,
                                         out=np.zeros_like(nir), 
                                         where=(nir + red + L) != 0)
    
    # NDWI (Normalized Difference Water Index)
    indices['NDWI'] = np.divide(nir - swir1, nir + swir1,
                               out=np.zeros_like(nir), where=(nir + swir1) != 0)
    
    return indices

def extract_pixel_values_at_points(image: np.ndarray, transform, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Extract pixel values at specific geographic coordinates
    
    Args:
        image: Satellite image array
        transform: Rasterio transform object
        points: List of (longitude, latitude) tuples
    
    Returns:
        Array of pixel values at each point
    """
    from rasterio.transform import rowcol
    
    values = []
    for lon, lat in points:
        try:
            row, col = rowcol(transform, lon, lat)
            if 0 <= row < image.shape[-2] and 0 <= col < image.shape[-1]:
                if len(image.shape) == 3:  # Multi-band
                    pixel_values = image[:, row, col]
                else:  # Single band
                    pixel_values = image[row, col]
                values.append(pixel_values)
            else:
                # Point outside image bounds
                values.append(np.full(image.shape[0] if len(image.shape) == 3 else 1, np.nan))
        except:
            values.append(np.full(image.shape[0] if len(image.shape) == 3 else 1, np.nan))
    
    return np.array(values)

def load_weather_data(file_path: str) -> pd.DataFrame:
    """
    Load weather data from CSV file
    
    Expected columns: date, lat, lon, temperature, humidity, precipitation, wind_speed, wind_direction
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_disease_labels(file_path: str) -> pd.DataFrame:
    """
    Load disease occurrence data from CSV file
    
    Expected columns: date, lat, lon, farm_id, disease_type, severity
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_farm_locations(file_path: str) -> pd.DataFrame:
    """
    Load farm location data from CSV file
    
    Expected columns: farm_id, lat, lon, crop_type, area_hectares
    """
    df = pd.read_csv(file_path)
    return df

def create_sample_data(output_dir: str, num_farms: int = 100, num_time_steps: int = 12):
    """
    Create sample datasets for testing (when real data is not available)
    """
    np.random.seed(42)
    
    # Sample farm locations (around a central point)
    center_lat, center_lon = 40.0, -95.0  # Central US
    
    farms_data = []
    for i in range(num_farms):
        farm_id = f"farm_{i:03d}"
        lat = center_lat + np.random.normal(0, 0.5)  # ~50km spread
        lon = center_lon + np.random.normal(0, 0.5)
        crop_type = np.random.choice(['corn', 'soybean', 'wheat'])
        area = np.random.uniform(10, 500)  # hectares
        
        farms_data.append({
            'farm_id': farm_id,
            'lat': lat,
            'lon': lon,
            'crop_type': crop_type,
            'area_hectares': area
        })
    
    farms_df = pd.DataFrame(farms_data)
    farms_df.to_csv(f"{output_dir}/farm_locations/sample_farms.csv", index=False)
    
    # Sample weather data
    weather_data = []
    dates = pd.date_range('2023-01-01', periods=num_time_steps, freq='M')
    
    for date in dates:
        for _, farm in farms_df.iterrows():
            weather_data.append({
                'date': date,
                'lat': farm['lat'],
                'lon': farm['lon'],
                'temperature': np.random.normal(20, 10),  # Celsius
                'humidity': np.random.uniform(30, 90),    # Percentage
                'precipitation': np.random.exponential(2), # mm
                'wind_speed': np.random.uniform(0, 15),   # m/s
                'wind_direction': np.random.uniform(0, 360) # degrees
            })
    
    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv(f"{output_dir}/weather/sample_weather.csv", index=False)
    
    # Sample disease labels
    disease_data = []
    disease_types = ['Healthy', 'Blight', 'Rust', 'Mosaic', 'Bacterial']
    
    for date in dates:
        for _, farm in farms_df.iterrows():
            # Higher probability of healthy crops
            disease_probs = [0.7, 0.1, 0.1, 0.05, 0.05]
            disease = np.random.choice(disease_types, p=disease_probs)
            severity = np.random.uniform(0, 1) if disease != 'Healthy' else 0
            
            disease_data.append({
                'date': date,
                'lat': farm['lat'],
                'lon': farm['lon'],
                'farm_id': farm['farm_id'],
                'disease_type': disease,
                'severity': severity
            })
    
    disease_df = pd.DataFrame(disease_data)
    disease_df.to_csv(f"{output_dir}/disease_labels/sample_diseases.csv", index=False)
    
    print(f"Sample data created in {output_dir}")
    print(f"- {len(farms_df)} farms")
    print(f"- {len(weather_df)} weather records")
    print(f"- {len(disease_df)} disease records")

def preprocess_temporal_data(df: pd.DataFrame, date_col: str = 'date', 
                           location_cols: List[str] = ['lat', 'lon']) -> pd.DataFrame:
    """
    Preprocess temporal data by sorting and handling missing values
    """
    # Sort by date and location
    df = df.sort_values([date_col] + location_cols)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df