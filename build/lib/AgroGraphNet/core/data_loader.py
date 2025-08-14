import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from ..utils.validation import validate_data_format
from ..utils.logger import get_logger

class DataLoader:
    """Load and validate user datasets"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        self.raw_data_path = Path(config.data.raw_path)
    
    def load_farm_data(self) -> pd.DataFrame:
        """Load farm characteristics data"""
        file_path = self.raw_data_path / "farms.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Farm data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        validate_data_format(df, 'farms')
        
        self.logger.info(f"Loaded {len(df)} farms")
        return df
    
    def load_weather_data(self) -> pd.DataFrame:
        """Load weather/climate data"""
        file_path = self.raw_data_path / "weather.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Weather data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        validate_data_format(df, 'weather')
        
        self.logger.info(f"Loaded weather data: {len(df)} records")
        return df
    
    def load_satellite_data(self) -> pd.DataFrame:
        """Load satellite/vegetation indices data"""
        file_path = self.raw_data_path / "satellite.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Satellite data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        validate_data_format(df, 'satellite')
        
        self.logger.info(f"Loaded satellite data: {len(df)} records")
        return df
    
    def load_labels(self) -> pd.DataFrame:
        """Load disease labels for training"""
        file_path = self.raw_data_path / "labels.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Labels not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        validate_data_format(df, 'labels')
        
        self.logger.info(f"Loaded labels: {len(df)} records")
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data"""
        data = {}
        
        try:
            data['farms'] = self.load_farm_data()
        except FileNotFoundError as e:
            self.logger.warning(f"Farm data not available: {e}")
        
        try:
            data['weather'] = self.load_weather_data()
        except FileNotFoundError as e:
            self.logger.warning(f"Weather data not available: {e}")
        
        try:
            data['satellite'] = self.load_satellite_data()
        except FileNotFoundError as e:
            self.logger.warning(f"Satellite data not available: {e}")
        
        try:
            data['labels'] = self.load_labels()
        except FileNotFoundError as e:
            self.logger.warning(f"Labels not available: {e}")
        
        return data
    
    def create_merged_dataset(self) -> pd.DataFrame:
        """Create a merged dataset from all sources"""
        data = self.load_all_data()
        
        if 'farms' not in data:
            raise ValueError("Farm data is required for creating merged dataset")
        
        # Start with farm data
        merged = data['farms'].copy()
        
        # Merge weather data
        if 'weather' in data:
            weather_agg = data['weather'].groupby('farm_id').agg({
                'temperature': ['mean', 'std'],
                'humidity': ['mean', 'std'],
                'precipitation': ['sum', 'mean'],
                'wind_speed': 'mean'
            }).round(3)
            
            # Flatten column names
            weather_agg.columns = ['_'.join(col).strip() for col in weather_agg.columns]
            weather_agg = weather_agg.reset_index()
            
            merged = merged.merge(weather_agg, on='farm_id', how='left')
        
        # Merge satellite data
        if 'satellite' in data:
            satellite_agg = data['satellite'].groupby('farm_id').agg({
                'ndvi': ['mean', 'std', 'min', 'max'],
                'evi': ['mean', 'std'] if 'evi' in data['satellite'].columns else 'mean',
                'savi': ['mean', 'std'] if 'savi' in data['satellite'].columns else 'mean',
                'ndwi': ['mean', 'std'] if 'ndwi' in data['satellite'].columns else 'mean'
            }).round(4)
            
            # Flatten column names
            satellite_agg.columns = ['_'.join(col).strip() for col in satellite_agg.columns]
            satellite_agg = satellite_agg.reset_index()
            
            merged = merged.merge(satellite_agg, on='farm_id', how='left')
        
        # Merge labels
        if 'labels' in data:
            # Get most recent label for each farm
            latest_labels = data['labels'].sort_values('date').groupby('farm_id').tail(1)
            label_data = latest_labels[['farm_id', 'disease_type', 'severity']].copy()
            
            merged = merged.merge(label_data, on='farm_id', how='left')
        
        self.logger.info(f"Created merged dataset with {len(merged)} records and {len(merged.columns)} features")
        
        return merged
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        merged_data = self.create_merged_dataset()
        
        # Separate features and labels
        if 'disease_type' not in merged_data.columns:
            raise ValueError("Disease labels not found. Cannot prepare training data.")
        
        # Create label mapping
        disease_to_id = {v: k for k, v in self.config.disease_classes.items()}
        
        # Convert disease names to numeric labels
        y = merged_data['disease_type'].map(disease_to_id)
        
        # Remove non-feature columns
        feature_cols = merged_data.columns.drop(['farm_id', 'disease_type', 'severity'])
        X = merged_data[feature_cols].copy()
        
        # Handle categorical variables FIRST
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing values AFTER categorical encoding
        X = X.fillna(X.mean())
        
        self.logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y