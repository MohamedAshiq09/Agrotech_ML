#!/usr/bin/env python3
"""
Debug the training process to identify the categorical data issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pathlib import Path

# Import from the installed package
try:
    from AgroGraphNet.core.config import Config
    from AgroGraphNet.core.data_loader import DataLoader
except ImportError:
    # Fallback to local import
    sys.path.append('../AgroGraphNet')
    from core.config import Config
    from core.data_loader import DataLoader

def debug_data_processing():
    """Debug the data processing pipeline"""
    
    # Load config
    config = Config.from_file('config.yaml')
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    print("=== Loading raw data ===")
    data = data_loader.load_all_data()
    
    for key, df in data.items():
        print(f"\n{key.upper()} data:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        if key == 'farms':
            print(f"Crop types: {df['crop_type'].unique()}")
    
    print("\n=== Creating merged dataset ===")
    merged = data_loader.create_merged_dataset()
    print(f"Merged shape: {merged.shape}")
    print(f"Merged columns: {list(merged.columns)}")
    print(f"Merged dtypes:\n{merged.dtypes}")
    
    print("\n=== Preparing training data ===")
    try:
        X, y = data_loader.prepare_training_data()
        print(f"X shape: {X.shape}")
        print(f"X columns: {list(X.columns)}")
        print(f"X dtypes:\n{X.dtypes}")
        print(f"y shape: {y.shape}")
        print(f"y values: {y.values}")
        
        # Check for any remaining object columns
        object_cols = X.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"\n⚠️  WARNING: Object columns still present: {list(object_cols)}")
            for col in object_cols:
                print(f"  {col}: {X[col].unique()}")
        
        # Try converting to numpy
        print("\n=== Testing numpy conversion ===")
        try:
            X_numpy = X.values
            print(f"✅ Successfully converted to numpy: {X_numpy.shape}")
            print(f"Data type: {X_numpy.dtype}")
        except Exception as e:
            print(f"❌ Failed to convert to numpy: {e}")
            
            # Check each column individually
            for col in X.columns:
                try:
                    col_values = X[col].values
                    print(f"  ✅ {col}: {col_values.dtype}")
                except Exception as col_e:
                    print(f"  ❌ {col}: {col_e}")
                    print(f"    Sample values: {X[col].head().tolist()}")
        
    except Exception as e:
        print(f"❌ Failed to prepare training data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_processing()