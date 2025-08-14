import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from .logger import get_logger

def validate_data_format(df: pd.DataFrame, data_type: str) -> bool:
    """Validate user data against expected schema"""
    logger = get_logger(__name__)
    
    # Load schema
    schema_path = Path(__file__).parent.parent / "templates" / "data_schema.json"
    
    if not schema_path.exists():
        logger.warning("Data schema file not found, skipping validation")
        return True
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    if data_type not in schema:
        raise ValueError(f"Unknown data type: {data_type}")
    
    requirements = schema[data_type]
    
    # Check required columns
    required_cols = requirements["required_columns"]
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns for {data_type}: {missing_cols}")
    
    # Check data types
    type_mapping = {
        'string': 'object',
        'float': ['float64', 'float32', 'int64', 'int32'],  # Allow numeric types
        'int': ['int64', 'int32', 'float64', 'float32'],
        'datetime': 'datetime64[ns]'
    }
    
    for col, expected_type in requirements["data_types"].items():
        if col in df.columns:
            if expected_type == 'datetime':
                try:
                    pd.to_datetime(df[col])
                except:
                    logger.warning(f"Column {col} could not be parsed as datetime")
            else:
                expected_pandas_types = type_mapping.get(expected_type, expected_type)
                if isinstance(expected_pandas_types, list):
                    if not any(str(df[col].dtype).startswith(t.split('[')[0]) for t in expected_pandas_types):
                        logger.warning(f"Column {col} has type {df[col].dtype}, expected one of {expected_pandas_types}")
                else:
                    if not str(df[col].dtype).startswith(expected_pandas_types.split('[')[0]):
                        logger.warning(f"Column {col} has type {df[col].dtype}, expected {expected_pandas_types}")
    
    logger.info(f"✅ {data_type} data validation passed")
    return True

def check_data_completeness(df: pd.DataFrame, data_type: str) -> Dict[str, float]:
    """Check data completeness and quality"""
    completeness = {}
    
    # Overall completeness
    total_cells = df.size
    non_null_cells = df.count().sum()
    overall_completeness = (non_null_cells / total_cells) * 100
    
    completeness['overall'] = overall_completeness
    
    # Per-column completeness
    for col in df.columns:
        col_completeness = (df[col].count() / len(df)) * 100
        completeness[col] = col_completeness
    
    # Data quality checks
    quality_issues = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        quality_issues.append(f"{duplicates} duplicate rows")
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > len(df) * 0.05:  # More than 5% outliers
            quality_issues.append(f"{col}: {outliers} potential outliers")
    
    completeness['quality_issues'] = quality_issues
    
    return completeness

def validate_farm_coordinates(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate farm coordinate data"""
    issues = []
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Check coordinate ranges
        invalid_lat = ((df['latitude'] < -90) | (df['latitude'] > 90)).sum()
        invalid_lon = ((df['longitude'] < -180) | (df['longitude'] > 180)).sum()
        
        if invalid_lat > 0:
            issues.append(f"{invalid_lat} invalid latitude values")
        if invalid_lon > 0:
            issues.append(f"{invalid_lon} invalid longitude values")
        
        # Check for coordinates at (0,0) which might be missing data
        zero_coords = ((df['latitude'] == 0) & (df['longitude'] == 0)).sum()
        if zero_coords > 0:
            issues.append(f"{zero_coords} farms at coordinates (0,0) - possible missing data")
    
    return {'coordinate_issues': issues}

def validate_temporal_data(df: pd.DataFrame, date_col: str = 'date') -> Dict[str, Any]:
    """Validate temporal aspects of data"""
    issues = []
    
    if date_col in df.columns:
        # Check date range
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        date_range = max_date - min_date
        
        # Check for future dates
        future_dates = (df[date_col] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            issues.append(f"{future_dates} records with future dates")
        
        # Check for very old dates (more than 10 years ago)
        old_threshold = pd.Timestamp.now() - pd.Timedelta(days=3650)
        old_dates = (df[date_col] < old_threshold).sum()
        if old_dates > 0:
            issues.append(f"{old_dates} records older than 10 years")
        
        # Check temporal distribution
        temporal_gaps = check_temporal_gaps(df, date_col)
        if temporal_gaps:
            issues.extend(temporal_gaps)
    
    return {
        'temporal_issues': issues,
        'date_range': f"{min_date} to {max_date}" if date_col in df.columns else None
    }

def check_temporal_gaps(df: pd.DataFrame, date_col: str) -> List[str]:
    """Check for significant gaps in temporal data"""
    issues = []
    
    # Sort by date
    df_sorted = df.sort_values(date_col)
    dates = pd.to_datetime(df_sorted[date_col])
    
    # Check for gaps larger than expected
    date_diffs = dates.diff().dropna()
    
    # If we expect monthly data, flag gaps > 45 days
    large_gaps = date_diffs[date_diffs > pd.Timedelta(days=45)]
    if len(large_gaps) > 0:
        issues.append(f"{len(large_gaps)} temporal gaps > 45 days detected")
    
    return issues