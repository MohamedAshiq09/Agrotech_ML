# 🌾 AgroGraphNet Prediction System

## Overview

The AgroGraphNet Prediction System combines 7 different machine learning models to predict crop diseases from farm data. This system uses Graph Neural Networks (GNNs) and traditional machine learning models to provide accurate disease predictions for agricultural planning.

## Models Included

1. **GCN (Graph Convolutional Network)** - Advanced spatial modeling
2. **GraphSAGE** - Graph sampling and aggregation approach
3. **GAT (Graph Attention Network)** - Attention-based spatial relationships
4. **Random Forest** - Ensemble baseline model
5. **SVM (Support Vector Machine)** - Kernel-based classifier
6. **Logistic Regression** - Linear baseline model
7. **Temporal GNN** - Advanced temporal-spatial modeling

## Prerequisites

Before using the prediction system, ensure you have:

1. **Trained Models**: Run notebook `05_model_development.ipynb` to train all models
2. **Python Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Basic Usage

```bash
cd AgroGraphNet
python run_model.py
```

The system will prompt you to enter the path to your dataset.

### 2. Using Sample Dataset

```bash
python run_model.py
# When prompted for dataset path, enter: sample_dataset.csv
```

### 3. Using Your Own Data

Prepare a CSV file with the following columns:

**Required Columns:**
- `farm_id`: Unique identifier for each farm
- `lat`: Latitude coordinate
- `lon`: Longitude coordinate
- `crop_type`: Type of crop (e.g., wheat, corn, soybean)

**Optional Columns:**
- `area_hectares`: Farm area in hectares (default: 50.0)
- `date`: Date of data collection (default: current date)

Example CSV format:
```csv
farm_id,lat,lon,crop_type,area_hectares
FARM_001,40.7128,-74.0060,wheat,45.2
FARM_002,40.7589,-73.9851,corn,52.8
FARM_003,40.7505,-73.9934,soybean,38.1
```

## Output Format

The system provides predictions in a comprehensive format:

```
🌾 AGROGRAPHNET PREDICTION RESULTS
================================================================================

📊 Predictions for 10 farms:
     Farm ID   Latitude   Longitude Crop Type Ensemble Prediction Ensemble Confidence
    farm_001    40.7128    -74.0060     wheat             Healthy              0.857
    farm_002    40.7589    -73.9851      corn             Blight              0.723
    farm_003    40.7505    -73.9934  soybean             Healthy              0.912
    ...

📈 Prediction Summary:
----------------------------------------
Ensemble Predictions Distribution:
  - Healthy: 7 farms (70.0%)
  - Blight: 2 farms (20.0%)
  - Rust: 1 farms (10.0%)

🚨 Risk Assessment:
  - Healthy farms: 7
  - At-risk farms: 3
  - Overall health rate: 70.0%
```

## Disease Classes

The system predicts the following disease categories:

- **Healthy**: No disease detected
- **Blight**: Various blight diseases affecting crops
- **Rust**: Rust fungal diseases
- **Mosaic**: Mosaic virus infections
- **Bacterial**: Bacterial infections

## Technical Details

### Prediction Process

1. **Data Loading**: Loads user dataset and validates format
2. **Preprocessing**: Creates graph structure and node features
3. **Model Loading**: Loads all trained models from disk
4. **Prediction**: Runs inference on all 7 models
5. **Ensemble**: Combines predictions using majority voting
6. **Output**: Displays results and saves to CSV

### Model Architecture

Each GNN model consists of:
- Input layer with farm features (lat, lon, crop type, etc.)
- Multiple GNN layers for spatial relationship learning
- Output layer with softmax activation for classification
- Batch normalization and dropout for regularization

### Performance

- **Input Features**: ~20 features per farm including spatial, temporal, and environmental data
- **Graph Structure**: Farms connected based on geographic proximity (< 5km)
- **Training**: Uses temporal split (60% train, 20% validation, 20% test)

## Troubleshooting

### Common Issues

1. **"No trained models found"**
   - Run `05_model_development.ipynb` first to train models
   - Check that models are saved in the `models/` directory

2. **"Missing required columns"**
   - Ensure your CSV has columns: `farm_id`, `lat`, `lon`, `crop_type`
   - Check for proper formatting and data types

3. **CUDA errors**
   - The system automatically uses GPU if available
   - Set `CUDA_VISIBLE_DEVICES=""` to force CPU usage

4. **Memory errors**
   - Reduce batch size in `MODEL_CONFIG`
   - Process data in smaller chunks

### Getting Help

For issues or questions:
1. Check the log output for specific error messages
2. Verify all dependencies are installed
3. Ensure models are properly trained and saved
4. Check data format matches requirements

## Advanced Usage

### Custom Configuration

Modify `src/config.py` to adjust:
- Model hyperparameters
- Graph construction parameters
- Disease class definitions

### Batch Processing

For large datasets, modify the script to:
- Process data in chunks
- Save intermediate results
- Implement progress tracking

### Model Updates

To retrain models:
1. Run `05_model_development.ipynb`
2. Copy new models to `models/` directory
3. Restart prediction script

## License

This project is part of the AgroGraphNet research initiative for crop disease prediction using graph neural networks.
