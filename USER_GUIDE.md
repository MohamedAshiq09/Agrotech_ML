# AgroGraphNet User Guide

## 📖 Complete Guide to Using AgroGraphNet

This comprehensive guide will walk you through every aspect of using AgroGraphNet for agricultural disease prediction.

## 🎯 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Project Workflow](#project-workflow)
5. [CLI Commands Reference](#cli-commands-reference)
6. [Configuration](#configuration)
7. [Model Selection](#model-selection)
8. [Interpreting Results](#interpreting-results)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## 🔧 Installation

### System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- 2GB+ free disk space
- Optional: NVIDIA GPU with CUDA support

### Install AgroGraphNet

```bash
# Install from PyPI
pip install agrographnet

# Verify installation
agrographnet --help
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agrographnet.git
cd agrographnet

# Install in development mode
pip install -e .
```

## 🚀 Quick Start

### 1. Create Your First Project

```bash
# Create a new project
agrographnet init cotton_disease_prediction

# Navigate to project directory
cd cotton_disease_prediction
```

This creates the following structure:
```
cotton_disease_prediction/
├── data/
│   └── raw/          # Place your CSV files here
├── models/           # Trained models will be saved here
├── results/          # Analysis results and reports
├── logs/             # Execution logs
├── config.yaml       # Configuration file
└── README.md         # Project documentation
```

### 2. Prepare Your Data

Place your CSV files in the `data/raw/` directory:

- `farms.csv` - Farm locations and characteristics
- `weather.csv` - Weather/climate data
- `satellite.csv` - Vegetation indices
- `labels.csv` - Disease labels for training

### 3. Validate Your Data

```bash
agrographnet validate
```

This checks:
- File existence and format
- Required columns presence
- Data types and ranges
- Missing values and duplicates
- Temporal and spatial consistency

### 4. Train Models

```bash
# Train a single model
agrographnet train --model graphsage --epochs 100

# Train all models for comparison
agrographnet train --model all --epochs 150 --gpu
```

### 5. Make Predictions

```bash
agrographnet predict \
  --input data/new_farms.csv \
  --output results/predictions.csv \
  --model graphsage
```

### 6. Generate Analysis

```bash
agrographnet analyze \
  --predictions results/predictions.csv \
  --output results/analysis_report.html
```

## 📊 Data Preparation

### Data Format Requirements

#### farms.csv
Contains farm location and characteristic information.

**Required columns:**
- `farm_id` (string): Unique farm identifier
- `latitude` (float): Farm latitude in decimal degrees
- `longitude` (float): Farm longitude in decimal degrees  
- `crop_type` (string): Type of crop grown

**Optional columns:**
- `farm_size` (float): Farm area in hectares
- `soil_type` (string): Soil classification
- `irrigation_type` (string): Irrigation method
- `elevation` (float): Elevation in meters

**Example:**
```csv
farm_id,latitude,longitude,crop_type,farm_size,soil_type
FARM_001,40.1234,-95.5678,corn,150.5,loam
FARM_002,40.2345,-95.6789,soybean,200.0,clay
FARM_003,40.3456,-95.7890,wheat,175.2,sandy_loam
```

#### weather.csv
Contains weather and climate data for each farm over time.

**Required columns:**
- `farm_id` (string): Farm identifier (must match farms.csv)
- `date` (datetime): Date in YYYY-MM-DD format
- `temperature` (float): Average temperature in Celsius
- `humidity` (float): Relative humidity percentage (0-100)
- `precipitation` (float): Precipitation in millimeters

**Optional columns:**
- `wind_speed` (float): Wind speed in m/s
- `wind_direction` (float): Wind direction in degrees (0-360)
- `solar_radiation` (float): Solar radiation in MJ/m²
- `pressure` (float): Atmospheric pressure in hPa

**Example:**
```csv
farm_id,date,temperature,humidity,precipitation,wind_speed
FARM_001,2023-01-01,15.2,65.0,2.5,8.3
FARM_001,2023-02-01,18.7,70.2,1.2,6.1
FARM_002,2023-01-01,14.8,68.5,3.1,7.9
```

#### satellite.csv
Contains vegetation indices derived from satellite imagery.

**Required columns:**
- `farm_id` (string): Farm identifier
- `date` (datetime): Date of satellite observation
- `ndvi` (float): Normalized Difference Vegetation Index (-1 to 1)

**Optional columns:**
- `evi` (float): Enhanced Vegetation Index
- `savi` (float): Soil Adjusted Vegetation Index
- `ndwi` (float): Normalized Difference Water Index
- `gci` (float): Green Chlorophyll Index

**Example:**
```csv
farm_id,date,ndvi,evi,savi,ndwi
FARM_001,2023-01-01,0.65,0.45,0.55,0.25
FARM_001,2023-02-01,0.72,0.52,0.62,0.18
FARM_002,2023-01-01,0.68,0.48,0.58,0.22
```

#### labels.csv
Contains disease occurrence labels for training.

**Required columns:**
- `farm_id` (string): Farm identifier
- `date` (datetime): Date of disease observation
- `disease_type` (string): Disease classification

**Optional columns:**
- `severity` (float): Disease severity score (0-1)
- `confidence` (float): Label confidence (0-1)

**Supported disease types:**
- `Healthy` - No disease detected
- `Blight` - Various blight diseases
- `Rust` - Rust diseases
- `Mosaic` - Viral mosaic diseases
- `Bacterial` - Bacterial infections

**Example:**
```csv
farm_id,date,disease_type,severity,confidence
FARM_001,2023-01-01,Healthy,0.0,0.95
FARM_002,2023-01-01,Blight,0.7,0.88
FARM_003,2023-01-01,Rust,0.4,0.82
```

### Data Quality Guidelines

1. **Temporal Alignment**: Ensure weather, satellite, and label data cover similar time periods
2. **Spatial Coverage**: All farms should have corresponding weather and satellite data
3. **Missing Values**: Minimize missing values; use interpolation where appropriate
4. **Outlier Detection**: Remove or flag obvious data errors
5. **Consistency**: Use consistent units and formats across all files
6. **Sample Size**: Minimum 50 farms recommended; 200+ farms for robust results

## 🔄 Project Workflow

### Complete Workflow Example

```bash
# 1. Create project
agrographnet init my_disease_study
cd my_disease_study

# 2. Add your data files to data/raw/
# (Copy your CSV files)

# 3. Validate data
agrographnet validate --fix

# 4. Train multiple models
agrographnet train --model all --epochs 200

# 5. Make predictions on new data
agrographnet predict \
  --input data/raw/new_farms.csv \
  --output results/predictions.csv \
  --model graphsage

# 6. Generate comprehensive analysis
agrographnet analyze \
  --predictions results/predictions.csv \
  --output results/report.html \
  --format html

# 7. Review results
# Open results/report.html in your browser
```

## 📋 CLI Commands Reference

### agrographnet init

Initialize a new AgroGraphNet project.

```bash
agrographnet init PROJECT_NAME [OPTIONS]

Options:
  --template TEXT    Project template (default: standard)
  --force           Overwrite existing project
  --help            Show help message
```

**Examples:**
```bash
# Basic project creation
agrographnet init wheat_study

# Force overwrite existing project
agrographnet init wheat_study --force
```

### agrographnet validate

Validate data format and quality.

```bash
agrographnet validate [OPTIONS]

Options:
  -d, --data-dir PATH   Data directory (default: data/raw)
  --fix                 Attempt to fix common issues
  --help               Show help message
```

**Examples:**
```bash
# Basic validation
agrographnet validate

# Validate specific directory
agrographnet validate --data-dir /path/to/data

# Auto-fix common issues
agrographnet validate --fix
```

### agrographnet train

Train Graph Neural Network models.

```bash
agrographnet train [OPTIONS]

Options:
  -c, --config PATH     Configuration file (default: config.yaml)
  -m, --model CHOICE    Model: gcn, graphsage, gat, all (default: graphsage)
  -e, --epochs INTEGER  Training epochs (default: 100)
  --gpu                Use GPU if available
  -o, --output-dir PATH Output directory (default: models)
  --help               Show help message
```

**Examples:**
```bash
# Train GraphSAGE model
agrographnet train --model graphsage --epochs 150

# Train all models with GPU
agrographnet train --model all --epochs 200 --gpu

# Custom configuration
agrographnet train --config custom_config.yaml
```

### agrographnet predict

Make predictions using trained models.

```bash
agrographnet predict [OPTIONS]

Options:
  -c, --config PATH      Configuration file (default: config.yaml)
  -m, --model CHOICE     Model: gcn, graphsage, gat (default: graphsage)
  -i, --input PATH       Input CSV file [required]
  -o, --output PATH      Output CSV file [required]
  --model-path PATH      Custom model file path
  --help                Show help message
```

**Examples:**
```bash
# Basic prediction
agrographnet predict -i new_data.csv -o predictions.csv

# Use specific model
agrographnet predict -i data.csv -o pred.csv --model gat

# Custom model file
agrographnet predict -i data.csv -o pred.csv --model-path models/custom_model.pth
```

### agrographnet analyze

Generate analysis reports and visualizations.

```bash
agrographnet analyze [OPTIONS]

Options:
  -c, --config PATH         Configuration file (default: config.yaml)
  -p, --predictions PATH    Predictions CSV file
  -o, --output PATH         Output file (default: results/analysis_report.html)
  --format CHOICE          Format: html, pdf, json (default: html)
  --help                   Show help message
```

**Examples:**
```bash
# Generate HTML report
agrographnet analyze --predictions results/predictions.csv

# Generate PDF report
agrographnet analyze -p pred.csv -o report.pdf --format pdf

# Generate JSON data
agrographnet analyze -p pred.csv -o data.json --format json
```

## ⚙️ Configuration

### Configuration File Structure

The `config.yaml` file controls all aspects of model training and prediction:

```yaml
# Data paths
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  graph_path: "data/graphs"
  labels_path: "data/labels"

# Model architecture
model:
  hidden_dim: 64        # Hidden layer dimensions
  num_layers: 3         # Number of GNN layers
  dropout: 0.2          # Dropout rate
  learning_rate: 0.001  # Learning rate
  num_heads: 4          # Attention heads (GAT only)

# Training parameters
training:
  epochs: 100           # Maximum training epochs
  batch_size: 32        # Batch size (for large graphs)
  validation_split: 0.2 # Validation data percentage
  early_stopping: 10    # Early stopping patience
  use_gpu: false        # Enable GPU training
  random_seed: 42       # Random seed for reproducibility

# Graph construction
graph:
  distance_threshold_km: 5.0  # Max distance for farm connections
  min_neighbors: 2            # Minimum neighbors per farm
  max_neighbors: 10           # Maximum neighbors per farm
  edge_features:              # Features for graph edges
    - "distance"
    - "elevation_diff"
    - "weather_similarity"

# Output paths
paths:
  models: "models"
  results: "results"
  logs: "logs"

# Disease classification
disease_classes:
  0: "Healthy"
  1: "Blight"
  2: "Rust"
  3: "Mosaic"
  4: "Bacterial"
```

### Configuration Tips

1. **Model Size**: Increase `hidden_dim` for complex datasets
2. **Training**: Use early stopping to prevent overfitting
3. **Graph**: Adjust distance threshold based on farm density
4. **GPU**: Enable for faster training on large datasets
5. **Seeds**: Set random seed for reproducible results

## 🧠 Model Selection

### Model Comparison

| Model | Strengths | Best For | Training Time |
|-------|-----------|----------|---------------|
| **GCN** | Fast, simple, interpretable | Small-medium datasets | Fast |
| **GraphSAGE** | Scalable, robust, versatile | Large datasets, diverse features | Medium |
| **GAT** | Attention mechanism, interpretable | Complex relationships | Slow |

### When to Use Each Model

#### GCN (Graph Convolutional Network)
- **Use when**: You have a small to medium dataset (<1000 farms)
- **Advantages**: Fast training, simple architecture, good baseline
- **Disadvantages**: May not capture complex patterns

#### GraphSAGE (Graph Sample and Aggregate)
- **Use when**: You have a large dataset or diverse feature types
- **Advantages**: Scalable, handles missing data well, robust performance
- **Disadvantages**: More complex than GCN

#### GAT (Graph Attention Network)
- **Use when**: You need interpretable results or have complex spatial relationships
- **Advantages**: Attention weights show important connections, handles irregular graphs
- **Disadvantages**: Slower training, more memory intensive

### Model Selection Guidelines

1. **Start with GraphSAGE** - Good default choice for most cases
2. **Use GCN for quick experiments** - Fast prototyping and baseline
3. **Try GAT for complex problems** - When you need interpretability
4. **Train all models** - Compare performance with `--model all`

## 📊 Interpreting Results

### Prediction Output

The prediction CSV contains:

```csv
farm_id,latitude,longitude,crop_type,predicted_disease,confidence,risk_score
FARM_001,40.123,-95.456,corn,Healthy,0.92,0.08
FARM_002,40.234,-95.567,soybean,Blight,0.85,0.85
FARM_003,40.345,-95.678,wheat,Rust,0.78,0.78
```

**Columns explained:**
- `predicted_disease`: Most likely disease class
- `confidence`: Model confidence in prediction (0-1)
- `risk_score`: Disease risk level (0=healthy, 1=high risk)

### Analysis Report Sections

#### 1. Summary Statistics
- Total farms analyzed
- Disease distribution
- High-risk farm count
- Average risk score

#### 2. Model Performance
- Training accuracy
- Validation accuracy
- Classification metrics (precision, recall, F1-score)
- Confusion matrix

#### 3. Spatial Analysis
- Disease clustering patterns
- Geographic distribution maps
- Spatial correlation analysis

#### 4. Temporal Analysis
- Seasonal disease patterns
- Weather correlation analysis
- Vegetation index trends

#### 5. Recommendations
- High-priority farms for monitoring
- Treatment recommendations
- Prevention strategies

### Key Metrics

#### Accuracy
- **Good**: >80%
- **Excellent**: >90%
- **Poor**: <70%

#### Confidence Scores
- **High confidence**: >0.8
- **Medium confidence**: 0.6-0.8
- **Low confidence**: <0.6

#### Risk Scores
- **Low risk**: 0.0-0.3
- **Medium risk**: 0.3-0.7
- **High risk**: 0.7-1.0

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Data Validation Errors

**Problem**: Missing required columns
```
❌ Missing required columns for farms: {'latitude', 'longitude'}
```

**Solution**: Ensure all required columns are present in your CSV files.

#### 2. Memory Errors

**Problem**: Out of memory during training
```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce `batch_size` in config.yaml
- Reduce `hidden_dim` or `num_layers`
- Use CPU instead of GPU
- Process data in smaller chunks

#### 3. Poor Model Performance

**Problem**: Low accuracy (<70%)

**Solutions**:
- Check data quality and completeness
- Increase training epochs
- Adjust model architecture
- Add more training data
- Feature engineering

#### 4. Graph Construction Issues

**Problem**: No graph connections found
```
⚠️ No spatial connections found, creating chain graph
```

**Solutions**:
- Increase `distance_threshold_km`
- Check coordinate data quality
- Verify coordinate system (should be WGS84)

#### 5. Prediction Errors

**Problem**: Model file not found
```
❌ Model file not found: models/graphsage_best_model.pth
```

**Solution**: Train a model first using `agrographnet train`

### Getting Help

1. **Check logs**: Look in `logs/agrographnet.log` for detailed error messages
2. **Validate data**: Run `agrographnet validate --fix` to identify issues
3. **GitHub Issues**: Report bugs at https://github.com/yourusername/agrographnet/issues
4. **Documentation**: Check the full documentation at https://agrographnet.readthedocs.io

## ✅ Best Practices

### Data Collection

1. **Temporal Coverage**: Collect data over multiple growing seasons
2. **Spatial Distribution**: Ensure good geographic coverage
3. **Data Quality**: Regular quality checks and validation
4. **Consistent Timing**: Align satellite and weather data collection
5. **Ground Truth**: Verify disease labels with field experts

### Model Training

1. **Start Simple**: Begin with default parameters
2. **Cross-Validation**: Use multiple train/validation splits
3. **Early Stopping**: Prevent overfitting with patience
4. **Model Comparison**: Train multiple architectures
5. **Regular Retraining**: Update models with new data

### Production Deployment

1. **Version Control**: Track model versions and performance
2. **Monitoring**: Monitor prediction quality over time
3. **Feedback Loop**: Collect user feedback for improvements
4. **Documentation**: Document model assumptions and limitations
5. **Backup**: Maintain backup models and data

### Performance Optimization

1. **GPU Usage**: Use GPU for large datasets
2. **Batch Processing**: Process multiple farms together
3. **Feature Selection**: Remove irrelevant features
4. **Data Preprocessing**: Cache processed data
5. **Model Pruning**: Remove unnecessary model complexity

## 📞 Support and Community

### Getting Support

- **Email**: contact@agrographnet.com
- **GitHub Issues**: https://github.com/yourusername/agrographnet/issues
- **Documentation**: https://agrographnet.readthedocs.io
- **Community Forum**: https://community.agrographnet.com

### Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Citation

If you use AgroGraphNet in your research, please cite:

```bibtex
@software{agrographnet2024,
  title={AgroGraphNet: Graph Neural Networks for Agricultural Disease Prediction},
  author={AgroGraphNet Team},
  year={2024},
  url={https://github.com/yourusername/agrographnet}
}
```