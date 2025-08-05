# AgroGraphNet: Complete Setup and Running Guide

## 🚀 Quick Start

This guide will help you set up and run the complete AgroGraphNet project for crop disease prediction using Graph Neural Networks.

## 📋 Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- At least 4GB RAM
- 2GB free disk space

## 🛠️ Installation Steps

### 1. Set Up Python Environment

```bash
# Create virtual environment
python -m venv agro_env

# Activate virtual environment
# On Windows:
agro_env\Scripts\activate
# On Linux/Mac:
source agro_env/bin/activate
```

### 2. Install Dependencies

```bash
# Navigate to project directory
cd AgroGraphNet

# Install all required packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test imports
python -c "import torch; import torch_geometric; import pandas; import numpy; print('All packages installed successfully!')"
```

## 📁 Project Structure

```
AgroGraphNet/
├── notebooks/                          # Jupyter notebooks (run in order)
│   ├── 01_data_collection.ipynb        # Data setup and exploration
│   ├── 02_data_preprocessing.ipynb     # Data cleaning and processing
│   ├── 03_graph_construction.ipynb     # Build farm network graphs
│   ├── 04_feature_engineering.ipynb    # Create ML features
│   ├── 05_model_development.ipynb      # Train GNN models
│   ├── 06_evaluation_visualization.ipynb # Analyze results
│   └── 07_prediction_pipeline.ipynb    # Real-time predictions
├── src/                                # Utility functions
│   ├── config.py                       # Configuration settings
│   ├── data_utils.py                   # Data processing utilities
│   ├── graph_utils.py                  # Graph construction utilities
│   ├── model_utils.py                  # GNN model definitions
│   └── visualization.py                # Plotting and mapping functions
├── data/                               # Dataset storage (created automatically)
├── models/                             # Saved model checkpoints
├── results/                            # Output visualizations and reports
└── requirements.txt                    # Python dependencies
```

## 🏃‍♂️ How to Run the Project

### Step 1: Start Jupyter Notebook

```bash
# Make sure you're in the AgroGraphNet directory and virtual environment is activated
jupyter notebook
```

This will open Jupyter in your web browser.

### Step 2: Run Notebooks in Order

**IMPORTANT: Run the notebooks in the exact order listed below!**

#### 1. Data Collection (`01_data_collection.ipynb`)
- **Purpose**: Sets up data directories and creates sample datasets
- **Runtime**: 2-3 minutes
- **What it does**:
  - Creates project directory structure
  - Generates sample farm locations, weather data, and disease labels
  - Performs initial data exploration
  - Creates interactive maps

**To run**: Open the notebook and click "Run All" or run each cell individually.

#### 2. Data Preprocessing (`02_data_preprocessing.ipynb`)
- **Purpose**: Cleans and processes raw data
- **Runtime**: 3-5 minutes
- **What it does**:
  - Processes satellite imagery features
  - Calculates vegetation indices (NDVI, EVI, etc.)
  - Cleans weather and disease data
  - Creates comprehensive feature matrix

#### 3. Graph Construction (`03_graph_construction.ipynb`)
- **Purpose**: Builds farm network graphs
- **Runtime**: 2-4 minutes
- **What it does**:
  - Creates spatial adjacency matrices
  - Calculates environmental similarity
  - Builds NetworkX graphs
  - Converts to PyTorch Geometric format

#### 4. Feature Engineering (`04_feature_engineering.ipynb`)
- **Purpose**: Engineers features for machine learning
- **Runtime**: 3-5 minutes
- **What it does**:
  - Creates advanced node and edge features
  - Performs feature selection
  - Analyzes feature importance
  - Prepares final feature matrices

#### 5. Model Development (`05_model_development.ipynb`)
- **Purpose**: Trains GNN models
- **Runtime**: 10-20 minutes (depending on hardware)
- **What it does**:
  - Trains baseline models (Random Forest, SVM, etc.)
  - Trains GNN models (GCN, GraphSAGE, GAT)
  - Compares model performance
  - Saves best model

#### 6. Evaluation & Visualization (`06_evaluation_visualization.ipynb`)
- **Purpose**: Analyzes results and creates visualizations
- **Runtime**: 5-8 minutes
- **What it does**:
  - Creates comprehensive performance analysis
  - Generates spatial prediction maps
  - Analyzes model interpretability
  - Creates interactive dashboards

#### 7. Prediction Pipeline (`07_prediction_pipeline.ipynb`)
- **Purpose**: Real-time prediction system
- **Runtime**: 2-3 minutes
- **What it does**:
  - Loads trained models
  - Simulates real-time data
  - Makes predictions
  - Generates alerts and reports

## 📊 Expected Outputs

After running all notebooks, you'll have:

### Generated Files:
- **Interactive Maps**: `results/*.html` files
- **Model Files**: `models/best_*_model.pth`
- **Predictions**: `results/spatial_predictions.csv`
- **Visualizations**: `results/*.png` files
- **Reports**: `results/final_report.md`

### Key Results:
- **Model Accuracy**: Typically 75-85% for disease classification
- **Interactive Maps**: Showing farm locations, disease predictions, and risk levels
- **Performance Comparisons**: GNN vs baseline models
- **Feature Importance**: Which factors most influence disease prediction

## 🔧 Troubleshooting

### Common Issues and Solutions:

#### 1. Import Errors
```bash
# If you get import errors, reinstall packages:
pip install --upgrade torch torch-geometric
pip install --force-reinstall -r requirements.txt
```

#### 2. Memory Issues
```python
# If you run out of memory, reduce batch size in src/config.py:
MODEL_CONFIG = {
    'batch_size': 16,  # Reduce from 32 to 16
    # ... other settings
}
```

#### 3. CUDA Issues
```python
# If CUDA is not available, the code will automatically use CPU
# To force CPU usage, add this to notebook cells:
import torch
torch.cuda.is_available = lambda: False
```

#### 4. Jupyter Kernel Issues
```bash
# If kernel dies, restart and run:
pip install ipykernel
python -m ipykernel install --user --name=agro_env
```

#### 5. File Not Found Errors
- Make sure you run notebooks in order
- Check that the virtual environment is activated
- Verify you're in the correct directory

## 📈 Performance Expectations

### Hardware Requirements:
- **Minimum**: 4GB RAM, CPU-only
- **Recommended**: 8GB RAM, GPU (optional)

### Runtime Estimates:
- **Total Runtime**: 30-45 minutes for all notebooks
- **Model Training**: 10-20 minutes (notebook 05)
- **Data Processing**: 15-20 minutes (notebooks 01-04)
- **Visualization**: 5-10 minutes (notebooks 06-07)

### Expected Accuracy:
- **Baseline Models**: 65-75%
- **GNN Models**: 75-85%
- **Best Model**: Usually GraphSAGE or GAT

## 🎯 Understanding the Results

### Key Metrics:
- **Accuracy**: Overall correct predictions
- **Precision/Recall**: Per-disease performance
- **F1-Score**: Balanced performance measure
- **Confidence**: Model certainty in predictions

### Interpreting Maps:
- **Green**: Healthy/Low risk farms
- **Orange**: Medium risk farms
- **Red**: High risk/Diseased farms

### Using Predictions:
- Check `results/07_current_predictions.csv` for detailed predictions
- Review `results/07_current_alerts.csv` for actionable alerts
- Open `results/07_prediction_dashboard.html` for interactive exploration

## 🔄 Running with Your Own Data

To use your own data instead of sample data:

1. **Farm Locations**: Place CSV in `data/raw/farm_locations/`
   - Required columns: `farm_id`, `lat`, `lon`, `crop_type`, `area_hectares`

2. **Weather Data**: Place CSV in `data/raw/weather/`
   - Required columns: `date`, `lat`, `lon`, `temperature`, `humidity`, `precipitation`, `wind_speed`

3. **Disease Data**: Place CSV in `data/raw/disease_labels/`
   - Required columns: `date`, `lat`, `lon`, `farm_id`, `disease_type`, `severity`

4. **Satellite Imagery**: Place GeoTIFF files in `data/raw/satellite/`
   - Sentinel-2 format recommended

## 📞 Support

If you encounter issues:

1. **Check Error Messages**: Read the full error message carefully
2. **Verify Environment**: Ensure virtual environment is activated
3. **Check Dependencies**: Run `pip list` to verify installations
4. **Restart Kernel**: In Jupyter, go to Kernel → Restart & Clear Output
5. **Run in Order**: Make sure you've run previous notebooks successfully

## 🎉 Success Indicators

You'll know the project is working correctly when:

✅ All notebooks run without errors
✅ Interactive maps are generated in `results/`
✅ Model accuracy is above 70%
✅ Prediction dashboard shows farm locations with risk levels
✅ Final report is generated with comprehensive results

## 📝 Next Steps

After successfully running the project:

1. **Experiment**: Try different model parameters in `src/config.py`
2. **Extend**: Add new features or data sources
3. **Deploy**: Use the prediction pipeline for real-time monitoring
4. **Validate**: Compare predictions with actual field observations
5. **Scale**: Apply to larger geographic regions

---

**Happy Farming with AI! 🌾🤖**