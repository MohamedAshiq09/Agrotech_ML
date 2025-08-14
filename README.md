# AgroGraphNet CLI Package

🌾 **Graph Neural Networks for Agricultural Disease Prediction**

AgroGraphNet is a professional CLI tool that enables agricultural researchers and practitioners to train and deploy Graph Neural Network models for crop disease prediction using their own datasets.

## 🚀 Quick Start

### Installation

```bash
pip install agrographnet
```

### Create Your First Project

```bash
# Create a new project
agrographnet init my_farm_project

# Navigate to project
cd my_farm_project

# Add your data files to data/raw/
# - farms.csv
# - weather.csv  
# - satellite.csv
# - labels.csv

# Validate your data
agrographnet validate

# Train models
agrographnet train --model graphsage --epochs 100

# Make predictions
agrographnet predict --input data/new_data.csv --output results/predictions.csv

# Generate analysis report
agrographnet analyze --output results/analysis_report.html
```

## 📊 Data Requirements

AgroGraphNet expects 4 CSV files in your `data/raw/` directory:

### 1. farms.csv - Farm Information
```csv
farm_id,latitude,longitude,crop_type,farm_size,soil_type
farm_001,40.123,-95.456,corn,150.5,loam
farm_002,40.234,-95.567,soybean,200.0,clay
```

**Required columns:** `farm_id`, `latitude`, `longitude`, `crop_type`  
**Optional columns:** `farm_size`, `soil_type`, `irrigation_type`

### 2. weather.csv - Weather Data
```csv
farm_id,date,temperature,humidity,precipitation,wind_speed
farm_001,2023-01-01,15.2,65.0,2.5,8.3
farm_001,2023-02-01,18.7,70.2,1.2,6.1
```

**Required columns:** `farm_id`, `date`, `temperature`, `humidity`, `precipitation`  
**Optional columns:** `wind_speed`, `wind_direction`, `solar_radiation`

### 3. satellite.csv - Vegetation Indices
```csv
farm_id,date,ndvi,evi,savi,ndwi
farm_001,2023-01-01,0.65,0.45,0.55,0.25
farm_001,2023-02-01,0.72,0.52,0.62,0.18
```

**Required columns:** `farm_id`, `date`, `ndvi`  
**Optional columns:** `evi`, `savi`, `ndwi`

### 4. labels.csv - Disease Labels (for training)
```csv
farm_id,date,disease_type,severity,confidence
farm_001,2023-01-01,Healthy,0.0,0.95
farm_002,2023-01-01,Blight,0.7,0.88
```

**Required columns:** `farm_id`, `date`, `disease_type`  
**Optional columns:** `severity`, `confidence`

## 🛠️ CLI Commands

### Initialize Project
```bash
agrographnet init <project_name> [--template standard] [--force]
```

### Validate Data
```bash
agrographnet validate [--data-dir data/raw] [--fix]
```

### Train Models
```bash
agrographnet train [OPTIONS]

Options:
  -c, --config PATH     Configuration file (default: config.yaml)
  -m, --model CHOICE    Model type: gcn, graphsage, gat, all (default: graphsage)
  -e, --epochs INTEGER  Number of training epochs (default: 100)
  --gpu                 Use GPU if available
  -o, --output-dir PATH Output directory for models (default: models)
```

### Make Predictions
```bash
agrographnet predict [OPTIONS]

Options:
  -c, --config PATH      Configuration file (default: config.yaml)
  -m, --model CHOICE     Model type: gcn, graphsage, gat (default: graphsage)
  -i, --input PATH       Input data file (CSV) [required]
  -o, --output PATH      Output predictions file (CSV) [required]
  --model-path PATH      Path to trained model file
```

### Generate Analysis
```bash
agrographnet analyze [OPTIONS]

Options:
  -c, --config PATH         Configuration file (default: config.yaml)
  -p, --predictions PATH    Predictions CSV file to analyze
  -o, --output PATH         Output report file (default: results/analysis_report.html)
  --format CHOICE           Output format: html, pdf, json (default: html)
```

## ⚙️ Configuration

AgroGraphNet uses YAML configuration files. A default `config.yaml` is created when you initialize a project:

```yaml
# Model parameters
model:
  hidden_dim: 64
  num_layers: 3
  dropout: 0.2
  learning_rate: 0.001

# Training parameters  
training:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  early_stopping: 10
  use_gpu: false

# Graph construction
graph:
  distance_threshold_km: 5.0
  min_neighbors: 2
  max_neighbors: 10

# Disease classes
disease_classes:
  0: "Healthy"
  1: "Blight" 
  2: "Rust"
  3: "Mosaic"
  4: "Bacterial"
```

## 🎯 Model Types

AgroGraphNet supports three Graph Neural Network architectures:

- **GCN** (Graph Convolutional Network): Fast and efficient for large graphs
- **GraphSAGE** (default): Scalable and robust, works well with diverse features  
- **GAT** (Graph Attention Network): Uses attention mechanisms for better interpretability

## 📈 Output Files

After running the pipeline, you'll find:

```
my_farm_project/
├── models/
│   ├── graphsage_best_model.pth    # Trained model
│   └── training_summary.txt        # Training metrics
├── results/
│   ├── predictions.csv             # Disease predictions
│   ├── analysis_report.html        # Comprehensive analysis
│   └── visualizations/             # Charts and maps
└── logs/
    └── agrographnet.log            # Execution logs
```

## 🔬 Example Use Cases

### Agricultural Research
- Disease spread modeling across regions
- Climate impact assessment on crop health
- Precision agriculture optimization

### Farm Management
- Early disease detection and warning systems
- Risk assessment for insurance
- Treatment prioritization

### Government/NGO
- Regional crop monitoring
- Food security assessment
- Agricultural policy planning

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- 📧 Email: contact@agrographnet.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/agrographnet/issues)
- 📖 Documentation: [Read the Docs](https://agrographnet.readthedocs.io)

## 🙏 Citation

If you use AgroGraphNet in your research, please cite:

```bibtex
@software{agrographnet2024,
  title={AgroGraphNet: Graph Neural Networks for Agricultural Disease Prediction},
  author={AgroGraphNet Team},
  year={2024},
  url={https://github.com/yourusername/agrographnet}
}
```