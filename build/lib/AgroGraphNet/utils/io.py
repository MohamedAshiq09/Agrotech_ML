import os
import shutil
from pathlib import Path
from typing import Dict, List

def create_project_structure(project_path: Path, template: str = "standard"):
    """Create the directory structure for a new AgroGraphNet project"""
    
    # Define project structure
    directories = [
        "data/raw",
        "data/processed", 
        "data/graphs",
        "data/labels",
        "models",
        "results",
        "logs",
        "notebooks",
        "scripts"
    ]
    
    # Create project root
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for directory in directories:
        dir_path = project_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create .gitignore
    gitignore_content = """# Data files
data/raw/*.csv
data/processed/
data/graphs/
data/labels/

# Model files
models/*.pth
models/*.pkl

# Results
results/
logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    gitignore_path = project_path / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    # Create sample data directory structure info
    data_info = """# Data Directory Structure

Place your CSV files in the following locations:

## data/raw/
- farms.csv: Farm locations and characteristics
- weather.csv: Weather/climate data  
- satellite.csv: Vegetation indices from satellite data
- labels.csv: Disease labels for training

## Expected Data Formats

### farms.csv
Required columns: farm_id, latitude, longitude, crop_type
Optional columns: farm_size, soil_type, irrigation_type

### weather.csv  
Required columns: farm_id, date, temperature, humidity, precipitation
Optional columns: wind_speed, wind_direction, solar_radiation

### satellite.csv
Required columns: farm_id, date, ndvi
Optional columns: evi, savi, ndwi

### labels.csv
Required columns: farm_id, date, disease_type
Optional columns: severity, confidence

## Processed Data
The following directories will be populated automatically:
- data/processed/: Cleaned and preprocessed data
- data/graphs/: Graph structures for GNN models
- data/labels/: Processed labels and encodings
"""
    
    data_readme = project_path / "data" / "README.md"
    with open(data_readme, 'w') as f:
        f.write(data_info)

def copy_template_files(project_path: Path, template: str):
    """Copy template configuration files to project"""
    
    # This function is called from the CLI init command
    # Template files are copied there to avoid circular imports
    pass

def ensure_directory_exists(path: Path):
    """Ensure a directory exists, create if it doesn't"""
    path.mkdir(parents=True, exist_ok=True)

def safe_file_copy(src: Path, dst: Path, overwrite: bool = False):
    """Safely copy a file with overwrite protection"""
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file already exists: {dst}")
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def list_data_files(data_dir: Path) -> Dict[str, List[str]]:
    """List available data files in the project"""
    file_types = {
        'csv': [],
        'json': [],
        'yaml': [],
        'other': []
    }
    
    if not data_dir.exists():
        return file_types
    
    for file_path in data_dir.rglob('*'):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                file_types['csv'].append(str(file_path.relative_to(data_dir)))
            elif suffix == '.json':
                file_types['json'].append(str(file_path.relative_to(data_dir)))
            elif suffix in ['.yaml', '.yml']:
                file_types['yaml'].append(str(file_path.relative_to(data_dir)))
            else:
                file_types['other'].append(str(file_path.relative_to(data_dir)))
    
    return file_types