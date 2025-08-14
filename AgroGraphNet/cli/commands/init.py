import click
import os
import shutil
from pathlib import Path
from ...utils.io import create_project_structure
from ...utils.logger import setup_logger

@click.command()
@click.argument('project_name')
@click.option('--template', default='standard', help='Project template to use')
@click.option('--force', is_flag=True, help='Overwrite existing project')
def init(project_name, template, force):
    """Initialize a new AgroGraphNet project"""
    logger = setup_logger()
    
    project_path = Path(project_name)
    
    if project_path.exists() and not force:
        click.echo(f"Error: Project '{project_name}' already exists. Use --force to overwrite.")
        return
    
    try:
        # Create project structure
        create_project_structure(project_path, template)
        
        # Copy template files
        copy_template_files(project_path, template)
        
        click.echo(f"✅ Successfully created AgroGraphNet project: {project_name}")
        click.echo(f"📁 Project location: {project_path.absolute()}")
        click.echo("\n📋 Next steps:")
        click.echo("1. Navigate to your project: cd " + project_name)
        click.echo("2. Add your data to the data/raw/ directory")
        click.echo("   - farms.csv (farm locations and characteristics)")
        click.echo("   - weather.csv (weather/climate data)")
        click.echo("   - satellite.csv (vegetation indices)")
        click.echo("   - labels.csv (disease labels for training)")
        click.echo("3. Validate your data: agrographnet validate")
        click.echo("4. Train models: agrographnet train")
        
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        click.echo(f"❌ Error: {e}")

def copy_template_files(project_path: Path, template: str):
    """Copy template files to the new project"""
    # Get template directory
    template_dir = Path(__file__).parent.parent.parent / "templates"
    
    # Copy config file
    config_src = template_dir / "config.yaml"
    config_dst = project_path / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, config_dst)
    
    # Copy data schema
    schema_src = template_dir / "data_schema.json"
    schema_dst = project_path / "data_schema.json"
    if schema_src.exists():
        shutil.copy2(schema_src, schema_dst)
    
    # Create README
    readme_content = f"""# {project_path.name}

AgroGraphNet project for crop disease prediction using Graph Neural Networks.

## Data Requirements

Place your CSV files in the `data/raw/` directory:

1. **farms.csv** - Farm locations and characteristics
   - Required columns: farm_id, latitude, longitude, crop_type
   - Optional columns: farm_size, soil_type, irrigation_type

2. **weather.csv** - Weather/climate data
   - Required columns: farm_id, date, temperature, humidity, precipitation
   - Optional columns: wind_speed, wind_direction, solar_radiation

3. **satellite.csv** - Vegetation indices from satellite data
   - Required columns: farm_id, date, ndvi
   - Optional columns: evi, savi, ndwi

4. **labels.csv** - Disease labels for training
   - Required columns: farm_id, date, disease_type
   - Optional columns: severity, confidence

## Usage

1. Validate your data:
   ```bash
   agrographnet validate
   ```

2. Train models:
   ```bash
   agrographnet train --model graphsage --epochs 100
   ```

3. Make predictions:
   ```bash
   agrographnet predict --input data/new_data.csv --output results/predictions.csv
   ```

4. Generate analysis:
   ```bash
   agrographnet analyze --output results/analysis_report.html
   ```
"""
    
    readme_path = project_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)