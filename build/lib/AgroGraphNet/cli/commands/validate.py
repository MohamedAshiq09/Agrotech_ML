import click
import pandas as pd
from pathlib import Path
from ...utils.validation import validate_data_format, check_data_completeness
from ...utils.logger import setup_logger

@click.command()
@click.option('--data-dir', '-d', default='data/raw', 
              help='Directory containing data files')
@click.option('--fix', is_flag=True, 
              help='Attempt to fix common data issues')
def validate(data_dir, fix):
    """Validate user data format and completeness"""
    logger = setup_logger()
    
    data_path = Path(data_dir)
    if not data_path.exists():
        click.echo(f"❌ Data directory not found: {data_path}")
        click.echo("Please run 'agrographnet init <project_name>' first")
        raise click.Abort()
    
    click.echo(f"🔍 Validating data in: {data_path}")
    
    # Expected files
    expected_files = {
        'farms.csv': 'farms',
        'weather.csv': 'weather', 
        'satellite.csv': 'satellite',
        'labels.csv': 'labels'
    }
    
    validation_results = {}
    all_valid = True
    
    for filename, data_type in expected_files.items():
        file_path = data_path / filename
        
        click.echo(f"\n📋 Checking {filename}...")
        
        if not file_path.exists():
            click.echo(f"  ❌ File not found: {filename}")
            all_valid = False
            continue
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            click.echo(f"  📊 Loaded {len(df)} records")
            
            # Validate format
            is_valid = validate_data_format(df, data_type)
            
            if is_valid:
                click.echo(f"  ✅ Format validation passed")
                
                # Check completeness
                completeness = check_data_completeness(df, data_type)
                click.echo(f"  📈 Data completeness: {completeness['overall']:.1f}%")
                
                if completeness['overall'] < 80:
                    click.echo(f"  ⚠️  Warning: Low data completeness")
                    all_valid = False
                
                # Check for duplicates
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    click.echo(f"  ⚠️  Warning: {duplicates} duplicate records found")
                    if fix:
                        df_clean = df.drop_duplicates()
                        df_clean.to_csv(file_path, index=False)
                        click.echo(f"  🔧 Removed {duplicates} duplicates")
                
                validation_results[data_type] = {
                    'valid': True,
                    'records': len(df),
                    'completeness': completeness['overall'],
                    'duplicates': duplicates
                }
                
            else:
                click.echo(f"  ❌ Format validation failed")
                all_valid = False
                validation_results[data_type] = {'valid': False}
                
        except Exception as e:
            click.echo(f"  ❌ Error reading file: {e}")
            all_valid = False
            validation_results[data_type] = {'valid': False, 'error': str(e)}
    
    # Overall summary
    click.echo(f"\n{'='*50}")
    if all_valid:
        click.echo("✅ All data validation checks passed!")
        click.echo("You can now proceed with training:")
        click.echo("  agrographnet train --model graphsage")
    else:
        click.echo("❌ Data validation failed!")
        click.echo("Please fix the issues above before training.")
        
        # Provide helpful suggestions
        click.echo("\n💡 Common fixes:")
        click.echo("1. Ensure all required CSV files are present")
        click.echo("2. Check column names match the expected schema")
        click.echo("3. Verify date formats (YYYY-MM-DD)")
        click.echo("4. Remove or fill missing values")
        click.echo("5. Use --fix flag to auto-fix some issues")
    
    # Save validation report
    report_path = Path("validation_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    click.echo(f"\n📄 Detailed report saved to: {report_path}")
    
    if not all_valid:
        raise click.Abort()