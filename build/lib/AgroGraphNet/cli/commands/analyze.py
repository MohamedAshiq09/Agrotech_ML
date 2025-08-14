import click
import pandas as pd
from pathlib import Path
from ...core.analyzer import Analyzer
from ...core.config import Config
from ...utils.logger import setup_logger

@click.command()
@click.option('--config', '-c', type=click.Path(),
              default='config.yaml', help='Configuration file path')
@click.option('--predictions', '-p', type=click.Path(exists=True),
              help='Predictions CSV file to analyze')
@click.option('--output', '-o', default='results/analysis_report.html',
              help='Output analysis report file')
@click.option('--format', 'output_format', default='html',
              type=click.Choice(['html', 'pdf', 'json']),
              help='Output format for the report')
def analyze(config, predictions, output, output_format):
    """Generate comprehensive analysis and visualization reports"""
    logger = setup_logger()
    
    try:
        # Load configuration (create default if doesn't exist)
        if not Path(config).exists():
            click.echo("⚠️  No config.yaml found. Creating default configuration...")
            config_obj = Config.create_default()
            config_obj.save(config)
        else:
            config_obj = Config.from_file(config)
        
        # Initialize analyzer
        analyzer = Analyzer(config_obj)
        
        click.echo("📊 Generating analysis report...")
        
        # Load data for analysis
        data_sources = {}
        
        # Load predictions if provided
        if predictions:
            click.echo(f"📂 Loading predictions: {predictions}")
            data_sources['predictions'] = pd.read_csv(predictions)
        
        # Load training results if available
        models_dir = Path(config_obj.paths.get('models', 'models'))
        if models_dir.exists():
            training_summary = models_dir / "training_summary.txt"
            if training_summary.exists():
                data_sources['training_summary'] = training_summary
        
        # Load raw data for additional insights
        raw_data_dir = Path(config_obj.data.raw_path)
        for filename in ['farms.csv', 'weather.csv', 'satellite.csv', 'labels.csv']:
            file_path = raw_data_dir / filename
            if file_path.exists():
                data_sources[filename.replace('.csv', '')] = pd.read_csv(file_path)
        
        # Generate analysis
        analysis_results = analyzer.generate_comprehensive_analysis(data_sources)
        
        # Create output directory
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report based on format
        if output_format == 'html':
            analyzer.create_html_report(analysis_results, output_path)
        elif output_format == 'pdf':
            analyzer.create_pdf_report(analysis_results, output_path)
        elif output_format == 'json':
            analyzer.create_json_report(analysis_results, output_path)
        
        click.echo(f"✅ Analysis report generated: {output_path}")
        
        # Print key insights
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            click.echo(f"\n📈 Key Insights:")
            
            if 'total_farms' in summary:
                click.echo(f"  🏭 Total Farms Analyzed: {summary['total_farms']}")
            
            if 'disease_distribution' in summary:
                click.echo(f"  🦠 Disease Distribution:")
                for disease, count in summary['disease_distribution'].items():
                    click.echo(f"    {disease}: {count}")
            
            if 'model_performance' in summary:
                click.echo(f"  🎯 Best Model Accuracy: {summary['model_performance']:.3f}")
            
            if 'high_risk_farms' in summary:
                click.echo(f"  ⚠️  High Risk Farms: {summary['high_risk_farms']}")
        
        # Generate additional visualizations
        viz_dir = output_path.parent / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        analyzer.create_visualizations(analysis_results, viz_dir)
        click.echo(f"📊 Visualizations saved to: {viz_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        click.echo(f"❌ Error: {e}")
        raise click.Abort()