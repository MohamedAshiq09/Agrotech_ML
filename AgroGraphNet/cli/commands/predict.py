import click
import pandas as pd
from pathlib import Path
from ...core.predictor import Predictor
from ...core.config import Config
from ...utils.logger import setup_logger

@click.command()
@click.option('--config', '-c', type=str,
              default='config.yaml', help='Configuration file path')
@click.option('--model', '-m', default='graphsage',
              type=click.Choice(['gcn', 'graphsage', 'gat']),
              help='Model to use for prediction')
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input data file (CSV)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output predictions file (CSV)')
@click.option('--model-path', type=click.Path(exists=True),
              help='Path to trained model file')
def predict(config, model, input, output, model_path):
    """Make predictions on new data using trained models"""
    logger = setup_logger()
    
    try:
        # Load configuration (create default if doesn't exist)
        if not Path(config).exists():
            click.echo("⚠️  No config.yaml found. Creating default configuration...")
            config_obj = Config.create_default()
            config_obj.save(config)
        else:
            config_obj = Config.from_file(config)
        
        # Initialize predictor
        predictor = Predictor(config_obj)
        
        # Load the trained model
        if model_path:
            model_file = model_path
        else:
            model_file = Path(config_obj.paths.get('models', 'models')) / f"{model}_best_model.pth"
        
        if not Path(model_file).exists():
            click.echo(f"❌ Model file not found: {model_file}")
            click.echo("Please train a model first using: agrographnet train")
            raise click.Abort()
        
        click.echo(f"📂 Loading model: {model_file}")
        predictor.load_model(model_file, model)
        
        # Load input data
        click.echo(f"📊 Loading input data: {input}")
        input_data = pd.read_csv(input)
        
        # Make predictions
        click.echo(f"🔮 Making predictions...")
        predictions = predictor.predict(input_data)
        
        # Save predictions
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create output dataframe
        results_df = input_data.copy()
        results_df['predicted_disease'] = predictions['disease_names']
        results_df['confidence'] = predictions['confidence']
        results_df['risk_score'] = predictions['risk_score']
        
        results_df.to_csv(output_path, index=False)
        
        click.echo(f"✅ Predictions saved to: {output_path}")
        
        # Print summary
        disease_counts = pd.Series(predictions['disease_names']).value_counts()
        click.echo(f"\n📈 Prediction Summary:")
        for disease, count in disease_counts.items():
            percentage = (count / len(predictions['disease_names'])) * 100
            click.echo(f"  {disease}: {count} farms ({percentage:.1f}%)")
        
        avg_confidence = predictions['confidence'].mean()
        click.echo(f"\n🎯 Average Confidence: {avg_confidence:.3f}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"❌ Error: {e}")
        raise click.Abort()