import click
from pathlib import Path
from ...core.trainer import Trainer
from ...core.config import Config
from ...utils.logger import setup_logger

@click.command()
@click.option('--config', '-c', type=str,
              default='config.yaml', help='Configuration file path')
@click.option('--model', '-m', default='graphsage',
              type=click.Choice(['gcn', 'graphsage', 'gat', 'all']),
              help='Model architecture to train')
@click.option('--epochs', '-e', default=100, help='Number of training epochs')
@click.option('--gpu', is_flag=True, help='Use GPU if available')
@click.option('--output-dir', '-o', default='models', help='Output directory for trained models')
def train(config, model, epochs, gpu, output_dir):
    """Train AgroGraphNet models on your data"""
    logger = setup_logger()
    
    try:
        # Check if we're in an AgroGraphNet project
        if not Path(config).exists():
            click.echo("⚠️  No config.yaml found.")
            click.echo("💡 For best results, create a project first:")
            click.echo("   agrographnet init my_project")
            click.echo("   cd my_project")
            click.echo("   # Add your data to data/raw/")
            click.echo("   agrographnet train")
            click.echo("\n🔧 Creating default configuration for current directory...")
            
            # Create default config
            config_obj = Config.create_default()
            config_obj.save(config)
            click.echo(f"✅ Created default config at: {config}")
        else:
            # Load existing configuration
            config_obj = Config.from_file(config)
        config_obj.training.epochs = epochs
        config_obj.training.use_gpu = gpu
        
        # Set output directory
        config_obj.paths['models'] = output_dir
        
        # Check if data directory exists
        data_dir = Path(config_obj.data.raw_path)
        if not data_dir.exists():
            click.echo(f"📁 Creating data directory: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            click.echo("\n📋 To train models, you need to add your data files to:")
            click.echo(f"   {data_dir}/")
            click.echo("   - farms.csv (farm locations)")
            click.echo("   - weather.csv (weather data)")
            click.echo("   - satellite.csv (vegetation indices)")
            click.echo("   - labels.csv (disease labels)")
            click.echo("\n💡 Run 'agrographnet validate' to check your data format")
            return
        
        # Initialize trainer
        trainer = Trainer(config_obj)
        
        # Train model(s)
        if model == 'all':
            models = ['gcn', 'graphsage', 'gat']
        else:
            models = [model]
        
        results = {}
        for model_name in models:
            click.echo(f"🚀 Training {model_name.upper()} model...")
            
            # Train the model
            model_results = trainer.train(model_name)
            results[model_name] = model_results
            
            click.echo(f"✅ {model_name.upper()} training completed!")
            click.echo(f"   Accuracy: {model_results['test_accuracy']:.3f}")
            click.echo(f"   F1-Score: {model_results['classification_report']['weighted avg']['f1-score']:.3f}")
        
        click.echo(f"\n📁 Models saved to: {output_dir}")
        
        # Save training summary
        summary_path = Path(output_dir) / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("AgroGraphNet Training Summary\n")
            f.write("=" * 40 + "\n\n")
            for model_name, result in results.items():
                f.write(f"{model_name.upper()} Model:\n")
                f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n")
                f.write(f"  Best Val Accuracy: {result['best_val_accuracy']:.4f}\n")
                f.write(f"  Training Epochs: {len(result['train_losses'])}\n\n")
        
        click.echo(f"📊 Training summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"❌ Error: {e}")
        raise click.Abort()