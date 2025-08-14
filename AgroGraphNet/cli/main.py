import click
from .commands import init, train, predict, analyze, validate

@click.group()
@click.version_option()
def cli():
    """AgroGraphNet: Graph Neural Networks for Agricultural Disease Prediction
    
    A CLI tool for training and deploying GNN models on agricultural datasets.
    """
    pass

# Register commands
cli.add_command(init.init)
cli.add_command(train.train)
cli.add_command(predict.predict)
cli.add_command(analyze.analyze)
cli.add_command(validate.validate)

if __name__ == '__main__':
    cli()