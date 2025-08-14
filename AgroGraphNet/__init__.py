"""
AgroGraphNet: Graph Neural Networks for Agricultural Disease Prediction
"""
from .__version__ import __version__
from .core.config import Config
from .core.predictor import Predictor
from .core.trainer import Trainer

__all__ = ['__version__', 'Config', 'Predictor', 'Trainer']