import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

@dataclass
class DataConfig:
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    graph_path: str = "data/graphs"
    labels_path: str = "data/labels"
    
@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    num_heads: int = 4  # For GAT models
    
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping: int = 10
    use_gpu: bool = False
    random_seed: int = 42
    
@dataclass
class GraphConfig:
    distance_threshold_km: float = 5.0
    min_neighbors: int = 2
    max_neighbors: int = 10
    edge_features: List[str] = None
    
    def __post_init__(self):
        if self.edge_features is None:
            self.edge_features = ['distance', 'elevation_diff', 'weather_similarity']

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    graph: GraphConfig
    paths: Dict[str, str]
    disease_classes: Dict[int, str] = None
    
    def __post_init__(self):
        if self.disease_classes is None:
            self.disease_classes = {
                0: 'Healthy',
                1: 'Blight', 
                2: 'Rust',
                3: 'Mosaic',
                4: 'Bacterial'
            }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Create default config if it doesn't exist
            default_config = cls.create_default()
            default_config.save(config_path)
            return default_config
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            graph=GraphConfig(**config_dict.get('graph', {})),
            paths=config_dict.get('paths', {}),
            disease_classes=config_dict.get('disease_classes', None)
        )
    
    @classmethod
    def create_default(cls) -> 'Config':
        """Create default configuration"""
        return cls(
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            graph=GraphConfig(),
            paths={
                'models': 'models',
                'results': 'results',
                'logs': 'logs'
            }
        )
    
    def save(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'graph': asdict(self.graph),
            'paths': self.paths,
            'disease_classes': self.disease_classes
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
            else:
                # Handle nested updates
                parts = key.split('.')
                if len(parts) == 2:
                    section, param = parts
                    if hasattr(self, section):
                        section_obj = getattr(self, section)
                        if hasattr(section_obj, param):
                            setattr(section_obj, param, value)