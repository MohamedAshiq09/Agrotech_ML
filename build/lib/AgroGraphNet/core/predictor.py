import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from .models.gnn_models import create_model
from .data_loader import DataLoader
from .graph_builder import GraphBuilder
from torch_geometric.data import Data
from ..utils.logger import get_logger

class Predictor:
    """Prediction pipeline for trained AgroGraphNet models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.training.use_gpu else 'cpu')
        
        self.model = None
        self.model_metadata = None
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.graph_builder = GraphBuilder(config)
    
    def load_model(self, model_path: str, model_type: str = None):
        """Load a trained model"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Loading model from: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model_metadata = checkpoint
        
        # Determine model type
        if model_type is None:
            model_type = checkpoint.get('model_type', 'graphsage')
        
        # Create model architecture
        self.model = create_model(
            model_type=model_type,
            input_dim=checkpoint.get('input_dim', 10),  # Will be updated based on data
            hidden_dim=checkpoint.get('hidden_dim', 64),
            output_dim=checkpoint.get('output_dim', 5),
            num_layers=checkpoint.get('config', self.config).model.num_layers,
            dropout=checkpoint.get('config', self.config).model.dropout
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Model loaded successfully. Test accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
    
    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model()")
        
        self.logger.info(f"Making predictions on {len(input_data)} samples")
        
        # Prepare data for prediction
        prediction_data = self._prepare_prediction_data(input_data)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(
                prediction_data.x,
                prediction_data.edge_index
            )
            
            # Get probabilities and predictions
            probabilities = torch.exp(outputs)  # Convert from log_softmax
            predictions = outputs.argmax(dim=1)
            confidence_scores = probabilities.max(dim=1)[0]
            
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            confidence_np = confidence_scores.cpu().numpy()
        
        # Convert predictions to disease names
        disease_names = [self.config.disease_classes[pred] for pred in predictions_np]
        
        # Calculate risk scores (1 - probability of healthy)
        healthy_class_id = 0  # Assuming 'Healthy' is class 0
        risk_scores = 1 - probabilities_np[:, healthy_class_id]
        
        results = {
            'predictions': predictions_np,
            'disease_names': disease_names,
            'probabilities': probabilities_np,
            'confidence': confidence_np,
            'risk_score': risk_scores,
            'class_names': list(self.config.disease_classes.values())
        }
        
        self.logger.info("Predictions completed successfully")
        return results
    
    def _prepare_prediction_data(self, input_data: pd.DataFrame) -> Data:
        """Prepare input data for prediction"""
        
        # If input data doesn't have all required columns, we need to handle it
        # For now, assume it has the same structure as training data
        
        # Handle missing columns by filling with defaults or means
        required_features = self._get_expected_features()
        
        processed_data = input_data.copy()
        
        # Add missing columns with default values
        for feature in required_features:
            if feature not in processed_data.columns:
                if feature in ['latitude', 'longitude']:
                    # Use center coordinates as default
                    processed_data[feature] = 0.0
                elif feature.endswith('_mean') or feature.endswith('_std'):
                    processed_data[feature] = 0.0
                else:
                    processed_data[feature] = 0.0
        
        # Select only the required features
        feature_data = processed_data[required_features]
        
        # Handle categorical variables
        categorical_cols = feature_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            feature_data[col] = pd.Categorical(feature_data[col]).codes
        
        # Fill missing values
        feature_data = feature_data.fillna(feature_data.mean())
        
        # Build graph structure
        graph_data = self.graph_builder.build_graph(feature_data)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=torch.FloatTensor(feature_data.values),
            edge_index=torch.LongTensor(graph_data['edge_index']),
            edge_attr=torch.FloatTensor(graph_data['edge_attr']) if 'edge_attr' in graph_data else None
        )
        
        return data.to(self.device)
    
    def _get_expected_features(self) -> List[str]:
        """Get list of expected feature names from training"""
        # This would ideally be saved with the model
        # For now, return a basic set of expected features
        
        basic_features = [
            'latitude', 'longitude', 'crop_type',
            'temperature_mean', 'humidity_mean', 'precipitation_sum',
            'ndvi_mean', 'ndvi_std'
        ]
        
        return basic_features
    
    def predict_batch(self, data_files: List[str]) -> Dict[str, Any]:
        """Make predictions on multiple data files"""
        all_results = {}
        
        for file_path in data_files:
            self.logger.info(f"Processing file: {file_path}")
            
            # Load data
            input_data = pd.read_csv(file_path)
            
            # Make predictions
            results = self.predict(input_data)
            
            # Store results
            file_name = Path(file_path).stem
            all_results[file_name] = results
        
        return all_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_type": self.model_metadata.get('model_type', 'Unknown'),
            "test_accuracy": self.model_metadata.get('test_accuracy', 'N/A'),
            "input_dim": self.model_metadata.get('input_dim', 'N/A'),
            "output_dim": self.model_metadata.get('output_dim', 'N/A'),
            "hidden_dim": self.model_metadata.get('hidden_dim', 'N/A'),
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        return info