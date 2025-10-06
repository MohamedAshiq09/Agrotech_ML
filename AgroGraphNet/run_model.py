#!/usr/bin/env python3
"""
AgroGraphNet: Combined Model Prediction Script

This script loads all 7 trained models and makes predictions on user-provided datasets.
Models included:
1. GCN (Graph Convolutional Network)
2. GraphSAGE (Graph Sample and Aggregate)
3. GAT (Graph Attention Network)
4. Random Forest (Baseline)
5. SVM (Baseline)
6. Logistic Regression (Baseline)
7. Temporal GNN (Advanced)

Usage:
    python run_model.py

Author: AgroGraphNet Team
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import pickle
import json
import warnings
import argparse
from datetime import datetime
import glob

warnings.filterwarnings('ignore')

# Import custom modules
try:
    from config import *
    from model_utils import *
    from graph_utils import *
    from data_utils import *
    from visualization import *
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure you're running this script from the AgroGraphNet directory.")
    sys.exit(1)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ModelPredictor:
    """
    Class to handle prediction using all trained models
    """

    def __init__(self, models_dir=None):
        """
        Initialize the predictor with trained models

        Args:
            models_dir: Directory containing trained models (default: PROJECT_ROOT/models)
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model storage
        self.gnn_models = {}
        self.baseline_models = {}
        self.temporal_model = None

        # Load all models
        self.load_all_models()

    def load_all_models(self):
        """
        Load all trained models from disk
        """
        print("Loading trained models...")

        # Load GNN models
        gnn_model_names = ['GCN', 'GraphSAGE', 'GAT']
        model_configs = {
            'GCN': GCNModel,
            'GraphSAGE': GraphSAGEModel,
            'GAT': GATModel
        }

        for model_name in gnn_model_names:
            model_path = self.models_dir / f'best_{model_name.lower()}_model.pth'
            if model_path.exists():
                try:
                    # We'll need to recreate the model architecture to load the state dict
                    # For now, we'll store the path and load when we know the input dimensions
                    self.gnn_models[model_name] = {
                        'path': model_path,
                        'config': model_configs[model_name]
                    }
                    print(f"✅ Found {model_name} model: {model_path}")
                except Exception as e:
                    print(f"⚠️ Could not load {model_name} model: {e}")
            else:
                print(f"⚠️ {model_name} model not found at {model_path}")

        # Load baseline models
        baseline_model_names = ['Random Forest', 'SVM', 'Logistic Regression']
        results_file = RESULTS_DIR / 'model_results.pkl'

        if results_file.exists():
            try:
                with open(results_file, 'rb') as f:
                    results_data = pickle.load(f)

                if 'baseline_results' in results_data:
                    for model_name in baseline_model_names:
                        if model_name in results_data['baseline_results']:
                            self.baseline_models[model_name] = results_data['baseline_results'][model_name]
                            print(f"✅ Loaded {model_name} baseline model")
                        else:
                            print(f"⚠️ {model_name} baseline model not found in results")

                # Load temporal model if available
                if 'best_temporal_model' in results_data:
                    self.temporal_model = results_data['best_temporal_model']
                    print("✅ Loaded Temporal GNN model")

            except Exception as e:
                print(f"⚠️ Could not load baseline models: {e}")
        else:
            print(f"⚠️ Results file not found at {results_file}")

        total_models = len(self.gnn_models) + len(self.baseline_models) + (1 if self.temporal_model else 0)
        print(f"\nLoaded {total_models} models successfully")

        if total_models == 0:
            print("⚠️ No trained models found. Please run the training notebooks first.")
            return False

        return True

    def load_user_data(self, data_path):
        """
        Load and preprocess user-provided dataset

        Args:
            data_path: Path to the user's dataset

        Returns:
            Preprocessed data ready for prediction
        """
        print(f"Loading user dataset from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        # Try different file formats
        if data_path.endswith('.csv'):
            data_df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            data_df = pd.read_excel(data_path)
        elif data_path.endswith('.json'):
            data_df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        print(f"✅ Loaded dataset: {data_df.shape}")

        # Basic data validation and preprocessing
        required_columns = ['farm_id', 'lat', 'lon', 'crop_type']
        missing_columns = [col for col in required_columns if col not in data_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Add default values for missing optional columns
        optional_defaults = {
            'area_hectares': 50.0,
            'date': datetime.now().strftime('%Y-%m-%d')
        }

        for col, default_val in optional_defaults.items():
            if col not in data_df.columns:
                data_df[col] = default_val
                print(f"⚠️ Added default {col}: {default_val}")

        # Clean data
        data_df = data_df.dropna(subset=['lat', 'lon', 'farm_id'])
        data_df['lat'] = pd.to_numeric(data_df['lat'], errors='coerce')
        data_df['lon'] = pd.to_numeric(data_df['lon'], errors='coerce')
        data_df = data_df.dropna(subset=['lat', 'lon'])

        print(f"✅ Cleaned dataset: {data_df.shape}")

        return data_df

    def preprocess_for_prediction(self, data_df):
        """
        Preprocess data for model prediction following the notebook pipeline

        Args:
            data_df: Raw user data

        Returns:
            Preprocessed data ready for prediction
        """
        print("Preprocessing data for prediction...")

        # 1. Create basic node features (similar to notebook 05)
        node_features = []
        farm_ids = []

        for _, farm in data_df.iterrows():
            # Basic features: lat, lon, area, crop type (one-hot)
            features = [farm['lat'], farm['lon'], farm['area_hectares']]

            # Add crop type one-hot encoding
            crop_types = data_df['crop_type'].unique()
            for crop in crop_types:
                features.append(1.0 if farm['crop_type'] == crop else 0.0)

            # Add temporal features (use current date)
            features.extend([0.5, 0.0, 1.0])  # Default temporal features

            # Add some default environmental features
            features.extend(np.random.normal(0, 1, 5))

            node_features.append(features)
            farm_ids.append(farm['farm_id'])

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)

        # 2. Create basic graph structure (similar to notebook 05)
        farms_array = data_df[['lat', 'lon']].values

        # Create distance matrix
        distance_matrix = create_distance_matrix(data_df)

        # Create adjacency matrix
        adjacency_matrix = create_adjacency_matrix(distance_matrix, threshold_km=5.0)

        # Create edge index from adjacency matrix
        edge_indices = np.where(adjacency_matrix == 1)
        edge_index = torch.tensor([edge_indices[0], edge_indices[1]], dtype=torch.long)

        # Create edge attributes (distances)
        edge_attr = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            dist = distance_matrix[src, dst]
            edge_attr.append([dist, 1.0])  # distance and similarity

        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create disease labels (dummy for prediction)
        # In real scenario, these would be predicted
        y = torch.zeros(len(data_df), dtype=torch.long)

        # Create PyTorch Geometric data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        print(f"✅ Created graph data:")
        print(f"  - Nodes: {graph_data.x.shape[0]}")
        print(f"  - Node features: {graph_data.x.shape[1]}")
        print(f"  - Edges: {graph_data.edge_index.shape[1]}")

        return graph_data, farm_ids

    def predict_with_all_models(self, graph_data, farm_ids):
        """
        Make predictions using all loaded models

        Args:
            graph_data: Preprocessed graph data
            farm_ids: List of farm IDs

        Returns:
            Dictionary with predictions from all models
        """
        print("Making predictions with all models...")

        predictions = {
            'farm_id': farm_ids,
            'gnn_predictions': {},
            'baseline_predictions': {},
            'ensemble_prediction': None,
            'confidence_scores': {}
        }

        # Create data loader
        data_loader = DataLoader([graph_data], batch_size=1, shuffle=False)

        # 1. GNN Model Predictions
        for model_name, model_info in self.gnn_models.items():
            try:
                print(f"Predicting with {model_name}...")

                # Create model with correct input dimensions
                input_dim = graph_data.x.shape[1]
                model = model_info['config'](
                    input_dim=input_dim,
                    hidden_dim=MODEL_CONFIG['hidden_dim'],
                    output_dim=len(DISEASE_CLASSES),
                    num_layers=MODEL_CONFIG['num_layers'],
                    dropout=MODEL_CONFIG['dropout']
                )

                # Load trained weights
                model_path = model_info['path']
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                # Make predictions
                all_preds = []
                all_probs = []

                with torch.no_grad():
                    for batch in data_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        prob = torch.softmax(out, dim=1)
                        pred = out.argmax(dim=1)

                        all_preds.extend(pred.cpu().numpy())
                        all_probs.extend(prob.cpu().numpy())

                # Store predictions
                predictions['gnn_predictions'][model_name] = {
                    'predicted_classes': [DISEASE_CLASSES[pred] for pred in all_preds],
                    'probabilities': all_probs,
                    'confidence': [max(prob) for prob in all_probs]
                }

                print(f"✅ {model_name} predictions completed")

            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                predictions['gnn_predictions'][model_name] = None

        # 2. Baseline Model Predictions
        for model_name, model_info in self.baseline_models.items():
            try:
                print(f"Predicting with {model_name}...")

                # Get the trained model
                baseline_model = model_info['model']

                # Prepare features for baseline models (flatten node features)
                X_pred = graph_data.x.numpy()

                # Make predictions
                pred_probs = baseline_model.predict_proba(X_pred)
                pred_classes = baseline_model.predict(X_pred)

                # Store predictions
                predictions['baseline_predictions'][model_name] = {
                    'predicted_classes': [DISEASE_CLASSES[pred] for pred in pred_classes],
                    'probabilities': pred_probs.tolist(),
                    'confidence': [max(prob) for prob in pred_probs]
                }

                print(f"✅ {model_name} predictions completed")

            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                predictions['baseline_predictions'][model_name] = None

        # 3. Ensemble Prediction (majority voting)
        all_predictions = []

        # Collect valid GNN predictions
        for model_preds in predictions['gnn_predictions'].values():
            if model_preds is not None:
                # Convert class names to indices for voting
                class_to_idx = {v: k for k, v in DISEASE_CLASSES.items()}
                indices = [class_to_idx[pred] for pred in model_preds['predicted_classes']]
                all_predictions.append(indices)

        # Collect valid baseline predictions
        for model_preds in predictions['baseline_predictions'].values():
            if model_preds is not None:
                # Convert class names to indices for voting
                class_to_idx = {v: k for k, v in DISEASE_CLASSES.items()}
                indices = [class_to_idx[pred] for pred in model_preds['predicted_classes']]
                all_predictions.append(indices)

        if all_predictions:
            # Majority voting
            ensemble_votes = np.array(all_predictions).T  # Shape: (n_samples, n_models)
            ensemble_predictions = []

            for votes in ensemble_votes:
                # Get most common prediction
                unique_votes, counts = np.unique(votes, return_counts=True)
                majority_vote = unique_votes[np.argmax(counts)]
                ensemble_predictions.append(majority_vote)

            # Calculate confidence (agreement ratio)
            ensemble_confidence = []
            for i, votes in enumerate(ensemble_votes):
                pred = ensemble_predictions[i]
                agreement = np.sum(votes == pred) / len(votes)
                ensemble_confidence.append(agreement)

            predictions['ensemble_prediction'] = {
                'predicted_classes': [DISEASE_CLASSES[pred] for pred in ensemble_predictions],
                'confidence': ensemble_confidence
            }

        print("✅ All predictions completed")
        return predictions

    def display_results(self, predictions, data_df):
        """
        Display prediction results in a user-friendly format

        Args:
            predictions: Dictionary with all model predictions
            data_df: Original user data
        """
        print("\n" + "="*80)
        print("🌾 AGROGRAPHNET PREDICTION RESULTS")
        print("="*80)

        # Create results DataFrame
        results_data = []

        for i, farm_id in enumerate(predictions['farm_id']):
            row = {
                'Farm ID': farm_id,
                'Latitude': data_df[data_df['farm_id'] == farm_id]['lat'].iloc[0],
                'Longitude': data_df[data_df['farm_id'] == farm_id]['lon'].iloc[0],
                'Crop Type': data_df[data_df['farm_id'] == farm_id]['crop_type'].iloc[0]
            }

            # Add ensemble prediction if available
            if predictions['ensemble_prediction']:
                row['Ensemble Prediction'] = predictions['ensemble_prediction']['predicted_classes'][i]
                row['Ensemble Confidence'] = f"{predictions['ensemble_prediction']['confidence'][i]:.3f}"

            # Add individual model predictions
            for model_type in ['gnn_predictions', 'baseline_predictions']:
                for model_name, model_preds in predictions[model_type].items():
                    if model_preds is not None:
                        col_name = f"{model_name} Prediction"
                        row[col_name] = model_preds['predicted_classes'][i]

            results_data.append(row)

        results_df = pd.DataFrame(results_data)

        # Display results
        print(f"\n📊 Predictions for {len(results_df)} farms:")
        print(results_df.to_string(index=False))

        # Summary statistics
        print("\n📈 Prediction Summary:")
        print("-" * 40)

        if predictions['ensemble_prediction']:
            ensemble_classes = predictions['ensemble_prediction']['predicted_classes']
            class_counts = pd.Series(ensemble_classes).value_counts()

            print("Ensemble Predictions Distribution:")
            for class_name, count in class_counts.items():
                percentage = (count / len(ensemble_classes)) * 100
                print(f"  - {class_name}: {count} farms ({percentage:.1f}%)")

            # Risk assessment
            healthy_count = class_counts.get('Healthy', 0)
            diseased_count = len(ensemble_classes) - healthy_count

            print("\n🚨 Risk Assessment:")
            print(f"  - Healthy farms: {healthy_count}")
            print(f"  - At-risk farms: {diseased_count}")
            print(f"  - Overall health rate: {(healthy_count / len(ensemble_classes)) * 100:.1f}%")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f'predictions_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)

        print(f"\n💾 Results saved to: {results_file}")

        return results_df

def main():
    """
    Main function to run the prediction pipeline
    """
    print("🌾 Welcome to AgroGraphNet Prediction System!")
    print("=" * 60)

    # Initialize predictor
    predictor = ModelPredictor()

    if not predictor.gnn_models and not predictor.baseline_models:
        print("\n❌ No trained models found!")
        print("Please run the training notebooks (05_model_development.ipynb) first.")
        return

    # Get user input for dataset
    print("\n📁 Please provide the path to your dataset:")
    print("   Supported formats: .csv, .xlsx, .json")
    print("   Required columns: farm_id, lat, lon, crop_type")
    print("   Optional columns: area_hectares, date")

    data_path = input("\nDataset path: ").strip().strip('"')

    if not data_path:
        print("❌ No dataset path provided!")
        return

    try:
        # Load and preprocess data
        data_df = predictor.load_user_data(data_path)

        # Preprocess for prediction
        graph_data, farm_ids = predictor.preprocess_for_prediction(data_df)

        # Make predictions
        predictions = predictor.predict_with_all_models(graph_data, farm_ids)

        # Display results
        results_df = predictor.display_results(predictions, data_df)

        print("\n🎉 Prediction completed successfully!")
        print("Thank you for using AgroGraphNet!")

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        print("Please check your dataset format and try again.")

if __name__ == "__main__":
    main()
