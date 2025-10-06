#!/usr/bin/env python3
"""
Fixed Model Training Script for AgroGraphNet

This script trains all models and saves them properly for the prediction system.
Run this instead of the notebook if you want to ensure all models are saved correctly.

Usage:
    python train_models.py
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

def main():
    """
    Main training function that saves all models properly
    """
    print("🌾 AgroGraphNet Model Training (Fixed Version)")
    print("=" * 60)

    # Create directories
    for dir_path in [MODELS_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load graph data (similar to notebook)
    print("Loading graph data...")

    graph_file = GRAPHS_DIR / 'farm_graphs.pkl'
    if graph_file.exists():
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)

        pytorch_graphs = graph_data['pytorch_graphs']
        farms_df = graph_data['farms_df']
        time_points = graph_data['time_points']

        print(f"✅ Loaded graph data: {len(pytorch_graphs)} graphs")

    else:
        print("⚠️ Graph data not found. Creating basic graphs from available data...")

        # Load basic data and create simple graphs
        farm_files = list(FARM_LOCATIONS_DIR.glob('*.csv'))
        if not farm_files:
            raise FileNotFoundError("No farm data found. Please run previous notebooks first.")

        farms_df = pd.read_csv(farm_files[0])

        # Create basic graphs
        pytorch_graphs = create_basic_graphs(farms_df)
        time_points = [f"2023-{i+1:02d}-01" for i in range(len(pytorch_graphs))]

        print(f"✅ Created basic graphs: {len(pytorch_graphs)} graphs")

    # Data Splitting
    print("Splitting data into train/validation/test sets...")

    # Temporal split: use first 60% for training, next 20% for validation, last 20% for testing
    n_time_points = len(pytorch_graphs)
    train_end = int(0.6 * n_time_points)
    val_end = int(0.8 * n_time_points)

    train_graphs = pytorch_graphs[:train_end]
    val_graphs = pytorch_graphs[train_end:val_end]
    test_graphs = pytorch_graphs[val_end:]

    print(f"Data split:")
    print(f"- Training: {len(train_graphs)} time points")
    print(f"- Validation: {len(val_graphs)} time points")
    print(f"- Testing: {len(test_graphs)} time points")

    # Create data loaders
    batch_size = MODEL_CONFIG['batch_size']

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    print(f"\n✅ Data loaders created with batch size: {batch_size}")

    # Train baseline models
    print("Training baseline models...")

    # Prepare data for baseline models (flatten temporal dimension)
    X_train = torch.cat([data.x for data in train_graphs]).numpy()
    y_train = torch.cat([data.y for data in train_graphs]).numpy()

    X_val = torch.cat([data.x for data in val_graphs]).numpy()
    y_val = torch.cat([data.y for data in val_graphs]).numpy()

    X_test = torch.cat([data.x for data in test_graphs]).numpy()
    y_test = torch.cat([data.y for data in test_graphs]).numpy()

    print(f"Baseline data shapes:")
    print(f"- X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"- X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"- X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Train baseline models
    baseline_results = create_baseline_models(X_train, y_train, X_test, y_test)

    print("\n✅ Baseline models trained")

    # Train GNN models
    print("Training GNN models...")

    # Model parameters
    input_dim = pytorch_graphs[0].x.shape[1]
    hidden_dim = MODEL_CONFIG['hidden_dim']
    output_dim = len(DISEASE_CLASSES)
    num_layers = MODEL_CONFIG['num_layers']
    dropout = MODEL_CONFIG['dropout']
    learning_rate = MODEL_CONFIG['learning_rate']
    num_epochs = MODEL_CONFIG['num_epochs']
    early_stopping_patience = MODEL_CONFIG['early_stopping_patience']

    print(f"Model configuration:")
    print(f"- Input dim: {input_dim}")
    print(f"- Hidden dim: {hidden_dim}")
    print(f"- Output dim: {output_dim}")
    print(f"- Num layers: {num_layers}")
    print(f"- Dropout: {dropout}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Epochs: {num_epochs}")

    # Dictionary to store results
    gnn_results = {}

    # Train GCN model
    print("\n" + "="*50)
    print("Training GCN Model")
    print("="*50)

    gcn_model = GCNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    gcn_results = train_and_evaluate(
        model=gcn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        device=device
    )

    gnn_results['GCN'] = gcn_results

    # Train GraphSAGE model
    print("\n" + "="*50)
    print("Training GraphSAGE Model")
    print("="*50)

    sage_model = GraphSAGEModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    sage_results = train_and_evaluate(
        model=sage_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        device=device
    )

    gnn_results['GraphSAGE'] = sage_results

    # Train GAT model
    print("\n" + "="*50)
    print("Training GAT Model")
    print("="*50)

    gat_model = GATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    gat_results = train_and_evaluate(
        model=gat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        device=device
    )

    gnn_results['GAT'] = gat_results

    print("\n✅ All GNN models trained successfully!")

    # Model Comparison
    print("Model Performance Comparison:")
    print("=" * 60)

    # Combine baseline and GNN results
    all_results = {}

    # Add baseline results
    for model_name, results in baseline_results.items():
        all_results[model_name] = {
            'test_accuracy': results['test_accuracy'],
            'model_type': 'Baseline'
        }

    # Add GNN results
    for model_name, results in gnn_results.items():
        all_results[model_name] = {
            'test_accuracy': results['test_accuracy'],
            'model_type': 'GNN'
        }

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {'Model': name, 'Test Accuracy': results['test_accuracy'], 'Type': results['model_type']}
        for name, results in all_results.items()
    ]).sort_values('Test Accuracy', ascending=False)

    print(comparison_df.to_string(index=False))

    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Test Accuracy']

    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    # Get best GNN model
    best_gnn = max(gnn_results.items(), key=lambda x: x[1]['test_accuracy'])
    best_gnn_name, best_gnn_results = best_gnn

    print(f"🥇 Best GNN Model: {best_gnn_name} (Accuracy: {best_gnn_results['test_accuracy']:.4f})")

    # SAVE ALL MODELS (This is the fix!)
    print("Saving all trained models...")

    # Save all trained GNN models individually
    for model_name, results in gnn_results.items():
        model_path = MODELS_DIR / f'best_{model_name.lower()}_model.pth'
        torch.save(results['model_state_dict'], model_path)
        print(f"✅ {model_name} model saved to {model_path}")

    # Save best GNN model (duplicate for compatibility)
    best_model_path = MODELS_DIR / f'best_{best_gnn_name.lower()}_model.pth'
    torch.save(best_gnn_results['model_state_dict'], best_model_path)
    print(f"✅ Best model saved to {best_model_path}")

    # Save all results
    results_data = {
        'baseline_results': baseline_results,
        'gnn_results': gnn_results,
        'best_model_name': best_gnn_name,
        'best_model_accuracy': best_gnn_results['test_accuracy'],
        'model_comparison': comparison_df.to_dict('records'),
        'model_config': MODEL_CONFIG,
        'graph_config': GRAPH_CONFIG
    }

    with open(RESULTS_DIR / 'model_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)

    print(f"✅ Results saved to {RESULTS_DIR / 'model_results.pkl'}")

    # Save model summary as JSON
    summary_data = {
        'best_model': best_gnn_name,
        'best_accuracy': float(best_gnn_results['test_accuracy']),
        'model_comparison': {
            name: float(results['test_accuracy'])
            for name, results in all_results.items()
        },
        'training_config': MODEL_CONFIG,
        'num_graphs': len(pytorch_graphs),
        'num_nodes': pytorch_graphs[0].x.shape[0],
        'node_features': pytorch_graphs[0].x.shape[1]
    }

    with open(RESULTS_DIR / 'model_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"✅ Model summary saved to {RESULTS_DIR / 'model_summary.json'}")

    print("\n🎉 Model training completed successfully!")
    print("\nFinal Results Summary:")
    print(f"- Best Model: {best_gnn_name}")
    print(f"- Best Accuracy: {best_gnn_results['test_accuracy']:.4f}")
    print(f"- Improvement over best baseline: {best_gnn_results['test_accuracy'] - max(baseline_results.values(), key=lambda x: x['test_accuracy'])['test_accuracy']:.4f}")

    print("\n✅ All models saved! You can now run the prediction script:")
    print("   python run_model.py")

def create_basic_graphs(farms_df, num_time_points=12):
    """
    Create basic graphs when processed data is not available
    """
    print("Creating basic graphs...")

    graphs = []

    # Create distance matrix
    distance_matrix = create_distance_matrix(farms_df)

    # Create adjacency matrix
    adjacency_matrix = create_adjacency_matrix(distance_matrix, threshold_km=5.0)

    for t in range(num_time_points):
        # Create basic node features
        node_features = []
        labels = []

        for _, farm in farms_df.iterrows():
            # Basic features: lat, lon, area, crop type (one-hot)
            features = [farm['lat'], farm['lon'], farm['area_hectares']]

            # Add crop type one-hot encoding
            crop_types = farms_df['crop_type'].unique()
            for crop in crop_types:
                features.append(1.0 if farm['crop_type'] == crop else 0.0)

            # Add temporal features
            features.extend([t / num_time_points, np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)])

            # Add some random environmental features
            features.extend(np.random.normal(0, 1, 5))

            node_features.append(features)

            # Random disease labels with higher probability of healthy
            label = np.random.choice([0, 1, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.05, 0.05])
            labels.append(label)

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

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

        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graphs.append(data)

    return graphs

if __name__ == "__main__":
    main()
