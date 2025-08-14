"""
Graph Neural Network model definitions for AgroGraphNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class GCNModel(nn.Module):
    """
    Graph Convolutional Network for crop disease prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        # Apply GCN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for crop disease prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGE(input_dim, hidden_dim, num_layers=1))
        
        for _ in range(num_layers - 2):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=1))
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        # Apply GraphSAGE layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

class GATModel(nn.Module):
    """
    Graph Attention Network for crop disease prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.2):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, 
                                 heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, 
                                     heads=num_heads, dropout=dropout))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim, output_dim, 
                                 heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index, batch=None):
        # Apply GAT layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network using LSTM + GNN
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 gnn_type: str = 'GCN', num_gnn_layers: int = 2, 
                 lstm_hidden_dim: int = 64, dropout: float = 0.2):
        super(TemporalGNN, self).__init__()
        
        self.gnn_type = gnn_type
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # GNN component
        if gnn_type == 'GCN':
            self.gnn = GCNModel(input_dim, hidden_dim, hidden_dim, 
                               num_gnn_layers, dropout)
        elif gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGEModel(input_dim, hidden_dim, hidden_dim, 
                                     num_gnn_layers, dropout)
        elif gnn_type == 'GAT':
            self.gnn = GATModel(input_dim, hidden_dim, hidden_dim, 
                               num_gnn_layers, dropout=dropout)
        
        # LSTM component
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Linear(lstm_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graph_sequence: List[Data]):
        # Process each graph in the sequence
        gnn_outputs = []
        
        for graph in graph_sequence:
            # Get GNN embeddings (remove log_softmax for intermediate representation)
            if self.gnn_type == 'GCN':
                # Modify GCN forward to return embeddings
                x = graph.x
                edge_index = graph.edge_index
                
                for i in range(self.gnn.num_layers - 1):
                    x = self.gnn.convs[i](x, edge_index)
                    x = self.gnn.batch_norms[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.gnn.dropout, training=self.training)
                
                # Don't apply final layer, use embeddings
                gnn_output = x
            else:
                # For other models, we need to modify similarly
                gnn_output = self.gnn(graph.x, graph.edge_index)
            
            gnn_outputs.append(gnn_output)
        
        # Stack temporal sequence
        temporal_features = torch.stack(gnn_outputs, dim=1)  # (nodes, time, features)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(temporal_features)
        
        # Use last time step output
        final_features = lstm_out[:, -1, :]  # (nodes, lstm_hidden_dim)
        
        # Apply dropout and classify
        final_features = self.dropout(final_features)
        output = self.classifier(final_features)
        
        return F.log_softmax(output, dim=1)

def train_model(model: nn.Module, train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: torch.device) -> float:
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        Tuple of (accuracy, predictions, true_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, np.array(all_preds), np.array(all_labels)

def train_and_evaluate(model: nn.Module, train_loader: DataLoader, 
                      val_loader: DataLoader, test_loader: DataLoader,
                      num_epochs: int = 100, learning_rate: float = 0.001,
                      early_stopping_patience: int = 10,
                      device: torch.device = None) -> Dict:
    """
    Complete training and evaluation pipeline
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        early_stopping_patience: Patience for early stopping
        device: Device to run on
    
    Returns:
        Dictionary with training results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # For log_softmax output
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validation
        val_acc, _, _ = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_acc, test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    # Generate classification report
    class_names = ['Healthy', 'Blight', 'Rust', 'Mosaic', 'Bacterial']
    report = classification_report(test_labels, test_preds, 
                                 target_names=class_names, output_dict=True)
    
    results = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'test_predictions': test_preds,
        'test_labels': test_labels,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(test_labels, test_preds),
        'model_state_dict': best_model_state
    }
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return results

def create_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Create baseline models for comparison
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with baseline results
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        class_names = ['Healthy', 'Blight', 'Rust', 'Mosaic', 'Bacterial']
        report = classification_report(y_test, test_pred, 
                                     target_names=class_names, output_dict=True)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'predictions': test_pred,
            'classification_report': report,
            'model': model
        }
        
        print(f"{name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return results