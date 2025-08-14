"""
Graph Neural Network model definitions adapted from the original AgroGraphNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from typing import Optional

class GCNModel(nn.Module):
    """Graph Convolutional Network for crop disease prediction"""
    
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
    """GraphSAGE model for crop disease prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
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
    """Graph Attention Network for crop disease prediction"""
    
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

def create_model(model_type: str, input_dim: int, hidden_dim: int, 
                output_dim: int, **kwargs) -> nn.Module:
    """Factory function to create models"""
    
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return GCNModel(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_type == 'graphsage':
        return GraphSAGEModel(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_type == 'gat':
        return GATModel(input_dim, hidden_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_info(model: nn.Module) -> dict:
    """Get information about a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': model.__class__.__name__
    }