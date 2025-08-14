import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch_geometric.data import Data, DataLoader
from .models.gnn_models import create_model
from .data_loader import DataLoader as AgroDataLoader
from .graph_builder import GraphBuilder
from ..utils.logger import get_logger

class Trainer:
    """Training pipeline for AgroGraphNet models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.training.use_gpu else 'cpu')
        
        # Initialize components
        self.data_loader = AgroDataLoader(config)
        self.graph_builder = GraphBuilder(config)
        
        # Set random seeds
        torch.manual_seed(config.training.random_seed)
        np.random.seed(config.training.random_seed)
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def prepare_data(self):
        """Prepare data for training"""
        self.logger.info("Preparing training data...")
        
        # Load and merge data
        X, y = self.data_loader.prepare_training_data()
        
        # Build graph
        graph_data = self.graph_builder.build_graph(X, y)
        
        # Split data
        train_mask, val_mask, test_mask = self._create_data_splits(len(X))
        
        # Create PyTorch Geometric data object
        data = Data(
            x=torch.FloatTensor(X.values),
            edge_index=torch.LongTensor(graph_data['edge_index']),
            edge_attr=torch.FloatTensor(graph_data['edge_attr']) if 'edge_attr' in graph_data else None,
            y=torch.LongTensor(y.values),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        self.logger.info(f"Graph created: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def _create_data_splits(self, num_samples):
        """Create train/validation/test splits"""
        indices = np.arange(num_samples)
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=0.2, 
            random_state=self.config.training.random_seed,
            stratify=None  # We'll handle stratification in the graph context
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.config.training.validation_split,
            random_state=self.config.training.random_seed
        )
        
        # Create masks
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        return train_mask, val_mask, test_mask
    
    def train(self, model_type: str = 'graphsage'):
        """Train a model"""
        self.logger.info(f"Starting training for {model_type} model")
        
        # Prepare data
        data = self.prepare_data()
        data = data.to(self.device)
        
        # Create model
        model_kwargs = {
            'num_layers': self.config.model.num_layers,
            'dropout': self.config.model.dropout
        }
        
        # Add num_heads only for GAT models
        if model_type == 'gat':
            model_kwargs['num_heads'] = self.config.model.num_heads
        
        model = create_model(
            model_type=model_type,
            input_dim=data.x.shape[1],
            hidden_dim=self.config.model.hidden_dim,
            output_dim=len(self.config.disease_classes),
            **model_kwargs
        )
        
        model = model.to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.model.learning_rate)
        criterion = nn.NLLLoss()
        
        # Training loop
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        patience_counter = 0
        best_model_state = model.state_dict().copy()  # Initialize with current state
        
        self.logger.info(f"Training on {self.device} for {self.config.training.epochs} epochs")
        
        for epoch in range(self.config.training.epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_pred = val_out[data.val_mask].argmax(dim=1)
                val_acc = accuracy_score(
                    data.y[data.val_mask].cpu().numpy(),
                    val_pred.cpu().numpy()
                )
                val_accuracies.append(val_acc)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch:3d}: Train Loss = {loss.item():.4f}, Val Acc = {val_acc:.4f}")
            
            if patience_counter >= self.config.training.early_stopping:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        test_results = self._evaluate_model(model, data)
        
        # Save model
        model_path = self._save_model(model, model_type, test_results)
        
        results = {
            'model_type': model_type,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_results['accuracy'],
            'test_predictions': test_results['predictions'],
            'test_labels': test_results['labels'],
            'classification_report': test_results['classification_report'],
            'confusion_matrix': test_results['confusion_matrix'],
            'model_path': str(model_path)
        }
        
        self.logger.info(f"Training completed. Test accuracy: {test_results['accuracy']:.4f}")
        return results
    
    def _evaluate_model(self, model, data):
        """Evaluate model on test set"""
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            test_pred = out[data.test_mask].argmax(dim=1)
            test_labels = data.y[data.test_mask]
            
            test_pred_np = test_pred.cpu().numpy()
            test_labels_np = test_labels.cpu().numpy()
            
            accuracy = accuracy_score(test_labels_np, test_pred_np)
            
            # Classification report
            class_names = list(self.config.disease_classes.values())
            class_labels = list(self.config.disease_classes.keys())
            report = classification_report(
                test_labels_np, test_pred_np,
                labels=class_labels,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            class_labels = list(self.config.disease_classes.keys())
            cm = confusion_matrix(test_labels_np, test_pred_np, labels=class_labels)
            
            return {
                'accuracy': accuracy,
                'predictions': test_pred_np,
                'labels': test_labels_np,
                'classification_report': report,
                'confusion_matrix': cm
            }
    
    def _save_model(self, model, model_type, test_results):
        """Save trained model"""
        models_dir = Path(self.config.paths.get('models', 'models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{model_type}_best_model.pth"
        
        # Save model state and metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'config': self.config,
            'test_accuracy': test_results['accuracy'],
            'input_dim': model.convs[0].in_channels if hasattr(model, 'convs') else None,
            'output_dim': len(self.config.disease_classes),
            'hidden_dim': self.config.model.hidden_dim
        }, model_path)
        
        self.logger.info(f"Model saved to: {model_path}")
        return model_path