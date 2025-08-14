"""
Visualization utilities for AgroGraphNet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_farm_locations(farm_locations: pd.DataFrame, 
                       disease_data: pd.DataFrame = None,
                       save_path: str = None) -> folium.Map:
    """
    Create interactive map of farm locations with disease information
    
    Args:
        farm_locations: DataFrame with farm coordinates
        disease_data: Optional disease data for coloring
        save_path: Optional path to save the map
    
    Returns:
        Folium map object
    """
    # Calculate map center
    center_lat = farm_locations['lat'].mean()
    center_lon = farm_locations['lon'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Color mapping for diseases
    disease_colors = {
        'Healthy': 'green',
        'Blight': 'red',
        'Rust': 'orange',
        'Mosaic': 'purple',
        'Bacterial': 'darkred'
    }
    
    # Add farm markers
    for _, farm in farm_locations.iterrows():
        # Get disease info if available
        if disease_data is not None:
            farm_disease = disease_data[disease_data['farm_id'] == farm['farm_id']]
            if len(farm_disease) > 0:
                disease_type = farm_disease.iloc[-1]['disease_type']  # Most recent
                color = disease_colors.get(disease_type, 'blue')
                popup_text = f"Farm: {farm['farm_id']}<br>Disease: {disease_type}"
            else:
                color = 'blue'
                popup_text = f"Farm: {farm['farm_id']}<br>No disease data"
        else:
            color = 'blue'
            popup_text = f"Farm: {farm['farm_id']}"
        
        folium.CircleMarker(
            location=[farm['lat'], farm['lon']],
            radius=8,
            popup=popup_text,
            color='black',
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add legend
    if disease_data is not None:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Disease Status</b></p>
        '''
        for disease, color in disease_colors.items():
            legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {disease}</p>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
    
    if save_path:
        m.save(save_path)
    
    return m

def plot_graph_network(G: nx.Graph, farm_locations: pd.DataFrame,
                      node_labels: np.ndarray = None,
                      save_path: str = None) -> plt.Figure:
    """
    Visualize the farm network graph
    
    Args:
        G: NetworkX graph
        farm_locations: Farm location data
        node_labels: Optional node labels for coloring
        save_path: Optional path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Geographic layout
    pos_geo = {}
    for i, (_, farm) in enumerate(farm_locations.iterrows()):
        pos_geo[i] = (farm['lon'], farm['lat'])
    
    if node_labels is not None:
        node_colors = node_labels
        cmap = plt.cm.Set1
    else:
        node_colors = 'lightblue'
        cmap = None
    
    nx.draw(G, pos_geo, ax=ax1, node_color=node_colors, cmap=cmap,
            node_size=100, edge_color='gray', alpha=0.7, with_labels=False)
    ax1.set_title('Farm Network (Geographic Layout)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Plot 2: Spring layout
    pos_spring = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos_spring, ax=ax2, node_color=node_colors, cmap=cmap,
            node_size=100, edge_color='gray', alpha=0.7, with_labels=False)
    ax2.set_title('Farm Network (Spring Layout)')
    
    # Add colorbar if labels provided
    if node_labels is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                  norm=plt.Normalize(vmin=node_labels.min(), 
                                                   vmax=node_labels.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Disease Class')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_satellite_imagery(image: np.ndarray, bands: List[str],
                          vegetation_indices: Dict[str, np.ndarray] = None,
                          save_path: str = None) -> plt.Figure:
    """
    Visualize satellite imagery and vegetation indices
    
    Args:
        image: Satellite image array
        bands: List of band names
        vegetation_indices: Optional vegetation indices
        save_path: Optional path to save the plot
    
    Returns:
        Matplotlib figure
    """
    n_plots = len(bands) + (len(vegetation_indices) if vegetation_indices else 0)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot satellite bands
    for i, band_name in enumerate(bands):
        if i < image.shape[0]:
            im = axes[plot_idx].imshow(image[i], cmap='gray')
            axes[plot_idx].set_title(f'{band_name} Band')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1
    
    # Plot vegetation indices
    if vegetation_indices:
        for name, index_array in vegetation_indices.items():
            im = axes[plot_idx].imshow(index_array, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[plot_idx].set_title(f'{name} Index')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_training_history(train_losses: List[float], 
                         val_accuracies: List[float],
                         save_path: str = None) -> plt.Figure:
    """
    Plot training history
    
    Args:
        train_losses: List of training losses
        val_accuracies: List of validation accuracies
        save_path: Optional path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies, 'r-', linewidth=2)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str] = None,
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Optional path to save the plot
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Healthy', 'Blight', 'Rust', 'Mosaic', 'Bacterial']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_model_comparison(baseline_results: Dict, gnn_results: Dict,
                         save_path: str = None) -> plt.Figure:
    """
    Compare model performance
    
    Args:
        baseline_results: Results from baseline models
        gnn_results: Results from GNN model
        save_path: Optional path to save the plot
    
    Returns:
        Matplotlib figure
    """
    # Prepare data
    models = list(baseline_results.keys()) + ['GNN']
    accuracies = [baseline_results[model]['test_accuracy'] for model in baseline_results.keys()]
    accuracies.append(gnn_results['test_accuracy'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=['skyblue'] * len(baseline_results) + ['orange'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_disease_spread_animation(farm_locations: pd.DataFrame,
                                  disease_data: pd.DataFrame,
                                  time_steps: List[str],
                                  save_path: str = None) -> folium.Map:
    """
    Create animated map showing disease spread over time
    
    Args:
        farm_locations: Farm location data
        disease_data: Disease occurrence data
        time_steps: List of time steps
        save_path: Optional path to save the map
    
    Returns:
        Folium map with time slider
    """
    # Calculate map center
    center_lat = farm_locations['lat'].mean()
    center_lon = farm_locations['lon'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Color mapping
    disease_colors = {
        'Healthy': 'green',
        'Blight': 'red',
        'Rust': 'orange',
        'Mosaic': 'purple',
        'Bacterial': 'darkred'
    }
    
    # Create features for each time step
    features = []
    
    for time_step in time_steps:
        time_disease = disease_data[disease_data['date'] == time_step]
        
        for _, farm in farm_locations.iterrows():
            farm_disease = time_disease[time_disease['farm_id'] == farm['farm_id']]
            
            if len(farm_disease) > 0:
                disease_type = farm_disease.iloc[0]['disease_type']
                color = disease_colors.get(disease_type, 'blue')
            else:
                disease_type = 'Healthy'
                color = 'green'
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [farm['lon'], farm['lat']]
                },
                'properties': {
                    'time': time_step,
                    'popup': f"Farm: {farm['farm_id']}<br>Disease: {disease_type}",
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': color,
                        'fillOpacity': 0.8,
                        'stroke': 'true',
                        'radius': 8
                    }
                }
            }
            features.append(feature)
    
    # Add time slider
    plugins.TimestampedGeoJson({
        'type': 'FeatureCollection',
        'features': features
    }, period='P1M', add_last_point=True).add_to(m)
    
    if save_path:
        m.save(save_path)
    
    return m

def plot_feature_importance(feature_names: List[str], 
                           importance_scores: np.ndarray,
                           save_path: str = None) -> plt.Figure:
    """
    Plot feature importance scores
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores
        save_path: Optional path to save the plot
    
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_scores = importance_scores[sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    bars = ax.barh(range(len(sorted_names)), sorted_scores, color='skyblue')
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_dashboard_summary(results: Dict, save_path: str = None) -> plt.Figure:
    """
    Create a comprehensive dashboard summary
    
    Args:
        results: Dictionary containing all results
        save_path: Optional path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training history
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_losses' in results:
        ax1.plot(results['train_losses'], 'b-', label='Train Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
    
    # 2. Validation accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if 'val_accuracies' in results:
        ax2.plot(results['val_accuracies'], 'r-', label='Val Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
    
    # 3. Model comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if 'model_comparison' in results:
        models = list(results['model_comparison'].keys())
        accuracies = [results['model_comparison'][m]['test_accuracy'] for m in models]
        ax3.bar(models, accuracies, color='skyblue')
        ax3.set_title('Model Comparison')
        ax3.set_ylabel('Test Accuracy')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Confusion matrix
    ax4 = fig.add_subplot(gs[1, :2])
    if 'confusion_matrix' in results:
        class_names = ['Healthy', 'Blight', 'Rust', 'Mosaic', 'Bacterial']
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', xticklabels=class_names, 
                   yticklabels=class_names, ax=ax4)
        ax4.set_title('Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
    
    # 5. Performance metrics
    ax5 = fig.add_subplot(gs[1, 2])
    if 'classification_report' in results:
        report = results['classification_report']
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Healthy', 'Blight', 'Rust', 'Mosaic', 'Bacterial']
        
        metric_data = []
        for metric in metrics:
            values = [report[cls][metric] for cls in classes if cls in report]
            metric_data.append(values)
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, (metric, values) in enumerate(zip(metrics, metric_data)):
            ax5.bar(x + i*width, values, width, label=metric)
        
        ax5.set_title('Performance by Class')
        ax5.set_xlabel('Disease Class')
        ax5.set_ylabel('Score')
        ax5.set_xticks(x + width)
        ax5.set_xticklabels(classes, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create summary text
    summary_text = "Model Performance Summary\n" + "="*50 + "\n"
    
    if 'test_accuracy' in results:
        summary_text += f"Test Accuracy: {results['test_accuracy']:.4f}\n"
    
    if 'best_val_accuracy' in results:
        summary_text += f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}\n"
    
    if 'classification_report' in results:
        report = results['classification_report']
        if 'macro avg' in report:
            summary_text += f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}\n"
        if 'weighted avg' in report:
            summary_text += f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}\n"
    
    ax6.text(0.1, 0.8, summary_text, fontsize=12, fontfamily='monospace',
             verticalalignment='top', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('AgroGraphNet: Crop Disease Prediction Results', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig