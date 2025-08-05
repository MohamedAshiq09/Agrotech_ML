"""
Graph construction and manipulation utilities for AgroGraphNet
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    R = 6371
    return R * c

def create_distance_matrix(locations: pd.DataFrame) -> np.ndarray:
    """
    Create distance matrix between farm locations
    
    Args:
        locations: DataFrame with 'lat' and 'lon' columns
    
    Returns:
        Distance matrix in kilometers
    """
    coords = locations[['lat', 'lon']].values
    n_locations = len(coords)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(i+1, n_locations):
            dist = calculate_haversine_distance(
                coords[i, 0], coords[i, 1], 
                coords[j, 0], coords[j, 1]
            )
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix

def create_adjacency_matrix(distance_matrix: np.ndarray, 
                          threshold_km: float = 5.0,
                          min_neighbors: int = 2,
                          max_neighbors: int = 10) -> np.ndarray:
    """
    Create adjacency matrix based on distance threshold and neighbor constraints
    
    Args:
        distance_matrix: Distance matrix between locations
        threshold_km: Maximum distance for connections
        min_neighbors: Minimum neighbors per node
        max_neighbors: Maximum neighbors per node
    
    Returns:
        Binary adjacency matrix
    """
    n_nodes = distance_matrix.shape[0]
    adjacency = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        # Get distances to all other nodes
        distances = distance_matrix[i]
        
        # Find nodes within threshold (excluding self)
        candidates = np.where((distances > 0) & (distances <= threshold_km))[0]
        
        if len(candidates) < min_neighbors:
            # If not enough neighbors within threshold, connect to closest nodes
            sorted_indices = np.argsort(distances)
            candidates = sorted_indices[1:min_neighbors+1]  # Exclude self (index 0)
        elif len(candidates) > max_neighbors:
            # If too many neighbors, keep only the closest ones
            candidate_distances = distances[candidates]
            closest_indices = np.argsort(candidate_distances)[:max_neighbors]
            candidates = candidates[closest_indices]
        
        # Create connections
        adjacency[i, candidates] = 1
        adjacency[candidates, i] = 1  # Make symmetric
    
    return adjacency

def calculate_environmental_similarity(weather_data: pd.DataFrame, 
                                     farm_locations: pd.DataFrame) -> np.ndarray:
    """
    Calculate environmental similarity matrix between farms
    
    Args:
        weather_data: Weather data with farm locations
        farm_locations: Farm location data
    
    Returns:
        Environmental similarity matrix
    """
    # Aggregate weather data by farm
    weather_features = ['temperature', 'humidity', 'precipitation', 'wind_speed']
    
    farm_weather = []
    for _, farm in farm_locations.iterrows():
        farm_data = weather_data[
            (abs(weather_data['lat'] - farm['lat']) < 0.01) & 
            (abs(weather_data['lon'] - farm['lon']) < 0.01)
        ]
        
        if len(farm_data) > 0:
            avg_weather = farm_data[weather_features].mean().values
        else:
            # Use overall averages if no specific data
            avg_weather = weather_data[weather_features].mean().values
        
        farm_weather.append(avg_weather)
    
    farm_weather = np.array(farm_weather)
    
    # Normalize features
    scaler = StandardScaler()
    farm_weather_norm = scaler.fit_transform(farm_weather)
    
    # Calculate similarity (inverse of Euclidean distance)
    distances = pdist(farm_weather_norm, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Convert to similarity (higher values = more similar)
    max_distance = np.max(distance_matrix)
    similarity_matrix = 1 - (distance_matrix / max_distance)
    
    return similarity_matrix

def create_networkx_graph(adjacency_matrix: np.ndarray,
                         node_features: np.ndarray,
                         edge_features: Dict[str, np.ndarray],
                         farm_locations: pd.DataFrame) -> nx.Graph:
    """
    Create NetworkX graph from adjacency matrix and features
    
    Args:
        adjacency_matrix: Binary adjacency matrix
        node_features: Node feature matrix
        edge_features: Dictionary of edge feature matrices
        farm_locations: Farm location data
    
    Returns:
        NetworkX graph object
    """
    G = nx.Graph()
    
    # Add nodes with features
    for i in range(len(farm_locations)):
        G.add_node(i, 
                  farm_id=farm_locations.iloc[i]['farm_id'],
                  lat=farm_locations.iloc[i]['lat'],
                  lon=farm_locations.iloc[i]['lon'],
                  features=node_features[i])
    
    # Add edges with features
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i+1, adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:
                edge_attrs = {}
                for feature_name, feature_matrix in edge_features.items():
                    edge_attrs[feature_name] = feature_matrix[i, j]
                
                G.add_edge(i, j, **edge_attrs)
    
    return G

def networkx_to_pytorch_geometric(G: nx.Graph, 
                                 node_labels: Optional[np.ndarray] = None) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    
    Args:
        G: NetworkX graph
        node_labels: Optional node labels for supervised learning
    
    Returns:
        PyTorch Geometric Data object
    """
    # Get node features
    node_features = []
    for node in G.nodes():
        node_features.append(G.nodes[node]['features'])
    
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Get edge indices
    edge_index = []
    edge_attr = []
    
    for edge in G.edges():
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])  # Add reverse edge for undirected graph
        
        # Get edge attributes
        edge_data = G.edges[edge]
        edge_features = [edge_data.get('distance', 0), 
                        edge_data.get('environmental_similarity', 0)]
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)  # Same features for reverse edge
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if node_labels is not None:
        data.y = torch.tensor(node_labels, dtype=torch.long)
    
    return data

def create_temporal_graphs(farm_locations: pd.DataFrame,
                          weather_data: pd.DataFrame,
                          disease_data: pd.DataFrame,
                          node_features: Dict[str, np.ndarray],
                          time_steps: List[str]) -> List[Data]:
    """
    Create temporal sequence of graphs
    
    Args:
        farm_locations: Farm location data
        weather_data: Weather data
        disease_data: Disease occurrence data
        node_features: Dictionary of node feature arrays
        time_steps: List of time step identifiers
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    graphs = []
    
    # Create base adjacency matrix (doesn't change over time)
    distance_matrix = create_distance_matrix(farm_locations)
    adjacency_matrix = create_adjacency_matrix(distance_matrix)
    
    for time_step in time_steps:
        # Filter data for this time step
        time_weather = weather_data[weather_data['date'] == time_step]
        time_disease = disease_data[disease_data['date'] == time_step]
        
        # Calculate environmental similarity for this time step
        env_similarity = calculate_environmental_similarity(time_weather, farm_locations)
        
        # Combine node features for this time step
        combined_features = []
        for i, farm in farm_locations.iterrows():
            farm_features = []
            
            # Add static features
            for feature_name, feature_array in node_features.items():
                if feature_array.ndim == 1:
                    farm_features.append(feature_array[i])
                else:
                    farm_features.extend(feature_array[i])
            
            # Add temporal weather features
            farm_weather = time_weather[
                (abs(time_weather['lat'] - farm['lat']) < 0.01) & 
                (abs(time_weather['lon'] - farm['lon']) < 0.01)
            ]
            
            if len(farm_weather) > 0:
                weather_features = farm_weather[['temperature', 'humidity', 'precipitation', 'wind_speed']].mean().values
            else:
                weather_features = time_weather[['temperature', 'humidity', 'precipitation', 'wind_speed']].mean().values
            
            farm_features.extend(weather_features)
            combined_features.append(farm_features)
        
        combined_features = np.array(combined_features)
        
        # Create edge features
        edge_features = {
            'distance': distance_matrix,
            'environmental_similarity': env_similarity
        }
        
        # Create NetworkX graph
        G = create_networkx_graph(adjacency_matrix, combined_features, edge_features, farm_locations)
        
        # Get labels for this time step
        labels = []
        disease_mapping = {'Healthy': 0, 'Blight': 1, 'Rust': 2, 'Mosaic': 3, 'Bacterial': 4}
        
        for _, farm in farm_locations.iterrows():
            farm_disease = time_disease[time_disease['farm_id'] == farm['farm_id']]
            if len(farm_disease) > 0:
                disease_type = farm_disease.iloc[0]['disease_type']
                labels.append(disease_mapping.get(disease_type, 0))
            else:
                labels.append(0)  # Default to healthy
        
        # Convert to PyTorch Geometric
        data = networkx_to_pytorch_geometric(G, np.array(labels))
        graphs.append(data)
    
    return graphs

def analyze_graph_properties(G: nx.Graph) -> Dict[str, float]:
    """
    Analyze basic properties of the graph
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary of graph properties
    """
    properties = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
    }
    
    if nx.is_connected(G):
        properties['diameter'] = nx.diameter(G)
        properties['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        properties['diameter'] = float('inf')
        properties['avg_path_length'] = float('inf')
        properties['num_components'] = nx.number_connected_components(G)
    
    return properties