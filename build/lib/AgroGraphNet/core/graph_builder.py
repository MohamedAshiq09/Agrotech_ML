import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from typing import Dict, List, Tuple, Any
from ..utils.logger import get_logger

class GraphBuilder:
    """Build graph structures for GNN models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
    
    def build_graph(self, features: pd.DataFrame, labels: pd.Series = None) -> Dict[str, Any]:
        """Build graph from farm data"""
        self.logger.info("Building graph structure...")
        
        # Extract coordinates if available
        if 'latitude' in features.columns and 'longitude' in features.columns:
            coords = features[['latitude', 'longitude']].values
            edge_index, edge_attr = self._build_spatial_graph(coords, features)
        else:
            # Build feature-based graph if no coordinates
            edge_index, edge_attr = self._build_feature_graph(features)
        
        graph_data = {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_nodes': len(features),
            'num_edges': edge_index.shape[1]
        }
        
        self.logger.info(f"Graph built: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
        return graph_data
    
    def _build_spatial_graph(self, coords: np.ndarray, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph based on spatial proximity"""
        
        # Convert coordinates to radians for haversine distance
        coords_rad = np.radians(coords)
        
        # Calculate pairwise distances (in radians)
        distances = haversine_distances(coords_rad)
        
        # Convert to kilometers (Earth radius ≈ 6371 km)
        distances_km = distances * 6371
        
        # Find neighbors within threshold
        threshold_km = self.config.graph.distance_threshold_km
        
        edge_list = []
        edge_features = []
        
        for i in range(len(coords)):
            # Find neighbors within threshold
            neighbors = np.where((distances_km[i] <= threshold_km) & (distances_km[i] > 0))[0]
            
            # Limit number of neighbors
            if len(neighbors) > self.config.graph.max_neighbors:
                # Keep closest neighbors
                neighbor_distances = distances_km[i][neighbors]
                closest_indices = np.argsort(neighbor_distances)[:self.config.graph.max_neighbors]
                neighbors = neighbors[closest_indices]
            
            # Add edges
            for j in neighbors:
                edge_list.append([i, j])
                
                # Calculate edge features
                edge_feat = self._calculate_edge_features(i, j, distances_km[i, j], features)
                edge_features.append(edge_feat)
        
        # Ensure minimum connectivity
        edge_list, edge_features = self._ensure_connectivity(
            edge_list, edge_features, distances_km, features
        )
        
        if len(edge_list) == 0:
            # Fallback: create a simple chain graph
            self.logger.warning("No spatial connections found, creating chain graph")
            edge_list = [[i, i+1] for i in range(len(coords)-1)]
            edge_features = [[1.0] for _ in edge_list]
        
        # Convert to numpy arrays
        edge_index = np.array(edge_list).T  # Shape: [2, num_edges]
        edge_attr = np.array(edge_features)  # Shape: [num_edges, num_features]
        
        return edge_index, edge_attr
    
    def _build_feature_graph(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph based on feature similarity"""
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.select_dtypes(include=[np.number]))
        
        # Use k-nearest neighbors
        k = min(self.config.graph.max_neighbors, len(features) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(features_scaled)
        
        distances, indices = nbrs.kneighbors(features_scaled)
        
        edge_list = []
        edge_features = []
        
        for i in range(len(features)):
            # Skip self (first neighbor)
            for j_idx in range(1, len(indices[i])):
                j = indices[i][j_idx]
                distance = distances[i][j_idx]
                
                edge_list.append([i, j])
                edge_features.append([1.0 / (1.0 + distance)])  # Similarity score
        
        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_features)
        
        return edge_index, edge_attr
    
    def _calculate_edge_features(self, i: int, j: int, distance: float, 
                               features: pd.DataFrame) -> List[float]:
        """Calculate features for an edge between nodes i and j"""
        
        edge_feat = []
        
        # Distance feature (normalized)
        max_distance = self.config.graph.distance_threshold_km
        normalized_distance = distance / max_distance
        edge_feat.append(normalized_distance)
        
        # Feature similarity (if we have numeric features)
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            feat_i = numeric_features.iloc[i].values
            feat_j = numeric_features.iloc[j].values
            
            # Cosine similarity
            dot_product = np.dot(feat_i, feat_j)
            norm_i = np.linalg.norm(feat_i)
            norm_j = np.linalg.norm(feat_j)
            
            if norm_i > 0 and norm_j > 0:
                similarity = dot_product / (norm_i * norm_j)
            else:
                similarity = 0.0
            
            edge_feat.append(similarity)
        
        # Crop type similarity (if available)
        if 'crop_type' in features.columns:
            crop_similarity = 1.0 if features.iloc[i]['crop_type'] == features.iloc[j]['crop_type'] else 0.0
            edge_feat.append(crop_similarity)
        
        return edge_feat
    
    def _ensure_connectivity(self, edge_list: List[List[int]], edge_features: List[List[float]],
                           distances: np.ndarray, features: pd.DataFrame) -> Tuple[List[List[int]], List[List[float]]]:
        """Ensure graph connectivity by adding edges to isolated nodes"""
        
        # Find nodes with too few connections
        node_degrees = {}
        for edge in edge_list:
            node_degrees[edge[0]] = node_degrees.get(edge[0], 0) + 1
            node_degrees[edge[1]] = node_degrees.get(edge[1], 0) + 1
        
        # Add connections for isolated or poorly connected nodes
        for node in range(len(features)):
            current_degree = node_degrees.get(node, 0)
            
            if current_degree < self.config.graph.min_neighbors:
                # Find closest unconnected nodes
                existing_neighbors = set()
                for edge in edge_list:
                    if edge[0] == node:
                        existing_neighbors.add(edge[1])
                    elif edge[1] == node:
                        existing_neighbors.add(edge[0])
                
                # Get distances to all other nodes
                node_distances = [(distances[node][j], j) for j in range(len(features)) 
                                if j != node and j not in existing_neighbors]
                
                # Sort by distance and add closest connections
                node_distances.sort()
                needed_connections = self.config.graph.min_neighbors - current_degree
                
                for dist, neighbor in node_distances[:needed_connections]:
                    edge_list.append([node, neighbor])
                    edge_feat = self._calculate_edge_features(node, neighbor, dist, features)
                    edge_features.append(edge_feat)
        
        return edge_list, edge_features