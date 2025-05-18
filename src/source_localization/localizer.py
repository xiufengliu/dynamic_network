"""
Source localization algorithm implementation.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Union
from ..network.graph import DynamicNetwork
from ..pathway_detection.definition import PropagationPathway


class SourceLocalizer:
    """
    A class for localizing event sources.
    
    Attributes:
        use_pathways (bool): Whether to use detected pathways for localization.
    """
    
    def __init__(self, use_pathways: bool = True):
        """
        Initialize the source localizer.
        
        Args:
            use_pathways: Whether to use detected pathways for localization.
        """
        self.use_pathways = use_pathways
    
    def localize(self, network: DynamicNetwork, features: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                pathways: Optional[List[PropagationPathway]] = None) -> List[Union[int, str]]:
        """
        Localize event sources.
        
        Args:
            network: The network.
            features: Dictionary with features extracted from time-series data.
            pathways: List of detected pathways (optional).
            
        Returns:
            List of source node IDs.
        """
        # Get active nodes and their activation times
        active_nodes = []
        activation_times = {}
        
        for i in features['amplitude']:
            if i in features['activation_time'] and features['activation_time'][i] is not None:
                active_nodes.append(i)
                activation_times[i] = features['activation_time'][i]
        
        # If no pathways are provided or use_pathways is False, use all active nodes
        if pathways is None or not self.use_pathways:
            return self._localize_from_activation_times(network, active_nodes, activation_times)
        else:
            return self._localize_from_pathways(network, pathways, activation_times)
    
    def _localize_from_activation_times(self, network: DynamicNetwork, active_nodes: List[int],
                                       activation_times: Dict[int, float]) -> List[Union[int, str]]:
        """
        Localize sources based on earliest activation times.
        
        Args:
            network: The network.
            active_nodes: List of active node indices.
            activation_times: Dictionary mapping node indices to activation times.
            
        Returns:
            List of source node IDs.
        """
        # Find nodes with earliest activation times
        min_time = float('inf')
        min_nodes = []
        
        for idx in active_nodes:
            if idx in activation_times:
                time = activation_times[idx]
                if time < min_time:
                    min_time = time
                    min_nodes = [idx]
                elif time == min_time:
                    min_nodes.append(idx)
        
        # Convert indices to node IDs
        source_nodes = [network.index_to_node(idx) for idx in min_nodes]
        
        return source_nodes
    
    def _localize_from_pathways(self, network: DynamicNetwork, pathways: List[PropagationPathway],
                               activation_times: Dict[int, float]) -> List[Union[int, str]]:
        """
        Localize sources based on detected pathways.
        
        Args:
            network: The network.
            pathways: List of detected pathways.
            activation_times: Dictionary mapping node indices to activation times.
            
        Returns:
            List of source node IDs.
        """
        # Get unique starting nodes from all pathways
        source_candidates = set()
        
        for pathway in pathways:
            source_candidates.add(pathway.get_source())
        
        # If no pathways, fall back to activation times
        if not source_candidates:
            active_nodes = list(activation_times.keys())
            return self._localize_from_activation_times(network, active_nodes, activation_times)
        
        # Find nodes with earliest activation times among source candidates
        min_time = float('inf')
        min_nodes = []
        
        for node_id in source_candidates:
            # Convert node ID to index
            idx = network.node_to_index(node_id)
            
            if idx in activation_times:
                time = activation_times[idx]
                if time < min_time:
                    min_time = time
                    min_nodes = [node_id]
                elif time == min_time:
                    min_nodes.append(node_id)
        
        return min_nodes
    
    def rank_sources(self, network: DynamicNetwork, features: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                    pathways: Optional[List[PropagationPathway]] = None) -> List[Tuple[Union[int, str], float]]:
        """
        Rank potential sources based on various criteria.
        
        Args:
            network: The network.
            features: Dictionary with features extracted from time-series data.
            pathways: List of detected pathways (optional).
            
        Returns:
            List of tuples (node_id, score) sorted by score in descending order.
        """
        # Get active nodes and their activation times
        active_nodes = []
        activation_times = {}
        
        for i in features['amplitude']:
            if i in features['activation_time'] and features['activation_time'][i] is not None:
                active_nodes.append(i)
                activation_times[i] = features['activation_time'][i]
        
        # Calculate scores based on activation times
        scores = {}
        min_time = min(activation_times.values()) if activation_times else 0
        
        for idx in active_nodes:
            if idx in activation_times:
                # Score based on how early the node was activated
                time_score = 1.0 / (1.0 + activation_times[idx] - min_time)
                
                # Convert index to node ID
                node_id = network.index_to_node(idx)
                
                scores[node_id] = time_score
        
        # If pathways are provided, incorporate pathway information
        if pathways is not None and self.use_pathways:
            # Count how many times each node appears as a source in pathways
            source_counts = {}
            
            for pathway in pathways:
                source = pathway.get_source()
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Normalize counts
            max_count = max(source_counts.values()) if source_counts else 1
            
            for node_id, count in source_counts.items():
                # Score based on how often the node appears as a source
                pathway_score = count / max_count
                
                # Combine with time score
                if node_id in scores:
                    scores[node_id] = 0.7 * scores[node_id] + 0.3 * pathway_score
                else:
                    scores[node_id] = 0.3 * pathway_score
        
        # Sort nodes by score in descending order
        ranked_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_sources
