"""
Event impact modeling.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Set
from ..network.graph import DynamicNetwork
from ..pathway_detection.definition import PropagationPathway


class ImpactModel:
    """
    A class for modeling event impact.
    
    Attributes:
        alpha (float): Exponent for impact calculation.
        beta_range (Tuple[float, float]): Range for transmission factors.
    """
    
    def __init__(self, alpha: float = 2.0, beta_range: Tuple[float, float] = (0.8, 1.0)):
        """
        Initialize the impact model.
        
        Args:
            alpha: Exponent for impact calculation.
            beta_range: Range for transmission factors.
        """
        self.alpha = alpha
        self.beta_range = beta_range
        self.transmission_factors = {}  # Edge-specific transmission factors
    
    def calculate_initial_impacts(self, network: DynamicNetwork, 
                                 features: Dict[str, Dict[int, Dict[str, np.ndarray]]]) -> Dict[Union[int, str], float]:
        """
        Calculate initial impacts for all nodes.
        
        Args:
            network: The network.
            features: Dictionary with features extracted from time-series data.
            
        Returns:
            Dictionary mapping node IDs to impact values.
        """
        impacts = {}
        
        # Calculate impact based on amplitude
        for i in features['amplitude']:
            if i in features['amplitude']:
                amplitude = features['amplitude'][i]
                
                # Calculate impact as the integral of squared amplitude
                impact = np.sum(amplitude ** self.alpha)
                
                # Convert index to node ID
                node_id = network.index_to_node(i)
                
                impacts[node_id] = impact
        
        return impacts
    
    def generate_transmission_factors(self, network: DynamicNetwork, seed: Optional[int] = None) -> None:
        """
        Generate transmission factors for all edges.
        
        Args:
            network: The network.
            seed: Random seed.
        """
        rng = np.random.RandomState(seed)
        
        for u, v, _ in network.get_edges():
            self.transmission_factors[(u, v)] = rng.uniform(self.beta_range[0], self.beta_range[1])
    
    def set_transmission_factor(self, source: Union[int, str], target: Union[int, str], 
                               factor: float) -> None:
        """
        Set the transmission factor for a specific edge.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            factor: Transmission factor.
        """
        self.transmission_factors[(source, target)] = factor
    
    def get_transmission_factor(self, source: Union[int, str], target: Union[int, str]) -> float:
        """
        Get the transmission factor for a specific edge.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            
        Returns:
            Transmission factor.
        """
        return self.transmission_factors.get((source, target), 0.9)  # Default value if not set
    
    def calculate_impacts_with_resources(self, network: DynamicNetwork, 
                                        initial_impacts: Dict[Union[int, str], float],
                                        pathways: List[PropagationPathway],
                                        resources: Dict[Union[int, str], float]) -> Dict[Union[int, str], float]:
        """
        Calculate impacts with resources deployed.
        
        Args:
            network: The network.
            initial_impacts: Dictionary mapping node IDs to initial impact values.
            pathways: List of detected pathways.
            resources: Dictionary mapping node IDs to resource capabilities.
            
        Returns:
            Dictionary mapping node IDs to impact values after resource deployment.
        """
        # Create a directed graph for impact propagation
        impact_graph = nx.DiGraph()
        
        # Add all nodes with initial impacts
        for node_id, impact in initial_impacts.items():
            impact_graph.add_node(node_id, initial_impact=impact)
        
        # Add edges from pathways
        for pathway in pathways:
            for i in range(len(pathway.nodes) - 1):
                source = pathway.nodes[i]
                target = pathway.nodes[i + 1]
                
                # Get or generate transmission factor
                beta = self.get_transmission_factor(source, target)
                
                # Add edge with transmission factor
                impact_graph.add_edge(source, target, beta=beta)
        
        # Calculate impacts with resources
        impacts = {}
        
        # First, apply resources to reduce initial impacts
        for node_id, impact in initial_impacts.items():
            if node_id in resources:
                # Reduce impact by resource capability
                reduced_impact = impact * (1 - resources[node_id]) ** self.alpha
                impacts[node_id] = reduced_impact
            else:
                impacts[node_id] = impact
        
        # Propagate impacts through the graph
        # Sort nodes in topological order to ensure proper propagation
        try:
            for node in nx.topological_sort(impact_graph):
                # Skip if node has no incoming edges (source nodes)
                if impact_graph.in_degree(node) == 0:
                    continue
                
                # Calculate incoming impact
                incoming_impact = 0
                
                for pred in impact_graph.predecessors(node):
                    if pred in impacts:
                        edge_data = impact_graph[pred][node]
                        beta = edge_data.get('beta', 0.9)
                        
                        # Add impact from predecessor
                        incoming_impact += impacts[pred] * beta
                
                # Apply resource if present
                if node in resources:
                    incoming_impact *= (1 - resources[node]) ** self.alpha
                
                # Update impact
                impacts[node] = incoming_impact
        except nx.NetworkXUnfeasible:
            # Handle cycles in the graph
            print("Warning: Cycles detected in the impact graph. Using approximate impact calculation.")
            
            # Simple approximation: iterate a fixed number of times
            for _ in range(10):
                for node in impact_graph.nodes():
                    # Skip if node has no incoming edges (source nodes)
                    if impact_graph.in_degree(node) == 0:
                        continue
                    
                    # Calculate incoming impact
                    incoming_impact = 0
                    
                    for pred in impact_graph.predecessors(node):
                        if pred in impacts:
                            edge_data = impact_graph[pred][node]
                            beta = edge_data.get('beta', 0.9)
                            
                            # Add impact from predecessor
                            incoming_impact += impacts[pred] * beta
                    
                    # Apply resource if present
                    if node in resources:
                        incoming_impact *= (1 - resources[node]) ** self.alpha
                    
                    # Update impact
                    impacts[node] = incoming_impact
        
        return impacts
