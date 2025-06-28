'''
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
        self.transmission_factors = {}

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
        
        for i in features['amplitude']:
            if i in features['amplitude']:
                amplitude = features['amplitude'][i]
                impact = np.sum(amplitude ** self.alpha)
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

    def calculate_impacts_with_resources(self, network: DynamicNetwork, 
                                        initial_impacts: Dict[Union[int, str], float],
                                        pathways: List[PropagationPathway],
                                        resources: Dict[Union[int, str], float],
                                        node_vulnerability: Optional[Dict[Union[int, str], float]] = None) -> Dict[Union[int, str], float]:
        """
        Calculate impacts with resources deployed.
        
        Args:
            network: The network.
            initial_impacts: Dictionary mapping node IDs to initial impact values.
            pathways: List of detected pathways.
            resources: Dictionary mapping node IDs to resource capabilities.
            node_vulnerability: Dictionary mapping node IDs to vulnerability scores.
            
        Returns:
            Dictionary mapping node IDs to impact values after resource deployment.
        """
        impacts = initial_impacts.copy()

        # Apply resources to reduce initial impacts
        for node_id, resource_capability in resources.items():
            if node_id in impacts:
                impacts[node_id] *= (1 - resource_capability)

        # Propagate impacts through the pathways
        for pathway in pathways:
            for i in range(len(pathway.nodes) - 1):
                source_node = pathway.nodes[i]
                target_node = pathway.nodes[i+1]

                # Get transmission factor
                transmission_factor = self.transmission_factors.get((source_node, target_node), 0.9)

                # Get source impact
                source_impact = impacts.get(source_node, 0.0)

                # Calculate propagated impact
                propagated_impact = source_impact * transmission_factor

                # Update target impact
                if target_node in impacts:
                    impacts[target_node] += propagated_impact
                else:
                    impacts[target_node] = propagated_impact

        # Apply node vulnerability
        if node_vulnerability:
            for node_id, vulnerability in node_vulnerability.items():
                if node_id in impacts:
                    impacts[node_id] *= vulnerability

        return impacts
'''
