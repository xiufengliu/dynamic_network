"""
Resource allocation optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from ..network.graph import DynamicNetwork
from ..pathway_detection.definition import PropagationPathway
from .impact_model import ImpactModel


class ResourceOptimizer:
    """
    A class for optimizing resource allocation.
    
    Attributes:
        impact_model (ImpactModel): Model for calculating event impact.
    """
    
    def __init__(self, impact_model: Optional[ImpactModel] = None):
        """
        Initialize the resource optimizer.
        
        Args:
            impact_model: Model for calculating event impact.
        """
        self.impact_model = impact_model if impact_model is not None else ImpactModel()
    
    def optimize(self, network: DynamicNetwork, pathways: List[PropagationPathway],
                initial_impacts: Dict[Union[int, str], float],
                critical_nodes: Optional[List[Union[int, str]]] = None,
                max_impact: float = 0.1, budget: Optional[float] = None,
                costs: Optional[Dict[Union[int, str], float]] = None,
                capabilities: Optional[Dict[Union[int, str], float]] = None) -> Dict[Union[int, str], float]:
        """
        Optimize resource allocation.
        
        Args:
            network: The network.
            pathways: List of detected pathways.
            initial_impacts: Dictionary mapping node IDs to initial impact values.
            critical_nodes: List of critical node IDs.
            max_impact: Maximum permissible impact at critical nodes.
            budget: Maximum budget for resource allocation.
            costs: Dictionary mapping node IDs to resource costs.
            capabilities: Dictionary mapping node IDs to resource capabilities.
            
        Returns:
            Dictionary mapping node IDs to allocated resource capabilities.
        """
        # If critical nodes are not specified, use all nodes
        if critical_nodes is None:
            critical_nodes = list(initial_impacts.keys())
        
        # If costs are not specified, use unit costs
        if costs is None:
            costs = {node_id: 1.0 for node_id in network.get_nodes()}
        
        # If capabilities are not specified, use default capabilities
        if capabilities is None:
            capabilities = {node_id: 0.5 for node_id in network.get_nodes()}
        
        # Use greedy heuristic for optimization
        from .greedy_heuristic import GreedyHeuristic
        greedy = GreedyHeuristic(self.impact_model)
        
        # Optimize resource allocation
        allocation = greedy.optimize(
            network=network,
            pathways=pathways,
            initial_impacts=initial_impacts,
            critical_nodes=critical_nodes,
            max_impact=max_impact,
            budget=budget,
            costs=costs,
            capabilities=capabilities
        )
        
        return allocation
    
    def evaluate_allocation(self, network: DynamicNetwork, pathways: List[PropagationPathway],
                           initial_impacts: Dict[Union[int, str], float],
                           allocation: Dict[Union[int, str], float],
                           critical_nodes: Optional[List[Union[int, str]]] = None) -> Dict[str, float]:
        """
        Evaluate a resource allocation.
        
        Args:
            network: The network.
            pathways: List of detected pathways.
            initial_impacts: Dictionary mapping node IDs to initial impact values.
            allocation: Dictionary mapping node IDs to allocated resource capabilities.
            critical_nodes: List of critical node IDs.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # If critical nodes are not specified, use all nodes
        if critical_nodes is None:
            critical_nodes = list(initial_impacts.keys())
        
        # Calculate impacts with resources
        impacts = self.impact_model.calculate_impacts_with_resources(
            network=network,
            initial_impacts=initial_impacts,
            pathways=pathways,
            resources=allocation
        )
        
        # Calculate total impact reduction
        total_initial_impact = sum(initial_impacts.values())
        total_reduced_impact = sum(impacts.values())
        total_impact_reduction = total_initial_impact - total_reduced_impact
        
        # Calculate impact reduction at critical nodes
        critical_initial_impact = sum(initial_impacts.get(node, 0) for node in critical_nodes)
        critical_reduced_impact = sum(impacts.get(node, 0) for node in critical_nodes)
        critical_impact_reduction = critical_initial_impact - critical_reduced_impact
        
        # Calculate maximum impact at critical nodes
        max_critical_impact = max(impacts.get(node, 0) for node in critical_nodes) if critical_nodes else 0
        
        # Calculate total cost
        total_cost = sum(allocation.values())
        
        # Calculate cost-effectiveness ratio
        cost_effectiveness = total_impact_reduction / total_cost if total_cost > 0 else float('inf')
        
        # Return evaluation metrics
        return {
            'total_initial_impact': total_initial_impact,
            'total_reduced_impact': total_reduced_impact,
            'total_impact_reduction': total_impact_reduction,
            'critical_initial_impact': critical_initial_impact,
            'critical_reduced_impact': critical_reduced_impact,
            'critical_impact_reduction': critical_impact_reduction,
            'max_critical_impact': max_critical_impact,
            'total_cost': total_cost,
            'cost_effectiveness': cost_effectiveness
        }
