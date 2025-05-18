"""
Greedy heuristic for resource allocation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from ..network.graph import DynamicNetwork
from ..pathway_detection.definition import PropagationPathway
from .impact_model import ImpactModel


class GreedyHeuristic:
    """
    A class implementing a greedy heuristic for resource allocation.
    
    Attributes:
        impact_model (ImpactModel): Model for calculating event impact.
    """
    
    def __init__(self, impact_model: Optional[ImpactModel] = None):
        """
        Initialize the greedy heuristic.
        
        Args:
            impact_model: Model for calculating event impact.
        """
        self.impact_model = impact_model if impact_model is not None else ImpactModel()
    
    def optimize(self, network: DynamicNetwork, pathways: List[PropagationPathway],
                initial_impacts: Dict[Union[int, str], float],
                critical_nodes: List[Union[int, str]],
                max_impact: float = 0.1, budget: Optional[float] = None,
                costs: Optional[Dict[Union[int, str], float]] = None,
                capabilities: Optional[Dict[Union[int, str], float]] = None) -> Dict[Union[int, str], float]:
        """
        Optimize resource allocation using a greedy heuristic.
        
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
        # If costs are not specified, use unit costs
        if costs is None:
            costs = {node_id: 1.0 for node_id in network.get_nodes()}
        
        # If capabilities are not specified, use default capabilities
        if capabilities is None:
            capabilities = {node_id: 0.5 for node_id in network.get_nodes()}
        
        # Initialize allocation
        allocation = {}
        
        # Calculate initial impacts at critical nodes
        impacts = self.impact_model.calculate_impacts_with_resources(
            network=network,
            initial_impacts=initial_impacts,
            pathways=pathways,
            resources=allocation
        )
        
        # Check if any critical node exceeds the maximum impact
        max_critical_impact = max(impacts.get(node, 0) for node in critical_nodes) if critical_nodes else 0
        
        # If all critical nodes are already below the threshold, return empty allocation
        if max_critical_impact <= max_impact:
            return allocation
        
        # Initialize total cost
        total_cost = 0.0
        
        # Iteratively add resources until all critical nodes are below the threshold or budget is exhausted
        while max_critical_impact > max_impact:
            # If budget is specified and exhausted, break
            if budget is not None and total_cost >= budget:
                break
            
            # Find the best node to add a resource
            best_node = None
            best_benefit = -float('inf')
            best_cost = float('inf')
            
            for node_id in network.get_nodes():
                # Skip if node already has a resource
                if node_id in allocation:
                    continue
                
                # Skip if node has no capability
                if node_id not in capabilities or capabilities[node_id] <= 0:
                    continue
                
                # Skip if node cost exceeds remaining budget
                if budget is not None and costs.get(node_id, 0) > budget - total_cost:
                    continue
                
                # Calculate impact with resource at this node
                temp_allocation = allocation.copy()
                temp_allocation[node_id] = capabilities[node_id]
                
                temp_impacts = self.impact_model.calculate_impacts_with_resources(
                    network=network,
                    initial_impacts=initial_impacts,
                    pathways=pathways,
                    resources=temp_allocation
                )
                
                # Calculate benefit (reduction in maximum critical impact)
                temp_max_critical_impact = max(temp_impacts.get(node, 0) for node in critical_nodes) if critical_nodes else 0
                benefit = max_critical_impact - temp_max_critical_impact
                
                # Calculate cost
                cost = costs.get(node_id, 1.0)
                
                # Calculate benefit-to-cost ratio
                if cost > 0:
                    benefit_cost_ratio = benefit / cost
                else:
                    benefit_cost_ratio = benefit  # Infinite ratio for zero cost
                
                # Update best node if this one is better
                if benefit_cost_ratio > best_benefit or (benefit_cost_ratio == best_benefit and cost < best_cost):
                    best_node = node_id
                    best_benefit = benefit_cost_ratio
                    best_cost = cost
            
            # If no beneficial node is found, break
            if best_node is None or best_benefit <= 0:
                break
            
            # Add resource at the best node
            allocation[best_node] = capabilities[best_node]
            total_cost += costs.get(best_node, 1.0)
            
            # Update impacts
            impacts = self.impact_model.calculate_impacts_with_resources(
                network=network,
                initial_impacts=initial_impacts,
                pathways=pathways,
                resources=allocation
            )
            
            # Update maximum critical impact
            max_critical_impact = max(impacts.get(node, 0) for node in critical_nodes) if critical_nodes else 0
        
        return allocation
