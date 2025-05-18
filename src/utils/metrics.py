"""
Evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union
from ..pathway_detection.definition import PropagationPathway


def pathway_jaccard_index(predicted_pathways: List[PropagationPathway], 
                         true_pathways: List[PropagationPathway]) -> float:
    """
    Calculate the Jaccard index between predicted and true pathways.
    
    Args:
        predicted_pathways: List of predicted pathways.
        true_pathways: List of true pathways.
        
    Returns:
        Jaccard index.
    """
    # Extract edges from pathways
    predicted_edges = set()
    for pathway in predicted_pathways:
        predicted_edges.update(pathway.get_edges())
    
    true_edges = set()
    for pathway in true_pathways:
        true_edges.update(pathway.get_edges())
    
    # Calculate Jaccard index
    intersection = len(predicted_edges.intersection(true_edges))
    union = len(predicted_edges.union(true_edges))
    
    return intersection / union if union > 0 else 0.0


def precision_recall_f1(predicted_pathways: List[PropagationPathway], 
                       true_pathways: List[PropagationPathway]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score for pathway detection.
    
    Args:
        predicted_pathways: List of predicted pathways.
        true_pathways: List of true pathways.
        
    Returns:
        Tuple of (precision, recall, F1-score).
    """
    # Extract edges from pathways
    predicted_edges = set()
    for pathway in predicted_pathways:
        predicted_edges.update(pathway.get_edges())
    
    true_edges = set()
    for pathway in true_pathways:
        true_edges.update(pathway.get_edges())
    
    # Calculate precision and recall
    true_positives = len(predicted_edges.intersection(true_edges))
    false_positives = len(predicted_edges - true_edges)
    false_negatives = len(true_edges - predicted_edges)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Calculate F1-score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def total_impact_reduction(initial_impacts: Dict[Union[int, str], float],
                          final_impacts: Dict[Union[int, str], float]) -> float:
    """
    Calculate the total impact reduction.
    
    Args:
        initial_impacts: Dictionary mapping node IDs to initial impact values.
        final_impacts: Dictionary mapping node IDs to final impact values.
        
    Returns:
        Total impact reduction.
    """
    # Calculate total initial impact
    total_initial_impact = sum(initial_impacts.values())
    
    # Calculate total final impact
    total_final_impact = sum(final_impacts.values())
    
    # Calculate total impact reduction
    total_impact_reduction = total_initial_impact - total_final_impact
    
    return total_impact_reduction


def cost_effectiveness_ratio(impact_reduction: float, total_cost: float) -> float:
    """
    Calculate the cost-effectiveness ratio.
    
    Args:
        impact_reduction: Total impact reduction.
        total_cost: Total cost of resource allocation.
        
    Returns:
        Cost-effectiveness ratio.
    """
    return impact_reduction / total_cost if total_cost > 0 else float('inf')


def constraint_satisfaction(impacts: Dict[Union[int, str], float],
                           critical_nodes: List[Union[int, str]],
                           max_impact: float) -> bool:
    """
    Check if the impact constraints are satisfied.
    
    Args:
        impacts: Dictionary mapping node IDs to impact values.
        critical_nodes: List of critical node IDs.
        max_impact: Maximum permissible impact at critical nodes.
        
    Returns:
        True if constraints are satisfied, False otherwise.
    """
    # Check if any critical node exceeds the maximum impact
    for node in critical_nodes:
        if node in impacts and impacts[node] > max_impact:
            return False
    
    return True
