"""
Evaluation metrics for source localization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set


def success_rate_at_k(predicted_sources: List[Union[int, str]], 
                     true_sources: List[Union[int, str]], k: int = 1) -> float:
    """
    Calculate the success rate at k.
    
    Args:
        predicted_sources: List of predicted source node IDs.
        true_sources: List of true source node IDs.
        k: Number of top predictions to consider.
        
    Returns:
        Success rate at k.
    """
    if not true_sources or not predicted_sources:
        return 0.0
    
    # Consider only the top k predictions
    top_k_predictions = predicted_sources[:k]
    
    # Check if any true source is in the top k predictions
    for true_source in true_sources:
        if true_source in top_k_predictions:
            return 1.0
    
    return 0.0


def mean_rank_of_true_source(ranked_sources: List[Tuple[Union[int, str], float]], 
                            true_sources: List[Union[int, str]]) -> float:
    """
    Calculate the mean rank of the true source.
    
    Args:
        ranked_sources: List of tuples (node_id, score) sorted by score.
        true_sources: List of true source node IDs.
        
    Returns:
        Mean rank of the true source.
    """
    if not true_sources or not ranked_sources:
        return float('inf')
    
    # Extract node IDs from ranked sources
    ranked_node_ids = [node_id for node_id, _ in ranked_sources]
    
    # Find the ranks of true sources
    ranks = []
    
    for true_source in true_sources:
        if true_source in ranked_node_ids:
            rank = ranked_node_ids.index(true_source) + 1  # 1-based indexing
            ranks.append(rank)
    
    # Return mean rank if any true source is found, otherwise infinity
    return np.mean(ranks) if ranks else float('inf')


def error_distance(predicted_sources: List[Union[int, str]], true_sources: List[Union[int, str]],
                  node_positions: Dict[Union[int, str], Tuple[float, float]]) -> float:
    """
    Calculate the error distance between predicted and true sources.
    
    Args:
        predicted_sources: List of predicted source node IDs.
        true_sources: List of true source node IDs.
        node_positions: Dictionary mapping node IDs to positions (x, y).
        
    Returns:
        Error distance.
    """
    if not true_sources or not predicted_sources or not node_positions:
        return float('inf')
    
    # Consider only the top prediction
    top_prediction = predicted_sources[0]
    
    # Calculate distances to all true sources
    distances = []
    
    for true_source in true_sources:
        if true_source in node_positions and top_prediction in node_positions:
            true_pos = node_positions[true_source]
            pred_pos = node_positions[top_prediction]
            
            # Euclidean distance
            distance = np.sqrt((true_pos[0] - pred_pos[0])**2 + (true_pos[1] - pred_pos[1])**2)
            distances.append(distance)
    
    # Return minimum distance if any distance is calculated, otherwise infinity
    return min(distances) if distances else float('inf')
