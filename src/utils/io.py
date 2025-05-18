"""
Input/output utilities.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from ..network.graph import DynamicNetwork
from ..pathway_detection.definition import PropagationPathway


def save_network(network: DynamicNetwork, filename: str) -> None:
    """
    Save a network to a file.
    
    Args:
        network: The network to save.
        filename: The name of the file.
    """
    network.save_to_file(filename)


def load_network(filename: str) -> DynamicNetwork:
    """
    Load a network from a file.
    
    Args:
        filename: The name of the file.
        
    Returns:
        The loaded network.
    """
    network = DynamicNetwork()
    network.load_from_file(filename)
    return network


def save_time_series(time: np.ndarray, signals: Dict[int, np.ndarray], filename: str) -> None:
    """
    Save time series data to a file.
    
    Args:
        time: Time array.
        signals: Dictionary mapping node indices to signal arrays.
        filename: The name of the file.
    """
    data = {
        'time': time,
        'signals': signals
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_time_series(filename: str) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Load time series data from a file.
    
    Args:
        filename: The name of the file.
        
    Returns:
        Tuple of (time, signals).
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data['time'], data['signals']


def save_features(features: Dict[str, Dict[int, Dict[str, np.ndarray]]], filename: str) -> None:
    """
    Save features to a file.
    
    Args:
        features: Dictionary with features.
        filename: The name of the file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(features, f)


def load_features(filename: str) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Load features from a file.
    
    Args:
        filename: The name of the file.
        
    Returns:
        Dictionary with features.
    """
    with open(filename, 'rb') as f:
        features = pickle.load(f)
    
    return features


def save_pathways(pathways: List[PropagationPathway], filename: str) -> None:
    """
    Save pathways to a file.
    
    Args:
        pathways: List of pathways.
        filename: The name of the file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(pathways, f)


def load_pathways(filename: str) -> List[PropagationPathway]:
    """
    Load pathways from a file.
    
    Args:
        filename: The name of the file.
        
    Returns:
        List of pathways.
    """
    with open(filename, 'rb') as f:
        pathways = pickle.load(f)
    
    return pathways


def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary with results.
        filename: The name of the file.
    """
    # Convert numpy arrays to lists
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    results_converted = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(results_converted, f, indent=4)


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load results from a JSON file.
    
    Args:
        filename: The name of the file.
        
    Returns:
        Dictionary with results.
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results


def save_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Save data to a CSV file.
    
    Args:
        data: DataFrame with data.
        filename: The name of the file.
    """
    data.to_csv(filename, index=False)


def load_csv(filename: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filename: The name of the file.
        
    Returns:
        DataFrame with data.
    """
    return pd.read_csv(filename)
