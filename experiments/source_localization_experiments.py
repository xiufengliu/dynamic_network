"""
Source localization experiments on synthetic data.

This script implements the source localization experiments described in Section 4.2.2 of the paper.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Set
import multiprocessing as mp
from functools import partial
import pickle
import networkx as nx
from scipy import stats

# Add the parent directory to the path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network.graph import DynamicNetwork
from src.network.generators import (
    generate_barabasi_albert_network,
    generate_erdos_renyi_network,
    generate_watts_strogatz_network,
    generate_grid_network
)
from src.feature_extraction.stft import STFT
from src.pathway_detection.detector import PathwayDetector
from src.pathway_detection.definition import PropagationPathway
from src.source_localization.localizer import SourceLocalizer
from src.source_localization.evaluation import (
    success_rate_at_k,
    mean_rank_of_true_source,
    error_distance
)
from src.utils.visualization import plot_network, plot_pathways, plot_time_series
from src.utils.io import save_results

# Import the synthetic data generation function from the main experiments script
from synthetic_experiments import generate_synthetic_data

# Create output directories
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


class PropagationCentrality:
    """
    Implementation of Propagation Centrality for source localization.
    
    This is a baseline method that ranks nodes based on their centrality in the network,
    weighted by their activation times.
    """
    
    def __init__(self):
        """Initialize the Propagation Centrality localizer."""
        pass
    
    def localize(self, network: DynamicNetwork, activation_times: Dict[int, float]) -> List[Union[int, str]]:
        """
        Localize the source using Propagation Centrality.
        
        Args:
            network: The network.
            activation_times: Dictionary mapping node indices to activation times.
            
        Returns:
            List of source node IDs, ranked by likelihood.
        """
        # Convert activation times to node IDs
        node_activation_times = {}
        for idx, time in activation_times.items():
            node_id = network.index_to_node(idx)
            node_activation_times[node_id] = time
        
        # Calculate propagation centrality for each node
        pc_scores = {}
        G = network.graph
        
        for node in G.nodes():
            if node not in node_activation_times:
                pc_scores[node] = 0
                continue
            
            # Calculate weighted sum of activation time differences
            score = 0
            for other_node in G.nodes():
                if other_node not in node_activation_times or other_node == node:
                    continue
                
                # Calculate shortest path length
                try:
                    path_length = nx.shortest_path_length(G, node, other_node)
                except nx.NetworkXNoPath:
                    continue
                
                # Calculate time difference
                time_diff = node_activation_times[other_node] - node_activation_times[node]
                
                # Only consider nodes activated after this node
                if time_diff > 0:
                    # Higher score for nodes that explain the activation pattern well
                    # (i.e., activation time difference proportional to path length)
                    score += 1.0 / (1.0 + abs(time_diff - path_length))
            
            pc_scores[node] = score
        
        # Rank nodes by PC score in descending order
        ranked_nodes = sorted(pc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in ranked_nodes]


class EarliestActivator:
    """
    Implementation of Earliest Activator for source localization.
    
    This is a baseline method that selects the node(s) with the earliest activation time.
    """
    
    def __init__(self):
        """Initialize the Earliest Activator localizer."""
        pass
    
    def localize(self, network: DynamicNetwork, activation_times: Dict[int, float]) -> List[Union[int, str]]:
        """
        Localize the source using Earliest Activator.
        
        Args:
            network: The network.
            activation_times: Dictionary mapping node indices to activation times.
            
        Returns:
            List of source node IDs, ranked by likelihood.
        """
        # Convert activation times to node IDs
        node_activation_times = {}
        for idx, time in activation_times.items():
            node_id = network.index_to_node(idx)
            node_activation_times[node_id] = time
        
        # Sort nodes by activation time
        sorted_nodes = sorted(node_activation_times.items(), key=lambda x: x[1])
        
        return [node for node, _ in sorted_nodes]


class CentralityBasedLocalizer:
    """
    Implementation of centrality-based source localization.
    
    This is a baseline method that ranks nodes based on their centrality in the network.
    """
    
    def __init__(self, centrality_type: str = 'degree'):
        """
        Initialize the centrality-based localizer.
        
        Args:
            centrality_type: Type of centrality to use ('degree' or 'betweenness').
        """
        self.centrality_type = centrality_type
    
    def localize(self, network: DynamicNetwork, activation_times: Dict[int, float]) -> List[Union[int, str]]:
        """
        Localize the source using centrality.
        
        Args:
            network: The network.
            activation_times: Dictionary mapping node indices to activation times.
            
        Returns:
            List of source node IDs, ranked by likelihood.
        """
        G = network.graph
        
        # Calculate centrality
        if self.centrality_type == 'degree':
            centrality = nx.degree_centrality(G)
        elif self.centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        else:
            raise ValueError(f"Unknown centrality type: {self.centrality_type}")
        
        # Rank nodes by centrality in descending order
        ranked_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in ranked_nodes]


def run_source_localization_experiment(
    network_types: List[str] = ['ba', 'er', 'ws', 'grid'],
    n_nodes_list: List[int] = [100, 500, 1000],
    snr_db_list: List[float] = [5, 10, 20],
    sparse_observation_ratios: List[float] = [1.0, 0.75, 0.5],
    n_runs: int = 10,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    Run source localization experiment.
    
    Args:
        network_types: List of network types to test.
        n_nodes_list: List of network sizes to test.
        snr_db_list: List of SNR values to test.
        sparse_observation_ratios: List of observation ratios to test.
        n_runs: Number of runs per configuration.
        base_seed: Base random seed.
        
    Returns:
        DataFrame with results.
    """
    results = []
    
    # Define localizers
    localizers = {
        'Our Method': SourceLocalizer(use_pathways=True),
        'Our Method (No Pathways)': SourceLocalizer(use_pathways=False),
        'Propagation Centrality': PropagationCentrality(),
        'Earliest Activator': EarliestActivator(),
        'Degree Centrality': CentralityBasedLocalizer(centrality_type='degree'),
        'Betweenness Centrality': CentralityBasedLocalizer(centrality_type='betweenness')
    }
    
    # Configure STFT and pathway detector
    stft = STFT(window_size=256, overlap=0.75)
    detector = PathwayDetector(delay_tolerance=0.5, phase_tolerance=np.pi/4, amplitude_threshold=0.2)
    event_freq = 0.1
    
    # Run experiments
    for network_type in network_types:
        for n_nodes in n_nodes_list:
            for snr_db in snr_db_list:
                for obs_ratio in sparse_observation_ratios:
                    for run in range(n_runs):
                        seed = base_seed + run
                        
                        # Generate synthetic data
                        network, signals, time, true_sources, true_pathways = generate_synthetic_data(
                            network_type=network_type,
                            n_nodes=n_nodes,
                            snr_db=snr_db,
                            delay_uncertainty=0.1,  # Fixed value
                            seed=seed
                        )
                        
                        # Apply sparse observation if needed
                        if obs_ratio < 1.0:
                            # Randomly select nodes to observe
                            rng = np.random.RandomState(seed)
                            n_observe = int(len(network.get_nodes()) * obs_ratio)
                            observe_indices = rng.choice(len(network.get_nodes()), size=n_observe, replace=False)
                            
                            # Create sparse signals
                            sparse_signals = {}
                            for i in observe_indices:
                                sparse_signals[i] = signals[i]
                        else:
                            sparse_signals = signals
                        
                        # Extract features
                        features = stft.extract_features(
                            np.array([sparse_signals[i] for i in sparse_signals.keys()]),
                            freq=event_freq,
                            amplitude_threshold=0.2
                        )
                        
                        # Detect pathways
                        detected_pathways = detector.detect(network, features, event_freq)
                        
                        # Get activation times
                        activation_times = {}
                        for i in features['activation_time']:
                            if features['activation_time'][i] is not None:
                                activation_times[i] = features['activation_time'][i]
                        
                        # Get node positions for error distance calculation
                        node_positions = nx.spring_layout(network.graph, seed=seed)
                        
                        # Run each localizer
                        for method_name, localizer in localizers.items():
                            # Skip pathway-based method if no pathways are detected
                            if method_name == 'Our Method' and not detected_pathways:
                                continue
                            
                            # Localize sources
                            if method_name in ['Our Method', 'Our Method (No Pathways)']:
                                predicted_sources = localizer.localize(network, features, detected_pathways if method_name == 'Our Method' else None)
                            else:
                                predicted_sources = localizer.localize(network, activation_times)
                            
                            # Calculate metrics
                            sr_1 = success_rate_at_k(predicted_sources, true_sources, k=1)
                            sr_3 = success_rate_at_k(predicted_sources, true_sources, k=3)
                            sr_5 = success_rate_at_k(predicted_sources, true_sources, k=5)
                            
                            # Calculate mean rank if possible
                            if method_name in ['Our Method', 'Our Method (No Pathways)']:
                                ranked_sources = localizer.rank_sources(network, features, detected_pathways if method_name == 'Our Method' else None)
                                mrts = mean_rank_of_true_source(ranked_sources, true_sources)
                            else:
                                mrts = mean_rank_of_true_source([(node, 0) for node in predicted_sources], true_sources)
                            
                            # Calculate error distance
                            ed = error_distance(predicted_sources, true_sources, node_positions)
                            
                            # Record results
                            results.append({
                                'network_type': network_type,
                                'n_nodes': n_nodes,
                                'snr_db': snr_db,
                                'obs_ratio': obs_ratio,
                                'run': run,
                                'method': method_name,
                                'sr_1': sr_1,
                                'sr_3': sr_3,
                                'sr_5': sr_5,
                                'mrts': mrts,
                                'ed': ed,
                                'n_detected_pathways': len(detected_pathways),
                                'n_observed_nodes': len(sparse_signals)
                            })
    
    return pd.DataFrame(results)


def plot_source_localization_results(results: pd.DataFrame, output_dir: str = 'results/figures'):
    """
    Plot source localization results.
    
    Args:
        results: DataFrame with results.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot SR@1 vs. SNR for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='snr_db',
        y='sr_1',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('Success Rate @ 1 vs. SNR by Network Type and Method')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Success Rate @ 1')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sr1_vs_snr.png'), dpi=300, bbox_inches='tight')
    
    # Plot SR@1 vs. observation ratio for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='obs_ratio',
        y='sr_1',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('Success Rate @ 1 vs. Observation Ratio by Network Type and Method')
    plt.xlabel('Observation Ratio')
    plt.ylabel('Success Rate @ 1')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sr1_vs_obs_ratio.png'), dpi=300, bbox_inches='tight')
    
    # Plot SR@1 vs. network size for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='n_nodes',
        y='sr_1',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('Success Rate @ 1 vs. Network Size by Network Type and Method')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Success Rate @ 1')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'sr1_vs_n_nodes.png'), dpi=300, bbox_inches='tight')
    
    # Create a summary table
    summary = results.groupby(['method', 'network_type']).agg({
        'sr_1': ['mean', 'std'],
        'sr_3': ['mean', 'std'],
        'sr_5': ['mean', 'std'],
        'mrts': ['mean', 'std'],
        'ed': ['mean', 'std']
    }).reset_index()
    
    # Save summary table
    summary.to_csv(os.path.join(output_dir, 'source_localization_summary.csv'), index=False)
    
    return summary


if __name__ == "__main__":
    # Run source localization experiment
    print("Running source localization experiment...")
    results = run_source_localization_experiment(
        network_types=['ba', 'er', 'ws', 'grid'],
        n_nodes_list=[100, 500],  # Reduced for faster execution
        snr_db_list=[5, 10, 20],
        sparse_observation_ratios=[1.0, 0.75, 0.5],
        n_runs=5  # Reduced for faster execution
    )
    
    # Save results
    results.to_csv('results/synthetic/source_localization_results.csv', index=False)
    
    # Plot results
    print("Plotting results...")
    summary = plot_source_localization_results(results)
    
    print("Experiment completed successfully!")
    print(f"Results saved to results/synthetic/source_localization_results.csv")
    print(f"Plots saved to results/figures/")
