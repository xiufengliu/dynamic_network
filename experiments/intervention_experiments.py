"""
Intervention optimization experiments on synthetic data.

This script implements the intervention optimization experiments described in Section 4.2.3 of the paper.
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
from src.intervention.impact_model import ImpactModel
from src.intervention.optimizer import ResourceOptimizer
from src.intervention.greedy_heuristic import GreedyHeuristic
from src.utils.metrics import (
    total_impact_reduction,
    cost_effectiveness_ratio,
    constraint_satisfaction
)
from src.utils.visualization import plot_network, plot_pathways, plot_time_series
from src.utils.io import save_results

# Import the synthetic data generation function from the main experiments script
from synthetic_experiments import generate_synthetic_data

# Create output directories
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


class RandomPlacement:
    """
    Implementation of Random Placement for intervention optimization.
    
    This is a baseline method that randomly allocates resources to nodes.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the Random Placement optimizer.
        
        Args:
            seed: Random seed.
        """
        self.rng = np.random.RandomState(seed)
    
    def optimize(self, network: DynamicNetwork, pathways: List[PropagationPathway],
                initial_impacts: Dict[Union[int, str], float],
                critical_nodes: List[Union[int, str]],
                max_impact: float = 0.1, budget: Optional[float] = None,
                costs: Optional[Dict[Union[int, str], float]] = None,
                capabilities: Optional[Dict[Union[int, str], float]] = None) -> Dict[Union[int, str], float]:
        """
        Optimize resource allocation using random placement.
        
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
        
        # Calculate number of nodes to allocate resources to
        if budget is None:
            # Allocate to 10% of nodes if no budget is specified
            n_allocate = max(1, int(0.1 * len(network.get_nodes())))
        else:
            # Calculate how many nodes we can allocate to within budget
            sorted_costs = sorted([(node, costs.get(node, 1.0)) for node in network.get_nodes()], key=lambda x: x[1])
            cumulative_cost = 0
            n_allocate = 0
            for node, cost in sorted_costs:
                if cumulative_cost + cost <= budget:
                    cumulative_cost += cost
                    n_allocate += 1
                else:
                    break
        
        # Randomly select nodes to allocate resources to
        all_nodes = list(network.get_nodes())
        selected_nodes = self.rng.choice(all_nodes, size=min(n_allocate, len(all_nodes)), replace=False)
        
        # Allocate resources
        allocation = {}
        for node in selected_nodes:
            allocation[node] = capabilities.get(node, 0.5)
        
        return allocation


class HighestImpactFirst:
    """
    Implementation of Highest Impact First for intervention optimization.
    
    This is a baseline method that allocates resources to nodes with the highest initial impact.
    """
    
    def optimize(self, network: DynamicNetwork, pathways: List[PropagationPathway],
                initial_impacts: Dict[Union[int, str], float],
                critical_nodes: List[Union[int, str]],
                max_impact: float = 0.1, budget: Optional[float] = None,
                costs: Optional[Dict[Union[int, str], float]] = None,
                capabilities: Optional[Dict[Union[int, str], float]] = None) -> Dict[Union[int, str], float]:
        """
        Optimize resource allocation using highest impact first.
        
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
        
        # Sort nodes by impact in descending order
        sorted_nodes = sorted(initial_impacts.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate resources
        allocation = {}
        total_cost = 0
        
        for node, impact in sorted_nodes:
            # Skip if node has no capability
            if node not in capabilities or capabilities[node] <= 0:
                continue
            
            # Calculate cost
            cost = costs.get(node, 1.0)
            
            # Check if we can afford it
            if budget is not None and total_cost + cost > budget:
                continue
            
            # Allocate resource
            allocation[node] = capabilities[node]
            total_cost += cost
            
            # Stop if we've allocated to all critical nodes
            if len(allocation) >= len(critical_nodes):
                break
        
        return allocation


class SourceProximity:
    """
    Implementation of Source Proximity for intervention optimization.
    
    This is a baseline method that allocates resources to nodes close to the source.
    """
    
    def optimize(self, network: DynamicNetwork, pathways: List[PropagationPathway],
                initial_impacts: Dict[Union[int, str], float],
                critical_nodes: List[Union[int, str]],
                max_impact: float = 0.1, budget: Optional[float] = None,
                costs: Optional[Dict[Union[int, str], float]] = None,
                capabilities: Optional[Dict[Union[int, str], float]] = None,
                sources: Optional[List[Union[int, str]]] = None) -> Dict[Union[int, str], float]:
        """
        Optimize resource allocation using source proximity.
        
        Args:
            network: The network.
            pathways: List of detected pathways.
            initial_impacts: Dictionary mapping node IDs to initial impact values.
            critical_nodes: List of critical node IDs.
            max_impact: Maximum permissible impact at critical nodes.
            budget: Maximum budget for resource allocation.
            costs: Dictionary mapping node IDs to resource costs.
            capabilities: Dictionary mapping node IDs to resource capabilities.
            sources: List of source node IDs.
            
        Returns:
            Dictionary mapping node IDs to allocated resource capabilities.
        """
        # If costs are not specified, use unit costs
        if costs is None:
            costs = {node_id: 1.0 for node_id in network.get_nodes()}
        
        # If capabilities are not specified, use default capabilities
        if capabilities is None:
            capabilities = {node_id: 0.5 for node_id in network.get_nodes()}
        
        # If sources are not specified, use the first node of each pathway
        if sources is None:
            sources = []
            for pathway in pathways:
                if pathway.nodes:
                    sources.append(pathway.nodes[0])
        
        # If still no sources, use nodes with highest impact
        if not sources:
            sorted_nodes = sorted(initial_impacts.items(), key=lambda x: x[1], reverse=True)
            if sorted_nodes:
                sources = [sorted_nodes[0][0]]
        
        # If still no sources, return empty allocation
        if not sources:
            return {}
        
        # Calculate distances from each node to the nearest source
        G = network.graph
        distances = {}
        
        for node in G.nodes():
            min_distance = float('inf')
            for source in sources:
                try:
                    distance = nx.shortest_path_length(G, source, node)
                    min_distance = min(min_distance, distance)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            if min_distance < float('inf'):
                distances[node] = min_distance
        
        # Sort nodes by distance in ascending order
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1])
        
        # Allocate resources
        allocation = {}
        total_cost = 0
        
        for node, distance in sorted_nodes:
            # Skip if node has no capability
            if node not in capabilities or capabilities[node] <= 0:
                continue
            
            # Calculate cost
            cost = costs.get(node, 1.0)
            
            # Check if we can afford it
            if budget is not None and total_cost + cost > budget:
                continue
            
            # Allocate resource
            allocation[node] = capabilities[node]
            total_cost += cost
            
            # Stop if we've allocated to enough nodes
            if budget is not None and total_cost >= budget:
                break
        
        return allocation


def run_intervention_experiment(
    network_types: List[str] = ['ba', 'er', 'ws', 'grid'],
    n_nodes_list: List[int] = [100, 500, 1000],
    snr_db_list: List[float] = [10],  # Fixed SNR for simplicity
    budget_ratios: List[float] = [0.05, 0.1, 0.2],  # Budget as fraction of network size
    critical_node_ratios: List[float] = [0.05, 0.1, 0.2],  # Critical nodes as fraction of network size
    n_runs: int = 10,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    Run intervention optimization experiment.
    
    Args:
        network_types: List of network types to test.
        n_nodes_list: List of network sizes to test.
        snr_db_list: List of SNR values to test.
        budget_ratios: List of budget ratios to test.
        critical_node_ratios: List of critical node ratios to test.
        n_runs: Number of runs per configuration.
        base_seed: Base random seed.
        
    Returns:
        DataFrame with results.
    """
    results = []
    
    # Define optimizers
    optimizers = {
        'Our Method': ResourceOptimizer(),
        'Random Placement': RandomPlacement(),
        'Highest Impact First': HighestImpactFirst(),
        'Source Proximity': SourceProximity()
    }
    
    # Configure STFT, pathway detector, and source localizer
    stft = STFT(window_size=256, overlap=0.75)
    detector = PathwayDetector(delay_tolerance=0.5, phase_tolerance=np.pi/4, amplitude_threshold=0.2)
    localizer = SourceLocalizer()
    event_freq = 0.1
    
    # Run experiments
    for network_type in network_types:
        for n_nodes in n_nodes_list:
            for snr_db in snr_db_list:
                for budget_ratio in budget_ratios:
                    for critical_ratio in critical_node_ratios:
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
                            
                            # Extract features
                            features = stft.extract_features(
                                np.array([signals[i] for i in range(len(network))]),
                                freq=event_freq,
                                amplitude_threshold=0.2
                            )
                            
                            # Detect pathways
                            detected_pathways = detector.detect(network, features, event_freq)
                            
                            # Localize sources
                            detected_sources = localizer.localize(network, features, detected_pathways)
                            
                            # Create impact model
                            impact_model = ImpactModel(alpha=2.0)
                            initial_impacts = impact_model.calculate_initial_impacts(network, features)
                            impact_model.generate_transmission_factors(network, seed=seed)
                            
                            # Select critical nodes (nodes with highest impact)
                            n_critical = max(1, int(len(network.get_nodes()) * critical_ratio))
                            critical_nodes = [node for node, impact in sorted(initial_impacts.items(), key=lambda x: x[1], reverse=True)[:n_critical]]
                            
                            # Calculate budget
                            budget = len(network.get_nodes()) * budget_ratio
                            
                            # Run each optimizer
                            for method_name, optimizer in optimizers.items():
                                # Skip our method if no pathways are detected
                                if method_name == 'Our Method' and not detected_pathways:
                                    continue
                                
                                # Optimize resource allocation
                                if method_name == 'Source Proximity':
                                    allocation = optimizer.optimize(
                                        network=network,
                                        pathways=detected_pathways,
                                        initial_impacts=initial_impacts,
                                        critical_nodes=critical_nodes,
                                        max_impact=0.1,
                                        budget=budget,
                                        sources=detected_sources
                                    )
                                else:
                                    allocation = optimizer.optimize(
                                        network=network,
                                        pathways=detected_pathways,
                                        initial_impacts=initial_impacts,
                                        critical_nodes=critical_nodes,
                                        max_impact=0.1,
                                        budget=budget
                                    )
                                
                                # Calculate impacts with resources
                                final_impacts = impact_model.calculate_impacts_with_resources(
                                    network=network,
                                    initial_impacts=initial_impacts,
                                    pathways=detected_pathways,
                                    resources=allocation
                                )
                                
                                # Calculate metrics
                                tir = total_impact_reduction(initial_impacts, final_impacts)
                                total_cost = sum(allocation.values())
                                cer = cost_effectiveness_ratio(tir, total_cost)
                                constraints_satisfied = constraint_satisfaction(final_impacts, critical_nodes, max_impact=0.1)
                                
                                # Record results
                                results.append({
                                    'network_type': network_type,
                                    'n_nodes': n_nodes,
                                    'snr_db': snr_db,
                                    'budget_ratio': budget_ratio,
                                    'critical_ratio': critical_ratio,
                                    'run': run,
                                    'method': method_name,
                                    'tir': tir,
                                    'cer': cer,
                                    'constraints_satisfied': constraints_satisfied,
                                    'n_allocated': len(allocation),
                                    'total_cost': total_cost,
                                    'n_detected_pathways': len(detected_pathways),
                                    'n_critical_nodes': len(critical_nodes)
                                })
    
    return pd.DataFrame(results)


def plot_intervention_results(results: pd.DataFrame, output_dir: str = 'results/figures'):
    """
    Plot intervention optimization results.
    
    Args:
        results: DataFrame with results.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot TIR vs. budget ratio for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='budget_ratio',
        y='tir',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('Total Impact Reduction vs. Budget Ratio by Network Type and Method')
    plt.xlabel('Budget Ratio')
    plt.ylabel('Total Impact Reduction')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tir_vs_budget_ratio.png'), dpi=300, bbox_inches='tight')
    
    # Plot CER vs. budget ratio for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='budget_ratio',
        y='cer',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('Cost-Effectiveness Ratio vs. Budget Ratio by Network Type and Method')
    plt.xlabel('Budget Ratio')
    plt.ylabel('Cost-Effectiveness Ratio')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cer_vs_budget_ratio.png'), dpi=300, bbox_inches='tight')
    
    # Plot constraint satisfaction rate vs. budget ratio for each network type and method
    plt.figure(figsize=(12, 8))
    constraint_data = results.groupby(['network_type', 'budget_ratio', 'method'])['constraints_satisfied'].mean().reset_index()
    sns.lineplot(
        data=constraint_data,
        x='budget_ratio',
        y='constraints_satisfied',
        hue='method',
        style='network_type',
        markers=True
    )
    plt.title('Constraint Satisfaction Rate vs. Budget Ratio by Network Type and Method')
    plt.xlabel('Budget Ratio')
    plt.ylabel('Constraint Satisfaction Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'constraint_satisfaction_vs_budget_ratio.png'), dpi=300, bbox_inches='tight')
    
    # Create a summary table
    summary = results.groupby(['method', 'network_type']).agg({
        'tir': ['mean', 'std'],
        'cer': ['mean', 'std'],
        'constraints_satisfied': ['mean', 'std']
    }).reset_index()
    
    # Save summary table
    summary.to_csv(os.path.join(output_dir, 'intervention_summary.csv'), index=False)
    
    return summary


if __name__ == "__main__":
    # Run intervention experiment
    print("Running intervention optimization experiment...")
    results = run_intervention_experiment(
        network_types=['ba', 'er', 'ws', 'grid'],
        n_nodes_list=[100, 500],  # Reduced for faster execution
        snr_db_list=[10],  # Fixed SNR for simplicity
        budget_ratios=[0.05, 0.1, 0.2],
        critical_node_ratios=[0.05, 0.1, 0.2],
        n_runs=5  # Reduced for faster execution
    )
    
    # Save results
    results.to_csv('results/synthetic/intervention_results.csv', index=False)
    
    # Plot results
    print("Plotting results...")
    summary = plot_intervention_results(results)
    
    print("Experiment completed successfully!")
    print(f"Results saved to results/synthetic/intervention_results.csv")
    print(f"Plots saved to results/figures/")
