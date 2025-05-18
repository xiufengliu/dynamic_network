"""
Scalability analysis.

This script implements the scalability analysis described in Section 4.5 of the paper.
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

# Import the synthetic data generation function from the main experiments script
from synthetic_experiments import generate_synthetic_data

# Create output directories
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


def measure_runtime(func, *args, **kwargs):
    """
    Measure the runtime of a function.
    
    Args:
        func: Function to measure.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        Tuple of (result, runtime_seconds).
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def run_scalability_analysis_network_size(
    n_nodes_list: List[int] = [100, 500, 1000, 2000, 5000],
    n_runs: int = 5,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    Run scalability analysis for network size.
    
    Args:
        n_nodes_list: List of network sizes to test.
        n_runs: Number of runs per configuration.
        base_seed: Base random seed.
        
    Returns:
        DataFrame with results.
    """
    results = []
    
    # Fixed parameters
    network_type = 'ba'
    snr_db = 10
    event_freq = 0.1
    
    # Configure STFT, pathway detector, source localizer, and optimizer
    stft = STFT(window_size=256, overlap=0.75)
    detector = PathwayDetector(delay_tolerance=0.5, phase_tolerance=np.pi/4, amplitude_threshold=0.2)
    localizer = SourceLocalizer()
    impact_model = ImpactModel(alpha=2.0)
    optimizer = ResourceOptimizer(impact_model)
    
    for n_nodes in n_nodes_list:
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
            
            # Measure feature extraction time
            _, feature_extraction_time = measure_runtime(
                stft.extract_features,
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=0.2
            )
            
            # Extract features for subsequent steps
            features = stft.extract_features(
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=0.2
            )
            
            # Measure pathway detection time
            _, pathway_detection_time = measure_runtime(
                detector.detect,
                network,
                features,
                event_freq
            )
            
            # Detect pathways for subsequent steps
            detected_pathways = detector.detect(network, features, event_freq)
            
            # Measure source localization time
            _, source_localization_time = measure_runtime(
                localizer.localize,
                network,
                features,
                detected_pathways
            )
            
            # Localize sources for subsequent steps
            detected_sources = localizer.localize(network, features, detected_pathways)
            
            # Calculate initial impacts
            initial_impacts = impact_model.calculate_initial_impacts(network, features)
            
            # Select critical nodes (10% of nodes)
            n_critical = max(1, int(len(network.get_nodes()) * 0.1))
            critical_nodes = [node for node, impact in sorted(initial_impacts.items(), key=lambda x: x[1], reverse=True)[:n_critical]]
            
            # Measure intervention optimization time
            _, intervention_time = measure_runtime(
                optimizer.optimize,
                network=network,
                pathways=detected_pathways,
                initial_impacts=initial_impacts,
                critical_nodes=critical_nodes,
                max_impact=0.1,
                budget=len(network.get_nodes()) * 0.1
            )
            
            # Record results
            results.append({
                'n_nodes': n_nodes,
                'n_edges': len(network.get_edges()),
                'run': run,
                'feature_extraction_time': feature_extraction_time,
                'pathway_detection_time': pathway_detection_time,
                'source_localization_time': source_localization_time,
                'intervention_time': intervention_time,
                'total_time': feature_extraction_time + pathway_detection_time + source_localization_time + intervention_time,
                'n_detected_pathways': len(detected_pathways)
            })
    
    return pd.DataFrame(results)


def run_scalability_analysis_signal_length(
    n_samples_list: List[int] = [500, 1000, 2000, 5000, 10000],
    n_runs: int = 5,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    Run scalability analysis for signal length.
    
    Args:
        n_samples_list: List of signal lengths to test.
        n_runs: Number of runs per configuration.
        base_seed: Base random seed.
        
    Returns:
        DataFrame with results.
    """
    results = []
    
    # Fixed parameters
    network_type = 'ba'
    n_nodes = 500
    snr_db = 10
    event_freq = 0.1
    
    # Configure STFT, pathway detector, source localizer, and optimizer
    detector = PathwayDetector(delay_tolerance=0.5, phase_tolerance=np.pi/4, amplitude_threshold=0.2)
    localizer = SourceLocalizer()
    impact_model = ImpactModel(alpha=2.0)
    optimizer = ResourceOptimizer(impact_model)
    
    for n_samples in n_samples_list:
        for run in range(n_runs):
            seed = base_seed + run
            
            # Generate synthetic data
            network, signals, time, true_sources, true_pathways = generate_synthetic_data(
                network_type=network_type,
                n_nodes=n_nodes,
                n_samples=n_samples,
                snr_db=snr_db,
                delay_uncertainty=0.1,  # Fixed value
                seed=seed
            )
            
            # Configure STFT with appropriate window size
            window_size = min(256, n_samples // 4)
            stft = STFT(window_size=window_size, overlap=0.75)
            
            # Measure feature extraction time
            _, feature_extraction_time = measure_runtime(
                stft.extract_features,
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=0.2
            )
            
            # Extract features for subsequent steps
            features = stft.extract_features(
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=0.2
            )
            
            # Measure pathway detection time
            _, pathway_detection_time = measure_runtime(
                detector.detect,
                network,
                features,
                event_freq
            )
            
            # Record results
            results.append({
                'n_samples': n_samples,
                'run': run,
                'feature_extraction_time': feature_extraction_time,
                'pathway_detection_time': pathway_detection_time,
                'total_time': feature_extraction_time + pathway_detection_time,
                'window_size': window_size
            })
    
    return pd.DataFrame(results)


def plot_scalability_results(network_results: pd.DataFrame, signal_results: pd.DataFrame, output_dir: str = 'results/figures'):
    """
    Plot scalability analysis results.
    
    Args:
        network_results: DataFrame with network size results.
        signal_results: DataFrame with signal length results.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot runtime vs. network size
    plt.figure(figsize=(12, 8))
    
    # Group by network size and calculate mean and std
    network_summary = network_results.groupby('n_nodes').agg({
        'feature_extraction_time': ['mean', 'std'],
        'pathway_detection_time': ['mean', 'std'],
        'source_localization_time': ['mean', 'std'],
        'intervention_time': ['mean', 'std'],
        'total_time': ['mean', 'std']
    })
    
    # Plot each component
    plt.errorbar(
        network_summary.index,
        network_summary[('feature_extraction_time', 'mean')],
        yerr=network_summary[('feature_extraction_time', 'std')],
        marker='o',
        label='Feature Extraction'
    )
    plt.errorbar(
        network_summary.index,
        network_summary[('pathway_detection_time', 'mean')],
        yerr=network_summary[('pathway_detection_time', 'std')],
        marker='s',
        label='Pathway Detection'
    )
    plt.errorbar(
        network_summary.index,
        network_summary[('source_localization_time', 'mean')],
        yerr=network_summary[('source_localization_time', 'std')],
        marker='^',
        label='Source Localization'
    )
    plt.errorbar(
        network_summary.index,
        network_summary[('intervention_time', 'mean')],
        yerr=network_summary[('intervention_time', 'std')],
        marker='d',
        label='Intervention Optimization'
    )
    plt.errorbar(
        network_summary.index,
        network_summary[('total_time', 'mean')],
        yerr=network_summary[('total_time', 'std')],
        marker='*',
        label='Total Time',
        linewidth=2
    )
    
    plt.title('Runtime vs. Network Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Runtime (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'runtime_vs_network_size.png'), dpi=300, bbox_inches='tight')
    
    # Plot runtime vs. signal length
    plt.figure(figsize=(12, 8))
    
    # Group by signal length and calculate mean and std
    signal_summary = signal_results.groupby('n_samples').agg({
        'feature_extraction_time': ['mean', 'std'],
        'pathway_detection_time': ['mean', 'std'],
        'total_time': ['mean', 'std']
    })
    
    # Plot each component
    plt.errorbar(
        signal_summary.index,
        signal_summary[('feature_extraction_time', 'mean')],
        yerr=signal_summary[('feature_extraction_time', 'std')],
        marker='o',
        label='Feature Extraction'
    )
    plt.errorbar(
        signal_summary.index,
        signal_summary[('pathway_detection_time', 'mean')],
        yerr=signal_summary[('pathway_detection_time', 'std')],
        marker='s',
        label='Pathway Detection'
    )
    plt.errorbar(
        signal_summary.index,
        signal_summary[('total_time', 'mean')],
        yerr=signal_summary[('total_time', 'std')],
        marker='*',
        label='Total Time',
        linewidth=2
    )
    
    plt.title('Runtime vs. Signal Length')
    plt.xlabel('Number of Samples')
    plt.ylabel('Runtime (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'runtime_vs_signal_length.png'), dpi=300, bbox_inches='tight')
    
    # Save summary tables
    network_summary.to_csv(os.path.join(output_dir, 'network_size_scalability_summary.csv'))
    signal_summary.to_csv(os.path.join(output_dir, 'signal_length_scalability_summary.csv'))
    
    return network_summary, signal_summary


if __name__ == "__main__":
    # Run scalability analysis for network size
    print("Running scalability analysis for network size...")
    network_results = run_scalability_analysis_network_size(
        n_nodes_list=[100, 500, 1000, 2000],  # Reduced for faster execution
        n_runs=3  # Reduced for faster execution
    )
    
    # Run scalability analysis for signal length
    print("Running scalability analysis for signal length...")
    signal_results = run_scalability_analysis_signal_length(
        n_samples_list=[500, 1000, 2000, 5000],  # Reduced for faster execution
        n_runs=3  # Reduced for faster execution
    )
    
    # Save results
    network_results.to_csv('results/synthetic/network_size_scalability_results.csv', index=False)
    signal_results.to_csv('results/synthetic/signal_length_scalability_results.csv', index=False)
    
    # Plot results
    print("Plotting results...")
    network_summary, signal_summary = plot_scalability_results(network_results, signal_results)
    
    print("Scalability analysis completed successfully!")
    print(f"Results saved to results/synthetic/")
    print(f"Plots saved to results/figures/")
