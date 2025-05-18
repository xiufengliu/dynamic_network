"""
Parameter sensitivity analysis.

This script implements the parameter sensitivity analysis described in Section 4.4 of the paper.
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
from src.feature_extraction.stft import STFT
from src.pathway_detection.detector import PathwayDetector
from src.pathway_detection.definition import PropagationPathway
from src.source_localization.localizer import SourceLocalizer
from src.utils.metrics import precision_recall_f1, pathway_jaccard_index
from src.source_localization.evaluation import success_rate_at_k

# Import the synthetic data generation function from the main experiments script
from synthetic_experiments import generate_synthetic_data

# Create output directories
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


def run_sensitivity_analysis(
    delay_tolerance_values: List[float] = [0.1, 0.2, 0.5, 1.0, 2.0],
    phase_tolerance_values: List[float] = [np.pi/16, np.pi/8, np.pi/4, np.pi/2, np.pi],
    amplitude_threshold_values: List[float] = [0.05, 0.1, 0.2, 0.5, 1.0],
    window_size_values: List[int] = [64, 128, 256, 512, 1024],
    n_runs: int = 10,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    Run parameter sensitivity analysis.
    
    Args:
        delay_tolerance_values: List of delay tolerance values to test.
        phase_tolerance_values: List of phase tolerance values to test.
        amplitude_threshold_values: List of amplitude threshold values to test.
        window_size_values: List of window size values to test.
        n_runs: Number of runs per configuration.
        base_seed: Base random seed.
        
    Returns:
        DataFrame with results.
    """
    results = []
    
    # Use a fixed network type and size for sensitivity analysis
    network_type = 'ba'
    n_nodes = 500
    snr_db = 10
    event_freq = 0.1
    
    # Run experiments for delay tolerance
    print("Testing delay tolerance...")
    for delay_tolerance in delay_tolerance_values:
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
            stft = STFT(window_size=256, overlap=0.75)  # Fixed window size
            features = stft.extract_features(
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=0.2  # Fixed amplitude threshold
            )
            
            # Detect pathways
            detector = PathwayDetector(
                delay_tolerance=delay_tolerance,
                phase_tolerance=np.pi/4,  # Fixed phase tolerance
                amplitude_threshold=0.2  # Fixed amplitude threshold
            )
            detected_pathways = detector.detect(network, features, event_freq)
            
            # Localize sources
            localizer = SourceLocalizer()
            detected_sources = localizer.localize(network, features, detected_pathways)
            
            # Calculate metrics
            if detected_pathways and true_pathways:
                precision, recall, f1 = precision_recall_f1(detected_pathways, true_pathways)
                pji = pathway_jaccard_index(detected_pathways, true_pathways)
            else:
                precision, recall, f1, pji = 0, 0, 0, 0
            
            sr_1 = success_rate_at_k(detected_sources, true_sources, k=1)
            
            # Record results
            results.append({
                'parameter': 'delay_tolerance',
                'value': delay_tolerance,
                'run': run,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pji': pji,
                'sr_1': sr_1,
                'n_detected_pathways': len(detected_pathways),
                'n_true_pathways': len(true_pathways)
            })
    
    # Run experiments for phase tolerance
    print("Testing phase tolerance...")
    for phase_tolerance in phase_tolerance_values:
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
            stft = STFT(window_size=256, overlap=0.75)  # Fixed window size
            features = stft.extract_features(
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=0.2  # Fixed amplitude threshold
            )
            
            # Detect pathways
            detector = PathwayDetector(
                delay_tolerance=0.5,  # Fixed delay tolerance
                phase_tolerance=phase_tolerance,
                amplitude_threshold=0.2  # Fixed amplitude threshold
            )
            detected_pathways = detector.detect(network, features, event_freq)
            
            # Localize sources
            localizer = SourceLocalizer()
            detected_sources = localizer.localize(network, features, detected_pathways)
            
            # Calculate metrics
            if detected_pathways and true_pathways:
                precision, recall, f1 = precision_recall_f1(detected_pathways, true_pathways)
                pji = pathway_jaccard_index(detected_pathways, true_pathways)
            else:
                precision, recall, f1, pji = 0, 0, 0, 0
            
            sr_1 = success_rate_at_k(detected_sources, true_sources, k=1)
            
            # Record results
            results.append({
                'parameter': 'phase_tolerance',
                'value': phase_tolerance,
                'run': run,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pji': pji,
                'sr_1': sr_1,
                'n_detected_pathways': len(detected_pathways),
                'n_true_pathways': len(true_pathways)
            })
    
    # Run experiments for amplitude threshold
    print("Testing amplitude threshold...")
    for amplitude_threshold in amplitude_threshold_values:
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
            stft = STFT(window_size=256, overlap=0.75)  # Fixed window size
            features = stft.extract_features(
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=amplitude_threshold
            )
            
            # Detect pathways
            detector = PathwayDetector(
                delay_tolerance=0.5,  # Fixed delay tolerance
                phase_tolerance=np.pi/4,  # Fixed phase tolerance
                amplitude_threshold=amplitude_threshold
            )
            detected_pathways = detector.detect(network, features, event_freq)
            
            # Localize sources
            localizer = SourceLocalizer()
            detected_sources = localizer.localize(network, features, detected_pathways)
            
            # Calculate metrics
            if detected_pathways and true_pathways:
                precision, recall, f1 = precision_recall_f1(detected_pathways, true_pathways)
                pji = pathway_jaccard_index(detected_pathways, true_pathways)
            else:
                precision, recall, f1, pji = 0, 0, 0, 0
            
            sr_1 = success_rate_at_k(detected_sources, true_sources, k=1)
            
            # Record results
            results.append({
                'parameter': 'amplitude_threshold',
                'value': amplitude_threshold,
                'run': run,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pji': pji,
                'sr_1': sr_1,
                'n_detected_pathways': len(detected_pathways),
                'n_true_pathways': len(true_pathways)
            })
    
    # Run experiments for window size
    print("Testing window size...")
    for window_size in window_size_values:
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
            stft = STFT(window_size=window_size, overlap=0.75)
            features = stft.extract_features(
                np.array([signals[i] for i in range(len(network))]),
                freq=event_freq,
                amplitude_threshold=0.2  # Fixed amplitude threshold
            )
            
            # Detect pathways
            detector = PathwayDetector(
                delay_tolerance=0.5,  # Fixed delay tolerance
                phase_tolerance=np.pi/4,  # Fixed phase tolerance
                amplitude_threshold=0.2  # Fixed amplitude threshold
            )
            detected_pathways = detector.detect(network, features, event_freq)
            
            # Localize sources
            localizer = SourceLocalizer()
            detected_sources = localizer.localize(network, features, detected_pathways)
            
            # Calculate metrics
            if detected_pathways and true_pathways:
                precision, recall, f1 = precision_recall_f1(detected_pathways, true_pathways)
                pji = pathway_jaccard_index(detected_pathways, true_pathways)
            else:
                precision, recall, f1, pji = 0, 0, 0, 0
            
            sr_1 = success_rate_at_k(detected_sources, true_sources, k=1)
            
            # Record results
            results.append({
                'parameter': 'window_size',
                'value': window_size,
                'run': run,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pji': pji,
                'sr_1': sr_1,
                'n_detected_pathways': len(detected_pathways),
                'n_true_pathways': len(true_pathways)
            })
    
    return pd.DataFrame(results)


def plot_sensitivity_results(results: pd.DataFrame, output_dir: str = 'results/figures'):
    """
    Plot sensitivity analysis results.
    
    Args:
        results: DataFrame with results.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group parameters
    parameters = results['parameter'].unique()
    
    for parameter in parameters:
        param_results = results[results['parameter'] == parameter]
        
        # Convert values to string for better plotting
        if parameter == 'phase_tolerance':
            param_results['value_str'] = param_results['value'].apply(lambda x: f'π/{int(np.pi/x)}' if x < np.pi else 'π')
        else:
            param_results['value_str'] = param_results['value'].astype(str)
        
        # Plot F1-score vs. parameter value
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=param_results,
            x='value',
            y='f1',
            marker='o',
            ci=95
        )
        plt.title(f'F1-Score vs. {parameter.replace("_", " ").title()}')
        plt.xlabel(parameter.replace('_', ' ').title())
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'f1_vs_{parameter}.png'), dpi=300, bbox_inches='tight')
        
        # Plot SR@1 vs. parameter value
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=param_results,
            x='value',
            y='sr_1',
            marker='o',
            ci=95
        )
        plt.title(f'Success Rate @ 1 vs. {parameter.replace("_", " ").title()}')
        plt.xlabel(parameter.replace('_', ' ').title())
        plt.ylabel('Success Rate @ 1')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'sr1_vs_{parameter}.png'), dpi=300, bbox_inches='tight')
        
        # Plot precision and recall vs. parameter value
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=param_results,
            x='value',
            y='precision',
            marker='o',
            label='Precision',
            ci=95
        )
        sns.lineplot(
            data=param_results,
            x='value',
            y='recall',
            marker='s',
            label='Recall',
            ci=95
        )
        plt.title(f'Precision and Recall vs. {parameter.replace("_", " ").title()}')
        plt.xlabel(parameter.replace('_', ' ').title())
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'precision_recall_vs_{parameter}.png'), dpi=300, bbox_inches='tight')
    
    # Create a summary table
    summary = results.groupby(['parameter', 'value']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'sr_1': ['mean', 'std']
    }).reset_index()
    
    # Save summary table
    summary.to_csv(os.path.join(output_dir, 'sensitivity_analysis_summary.csv'), index=False)
    
    return summary


if __name__ == "__main__":
    # Run sensitivity analysis
    print("Running parameter sensitivity analysis...")
    results = run_sensitivity_analysis(
        delay_tolerance_values=[0.1, 0.2, 0.5, 1.0, 2.0],
        phase_tolerance_values=[np.pi/16, np.pi/8, np.pi/4, np.pi/2, np.pi],
        amplitude_threshold_values=[0.05, 0.1, 0.2, 0.5, 1.0],
        window_size_values=[64, 128, 256, 512, 1024],
        n_runs=5  # Reduced for faster execution
    )
    
    # Save results
    results.to_csv('results/synthetic/sensitivity_analysis_results.csv', index=False)
    
    # Plot results
    print("Plotting results...")
    summary = plot_sensitivity_results(results)
    
    print("Sensitivity analysis completed successfully!")
    print(f"Results saved to results/synthetic/sensitivity_analysis_results.csv")
    print(f"Plots saved to results/figures/")
