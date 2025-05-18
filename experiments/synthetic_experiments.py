"""
Experimental evaluation on synthetic data.

This script implements the experiments described in Section 4.2 of the paper:
- Pathway Detection Accuracy
- Source Localization Precision
- Effectiveness of Optimized Intervention
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
    precision_recall_f1,
    pathway_jaccard_index,
    total_impact_reduction,
    cost_effectiveness_ratio,
    constraint_satisfaction
)
from src.utils.visualization import (
    plot_network,
    plot_pathways,
    plot_time_series
)
from src.utils.io import (
    save_network,
    save_time_series,
    save_features,
    save_pathways,
    save_results
)

# Create output directories
os.makedirs('data/synthetic/ba_networks', exist_ok=True)
os.makedirs('data/synthetic/er_networks', exist_ok=True)
os.makedirs('data/synthetic/ws_networks', exist_ok=True)
os.makedirs('data/synthetic/grid_networks', exist_ok=True)
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


def generate_synthetic_data(
    network_type: str,
    n_nodes: int,
    n_samples: int = 1000,
    event_freq: float = 0.1,
    snr_db: float = 10.0,
    delay_uncertainty: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[DynamicNetwork, Dict[int, np.ndarray], np.ndarray, List[int], List[PropagationPathway]]:
    """
    Generate synthetic data for experiments.

    Args:
        network_type: Type of network ('ba', 'er', 'ws', 'grid').
        n_nodes: Number of nodes.
        n_samples: Number of time samples.
        event_freq: Characteristic frequency of the event.
        snr_db: Signal-to-noise ratio in dB.
        delay_uncertainty: Standard deviation of delay noise as fraction of nominal delay.
        seed: Random seed.

    Returns:
        Tuple of (network, signals, time, true_sources, true_pathways).
    """
    # Set random seed
    rng = np.random.RandomState(seed)

    # Generate network based on type
    if network_type == 'ba':
        network = generate_barabasi_albert_network(n=n_nodes, m=4, seed=seed)
    elif network_type == 'er':
        # Calculate p to achieve similar average degree as BA with m=4
        p = 4 / (n_nodes - 1)
        network = generate_erdos_renyi_network(n=n_nodes, p=p, seed=seed)
    elif network_type == 'ws':
        network = generate_watts_strogatz_network(n=n_nodes, k=4, p=0.1, seed=seed)
    elif network_type == 'grid':
        # Calculate grid size to get approximately n_nodes
        grid_size = int(np.sqrt(n_nodes))
        network = generate_grid_network(n=grid_size, m=grid_size, seed=seed)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    # Generate time array
    fs = 1.0  # Sampling frequency
    time = np.arange(n_samples) / fs

    # Select a source node
    source_idx = rng.randint(0, len(network.get_nodes()))
    source_node = network.index_to_node(source_idx)
    true_sources = [source_node]

    # Initialize signals
    signals = {}
    for i in range(len(network.get_nodes())):
        signals[i] = np.zeros(n_samples)

    # Generate event at source node
    event_amplitude = 1.0
    event_phase = rng.uniform(0, 2 * np.pi)
    event_duration = n_samples // 5
    event_start = n_samples // 4

    # Create Gaussian pulse envelope
    t = np.arange(event_duration)
    envelope = np.exp(-(t - event_duration/2)**2 / (2 * (event_duration/6)**2))

    # Generate oscillatory event at source
    event = envelope * np.sin(2 * np.pi * event_freq * t / fs + event_phase)
    signals[source_idx][event_start:event_start+event_duration] = event_amplitude * event

    # Track true pathways
    true_pathways = []

    # Propagate event through the network
    activation_times = {source_idx: event_start}
    activation_phases = {source_idx: event_phase}
    processed_nodes = {source_idx}
    nodes_to_process = [source_idx]
    parent_map = {}  # Maps node index to its parent in the propagation tree

    while nodes_to_process:
        current_idx = nodes_to_process.pop(0)
        current_node = network.index_to_node(current_idx)

        # Get neighbors
        neighbors = network.get_neighbors(current_node)

        for neighbor in neighbors:
            neighbor_idx = network.node_to_index(neighbor)

            # Skip if already processed
            if neighbor_idx in processed_nodes:
                continue

            # Get propagation delay
            delay = network.get_nominal_delay(current_node, neighbor)
            delay_samples = int(delay * fs)

            # Add random variation to delay based on delay_uncertainty
            delay_variation = rng.normal(0, delay_uncertainty * delay)
            delay_samples += int(delay_variation * fs)

            # Ensure delay is positive
            delay_samples = max(1, delay_samples)

            # Calculate activation time
            activation_time = activation_times[current_idx] + delay_samples

            # Skip if activation time is too late
            if activation_time + event_duration >= n_samples:
                continue

            # Calculate attenuation
            attenuation = rng.uniform(0.7, 0.9)

            # Calculate phase shift
            phase_shift = 2 * np.pi * event_freq * delay

            # Generate event at neighbor
            neighbor_phase = activation_phases[current_idx] - phase_shift
            neighbor_event = envelope * np.sin(2 * np.pi * event_freq * t / fs + neighbor_phase)
            signals[neighbor_idx][activation_time:activation_time+event_duration] = event_amplitude * attenuation * neighbor_event

            # Add to activation times and phases
            activation_times[neighbor_idx] = activation_time
            activation_phases[neighbor_idx] = neighbor_phase

            # Record parent
            parent_map[neighbor_idx] = current_idx

            # Mark as processed and add to queue
            processed_nodes.add(neighbor_idx)
            nodes_to_process.append(neighbor_idx)

    # Construct true pathways
    for node_idx in processed_nodes:
        if node_idx == source_idx:
            continue

        # Trace back to source
        path = [node_idx]
        current = node_idx
        while current in parent_map:
            current = parent_map[current]
            path.append(current)

        # Reverse to get source -> node path
        path.reverse()

        # Convert indices to node IDs
        path_nodes = [network.index_to_node(idx) for idx in path]

        # Create pathway
        pathway = PropagationPathway(path_nodes, event_freq)

        # Add delays
        for i in range(len(path) - 1):
            source_idx = path[i]
            target_idx = path[i + 1]
            delay = (activation_times[target_idx] - activation_times[source_idx]) / fs
            pathway.delays.append(delay)

        # Add phases
        for idx in path:
            pathway.phases.append(activation_phases[idx])

        # Add activation times
        for idx in path:
            pathway.activation_times.append(activation_times[idx] / fs)

        true_pathways.append(pathway)

    # Add noise
    for i in range(len(network.get_nodes())):
        # Calculate signal power
        signal_power = np.mean(signals[i]**2)

        # Calculate noise power
        noise_power = signal_power / (10**(snr_db/10)) if signal_power > 0 else 0.01

        # Add noise
        noise = rng.normal(0, np.sqrt(noise_power), n_samples)
        signals[i] += noise

    return network, signals, time, true_sources, true_pathways


def load_synthetic_dataset(
    network_type: str,
    n_nodes: int,
    snr_db: float,
    delay_uncertainty: float,
    dataset_idx: int,
    data_dir: str = '../data/synthetic'
) -> Tuple[DynamicNetwork, Dict[int, np.ndarray], np.ndarray, List[Union[int, str]], List[PropagationPathway]]:
    """
    Load a pre-generated synthetic dataset.

    Args:
        network_type: Type of network ('ba', 'er', 'ws', 'grid').
        n_nodes: Number of nodes.
        snr_db: Signal-to-noise ratio in dB.
        delay_uncertainty: Delay uncertainty value.
        dataset_idx: Dataset index.
        data_dir: Directory containing the datasets.

    Returns:
        Tuple of (network, signals, time, true_sources, true_pathways).
    """
    # Create filename
    filename_base = f"{network_type}_{n_nodes}_{snr_db}db_{delay_uncertainty}delay_{dataset_idx}"
    network_filename = os.path.join(data_dir, f'{network_type}_networks', f"{filename_base}.graphml")
    data_filename = os.path.join(data_dir, f'{network_type}_networks', f"{filename_base}.pkl")

    # Check if files exist
    if not os.path.exists(network_filename) or not os.path.exists(data_filename):
        # If files don't exist, generate the dataset
        print(f"Dataset {filename_base} not found. Generating...")
        network, signals, time, true_sources, true_pathways = generate_synthetic_data(
            network_type=network_type,
            n_nodes=n_nodes,
            snr_db=snr_db,
            delay_uncertainty=delay_uncertainty,
            seed=42 + dataset_idx
        )

        # Save the dataset for future use
        os.makedirs(os.path.dirname(network_filename), exist_ok=True)
        network.save_to_file(network_filename)

        with open(data_filename, 'wb') as f:
            pickle.dump({
                'signals': signals,
                'time': time,
                'true_sources': true_sources,
                'true_pathways': true_pathways,
                'metadata': {
                    'network_type': network_type,
                    'n_nodes': n_nodes,
                    'snr_db': snr_db,
                    'delay_uncertainty': delay_uncertainty,
                    'seed': 42 + dataset_idx
                }
            }, f)
    else:
        # Load network
        network = DynamicNetwork()
        network.load_from_file(network_filename)

        # Load data
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        signals = data['signals']
        time = data['time']
        true_sources = data['true_sources']
        true_pathways = data['true_pathways']

    return network, signals, time, true_sources, true_pathways


def run_pathway_detection_experiment(
    network_types: List[str] = ['ba', 'er', 'ws', 'grid'],
    n_nodes_list: List[int] = [100, 500, 1000],
    snr_db_list: List[float] = [5, 10, 20],
    delay_uncertainty_list: List[float] = [0.05, 0.1, 0.2],
    n_runs: int = 10,
    base_seed: int = 42,
    use_pregenerated_data: bool = True,
    data_dir: str = '../data/synthetic'
) -> pd.DataFrame:
    """
    Run pathway detection experiment.

    Args:
        network_types: List of network types to test.
        n_nodes_list: List of network sizes to test.
        snr_db_list: List of SNR values to test.
        delay_uncertainty_list: List of delay uncertainty values to test.
        n_runs: Number of runs per configuration.
        base_seed: Base random seed.

    Returns:
        DataFrame with results.
    """
    results = []

    # Define baseline methods
    methods = {
        'Full': PathwayDetector(delay_tolerance=0.5, phase_tolerance=np.pi/4, amplitude_threshold=0.2),
        'No-Phase': PathwayDetector(delay_tolerance=0.5, phase_tolerance=np.inf, amplitude_threshold=0.2),
        'TCDC': PathwayDetector(delay_tolerance=0.5, phase_tolerance=np.inf, amplitude_threshold=0.2),
        'TC': PathwayDetector(delay_tolerance=np.inf, phase_tolerance=np.inf, amplitude_threshold=0.2)
    }

    # Configure STFT
    stft = STFT(window_size=256, overlap=0.75)
    event_freq = 0.1

    # Run experiments
    for network_type in network_types:
        for n_nodes in n_nodes_list:
            for snr_db in snr_db_list:
                for delay_uncertainty in delay_uncertainty_list:
                    for run in range(n_runs):
                        # Load or generate synthetic data
                        if use_pregenerated_data:
                            network, signals, time, true_sources, true_pathways = load_synthetic_dataset(
                                network_type=network_type,
                                n_nodes=n_nodes,
                                snr_db=snr_db,
                                delay_uncertainty=delay_uncertainty,
                                dataset_idx=run,
                                data_dir=data_dir
                            )
                        else:
                            seed = base_seed + run
                            network, signals, time, true_sources, true_pathways = generate_synthetic_data(
                                network_type=network_type,
                                n_nodes=n_nodes,
                                snr_db=snr_db,
                                delay_uncertainty=delay_uncertainty,
                                seed=seed
                            )

                        # Extract features
                        features = stft.extract_features(
                            np.array([signals[i] for i in range(len(network))]),
                            freq=event_freq,
                            amplitude_threshold=0.2
                        )

                        # Run each method
                        for method_name, detector in methods.items():
                            # Detect pathways
                            detected_pathways = detector.detect(network, features, event_freq)

                            # Calculate metrics
                            if detected_pathways and true_pathways:
                                precision, recall, f1 = precision_recall_f1(detected_pathways, true_pathways)
                                pji = pathway_jaccard_index(detected_pathways, true_pathways)
                            else:
                                precision, recall, f1, pji = 0, 0, 0, 0

                            # Record results
                            results.append({
                                'network_type': network_type,
                                'n_nodes': n_nodes,
                                'snr_db': snr_db,
                                'delay_uncertainty': delay_uncertainty,
                                'run': run,
                                'method': method_name,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'pji': pji,
                                'n_detected_pathways': len(detected_pathways),
                                'n_true_pathways': len(true_pathways)
                            })

    return pd.DataFrame(results)


def plot_pathway_detection_results(results: pd.DataFrame, output_dir: str = 'results/figures'):
    """
    Plot pathway detection results.

    Args:
        results: DataFrame with results.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot F1-score vs. SNR for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='snr_db',
        y='f1',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('F1-Score vs. SNR by Network Type and Method')
    plt.xlabel('SNR (dB)')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'f1_vs_snr.png'), dpi=300, bbox_inches='tight')

    # Plot F1-score vs. delay uncertainty for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='delay_uncertainty',
        y='f1',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('F1-Score vs. Delay Uncertainty by Network Type and Method')
    plt.xlabel('Delay Uncertainty')
    plt.ylabel('F1-Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'f1_vs_delay_uncertainty.png'), dpi=300, bbox_inches='tight')

    # Plot F1-score vs. network size for each network type and method
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=results,
        x='n_nodes',
        y='f1',
        hue='method',
        style='network_type',
        markers=True,
        ci=95
    )
    plt.title('F1-Score vs. Network Size by Network Type and Method')
    plt.xlabel('Number of Nodes')
    plt.ylabel('F1-Score')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'f1_vs_n_nodes.png'), dpi=300, bbox_inches='tight')

    # Create a summary table
    summary = results.groupby(['method', 'network_type']).agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'pji': ['mean', 'std']
    }).reset_index()

    # Save summary table
    summary.to_csv(os.path.join(output_dir, 'pathway_detection_summary.csv'), index=False)

    return summary


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run pathway detection experiment.')
    parser.add_argument('--network_types', type=str, nargs='+', default=['ba', 'er', 'ws', 'grid'],
                        help='Network types to test')
    parser.add_argument('--n_nodes_list', type=int, nargs='+', default=[100, 500],
                        help='Network sizes to test')
    parser.add_argument('--snr_db_list', type=float, nargs='+', default=[5, 10, 20],
                        help='SNR values to test')
    parser.add_argument('--delay_uncertainty_list', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Delay uncertainty values to test')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs per configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--use_pregenerated_data', action='store_true',
                        help='Use pre-generated datasets')
    parser.add_argument('--data_dir', type=str, default='../data/synthetic',
                        help='Directory containing the datasets')
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate datasets without running experiments')

    args = parser.parse_args()

    # Generate datasets if requested
    if args.generate_only:
        print("Generating synthetic datasets...")
        for network_type in args.network_types:
            for n_nodes in args.n_nodes_list:
                for snr_db in args.snr_db_list:
                    for delay_uncertainty in args.delay_uncertainty_list:
                        for run in range(args.n_runs):
                            print(f"Generating dataset: {network_type}, {n_nodes} nodes, {snr_db} dB, {delay_uncertainty} delay, run {run}")
                            load_synthetic_dataset(
                                network_type=network_type,
                                n_nodes=n_nodes,
                                snr_db=snr_db,
                                delay_uncertainty=delay_uncertainty,
                                dataset_idx=run,
                                data_dir=args.data_dir
                            )
        print("Dataset generation completed!")
        sys.exit(0)

    # Run pathway detection experiment
    print("Running pathway detection experiment...")
    results = run_pathway_detection_experiment(
        network_types=args.network_types,
        n_nodes_list=args.n_nodes_list,
        snr_db_list=args.snr_db_list,
        delay_uncertainty_list=args.delay_uncertainty_list,
        n_runs=args.n_runs,
        base_seed=args.seed,
        use_pregenerated_data=args.use_pregenerated_data,
        data_dir=args.data_dir
    )

    # Save results
    results.to_csv('results/synthetic/pathway_detection_results.csv', index=False)

    # Plot results
    print("Plotting results...")
    summary = plot_pathway_detection_results(results)

    print("Experiment completed successfully!")
    print(f"Results saved to results/synthetic/pathway_detection_results.csv")
    print(f"Plots saved to results/figures/")
