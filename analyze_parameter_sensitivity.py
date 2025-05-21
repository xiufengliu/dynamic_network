"""
Analyze the parameter sensitivity of our methods on real-world datasets.

This script evaluates how the performance of pathway detection, source localization,
and intervention optimization varies with different parameter settings.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from src.network.graph import DynamicNetwork
from src.utils.real_world_loader import (
    load_roadnet_ca,
    load_wiki_talk,
    load_email_eu_core,
    load_reddit_hyperlinks
)
from src.feature_extraction.stft import STFT
from src.pathway_detection.detector import PathwayDetector
from src.source_localization.localizer import SourceLocalizer
from src.intervention.optimizer import ResourceOptimizer

# Create output directories
os.makedirs('results/sensitivity', exist_ok=True)
os.makedirs('results/sensitivity/figures', exist_ok=True)
os.makedirs('results/sensitivity/data', exist_ok=True)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

# Default parameters
DEFAULT_STFT_WINDOW_SIZE = 256
DEFAULT_STFT_OVERLAP = 0.5
DEFAULT_DELAY_TOLERANCE = 0.5
DEFAULT_PHASE_TOLERANCE = np.pi/4
DEFAULT_AMPLITUDE_THRESHOLD = 0.1
DEFAULT_EVENT_FREQ = 0.1
NUM_SOURCES = 5
NUM_CRITICAL_NODES = 20
MAX_IMPACT = 0.1
NUM_RUNS = 3  # Number of runs for each parameter setting

# Parameter ranges to test
DELAY_TOLERANCE_RANGE = [0.1, 0.3, 0.5, 0.7, 1.0]
PHASE_TOLERANCE_RANGE = [np.pi/8, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
AMPLITUDE_THRESHOLD_RANGE = [0.05, 0.1, 0.2, 0.3, 0.5]
STFT_WINDOW_SIZE_RANGE = [128, 256, 512, 1024]

def simulate_event_propagation(network, sources, event_freq=0.1, snr_db=10.0, delay_uncertainty=0.1):
    """
    Simulate event propagation on a network.
    
    Args:
        network: The network to simulate on
        sources: List of source nodes
        event_freq: Characteristic frequency for oscillatory events
        snr_db: Signal-to-noise ratio in dB
        delay_uncertainty: Uncertainty in propagation delays
        
    Returns:
        signals: Dictionary of time-series data for each node
        time: Time points
        true_pathways: List of true propagation pathways
    """
    # Initialize
    G = network.graph
    nodes = list(G.nodes())
    N = len(nodes)
    
    # Create time vector (10 seconds at 100 Hz sampling rate)
    fs = 100  # Sampling frequency (Hz)
    duration = 10  # Duration (seconds)
    time = np.linspace(0, duration, int(fs * duration))
    
    # Initialize signals for all nodes
    signals = {}
    for node in nodes:
        signals[node] = np.zeros_like(time)
    
    # Initialize activation times and visited nodes
    activation_times = {}
    for source in sources:
        activation_times[source] = 0.0  # Sources activate at t=0
    
    visited = set(sources)
    queue = list(sources)
    
    # Track true pathways
    true_pathways = []
    for source in sources:
        true_pathways.append([source])  # Start each pathway with a source
    
    # Breadth-first propagation
    while queue:
        current = queue.pop(0)
        current_time = activation_times[current]
        
        # Process neighbors
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                # Get edge delay
                delay = G[current][neighbor].get('weight', 1.0)
                
                # Add uncertainty to delay
                noise = np.random.normal(0, delay_uncertainty * delay)
                actual_delay = max(0.1, delay + noise)  # Ensure positive delay
                
                # Calculate activation time
                neighbor_time = current_time + actual_delay
                if neighbor_time < duration:  # Only if within simulation time
                    activation_times[neighbor] = neighbor_time
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # Add to pathway
                    for path in true_pathways:
                        if path[-1] == current:
                            # Create a new branch if this is not the first neighbor
                            if len([p for p in true_pathways if p[-1] == current]) > 1:
                                new_path = path.copy()
                                new_path.append(neighbor)
                                true_pathways.append(new_path)
                            else:
                                path.append(neighbor)
    
    # Generate signals based on activation times
    for node in visited:
        t_activate = activation_times[node]
        idx_activate = int(t_activate * fs)
        
        # Create a Gaussian pulse centered at activation time
        pulse_width = 0.5 * fs  # 0.5 seconds
        t_indices = np.arange(len(time))
        gaussian_pulse = np.exp(-0.5 * ((t_indices - idx_activate) / pulse_width) ** 2)
        
        # Create oscillatory signal
        oscillation = gaussian_pulse * np.sin(2 * np.pi * event_freq * time + np.random.uniform(0, 2*np.pi))
        
        # Add to node's signal
        signals[node] = oscillation
    
    # Add noise based on SNR
    for node in nodes:
        if np.max(np.abs(signals[node])) > 0:  # Only add noise to active nodes
            signal_power = np.mean(signals[node] ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(time))
            signals[node] += noise
    
    # Create features dictionary
    features = {}
    for node in visited:
        features[node] = {
            'activation_time': activation_times[node],
            'signal': signals[node],
            'amplitude': np.max(np.abs(signals[node])),
            'phase': np.random.uniform(0, 2*np.pi)  # Random phase for simplicity
        }
    
    return features, time, true_pathways

def evaluate_pathway_detection(detected_pathways, true_pathways):
    """
    Evaluate pathway detection performance.
    
    Args:
        detected_pathways: List of detected pathways
        true_pathways: List of true pathways
        
    Returns:
        precision, recall, f1_score, pji
    """
    # Convert pathways to sets of edges for comparison
    true_edges = set()
    for path in true_pathways:
        for i in range(len(path) - 1):
            true_edges.add((path[i], path[i+1]))
    
    detected_edges = set()
    for path in detected_pathways:
        for i in range(len(path) - 1):
            detected_edges.add((path[i], path[i+1]))
    
    # Calculate precision, recall, F1
    if len(detected_edges) == 0:
        precision = 0.0
    else:
        precision = len(detected_edges.intersection(true_edges)) / len(detected_edges)
    
    if len(true_edges) == 0:
        recall = 0.0
    else:
        recall = len(detected_edges.intersection(true_edges)) / len(true_edges)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    # Calculate Pathway Jaccard Index
    if len(detected_edges) == 0 and len(true_edges) == 0:
        pji = 1.0
    elif len(detected_edges.union(true_edges)) == 0:
        pji = 0.0
    else:
        pji = len(detected_edges.intersection(true_edges)) / len(detected_edges.union(true_edges))
    
    return precision, recall, f1_score, pji

def evaluate_source_localization(detected_sources, true_sources, network):
    """
    Evaluate source localization performance.
    
    Args:
        detected_sources: List of detected sources
        true_sources: List of true sources
        network: The network
        
    Returns:
        sr@1, sr@3, sr@5, mrts, ed
    """
    # Success Rate @ k
    detected_set = set(detected_sources[:1])  # Top 1
    sr1 = len(detected_set.intersection(set(true_sources))) / len(true_sources)
    
    detected_set = set(detected_sources[:min(3, len(detected_sources))])  # Top 3
    sr3 = len(detected_set.intersection(set(true_sources))) / len(true_sources)
    
    detected_set = set(detected_sources[:min(5, len(detected_sources))])  # Top 5
    sr5 = len(detected_set.intersection(set(true_sources))) / len(true_sources)
    
    # Mean Rank of True Source
    ranks = []
    for true_source in true_sources:
        if true_source in detected_sources:
            ranks.append(detected_sources.index(true_source) + 1)
        else:
            ranks.append(len(detected_sources) + 1)  # Penalty for not found
    mrts = np.mean(ranks)
    
    # Error Distance (using shortest path in the network)
    distances = []
    G = network.graph
    for true_source in true_sources:
        min_dist = float('inf')
        for detected_source in detected_sources[:min(5, len(detected_sources))]:  # Consider top 5
            try:
                dist = nx.shortest_path_length(G, source=detected_source, target=true_source)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
        if min_dist == float('inf'):
            min_dist = 10  # Default penalty if no path exists
        distances.append(min_dist)
    ed = np.mean(distances)
    
    return sr1, sr3, sr5, mrts, ed

def analyze_delay_tolerance_sensitivity(network, features, true_pathways, sources):
    """
    Analyze sensitivity to delay tolerance parameter.
    
    Args:
        network: The network to analyze
        features: Dictionary of node features
        true_pathways: List of true propagation pathways
        sources: List of true source nodes
        
    Returns:
        results: Dictionary of sensitivity results
    """
    print("Analyzing sensitivity to delay tolerance...")
    
    results = {
        'delay_tolerance': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'pji': [],
        'sr@1': []
    }
    
    for delay_tolerance in DELAY_TOLERANCE_RANGE:
        print(f"  Testing delay_tolerance = {delay_tolerance}")
        
        # Run pathway detection
        detector = PathwayDetector(
            delay_tolerance=delay_tolerance,
            phase_tolerance=DEFAULT_PHASE_TOLERANCE,
            amplitude_threshold=DEFAULT_AMPLITUDE_THRESHOLD
        )
        detected_pathways = detector.detect(network, features, DEFAULT_EVENT_FREQ)
        
        # Evaluate pathway detection
        precision, recall, f1, pji = evaluate_pathway_detection(detected_pathways, true_pathways)
        
        # Run source localization
        localizer = SourceLocalizer()
        detected_sources = localizer.localize(network, features, detected_pathways)
        
        # Evaluate source localization (just SR@1 for simplicity)
        sr1, _, _, _, _ = evaluate_source_localization(detected_sources, sources, network)
        
        # Record results
        results['delay_tolerance'].append(delay_tolerance)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)
        results['pji'].append(pji)
        results['sr@1'].append(sr1)
    
    return results

def analyze_phase_tolerance_sensitivity(network, features, true_pathways, sources):
    """
    Analyze sensitivity to phase tolerance parameter.
    
    Args:
        network: The network to analyze
        features: Dictionary of node features
        true_pathways: List of true propagation pathways
        sources: List of true source nodes
        
    Returns:
        results: Dictionary of sensitivity results
    """
    print("Analyzing sensitivity to phase tolerance...")
    
    results = {
        'phase_tolerance': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'pji': [],
        'sr@1': []
    }
    
    for phase_tolerance in PHASE_TOLERANCE_RANGE:
        print(f"  Testing phase_tolerance = {phase_tolerance}")
        
        # Run pathway detection
        detector = PathwayDetector(
            delay_tolerance=DEFAULT_DELAY_TOLERANCE,
            phase_tolerance=phase_tolerance,
            amplitude_threshold=DEFAULT_AMPLITUDE_THRESHOLD
        )
        detected_pathways = detector.detect(network, features, DEFAULT_EVENT_FREQ)
        
        # Evaluate pathway detection
        precision, recall, f1, pji = evaluate_pathway_detection(detected_pathways, true_pathways)
        
        # Run source localization
        localizer = SourceLocalizer()
        detected_sources = localizer.localize(network, features, detected_pathways)
        
        # Evaluate source localization (just SR@1 for simplicity)
        sr1, _, _, _, _ = evaluate_source_localization(detected_sources, sources, network)
        
        # Record results
        results['phase_tolerance'].append(phase_tolerance)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)
        results['pji'].append(pji)
        results['sr@1'].append(sr1)
    
    return results

def analyze_amplitude_threshold_sensitivity(network, features, true_pathways, sources):
    """
    Analyze sensitivity to amplitude threshold parameter.
    
    Args:
        network: The network to analyze
        features: Dictionary of node features
        true_pathways: List of true propagation pathways
        sources: List of true source nodes
        
    Returns:
        results: Dictionary of sensitivity results
    """
    print("Analyzing sensitivity to amplitude threshold...")
    
    results = {
        'amplitude_threshold': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'pji': [],
        'sr@1': []
    }
    
    for amplitude_threshold in AMPLITUDE_THRESHOLD_RANGE:
        print(f"  Testing amplitude_threshold = {amplitude_threshold}")
        
        # Run pathway detection
        detector = PathwayDetector(
            delay_tolerance=DEFAULT_DELAY_TOLERANCE,
            phase_tolerance=DEFAULT_PHASE_TOLERANCE,
            amplitude_threshold=amplitude_threshold
        )
        detected_pathways = detector.detect(network, features, DEFAULT_EVENT_FREQ)
        
        # Evaluate pathway detection
        precision, recall, f1, pji = evaluate_pathway_detection(detected_pathways, true_pathways)
        
        # Run source localization
        localizer = SourceLocalizer()
        detected_sources = localizer.localize(network, features, detected_pathways)
        
        # Evaluate source localization (just SR@1 for simplicity)
        sr1, _, _, _, _ = evaluate_source_localization(detected_sources, sources, network)
        
        # Record results
        results['amplitude_threshold'].append(amplitude_threshold)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)
        results['pji'].append(pji)
        results['sr@1'].append(sr1)
    
    return results

def analyze_stft_window_size_sensitivity(network, signals, time, true_pathways, sources):
    """
    Analyze sensitivity to STFT window size parameter.
    
    Args:
        network: The network to analyze
        signals: Dictionary of node signals
        time: Time points
        true_pathways: List of true propagation pathways
        sources: List of true source nodes
        
    Returns:
        results: Dictionary of sensitivity results
    """
    print("Analyzing sensitivity to STFT window size...")
    
    results = {
        'window_size': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'pji': [],
        'sr@1': []
    }
    
    for window_size in STFT_WINDOW_SIZE_RANGE:
        print(f"  Testing window_size = {window_size}")
        
        # Extract features using STFT
        stft = STFT(window_size=window_size, overlap=DEFAULT_STFT_OVERLAP)
        features = {}
        for node, signal in signals.items():
            if np.max(np.abs(signal)) > 0:  # Only process active nodes
                features[node] = stft.extract_features(signal, DEFAULT_EVENT_FREQ)
        
        # Run pathway detection
        detector = PathwayDetector(
            delay_tolerance=DEFAULT_DELAY_TOLERANCE,
            phase_tolerance=DEFAULT_PHASE_TOLERANCE,
            amplitude_threshold=DEFAULT_AMPLITUDE_THRESHOLD
        )
        detected_pathways = detector.detect(network, features, DEFAULT_EVENT_FREQ)
        
        # Evaluate pathway detection
        precision, recall, f1, pji = evaluate_pathway_detection(detected_pathways, true_pathways)
        
        # Run source localization
        localizer = SourceLocalizer()
        detected_sources = localizer.localize(network, features, detected_pathways)
        
        # Evaluate source localization (just SR@1 for simplicity)
        sr1, _, _, _, _ = evaluate_source_localization(detected_sources, sources, network)
        
        # Record results
        results['window_size'].append(window_size)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)
        results['pji'].append(pji)
        results['sr@1'].append(sr1)
    
    return results

def run_parameter_sensitivity_analysis(network_name, network, num_runs=NUM_RUNS):
    """
    Run parameter sensitivity analysis on a network.
    
    Args:
        network_name: Name of the network
        network: The network to analyze
        num_runs: Number of runs for each parameter setting
        
    Returns:
        results: Dictionary of sensitivity results
    """
    print(f"\n=== Running parameter sensitivity analysis on {network_name} ===")
    
    # Take a smaller subgraph for analysis (for computational efficiency)
    nodes = list(network.graph.nodes())
    subgraph_size = min(2000, len(nodes))
    subgraph_nodes = nodes[:subgraph_size]
    subgraph = network.create_subgraph(subgraph_nodes)
    print(f"Created subgraph with {len(subgraph.get_nodes())} nodes, {len(subgraph.get_edges())} edges")
    
    # Initialize results
    all_results = {
        'delay_tolerance': {
            'runs': []
        },
        'phase_tolerance': {
            'runs': []
        },
        'amplitude_threshold': {
            'runs': []
        },
        'stft_window_size': {
            'runs': []
        }
    }
    
    # Run multiple times for statistical significance
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # Select random sources
        sources = np.random.choice(subgraph_nodes, size=min(NUM_SOURCES, len(subgraph_nodes)), replace=False)
        print(f"Selected sources: {sources}")
        
        # Simulate event propagation
        features, time_points, true_pathways = simulate_event_propagation(
            subgraph, 
            sources, 
            event_freq=DEFAULT_EVENT_FREQ, 
            snr_db=10.0, 
            delay_uncertainty=0.1
        )
        
        # Extract signals for STFT window size analysis
        signals = {node: feature['signal'] for node, feature in features.items() if 'signal' in feature}
        
        # Analyze sensitivity to each parameter
        delay_results = analyze_delay_tolerance_sensitivity(subgraph, features, true_pathways, sources)
        all_results['delay_tolerance']['runs'].append(delay_results)
        
        phase_results = analyze_phase_tolerance_sensitivity(subgraph, features, true_pathways, sources)
        all_results['phase_tolerance']['runs'].append(phase_results)
        
        amplitude_results = analyze_amplitude_threshold_sensitivity(subgraph, features, true_pathways, sources)
        all_results['amplitude_threshold']['runs'].append(amplitude_results)
        
        window_results = analyze_stft_window_size_sensitivity(subgraph, signals, time_points, true_pathways, sources)
        all_results['stft_window_size']['runs'].append(window_results)
    
    # Calculate average results across runs
    for param_type in all_results:
        runs = all_results[param_type]['runs']
        
        # Initialize average results
        avg_results = {}
        for key in runs[0]:
            avg_results[key] = []
        
        # Calculate averages
        for key in avg_results:
            if key == 'delay_tolerance' or key == 'phase_tolerance' or key == 'amplitude_threshold' or key == 'window_size':
                # Just take values from first run (they're the same across runs)
                avg_results[key] = runs[0][key]
            else:
                # Average across runs
                for i in range(len(runs[0][key])):
                    values = [run[key][i] for run in runs]
                    avg_results[key].append(np.mean(values))
        
        all_results[param_type]['avg'] = avg_results
    
    return all_results

def main():
    """Main function to run parameter sensitivity analysis on all real-world datasets."""
    print("Starting parameter sensitivity analysis on real-world datasets...")
    
    # Dictionary to store results for all networks
    all_results = {}
    
    # 1. email-Eu-core-temporal (smaller dataset for faster analysis)
    try:
        print("\n=== Loading email-Eu-core-temporal dataset ===")
        email = load_email_eu_core(EMAIL_PATH)
        print(f"Loaded email-Eu-core: {len(email.get_nodes())} nodes, {len(email.get_edges())} edges")
        
        # Run parameter sensitivity analysis
        email_results = run_parameter_sensitivity_analysis("email-Eu-core", email)
        all_results["email-Eu-core"] = email_results
    except Exception as e:
        print(f"Error processing email-Eu-core: {e}")
    
    # Save results
    save_results(all_results)
    
    # Generate visualizations
    generate_visualizations(all_results)
    
    print("\nParameter sensitivity analysis completed successfully!")
    print("Results saved to results/sensitivity/")

def save_results(results):
    """Save sensitivity results to files."""
    # Save as pickle for later analysis
    import pickle
    with open('results/sensitivity/data/sensitivity_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save as CSV for each network and parameter
    for network_name, network_results in results.items():
        for param_type, param_results in network_results.items():
            avg_results = param_results['avg']
            df = pd.DataFrame(avg_results)
            df.to_csv(f'results/sensitivity/data/{network_name.replace(" ", "_").lower()}_{param_type}.csv', index=False)

def generate_visualizations(results):
    """Generate visualizations of parameter sensitivity results."""
    for network_name, network_results in results.items():
        # 1. Delay Tolerance Sensitivity
        if 'delay_tolerance' in network_results:
            plt.figure(figsize=(12, 8))
            
            avg_results = network_results['delay_tolerance']['avg']
            plt.plot(avg_results['delay_tolerance'], avg_results['f1_score'], 'o-', label='F1 Score')
            plt.plot(avg_results['delay_tolerance'], avg_results['sr@1'], 's-', label='SR@1')
            
            plt.xlabel('Delay Tolerance')
            plt.ylabel('Performance')
            plt.title(f'{network_name} - Sensitivity to Delay Tolerance')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/sensitivity/figures/{network_name.replace(" ", "_").lower()}_delay_tolerance.png', dpi=300)
            plt.close()
        
        # 2. Phase Tolerance Sensitivity
        if 'phase_tolerance' in network_results:
            plt.figure(figsize=(12, 8))
            
            avg_results = network_results['phase_tolerance']['avg']
            # Convert phase tolerance from radians to degrees for better readability
            phase_degrees = [p * 180 / np.pi for p in avg_results['phase_tolerance']]
            
            plt.plot(phase_degrees, avg_results['f1_score'], 'o-', label='F1 Score')
            plt.plot(phase_degrees, avg_results['sr@1'], 's-', label='SR@1')
            
            plt.xlabel('Phase Tolerance (degrees)')
            plt.ylabel('Performance')
            plt.title(f'{network_name} - Sensitivity to Phase Tolerance')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/sensitivity/figures/{network_name.replace(" ", "_").lower()}_phase_tolerance.png', dpi=300)
            plt.close()
        
        # 3. Amplitude Threshold Sensitivity
        if 'amplitude_threshold' in network_results:
            plt.figure(figsize=(12, 8))
            
            avg_results = network_results['amplitude_threshold']['avg']
            plt.plot(avg_results['amplitude_threshold'], avg_results['f1_score'], 'o-', label='F1 Score')
            plt.plot(avg_results['amplitude_threshold'], avg_results['sr@1'], 's-', label='SR@1')
            
            plt.xlabel('Amplitude Threshold')
            plt.ylabel('Performance')
            plt.title(f'{network_name} - Sensitivity to Amplitude Threshold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/sensitivity/figures/{network_name.replace(" ", "_").lower()}_amplitude_threshold.png', dpi=300)
            plt.close()
        
        # 4. STFT Window Size Sensitivity
        if 'stft_window_size' in network_results:
            plt.figure(figsize=(12, 8))
            
            avg_results = network_results['stft_window_size']['avg']
            plt.plot(avg_results['window_size'], avg_results['f1_score'], 'o-', label='F1 Score')
            plt.plot(avg_results['window_size'], avg_results['sr@1'], 's-', label='SR@1')
            
            plt.xlabel('STFT Window Size')
            plt.ylabel('Performance')
            plt.title(f'{network_name} - Sensitivity to STFT Window Size')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/sensitivity/figures/{network_name.replace(" ", "_").lower()}_stft_window_size.png', dpi=300)
            plt.close()

if __name__ == "__main__":
    main()
