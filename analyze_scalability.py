"""
Analyze the scalability of our methods on real-world datasets.

This script measures the runtime of pathway detection, source localization, and
intervention optimization as a function of network size, using subgraphs of
increasing size from the real-world datasets.
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
os.makedirs('results/scalability', exist_ok=True)
os.makedirs('results/scalability/figures', exist_ok=True)
os.makedirs('results/scalability/data', exist_ok=True)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

# Experiment parameters
STFT_WINDOW_SIZE = 256
STFT_OVERLAP = 0.5
DELAY_TOLERANCE = 0.5
PHASE_TOLERANCE = np.pi/4
AMPLITUDE_THRESHOLD = 0.1
EVENT_FREQ = 0.1  # Characteristic frequency for oscillatory events
NUM_SOURCES = 5  # Number of sources to simulate
NUM_CRITICAL_NODES = 20  # Number of critical nodes for intervention
MAX_IMPACT = 0.1  # Maximum permissible impact for critical nodes
NUM_RUNS = 3  # Number of runs for each network size

# Network sizes to test
NETWORK_SIZES = [100, 200, 500, 1000, 2000, 5000]

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

def measure_scalability(network_name, network, sizes=NETWORK_SIZES, num_runs=NUM_RUNS):
    """
    Measure the scalability of our methods on a network.
    
    Args:
        network_name: Name of the network
        network: The network to experiment on
        sizes: List of network sizes to test
        num_runs: Number of runs for each size
        
    Returns:
        results: Dictionary of scalability results
    """
    print(f"\n=== Measuring scalability on {network_name} ===")
    
    # Initialize results
    results = {
        'network_size': [],
        'edge_count': [],
        'pathway_detection_time': [],
        'source_localization_time': [],
        'intervention_time': [],
        'feature_extraction_time': [],
        'total_time': []
    }
    
    # Get all nodes
    all_nodes = list(network.graph.nodes())
    
    # Test different network sizes
    for size in sizes:
        if size > len(all_nodes):
            print(f"Skipping size {size} (larger than network)")
            continue
        
        print(f"\nTesting network size: {size}")
        
        # Create subgraph
        subgraph_nodes = all_nodes[:size]
        subgraph = network.create_subgraph(subgraph_nodes)
        
        # Count edges
        edge_count = len(subgraph.get_edges())
        
        # Run multiple times for statistical significance
        pathway_times = []
        source_times = []
        intervention_times = []
        feature_times = []
        total_times = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            
            # Select random sources
            sources = np.random.choice(subgraph_nodes, size=min(NUM_SOURCES, size), replace=False)
            
            # Start timing
            start_total = time.time()
            
            # Simulate event propagation
            features, time_points, true_pathways = simulate_event_propagation(
                subgraph, 
                sources, 
                event_freq=EVENT_FREQ, 
                snr_db=10.0, 
                delay_uncertainty=0.1
            )
            
            # Measure feature extraction time
            start_feature = time.time()
            stft = STFT(window_size=STFT_WINDOW_SIZE, overlap=STFT_OVERLAP)
            for node, feature in features.items():
                if 'signal' in feature:
                    # Extract STFT features (simplified for timing purposes)
                    _ = stft.extract_features(feature['signal'], EVENT_FREQ)
            feature_time = time.time() - start_feature
            feature_times.append(feature_time)
            
            # Measure pathway detection time
            start_pathway = time.time()
            detector = PathwayDetector(
                delay_tolerance=DELAY_TOLERANCE,
                phase_tolerance=PHASE_TOLERANCE,
                amplitude_threshold=AMPLITUDE_THRESHOLD
            )
            detected_pathways = detector.detect(subgraph, features, EVENT_FREQ)
            pathway_time = time.time() - start_pathway
            pathway_times.append(pathway_time)
            
            # Measure source localization time
            start_source = time.time()
            localizer = SourceLocalizer()
            detected_sources = localizer.localize(subgraph, features, detected_pathways)
            source_time = time.time() - start_source
            source_times.append(source_time)
            
            # Select critical nodes
            critical_nodes = np.random.choice(subgraph_nodes, size=min(NUM_CRITICAL_NODES, size), replace=False)
            
            # Calculate initial impacts
            initial_impacts = {}
            for node in subgraph.graph.nodes():
                if node in features:
                    initial_impacts[node] = features[node].get('amplitude', 0) ** 2
                else:
                    initial_impacts[node] = 0.0
            
            # Measure intervention time
            start_intervention = time.time()
            optimizer = ResourceOptimizer()
            allocation = optimizer.optimize(
                network=subgraph,
                pathways=detected_pathways,
                initial_impacts=initial_impacts,
                critical_nodes=critical_nodes,
                max_impact=MAX_IMPACT
            )
            intervention_time = time.time() - start_intervention
            intervention_times.append(intervention_time)
            
            # Total time
            total_time = time.time() - start_total
            total_times.append(total_time)
        
        # Record average times
        results['network_size'].append(size)
        results['edge_count'].append(edge_count)
        results['pathway_detection_time'].append(np.mean(pathway_times))
        results['source_localization_time'].append(np.mean(source_times))
        results['intervention_time'].append(np.mean(intervention_times))
        results['feature_extraction_time'].append(np.mean(feature_times))
        results['total_time'].append(np.mean(total_times))
        
        print(f"  Average times (seconds):")
        print(f"    Feature extraction: {np.mean(feature_times):.4f}")
        print(f"    Pathway detection: {np.mean(pathway_times):.4f}")
        print(f"    Source localization: {np.mean(source_times):.4f}")
        print(f"    Intervention: {np.mean(intervention_times):.4f}")
        print(f"    Total: {np.mean(total_times):.4f}")
    
    return results

def main():
    """Main function to analyze scalability on all real-world datasets."""
    print("Starting scalability analysis on real-world datasets...")
    
    # Dictionary to store results for all networks
    all_results = {}
    
    # 1. roadNet-CA
    try:
        print("\n=== Loading roadNet-CA dataset ===")
        roadnet = load_roadnet_ca(ROADNET_PATH)
        print(f"Loaded roadNet-CA: {len(roadnet.get_nodes())} nodes, {len(roadnet.get_edges())} edges")
        
        # Measure scalability
        roadnet_results = measure_scalability("roadNet-CA", roadnet)
        all_results["roadNet-CA"] = roadnet_results
    except Exception as e:
        print(f"Error processing roadNet-CA: {e}")
    
    # 2. wiki-Talk
    try:
        print("\n=== Loading wiki-Talk dataset ===")
        wiki_talk = load_wiki_talk(WIKI_TALK_PATH)
        print(f"Loaded wiki-Talk: {len(wiki_talk.get_nodes())} nodes, {len(wiki_talk.get_edges())} edges")
        
        # Measure scalability
        wiki_talk_results = measure_scalability("wiki-Talk", wiki_talk)
        all_results["wiki-Talk"] = wiki_talk_results
    except Exception as e:
        print(f"Error processing wiki-Talk: {e}")
    
    # 3. email-Eu-core-temporal
    try:
        print("\n=== Loading email-Eu-core-temporal dataset ===")
        email = load_email_eu_core(EMAIL_PATH)
        print(f"Loaded email-Eu-core: {len(email.get_nodes())} nodes, {len(email.get_edges())} edges")
        
        # Measure scalability
        email_results = measure_scalability("email-Eu-core", email)
        all_results["email-Eu-core"] = email_results
    except Exception as e:
        print(f"Error processing email-Eu-core: {e}")
    
    # 4. soc-redditHyperlinks-body
    try:
        print("\n=== Loading soc-redditHyperlinks-body dataset ===")
        reddit = load_reddit_hyperlinks(REDDIT_PATH)
        print(f"Loaded Reddit Hyperlinks: {len(reddit.get_nodes())} nodes, {len(reddit.get_edges())} edges")
        
        # Measure scalability
        reddit_results = measure_scalability("Reddit Hyperlinks", reddit)
        all_results["Reddit Hyperlinks"] = reddit_results
    except Exception as e:
        print(f"Error processing Reddit Hyperlinks: {e}")
    
    # Save results
    save_results(all_results)
    
    # Generate visualizations
    generate_visualizations(all_results)
    
    print("\nScalability analysis completed successfully!")
    print("Results saved to results/scalability/")

def save_results(results):
    """Save scalability results to files."""
    # Save as pickle for later analysis
    import pickle
    with open('results/scalability/data/scalability_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save as CSV for each network
    for network_name, network_results in results.items():
        df = pd.DataFrame(network_results)
        df.to_csv(f'results/scalability/data/{network_name.replace(" ", "_").lower()}_scalability.csv', index=False)

def generate_visualizations(results):
    """Generate visualizations of scalability results."""
    # 1. Runtime vs. Network Size for each algorithm
    plt.figure(figsize=(12, 8))
    
    for network_name, network_results in results.items():
        plt.plot(
            network_results['network_size'], 
            network_results['pathway_detection_time'],
            'o-',
            label=f"{network_name} - Pathway Detection"
        )
    
    plt.xlabel('Network Size (nodes)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Pathway Detection Scalability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/scalability/figures/pathway_detection_scalability.png', dpi=300)
    plt.close()
    
    # 2. Runtime vs. Network Size for source localization
    plt.figure(figsize=(12, 8))
    
    for network_name, network_results in results.items():
        plt.plot(
            network_results['network_size'], 
            network_results['source_localization_time'],
            'o-',
            label=f"{network_name}"
        )
    
    plt.xlabel('Network Size (nodes)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Source Localization Scalability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/scalability/figures/source_localization_scalability.png', dpi=300)
    plt.close()
    
    # 3. Runtime vs. Network Size for intervention
    plt.figure(figsize=(12, 8))
    
    for network_name, network_results in results.items():
        plt.plot(
            network_results['network_size'], 
            network_results['intervention_time'],
            'o-',
            label=f"{network_name}"
        )
    
    plt.xlabel('Network Size (nodes)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Intervention Optimization Scalability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/scalability/figures/intervention_scalability.png', dpi=300)
    plt.close()
    
    # 4. Runtime vs. Edge Count for pathway detection
    plt.figure(figsize=(12, 8))
    
    for network_name, network_results in results.items():
        plt.plot(
            network_results['edge_count'], 
            network_results['pathway_detection_time'],
            'o-',
            label=f"{network_name}"
        )
    
    plt.xlabel('Edge Count')
    plt.ylabel('Runtime (seconds)')
    plt.title('Pathway Detection Scalability vs. Edge Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/scalability/figures/pathway_detection_edge_scalability.png', dpi=300)
    plt.close()
    
    # 5. Log-log plot of total runtime vs. network size
    plt.figure(figsize=(12, 8))
    
    for network_name, network_results in results.items():
        plt.loglog(
            network_results['network_size'], 
            network_results['total_time'],
            'o-',
            label=f"{network_name}"
        )
    
    plt.xlabel('Network Size (nodes) - log scale')
    plt.ylabel('Total Runtime (seconds) - log scale')
    plt.title('Total Algorithm Scalability (Log-Log)')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/scalability/figures/total_scalability_loglog.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
