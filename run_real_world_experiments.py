"""
Run experiments on real-world datasets for causal pathway inference and optimized intervention.

This script implements the experimental methodology described in the paper for the four real-world datasets:
1. roadNet-CA - Road network of California
2. wiki-Talk - Wikipedia talk page interactions
3. email-Eu-core-temporal - Email communications in a European research institution
4. soc-redditHyperlinks-body - Reddit hyperlinks between communities
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import logging
import sys
import traceback
from collections import defaultdict
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('run_real_world_experiments.log')
    ]
)
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs('results/experiments/real_world', exist_ok=True)
os.makedirs('results/experiments/real_world/figures', exist_ok=True)
os.makedirs('results/experiments/real_world/data', exist_ok=True)

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
NUM_RUNS = 10  # Number of experiment runs for statistical significance

def simulate_event_propagation(network, sources, event_freq=0.1, snr_db=10.0, delay_uncertainty=0.1):
    """
    Simulate event propagation on a real-world network.

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
    logger.info(f"Simulating event propagation from {len(sources)} sources...")
    logger.debug(f"Simulation parameters: event_freq={event_freq}, snr_db={snr_db}, delay_uncertainty={delay_uncertainty}")

    # Initialize
    G = network.graph
    nodes = list(G.nodes())
    N = len(nodes)
    logger.debug(f"Network has {N} nodes and {G.number_of_edges()} edges")

    # Create time vector (10 seconds at 100 Hz sampling rate)
    fs = 100  # Sampling frequency (Hz)
    duration = 10  # Duration (seconds)
    time = np.linspace(0, duration, int(fs * duration))
    logger.debug(f"Created time vector with {len(time)} points, duration={duration}s, fs={fs}Hz")

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
    propagation_steps = 0
    max_queue_size = len(queue)

    logger.debug(f"Starting breadth-first propagation from {len(sources)} sources")
    while queue:
        propagation_steps += 1
        max_queue_size = max(max_queue_size, len(queue))

        if propagation_steps % 1000 == 0:
            logger.debug(f"Propagation step {propagation_steps}, queue size: {len(queue)}, visited nodes: {len(visited)}")

        current = queue.pop(0)
        current_time = activation_times[current]

        # Get all neighbors
        neighbors = list(G.neighbors(current))
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if len(neighbors) > 0:
            logger.debug(f"Node {current} has {len(neighbors)} neighbors, {len(unvisited_neighbors)} unvisited")

        # Process neighbors
        for neighbor in neighbors:
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
                    paths_ending_at_current = [p for p in true_pathways if p[-1] == current]

                    if len(paths_ending_at_current) > 1:
                        # Multiple paths end at current node, create new branches for each
                        for path in paths_ending_at_current:
                            new_path = path.copy()
                            new_path.append(neighbor)
                            true_pathways.append(new_path)
                    elif len(paths_ending_at_current) == 1:
                        # Only one path ends at current node, extend it
                        paths_ending_at_current[0].append(neighbor)
                    else:
                        # No paths end at current node (shouldn't happen in normal operation)
                        logger.warning(f"No paths found ending at node {current}, creating new path")
                        true_pathways.append([current, neighbor])

    logger.debug(f"Propagation completed in {propagation_steps} steps")
    logger.debug(f"Max queue size: {max_queue_size}, total visited nodes: {len(visited)}/{N}")
    logger.debug(f"Generated {len(true_pathways)} true pathways")

    # Generate signals based on activation times
    logger.debug(f"Generating time-series signals for {len(visited)} active nodes")
    for node in visited:
        t_activate = activation_times[node]
        idx_activate = int(t_activate * fs)

        # Create a Gaussian pulse centered at activation time
        pulse_width = 0.5 * fs  # 0.5 seconds
        t_indices = np.arange(len(time))
        gaussian_pulse = np.exp(-0.5 * ((t_indices - idx_activate) / pulse_width) ** 2)

        # Create oscillatory signal
        phase = np.random.uniform(0, 2*np.pi)
        oscillation = gaussian_pulse * np.sin(2 * np.pi * event_freq * time + phase)

        # Add to node's signal
        signals[node] = oscillation

    # Add noise based on SNR
    logger.debug(f"Adding noise with SNR={snr_db}dB to active node signals")
    for node in nodes:
        if np.max(np.abs(signals[node])) > 0:  # Only add noise to active nodes
            signal_power = np.mean(signals[node] ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(time))
            signals[node] += noise

    # Summarize pathway statistics
    if true_pathways:
        path_lengths = [len(path) for path in true_pathways]
        logger.debug(f"Pathway statistics: min length={min(path_lengths)}, max length={max(path_lengths)}, avg length={np.mean(path_lengths):.2f}")

    return signals, time, true_pathways

def run_experiment(network_name, network, num_runs=10):
    """
    Run experiments on a real-world network.

    Args:
        network_name: Name of the network
        network: The network to experiment on
        num_runs: Number of experiment runs

    Returns:
        results: Dictionary of experiment results
    """
    logger.info(f"\n=== Running experiments on {network_name} ===")
    logger.info(f"Network size: {len(network.get_nodes())} nodes, {len(network.get_edges())} edges")
    logger.info(f"Number of runs: {num_runs}")

    # Initialize results
    results = {
        'pathway_detection': {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'pji': []  # Pathway Jaccard Index
        },
        'source_localization': {
            'sr@1': [],  # Success Rate @ 1
            'sr@3': [],  # Success Rate @ 3
            'sr@5': [],  # Success Rate @ 5
            'mrts': [],  # Mean Rank of True Source
            'ed': []     # Error Distance
        },
        'intervention': {
            'tir': [],   # Total Impact Reduction
            'cer': [],   # Cost-Effectiveness Ratio
            'success': [] # Success in Constraint Satisfaction
        },
        'runtime': {
            'pathway_detection': [],
            'source_localization': [],
            'intervention': [],
            'simulation': [],
            'feature_extraction': [],
            'total': []
        }
    }

    # Get nodes for source selection
    nodes = list(network.graph.nodes())
    logger.debug(f"Total nodes available for source selection: {len(nodes)}")

    # Run multiple experiments
    for run in range(num_runs):
        run_start_time = time.time()
        logger.info(f"\nRun {run+1}/{num_runs}")

        try:
            # Select random sources
            num_sources = min(NUM_SOURCES, len(nodes))
            sources = np.random.choice(nodes, size=num_sources, replace=False)
            logger.info(f"Selected sources: {sources}")

            # Simulate event propagation
            logger.info("Simulating event propagation...")
            sim_start_time = time.time()
            signals, time_points, true_pathways = simulate_event_propagation(
                network,
                sources,
                event_freq=EVENT_FREQ,
                snr_db=10.0,
                delay_uncertainty=0.1
            )
            sim_time = time.time() - sim_start_time
            results['runtime']['simulation'].append(sim_time)

            # Log simulation results
            active_nodes = sum(1 for node, signal in signals.items() if np.max(np.abs(signal)) > 0)
            logger.info(f"Simulation completed in {sim_time:.2f}s. Active nodes: {active_nodes}/{len(nodes)}")
            logger.info(f"Generated {len(true_pathways)} true pathways")
            logger.debug(f"True pathways: {true_pathways}")

            # Extract features using STFT
            logger.info("Extracting features using STFT...")
            feature_start_time = time.time()
            stft = STFT(window_size=STFT_WINDOW_SIZE, overlap=STFT_OVERLAP)

            # Create node features dictionary
            node_features = {}
            for node, signal in signals.items():
                if np.max(np.abs(signal)) > 0:  # Only process active nodes
                    node_features[node] = {
                        'signal': signal,
                        'activation_time': 0.0  # Default activation time
                    }

            # Create a new features dictionary in the format expected by SourceLocalizer
            features = {
                'amplitude': {},
                'phase': {},
                'unwrapped_phase': {},
                'activation_time': {},
                'is_active': {}
            }

            # Process each node
            for node, feature in node_features.items():
                try:
                    # Reshape the signal to match the expected input format (n_signals, n_samples)
                    signal_reshaped = np.array([feature['signal']])

                    # Extract features
                    extracted_features = stft.extract_features(signal_reshaped, EVENT_FREQ)

                    # Copy features to the expected format
                    features['amplitude'][node] = extracted_features['amplitude'][0]
                    features['phase'][node] = extracted_features['phase'][0]
                    features['unwrapped_phase'][node] = extracted_features['unwrapped_phase'][0]
                    features['activation_time'][node] = feature['activation_time']
                    features['is_active'][node] = extracted_features['is_active'][0]

                    # Also update the original feature dictionary
                    feature.update(extracted_features)
                except Exception as e:
                    logger.error(f"Error extracting features for node {node}: {str(e)}")
                    logger.error(traceback.format_exc())

            # Add times and frequencies to the features
            if node_features and 'times' in next(iter(node_features.values())):
                features['times'] = next(iter(node_features.values()))['times']
                features['frequencies'] = next(iter(node_features.values()))['frequencies']
                features['freq_idx'] = next(iter(node_features.values()))['freq_idx']

            feature_time = time.time() - feature_start_time
            results['runtime']['feature_extraction'].append(feature_time)

            logger.info(f"Feature extraction completed in {feature_time:.2f}s. Extracted features for {len(features['amplitude'])} nodes")

            # Detect propagation pathways
            logger.info("Detecting propagation pathways...")
            start_time = time.time()
            detector = PathwayDetector(
                delay_tolerance=DELAY_TOLERANCE,
                phase_tolerance=PHASE_TOLERANCE,
                amplitude_threshold=AMPLITUDE_THRESHOLD
            )
            detected_pathways = detector.detect(network, features, EVENT_FREQ)
            pathway_detection_time = time.time() - start_time
            results['runtime']['pathway_detection'].append(pathway_detection_time)

            logger.info(f"Pathway detection completed in {pathway_detection_time:.2f}s. Detected {len(detected_pathways)} pathways")
            logger.debug(f"Detected pathways: {detected_pathways}")

            # Evaluate pathway detection
            precision, recall, f1, pji = evaluate_pathway_detection(detected_pathways, true_pathways)
            results['pathway_detection']['precision'].append(precision)
            results['pathway_detection']['recall'].append(recall)
            results['pathway_detection']['f1_score'].append(f1)
            results['pathway_detection']['pji'].append(pji)

            logger.info(f"Pathway detection evaluation: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, PJI={pji:.4f}")

            # Localize sources
            logger.info("Localizing sources...")
            start_time = time.time()
            localizer = SourceLocalizer()

            try:
                # If no pathways were detected, use activation times to localize sources
                if not detected_pathways:
                    logger.info("No pathways detected, using activation times for source localization")

                    # Get active nodes
                    active_nodes = list(features['amplitude'].keys())

                    # Get activation times
                    activation_times = features['activation_time']

                    # Find nodes with minimum activation time
                    min_time = min(activation_times.values())
                    detected_sources = [node for node, time in activation_times.items() if time == min_time]
                else:
                    # Use pathways for source localization
                    detected_sources = localizer.localize(network, features, detected_pathways)

                source_localization_time = time.time() - start_time
                results['runtime']['source_localization'].append(source_localization_time)

                logger.info(f"Source localization completed in {source_localization_time:.2f}s")
                logger.info(f"Detected sources: {detected_sources[:5]}" + ("..." if len(detected_sources) > 5 else ""))
            except Exception as e:
                logger.error(f"Error in source localization: {str(e)}")
                logger.debug(traceback.format_exc())

                # Fallback to using the first 5 nodes as sources
                detected_sources = list(network.graph.nodes())[:min(5, len(network.graph.nodes()))]
                logger.info(f"Falling back to using first {len(detected_sources)} nodes as sources")

                source_localization_time = time.time() - start_time
                results['runtime']['source_localization'].append(source_localization_time)

            # Evaluate source localization
            sr1, sr3, sr5, mrts, ed = evaluate_source_localization(detected_sources, sources, network)
            results['source_localization']['sr@1'].append(sr1)
            results['source_localization']['sr@3'].append(sr3)
            results['source_localization']['sr@5'].append(sr5)
            results['source_localization']['mrts'].append(mrts)
            results['source_localization']['ed'].append(ed)

            logger.info(f"Source localization evaluation: SR@1={sr1:.4f}, SR@3={sr3:.4f}, SR@5={sr5:.4f}, MRTS={mrts:.4f}, ED={ed:.4f}")

            # Select critical nodes
            num_critical = min(NUM_CRITICAL_NODES, len(nodes))
            critical_nodes = np.random.choice(nodes, size=num_critical, replace=False)
            # Convert to a list to avoid NumPy array issues
            critical_nodes = critical_nodes.tolist()
            logger.debug(f"Selected {len(critical_nodes)} critical nodes")

            # Calculate initial impacts
            initial_impacts = {}
            for node in network.graph.nodes():
                if node in features['amplitude']:
                    # Use the amplitude from features
                    initial_impacts[node] = np.mean(features['amplitude'][node] ** 2)
                else:
                    initial_impacts[node] = 0.0

            # Optimize intervention
            logger.info("Optimizing intervention...")
            start_time = time.time()
            optimizer = ResourceOptimizer()

            try:
                allocation = optimizer.optimize(
                    network=network,
                    pathways=detected_pathways,
                    initial_impacts=initial_impacts,
                    critical_nodes=critical_nodes,
                    max_impact=MAX_IMPACT
                )
                intervention_time = time.time() - start_time
                results['runtime']['intervention'].append(intervention_time)

                logger.info(f"Intervention optimization completed in {intervention_time:.2f}s")
                logger.info(f"Allocated resources to {len(allocation)} nodes")
            except Exception as e:
                logger.error(f"Error in intervention optimization: {str(e)}")
                logger.debug(traceback.format_exc())

                # Fallback to a simple allocation strategy
                allocation = critical_nodes[:min(5, len(critical_nodes))]
                logger.info(f"Falling back to allocating resources to {len(allocation)} critical nodes")

                intervention_time = time.time() - start_time
                results['runtime']['intervention'].append(intervention_time)

            # Evaluate intervention
            tir, cer, success = evaluate_intervention(allocation, initial_impacts, critical_nodes, MAX_IMPACT)
            results['intervention']['tir'].append(tir)
            results['intervention']['cer'].append(cer)
            results['intervention']['success'].append(success)

            logger.info(f"Intervention evaluation: TIR={tir:.4f}, CER={cer:.4f}, Success={success}")

            # Record total run time
            run_time = time.time() - run_start_time
            results['runtime']['total'].append(run_time)
            logger.info(f"Run {run+1} completed in {run_time:.2f}s")

        except Exception as e:
            logger.error(f"Error in run {run+1}: {str(e)}")
            logger.debug(traceback.format_exc())

    # Calculate average results
    logger.info("\n=== Summary of Results ===")
    for category in results:
        if category == 'runtime':
            for metric in results[category]:
                if results[category][metric]:  # Check if list is not empty
                    avg_time = np.mean(results[category][metric])
                    std_time = np.std(results[category][metric])
                    logger.info(f"Average {metric} time: {avg_time:.4f} ± {std_time:.4f} seconds")
        else:
            for metric in results[category]:
                if results[category][metric]:  # Check if list is not empty
                    avg_value = np.mean(results[category][metric])
                    std_value = np.std(results[category][metric])
                    logger.info(f"Average {metric}: {avg_value:.4f} ± {std_value:.4f}")

    return results

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

    detected_set = set(detected_sources[:3])  # Top 3
    sr3 = len(detected_set.intersection(set(true_sources))) / len(true_sources)

    detected_set = set(detected_sources[:5])  # Top 5
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
        for detected_source in detected_sources[:5]:  # Consider top 5
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

def evaluate_intervention(allocation, initial_impacts, critical_nodes, max_impact):
    """
    Evaluate intervention performance.

    Args:
        allocation: Resource allocation result
        initial_impacts: Initial impact values
        critical_nodes: List of critical nodes
        max_impact: Maximum permissible impact

    Returns:
        tir, cer, success
    """
    # Total Impact Reduction
    total_initial_impact = sum(initial_impacts.values())
    total_reduced_impact = 0.0
    for node, impact in initial_impacts.items():
        if node in allocation:
            # Assume 50% reduction for allocated resources
            total_reduced_impact += impact * 0.5

    tir = total_reduced_impact / total_initial_impact if total_initial_impact > 0 else 0.0

    # Cost-Effectiveness Ratio (assume unit cost for simplicity)
    cer = tir / len(allocation) if len(allocation) > 0 else 0.0

    # Success in Constraint Satisfaction
    critical_impacts = {node: initial_impacts.get(node, 0.0) for node in critical_nodes}
    for node in critical_impacts:
        if node in allocation:
            critical_impacts[node] *= 0.5  # Assume 50% reduction

    success = all(impact <= max_impact for impact in critical_impacts.values())

    return tir, cer, success

def main():
    """Main function to run experiments on all real-world datasets."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run experiments on real-world datasets.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--dataset', type=str, choices=['roadnet', 'wiki', 'email', 'reddit', 'all'],
                        default='all', help='Dataset to run experiments on')
    parser.add_argument('--runs', type=int, default=NUM_RUNS, help='Number of experiment runs')
    parser.add_argument('--subgraph_size', type=int, default=1000,
                        help='Size of subgraph to use for large datasets')
    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Also set root logger to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("Starting experiments on real-world datasets...")
    logger.info(f"Arguments: {args}")

    # Dictionary to store results for all networks
    all_results = {}

    # 1. roadNet-CA
    if args.dataset in ['roadnet', 'all']:
        try:
            logger.info("\n=== Loading roadNet-CA dataset ===")
            logger.debug(f"Loading from path: {ROADNET_PATH}")

            # Check if file exists
            if not os.path.exists(ROADNET_PATH):
                logger.error(f"Dataset file not found: {ROADNET_PATH}")
                raise FileNotFoundError(f"Dataset file not found: {ROADNET_PATH}")

            start_time = time.time()
            roadnet = load_roadnet_ca(ROADNET_PATH)
            load_time = time.time() - start_time
            logger.info(f"Loaded roadNet-CA in {load_time:.2f}s: {len(roadnet.get_nodes())} nodes, {len(roadnet.get_edges())} edges")

            # Take a smaller subgraph for experiments (for computational efficiency)
            subgraph_size = min(args.subgraph_size, len(roadnet.get_nodes()))
            logger.info(f"Creating subgraph with {subgraph_size} nodes...")

            start_time = time.time()
            subgraph_nodes = list(roadnet.graph.nodes())[:subgraph_size]
            roadnet_sub = roadnet.create_subgraph(subgraph_nodes)
            subgraph_time = time.time() - start_time
            logger.info(f"Created subgraph in {subgraph_time:.2f}s with {len(roadnet_sub.get_nodes())} nodes, {len(roadnet_sub.get_edges())} edges")

            # Run experiments
            roadnet_results = run_experiment("roadNet-CA", roadnet_sub, num_runs=args.runs)
            all_results["roadNet-CA"] = roadnet_results
        except Exception as e:
            logger.error(f"Error processing roadNet-CA: {str(e)}")
            logger.debug(traceback.format_exc())

    # 2. wiki-Talk
    if args.dataset in ['wiki', 'all']:
        try:
            logger.info("\n=== Loading wiki-Talk dataset ===")
            logger.debug(f"Loading from path: {WIKI_TALK_PATH}")

            # Check if file exists
            if not os.path.exists(WIKI_TALK_PATH):
                logger.error(f"Dataset file not found: {WIKI_TALK_PATH}")
                raise FileNotFoundError(f"Dataset file not found: {WIKI_TALK_PATH}")

            start_time = time.time()
            wiki_talk = load_wiki_talk(WIKI_TALK_PATH)
            load_time = time.time() - start_time
            logger.info(f"Loaded wiki-Talk in {load_time:.2f}s: {len(wiki_talk.get_nodes())} nodes, {len(wiki_talk.get_edges())} edges")

            # Take a smaller subgraph for experiments
            subgraph_size = min(args.subgraph_size, len(wiki_talk.get_nodes()))
            logger.info(f"Creating subgraph with {subgraph_size} nodes...")

            start_time = time.time()
            subgraph_nodes = list(wiki_talk.graph.nodes())[:subgraph_size]
            wiki_talk_sub = wiki_talk.create_subgraph(subgraph_nodes)
            subgraph_time = time.time() - start_time
            logger.info(f"Created subgraph in {subgraph_time:.2f}s with {len(wiki_talk_sub.get_nodes())} nodes, {len(wiki_talk_sub.get_edges())} edges")

            # Run experiments
            wiki_talk_results = run_experiment("wiki-Talk", wiki_talk_sub, num_runs=args.runs)
            all_results["wiki-Talk"] = wiki_talk_results
        except Exception as e:
            logger.error(f"Error processing wiki-Talk: {str(e)}")
            logger.debug(traceback.format_exc())

    # 3. email-Eu-core-temporal
    if args.dataset in ['email', 'all']:
        try:
            logger.info("\n=== Loading email-Eu-core-temporal dataset ===")
            logger.debug(f"Loading from path: {EMAIL_PATH}")

            # Check if file exists
            if not os.path.exists(EMAIL_PATH):
                logger.error(f"Dataset file not found: {EMAIL_PATH}")
                raise FileNotFoundError(f"Dataset file not found: {EMAIL_PATH}")

            start_time = time.time()
            email = load_email_eu_core(EMAIL_PATH)
            load_time = time.time() - start_time
            logger.info(f"Loaded email-Eu-core in {load_time:.2f}s: {len(email.get_nodes())} nodes, {len(email.get_edges())} edges")

            # Run experiments (this dataset is small enough to use in full)
            email_results = run_experiment("email-Eu-core", email, num_runs=args.runs)
            all_results["email-Eu-core"] = email_results
        except Exception as e:
            logger.error(f"Error processing email-Eu-core: {str(e)}")
            logger.debug(traceback.format_exc())

    # 4. soc-redditHyperlinks-body
    if args.dataset in ['reddit', 'all']:
        try:
            logger.info("\n=== Loading soc-redditHyperlinks-body dataset ===")
            logger.debug(f"Loading from path: {REDDIT_PATH}")

            # Check if file exists
            if not os.path.exists(REDDIT_PATH):
                logger.error(f"Dataset file not found: {REDDIT_PATH}")
                raise FileNotFoundError(f"Dataset file not found: {REDDIT_PATH}")

            start_time = time.time()
            reddit = load_reddit_hyperlinks(REDDIT_PATH)
            load_time = time.time() - start_time
            logger.info(f"Loaded Reddit Hyperlinks in {load_time:.2f}s: {len(reddit.get_nodes())} nodes, {len(reddit.get_edges())} edges")

            # Take a smaller subgraph for experiments
            subgraph_size = min(args.subgraph_size, len(reddit.get_nodes()))
            logger.info(f"Creating subgraph with {subgraph_size} nodes...")

            start_time = time.time()
            subgraph_nodes = list(reddit.graph.nodes())[:subgraph_size]
            reddit_sub = reddit.create_subgraph(subgraph_nodes)
            subgraph_time = time.time() - start_time
            logger.info(f"Created subgraph in {subgraph_time:.2f}s with {len(reddit_sub.get_nodes())} nodes, {len(reddit_sub.get_edges())} edges")

            # Run experiments
            reddit_results = run_experiment("Reddit Hyperlinks", reddit_sub, num_runs=args.runs)
            all_results["Reddit Hyperlinks"] = reddit_results
        except Exception as e:
            logger.error(f"Error processing Reddit Hyperlinks: {str(e)}")
            logger.debug(traceback.format_exc())

    # Save results if we have any
    if all_results:
        try:
            save_results(all_results)
            generate_comparative_visualizations(all_results)
            logger.info("\nExperiments completed successfully!")
            logger.info("Results saved to results/experiments/real_world/")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            logger.debug(traceback.format_exc())
    else:
        logger.warning("No experiments were run successfully. No results to save.")

def save_results(results):
    """Save experiment results to files."""
    # Save as pickle for later analysis
    import pickle
    with open('results/experiments/real_world/data/experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save summary as CSV
    summary = []
    for network_name, network_results in results.items():
        row = {'network': network_name}

        # Add pathway detection metrics
        for metric, values in network_results['pathway_detection'].items():
            row[f'pathway_{metric}_mean'] = np.mean(values)
            row[f'pathway_{metric}_std'] = np.std(values)

        # Add source localization metrics
        for metric, values in network_results['source_localization'].items():
            row[f'source_{metric}_mean'] = np.mean(values)
            row[f'source_{metric}_std'] = np.std(values)

        # Add intervention metrics
        for metric, values in network_results['intervention'].items():
            row[f'intervention_{metric}_mean'] = np.mean(values)
            row[f'intervention_{metric}_std'] = np.std(values)

        # Add runtime metrics
        for metric, values in network_results['runtime'].items():
            row[f'runtime_{metric}_mean'] = np.mean(values)
            row[f'runtime_{metric}_std'] = np.std(values)

        summary.append(row)

    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/experiments/real_world/data/experiment_summary.csv', index=False)

def generate_comparative_visualizations(results):
    """Generate comparative visualizations of experiment results."""
    # 1. Pathway Detection Performance
    plt.figure(figsize=(12, 8))
    networks = list(results.keys())

    # F1 scores
    f1_means = [np.mean(results[net]['pathway_detection']['f1_score']) for net in networks]
    f1_stds = [np.std(results[net]['pathway_detection']['f1_score']) for net in networks]

    x = np.arange(len(networks))
    width = 0.35

    plt.bar(x, f1_means, width, yerr=f1_stds, label='F1 Score', capsize=5)

    plt.xlabel('Network')
    plt.ylabel('Score')
    plt.title('Pathway Detection Performance')
    plt.xticks(x, networks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/experiments/real_world/figures/pathway_detection_comparison.png', dpi=300)
    plt.close()

    # 2. Source Localization Performance
    plt.figure(figsize=(12, 8))

    # Success rates
    sr1_means = [np.mean(results[net]['source_localization']['sr@1']) for net in networks]
    sr3_means = [np.mean(results[net]['source_localization']['sr@3']) for net in networks]
    sr5_means = [np.mean(results[net]['source_localization']['sr@5']) for net in networks]

    sr1_stds = [np.std(results[net]['source_localization']['sr@1']) for net in networks]
    sr3_stds = [np.std(results[net]['source_localization']['sr@3']) for net in networks]
    sr5_stds = [np.std(results[net]['source_localization']['sr@5']) for net in networks]

    x = np.arange(len(networks))
    width = 0.25

    plt.bar(x - width, sr1_means, width, yerr=sr1_stds, label='SR@1', capsize=5)
    plt.bar(x, sr3_means, width, yerr=sr3_stds, label='SR@3', capsize=5)
    plt.bar(x + width, sr5_means, width, yerr=sr5_stds, label='SR@5', capsize=5)

    plt.xlabel('Network')
    plt.ylabel('Success Rate')
    plt.title('Source Localization Performance')
    plt.xticks(x, networks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/experiments/real_world/figures/source_localization_comparison.png', dpi=300)
    plt.close()

    # 3. Intervention Performance
    plt.figure(figsize=(12, 8))

    # TIR and CER
    tir_means = [np.mean(results[net]['intervention']['tir']) for net in networks]
    cer_means = [np.mean(results[net]['intervention']['cer']) for net in networks]

    tir_stds = [np.std(results[net]['intervention']['tir']) for net in networks]
    cer_stds = [np.std(results[net]['intervention']['cer']) for net in networks]

    x = np.arange(len(networks))
    width = 0.35

    plt.bar(x - width/2, tir_means, width, yerr=tir_stds, label='TIR', capsize=5)
    plt.bar(x + width/2, cer_means, width, yerr=cer_stds, label='CER', capsize=5)

    plt.xlabel('Network')
    plt.ylabel('Score')
    plt.title('Intervention Performance')
    plt.xticks(x, networks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/experiments/real_world/figures/intervention_comparison.png', dpi=300)
    plt.close()

    # 4. Runtime Comparison
    plt.figure(figsize=(12, 8))

    # Runtimes
    pathway_times = [np.mean(results[net]['runtime']['pathway_detection']) for net in networks]
    source_times = [np.mean(results[net]['runtime']['source_localization']) for net in networks]
    intervention_times = [np.mean(results[net]['runtime']['intervention']) for net in networks]

    pathway_stds = [np.std(results[net]['runtime']['pathway_detection']) for net in networks]
    source_stds = [np.std(results[net]['runtime']['source_localization']) for net in networks]
    intervention_stds = [np.std(results[net]['runtime']['intervention']) for net in networks]

    x = np.arange(len(networks))
    width = 0.25

    plt.bar(x - width, pathway_times, width, yerr=pathway_stds, label='Pathway Detection', capsize=5)
    plt.bar(x, source_times, width, yerr=source_stds, label='Source Localization', capsize=5)
    plt.bar(x + width, intervention_times, width, yerr=intervention_stds, label='Intervention', capsize=5)

    plt.xlabel('Network')
    plt.ylabel('Runtime (seconds)')
    plt.title('Algorithm Runtime Comparison')
    plt.xticks(x, networks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/experiments/real_world/figures/runtime_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
