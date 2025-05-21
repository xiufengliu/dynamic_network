"""
Run experiments step by step.

This script runs the experiments step by step, with pauses between each step
to allow for debugging and inspection of intermediate results.
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
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
        logging.FileHandler('step_by_step.log')
    ]
)
logger = logging.getLogger(__name__)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

# Default parameters
STFT_WINDOW_SIZE = 256
STFT_OVERLAP = 0.5
DELAY_TOLERANCE = 0.5
PHASE_TOLERANCE = np.pi/4
AMPLITUDE_THRESHOLD = 0.1
EVENT_FREQ = 0.1
NUM_SOURCES = 5
NUM_CRITICAL_NODES = 20
MAX_IMPACT = 0.1

def pause(message="Press Enter to continue..."):
    """Pause execution and wait for user input."""
    input(message)

def load_dataset(dataset_name, subgraph_size=100):
    """Load a dataset and create a subgraph."""
    logger.info(f"Loading {dataset_name} dataset...")

    # Check if file exists
    if dataset_name == "roadnet":
        path = ROADNET_PATH
        loader = load_roadnet_ca
    elif dataset_name == "wiki":
        path = WIKI_TALK_PATH
        loader = load_wiki_talk
    elif dataset_name == "email":
        path = EMAIL_PATH
        loader = load_email_eu_core
    elif dataset_name == "reddit":
        path = REDDIT_PATH
        loader = load_reddit_hyperlinks
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return None

    if not os.path.exists(path):
        logger.error(f"Dataset file not found: {path}")
        return None

    # Load dataset
    start_time = time.time()
    network = loader(path)
    load_time = time.time() - start_time
    logger.info(f"Loaded {dataset_name} in {load_time:.2f}s: {len(network.get_nodes())} nodes, {len(network.get_edges())} edges")

    # Create subgraph
    logger.info(f"Creating subgraph with {subgraph_size} nodes...")
    subgraph_size = min(subgraph_size, len(network.get_nodes()))
    subgraph_nodes = list(network.graph.nodes())[:subgraph_size]

    start_time = time.time()
    subgraph = network.create_subgraph(subgraph_nodes)
    subgraph_time = time.time() - start_time
    logger.info(f"Created subgraph in {subgraph_time:.2f}s with {len(subgraph.get_nodes())} nodes, {len(subgraph.get_edges())} edges")

    return subgraph

def simulate_event_propagation(network, sources):
    """Simulate event propagation on a network."""
    logger.info(f"Simulating event propagation from {len(sources)} sources...")

    # Initialize
    G = network.graph
    nodes = list(G.nodes())
    N = len(nodes)

    # Create time vector (10 seconds at 100 Hz sampling rate)
    fs = 100
    duration = 10
    time = np.linspace(0, duration, int(fs * duration))

    # Initialize signals for all nodes
    signals = {}
    for node in nodes:
        signals[node] = np.zeros_like(time)

    # Initialize activation times and visited nodes
    activation_times = {}
    for source in sources:
        activation_times[source] = 0.0

    visited = set(sources)
    queue = list(sources)

    # Track true pathways
    true_pathways = []
    for source in sources:
        true_pathways.append([source])

    # Breadth-first propagation
    propagation_steps = 0
    while queue:
        propagation_steps += 1

        if propagation_steps % 1000 == 0:
            logger.info(f"Propagation step {propagation_steps}, queue size: {len(queue)}, visited nodes: {len(visited)}")

        current = queue.pop(0)
        current_time = activation_times[current]

        # Process neighbors
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                delay = G[current][neighbor].get('weight', 1.0)
                neighbor_time = current_time + delay

                if neighbor_time < duration:
                    activation_times[neighbor] = neighbor_time
                    visited.add(neighbor)
                    queue.append(neighbor)

                    # Add to pathway
                    for path in true_pathways:
                        if path[-1] == current:
                            path.append(neighbor)

    logger.info(f"Propagation completed in {propagation_steps} steps")
    logger.info(f"Total visited nodes: {len(visited)}/{N}")

    # Generate signals based on activation times
    for node in visited:
        t_activate = activation_times[node]
        idx_activate = int(t_activate * fs)

        # Create a Gaussian pulse centered at activation time
        pulse_width = 0.5 * fs
        t_indices = np.arange(len(time))
        gaussian_pulse = np.exp(-0.5 * ((t_indices - idx_activate) / pulse_width) ** 2)

        # Create oscillatory signal
        oscillation = gaussian_pulse * np.sin(2 * np.pi * EVENT_FREQ * time + np.random.uniform(0, 2*np.pi))

        # Add to node's signal
        signals[node] = oscillation

    # Add noise based on SNR
    snr_db = 10.0
    for node in nodes:
        if np.max(np.abs(signals[node])) > 0:
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
            'phase': np.random.uniform(0, 2*np.pi)
        }

    logger.info(f"Generated {len(true_pathways)} true pathways")

    return features, time, true_pathways

def extract_features(features):
    """Extract features using STFT."""
    logger.info("Extracting features using STFT...")

    start_time = time.time()
    stft = STFT(window_size=STFT_WINDOW_SIZE, overlap=STFT_OVERLAP)

    # Create a new features dictionary in the format expected by SourceLocalizer
    stft_features = {
        'amplitude': {},
        'phase': {},
        'unwrapped_phase': {},
        'activation_time': {},
        'is_active': {}
    }

    for node, feature in features.items():
        if 'signal' in feature:
            # Reshape the signal to match the expected input format (n_signals, n_samples)
            signal_reshaped = np.array([feature['signal']])

            try:
                # Extract features
                extracted_features = stft.extract_features(signal_reshaped, EVENT_FREQ)

                # Copy features to the expected format
                stft_features['amplitude'][node] = extracted_features['amplitude'][0]
                stft_features['phase'][node] = extracted_features['phase'][0]
                stft_features['unwrapped_phase'][node] = extracted_features['unwrapped_phase'][0]
                stft_features['activation_time'][node] = feature['activation_time']
                stft_features['is_active'][node] = extracted_features['is_active'][0]

                # Also update the original feature dictionary
                feature.update(extracted_features)
            except Exception as e:
                logger.error(f"Error extracting features for node {node}: {str(e)}")
                logger.error(traceback.format_exc())

    # Add times and frequencies to the features
    if features and 'times' in next(iter(features.values())):
        stft_features['times'] = next(iter(features.values()))['times']
        stft_features['frequencies'] = next(iter(features.values()))['frequencies']
        stft_features['freq_idx'] = next(iter(features.values()))['freq_idx']

    feature_time = time.time() - start_time
    logger.info(f"Feature extraction completed in {feature_time:.2f}s for {len(features)} nodes")

    return stft_features

def detect_pathways(network, features):
    """Detect propagation pathways."""
    logger.info("Detecting propagation pathways...")

    start_time = time.time()
    detector = PathwayDetector(
        delay_tolerance=DELAY_TOLERANCE,
        phase_tolerance=PHASE_TOLERANCE,
        amplitude_threshold=AMPLITUDE_THRESHOLD
    )
    detected_pathways = detector.detect(network, features, EVENT_FREQ)
    pathway_time = time.time() - start_time

    logger.info(f"Pathway detection completed in {pathway_time:.2f}s")
    logger.info(f"Detected {len(detected_pathways)} pathways")

    return detected_pathways

def localize_sources(network, features, pathways):
    """Localize sources."""
    logger.info("Localizing sources...")

    start_time = time.time()
    localizer = SourceLocalizer()
    detected_sources = localizer.localize(network, features, pathways)
    source_time = time.time() - start_time

    logger.info(f"Source localization completed in {source_time:.2f}s")
    logger.info(f"Detected sources: {detected_sources[:5]}" + ("..." if len(detected_sources) > 5 else ""))

    return detected_sources

def optimize_intervention(network, pathways, features, critical_nodes):
    """Optimize intervention."""
    logger.info("Optimizing intervention...")

    # Calculate initial impacts
    initial_impacts = {}
    for node in network.graph.nodes():
        if node in features['amplitude']:
            # Use the amplitude from features
            initial_impacts[node] = np.mean(features['amplitude'][node] ** 2)
        else:
            initial_impacts[node] = 0.0

    start_time = time.time()
    optimizer = ResourceOptimizer()
    allocation = optimizer.optimize(
        network=network,
        pathways=pathways,
        initial_impacts=initial_impacts,
        critical_nodes=critical_nodes,
        max_impact=MAX_IMPACT
    )
    intervention_time = time.time() - start_time

    logger.info(f"Intervention optimization completed in {intervention_time:.2f}s")
    logger.info(f"Allocated resources to {len(allocation)} nodes")

    return allocation

def main():
    """Main function to run experiments step by step."""
    parser = argparse.ArgumentParser(description='Run experiments step by step.')
    parser.add_argument('--dataset', type=str, choices=['roadnet', 'wiki', 'email', 'reddit'],
                        default='email', help='Dataset to use')
    parser.add_argument('--subgraph_size', type=int, default=100, help='Size of subgraph to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no_pause', action='store_true', help='Run without pausing between steps')

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting step-by-step experiment...")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Subgraph size: {args.subgraph_size}")

    # Create output directories
    os.makedirs('results/step_by_step', exist_ok=True)

    try:
        # Step 1: Load dataset
        logger.info("\n=== Step 1: Load dataset ===")
        network = load_dataset(args.dataset, args.subgraph_size)
        if network is None:
            logger.error("Failed to load dataset")
            return False

        if not args.no_pause:
            pause()

        # Step 2: Select sources
        logger.info("\n=== Step 2: Select sources ===")
        nodes = list(network.graph.nodes())
        sources = np.random.choice(nodes, size=min(NUM_SOURCES, len(nodes)), replace=False)
        logger.info(f"Selected sources: {sources}")

        if not args.no_pause:
            pause()

        # Step 3: Simulate event propagation
        logger.info("\n=== Step 3: Simulate event propagation ===")
        features, time_points, true_pathways = simulate_event_propagation(network, sources)

        if not args.no_pause:
            pause()

        # Step 4: Extract features
        logger.info("\n=== Step 4: Extract features ===")
        features = extract_features(features)

        if not args.no_pause:
            pause()

        # Step 5: Detect pathways
        logger.info("\n=== Step 5: Detect pathways ===")
        detected_pathways = detect_pathways(network, features)

        if not args.no_pause:
            pause()

        # Step 6: Localize sources
        logger.info("\n=== Step 6: Localize sources ===")
        detected_sources = localize_sources(network, features, detected_pathways)

        if not args.no_pause:
            pause()

        # Step 7: Optimize intervention
        logger.info("\n=== Step 7: Optimize intervention ===")
        critical_nodes = np.random.choice(nodes, size=min(NUM_CRITICAL_NODES, len(nodes)), replace=False)
        logger.info(f"Selected {len(critical_nodes)} critical nodes")

        allocation = optimize_intervention(network, detected_pathways, features, critical_nodes)

        # Step 8: Evaluate results
        logger.info("\n=== Step 8: Evaluate results ===")
        logger.info(f"True sources: {sources}")
        logger.info(f"Detected sources: {detected_sources[:5]}" + ("..." if len(detected_sources) > 5 else ""))

        # Calculate source detection accuracy
        detected_set = set(detected_sources[:len(sources)])
        true_set = set(sources)
        accuracy = len(detected_set.intersection(true_set)) / len(true_set)
        logger.info(f"Source detection accuracy: {accuracy:.4f}")

        logger.info("Step-by-step experiment completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in step-by-step experiment: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
