"""
Run a minimal test on a small subset of the data.

This script runs a minimal test on a small subset of the data to quickly identify any issues.
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        logging.FileHandler('minimal_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

# Test parameters
SUBGRAPH_SIZE = 10  # Tiny subgraph for quick testing
NUM_SOURCES = 1
NUM_CRITICAL_NODES = 2
STFT_WINDOW_SIZE = 64  # Even smaller window size
STFT_OVERLAP = 0.5
DELAY_TOLERANCE = 0.5
PHASE_TOLERANCE = np.pi/4
AMPLITUDE_THRESHOLD = 0.1
EVENT_FREQ = 0.1

# Enable debug logging
logging.getLogger().setLevel(logging.DEBUG)

# Only run the synthetic test for now
RUN_REAL_NETWORK_TEST = False

def create_test_network():
    """Create a small test network."""
    logger.info("Creating test network...")

    network = DynamicNetwork()

    # Add nodes
    for i in range(10):
        network.add_node(i)

    # Add edges (simple line graph)
    for i in range(9):
        network.add_edge(i, i+1, weight=1.0)

    # Add some cross edges
    network.add_edge(0, 5, weight=2.0)
    network.add_edge(2, 7, weight=1.5)
    network.add_edge(4, 9, weight=1.8)

    logger.info(f"Created test network with {len(network.get_nodes())} nodes and {len(network.get_edges())} edges")
    return network

def simulate_event_propagation(network, sources):
    """Simulate event propagation on a network."""
    logger.info(f"Simulating event propagation from {len(sources)} sources...")

    # Initialize
    G = network.graph
    nodes = list(G.nodes())

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
    while queue:
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

    # Create features dictionary
    features = {}
    for node in visited:
        features[node] = {
            'activation_time': activation_times[node],
            'signal': signals[node],
            'amplitude': np.max(np.abs(signals[node])),
            'phase': np.random.uniform(0, 2*np.pi)
        }

    logger.info(f"Simulation completed. Active nodes: {len(visited)}/{len(nodes)}")
    logger.info(f"Generated {len(true_pathways)} true pathways")

    return features, time, true_pathways

def run_test_on_synthetic_network():
    """Run test on a synthetic network."""
    logger.info("\n=== Running test on synthetic network ===")

    try:
        # Create test network
        network = create_test_network()

        # Select sources
        sources = [0, 5]
        logger.info(f"Selected sources: {sources}")

        # Simulate event propagation
        features, time_points, true_pathways = simulate_event_propagation(network, sources)

        # Extract features using STFT
        logger.info("Extracting features using STFT...")
        stft = STFT(window_size=STFT_WINDOW_SIZE, overlap=STFT_OVERLAP)

        # Debug STFT parameters
        logger.debug(f"STFT parameters: window_size={STFT_WINDOW_SIZE}, overlap={STFT_OVERLAP}")

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
                signal = feature['signal']
                logger.debug(f"Node {node} signal shape: {signal.shape}, type: {type(signal)}")

                try:
                    # Reshape the signal to match the expected input format (n_signals, n_samples)
                    signal_reshaped = np.array([signal])
                    logger.debug(f"Reshaped signal shape: {signal_reshaped.shape}")

                    # Extract features
                    extracted_features = stft.extract_features(signal_reshaped, EVENT_FREQ)
                    logger.debug(f"Extracted features keys: {extracted_features.keys()}")

                    # Copy features to the expected format
                    stft_features['amplitude'][node] = extracted_features['amplitude'][0]
                    stft_features['phase'][node] = extracted_features['phase'][0]
                    stft_features['unwrapped_phase'][node] = extracted_features['unwrapped_phase'][0]
                    stft_features['activation_time'][node] = feature['activation_time']
                    stft_features['is_active'][node] = extracted_features['is_active'][0]

                    # Also update the original feature dictionary
                    feature.update(extracted_features)
                    logger.debug(f"Updated features for node {node}")
                except Exception as e:
                    logger.error(f"Error extracting features for node {node}: {str(e)}")
                    logger.error(traceback.format_exc())

        # Add times and frequencies to the features
        if features and 'times' in next(iter(features.values())):
            stft_features['times'] = next(iter(features.values()))['times']
            stft_features['frequencies'] = next(iter(features.values()))['frequencies']
            stft_features['freq_idx'] = next(iter(features.values()))['freq_idx']

        # Detect propagation pathways
        logger.info("Detecting propagation pathways...")
        detector = PathwayDetector(
            delay_tolerance=DELAY_TOLERANCE,
            phase_tolerance=PHASE_TOLERANCE,
            amplitude_threshold=AMPLITUDE_THRESHOLD
        )
        detected_pathways = detector.detect(network, features, EVENT_FREQ)

        logger.info(f"Detected {len(detected_pathways)} pathways")
        logger.info(f"True pathways: {true_pathways}")
        logger.info(f"Detected pathways: {detected_pathways}")

        # Localize sources
        logger.info("Localizing sources...")
        localizer = SourceLocalizer()
        detected_sources = localizer.localize(network, stft_features, detected_pathways)

        logger.info(f"True sources: {sources}")
        logger.info(f"Detected sources: {detected_sources[:5]}")

        # Optimize intervention
        logger.info("Optimizing intervention...")
        critical_nodes = [3, 7]

        # Calculate initial impacts
        initial_impacts = {}
        for node in network.graph.nodes():
            if node in stft_features['amplitude']:
                # Use the amplitude from stft_features
                initial_impacts[node] = np.mean(stft_features['amplitude'][node] ** 2)
            else:
                initial_impacts[node] = 0.0

        optimizer = ResourceOptimizer()
        allocation = optimizer.optimize(
            network=network,
            pathways=detected_pathways,
            initial_impacts=initial_impacts,
            critical_nodes=critical_nodes,
            max_impact=0.1
        )

        logger.info(f"Allocated resources to {len(allocation)} nodes: {allocation}")

        logger.info("Test on synthetic network completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in synthetic network test: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_test_on_real_network(dataset_name):
    """Run test on a real-world network."""
    logger.info(f"\n=== Running test on {dataset_name} dataset ===")

    try:
        # Load dataset
        if dataset_name == "email":
            logger.info(f"Loading {dataset_name} dataset...")
            if not os.path.exists(EMAIL_PATH):
                logger.error(f"Dataset file not found: {EMAIL_PATH}")
                return False

            network = load_email_eu_core(EMAIL_PATH)
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False

        logger.info(f"Loaded network with {len(network.get_nodes())} nodes and {len(network.get_edges())} edges")

        # Create subgraph
        logger.info(f"Creating subgraph with {SUBGRAPH_SIZE} nodes...")
        subgraph_nodes = list(network.graph.nodes())[:SUBGRAPH_SIZE]
        subgraph = network.create_subgraph(subgraph_nodes)

        logger.info(f"Created subgraph with {len(subgraph.get_nodes())} nodes and {len(subgraph.get_edges())} edges")

        # Select sources
        nodes = list(subgraph.graph.nodes())
        sources = np.random.choice(nodes, size=min(NUM_SOURCES, len(nodes)), replace=False)
        logger.info(f"Selected sources: {sources}")

        # Simulate event propagation
        features, time_points, true_pathways = simulate_event_propagation(subgraph, sources)

        # Extract features using STFT
        logger.info("Extracting features using STFT...")
        stft = STFT(window_size=STFT_WINDOW_SIZE, overlap=STFT_OVERLAP)

        # Debug STFT parameters
        logger.debug(f"STFT parameters: window_size={STFT_WINDOW_SIZE}, overlap={STFT_OVERLAP}")

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
                signal = feature['signal']
                logger.debug(f"Node {node} signal shape: {signal.shape}, type: {type(signal)}")

                try:
                    # Reshape the signal to match the expected input format (n_signals, n_samples)
                    signal_reshaped = np.array([signal])
                    logger.debug(f"Reshaped signal shape: {signal_reshaped.shape}")

                    # Extract features
                    extracted_features = stft.extract_features(signal_reshaped, EVENT_FREQ)
                    logger.debug(f"Extracted features keys: {extracted_features.keys()}")

                    # Copy features to the expected format
                    stft_features['amplitude'][node] = extracted_features['amplitude'][0]
                    stft_features['phase'][node] = extracted_features['phase'][0]
                    stft_features['unwrapped_phase'][node] = extracted_features['unwrapped_phase'][0]
                    stft_features['activation_time'][node] = feature['activation_time']
                    stft_features['is_active'][node] = extracted_features['is_active'][0]

                    # Also update the original feature dictionary
                    feature.update(extracted_features)
                    logger.debug(f"Updated features for node {node}")
                except Exception as e:
                    logger.error(f"Error extracting features for node {node}: {str(e)}")
                    logger.error(traceback.format_exc())

        # Add times and frequencies to the features
        if features and 'times' in next(iter(features.values())):
            stft_features['times'] = next(iter(features.values()))['times']
            stft_features['frequencies'] = next(iter(features.values()))['frequencies']
            stft_features['freq_idx'] = next(iter(features.values()))['freq_idx']

        # Detect propagation pathways
        logger.info("Detecting propagation pathways...")
        detector = PathwayDetector(
            delay_tolerance=DELAY_TOLERANCE,
            phase_tolerance=PHASE_TOLERANCE,
            amplitude_threshold=AMPLITUDE_THRESHOLD
        )
        detected_pathways = detector.detect(subgraph, features, EVENT_FREQ)

        logger.info(f"Detected {len(detected_pathways)} pathways")

        # Localize sources
        logger.info("Localizing sources...")
        localizer = SourceLocalizer()
        detected_sources = localizer.localize(subgraph, stft_features, detected_pathways)

        logger.info(f"True sources: {sources}")
        logger.info(f"Detected sources: {detected_sources[:5]}")

        # Optimize intervention
        logger.info("Optimizing intervention...")
        critical_nodes = np.random.choice(nodes, size=min(NUM_CRITICAL_NODES, len(nodes)), replace=False)

        # Calculate initial impacts
        initial_impacts = {}
        for node in subgraph.graph.nodes():
            if node in stft_features['amplitude']:
                # Use the amplitude from stft_features
                initial_impacts[node] = np.mean(stft_features['amplitude'][node] ** 2)
            else:
                initial_impacts[node] = 0.0

        optimizer = ResourceOptimizer()
        allocation = optimizer.optimize(
            network=subgraph,
            pathways=detected_pathways,
            initial_impacts=initial_impacts,
            critical_nodes=critical_nodes,
            max_impact=0.1
        )

        logger.info(f"Allocated resources to {len(allocation)} nodes")

        logger.info(f"Test on {dataset_name} dataset completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in {dataset_name} dataset test: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run minimal tests."""
    logger.info("Starting minimal tests...")

    # Create output directories
    os.makedirs('results/minimal_test', exist_ok=True)

    # Run test on synthetic network
    synthetic_ok = run_test_on_synthetic_network()

    # Run test on real network (email dataset is the smallest)
    real_ok = True
    if RUN_REAL_NETWORK_TEST:
        real_ok = run_test_on_real_network("email")

    # Print summary
    logger.info("\n=== Minimal Test Summary ===")
    logger.info(f"Synthetic network test: {'OK' if synthetic_ok else 'FAILED'}")
    if RUN_REAL_NETWORK_TEST:
        logger.info(f"Real network test: {'OK' if real_ok else 'FAILED'}")
    else:
        logger.info("Real network test: SKIPPED")

    # Return success if all tests are OK
    return synthetic_ok and real_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
