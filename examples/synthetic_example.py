"""
Example of using the framework with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network.graph import DynamicNetwork
from src.network.generators import generate_barabasi_albert_network
from src.feature_extraction.stft import STFT
from src.feature_extraction.amplitude import AmplitudeExtractor
from src.feature_extraction.phase import PhaseExtractor
from src.pathway_detection.detector import PathwayDetector
from src.source_localization.localizer import SourceLocalizer
from src.intervention.impact_model import ImpactModel
from src.intervention.optimizer import ResourceOptimizer
from src.utils.visualization import plot_network, plot_pathways, plot_time_series


def generate_synthetic_data(n_nodes: int = 100, n_samples: int = 1000,
                           event_freq: float = 0.1, snr: float = 10.0,
                           seed: Optional[int] = None) -> Tuple[DynamicNetwork, Dict[int, np.ndarray], np.ndarray]:
    """
    Generate synthetic data for testing.

    Args:
        n_nodes: Number of nodes.
        n_samples: Number of time samples.
        event_freq: Characteristic frequency of the event.
        snr: Signal-to-noise ratio.
        seed: Random seed.

    Returns:
        Tuple of (network, signals, time).
    """
    # Set random seed
    rng = np.random.RandomState(seed)

    # Generate network
    network = generate_barabasi_albert_network(n=n_nodes, m=4, seed=seed)

    # Generate time array
    fs = 1.0  # Sampling frequency
    time = np.arange(n_samples) / fs

    # Select a source node
    source_idx = rng.randint(0, n_nodes)
    source_node = network.index_to_node(source_idx)

    # Initialize signals
    signals = {}
    for i in range(n_nodes):
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

    # Propagate event through the network
    activation_times = {source_idx: event_start}
    processed_nodes = {source_idx}
    nodes_to_process = [source_idx]

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

            # Add random variation to delay
            delay_variation = rng.normal(0, 0.1 * delay)
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
            neighbor_phase = event_phase - phase_shift
            neighbor_event = envelope * np.sin(2 * np.pi * event_freq * t / fs + neighbor_phase)
            signals[neighbor_idx][activation_time:activation_time+event_duration] = event_amplitude * attenuation * neighbor_event

            # Add to activation times
            activation_times[neighbor_idx] = activation_time

            # Mark as processed and add to queue
            processed_nodes.add(neighbor_idx)
            nodes_to_process.append(neighbor_idx)

    # Add noise
    for i in range(n_nodes):
        # Calculate signal power
        signal_power = np.mean(signals[i]**2)

        # Calculate noise power
        noise_power = signal_power / (10**(snr/10)) if signal_power > 0 else 0.01

        # Add noise
        noise = rng.normal(0, np.sqrt(noise_power), n_samples)
        signals[i] += noise

    return network, signals, time


def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    network, signals, time = generate_synthetic_data(n_nodes=50, n_samples=1000, seed=42)

    # Extract features
    print("Extracting features...")
    stft = STFT(window_size=128, overlap=0.75)
    event_freq = 0.1
    features = stft.extract_features(
        np.array([signals[i] for i in range(len(network))]),
        freq=event_freq,
        amplitude_threshold=0.2
    )

    # Detect pathways
    print("Detecting pathways...")
    detector = PathwayDetector(
        delay_tolerance=0.5,
        phase_tolerance=np.pi/4,
        amplitude_threshold=0.2
    )
    pathways = detector.detect(network, features, event_freq)

    print(f"Detected {len(pathways)} pathways.")
    for i, pathway in enumerate(pathways[:5]):  # Show first 5 pathways
        print(f"Pathway {i+1}: {pathway}")

    # Localize sources
    print("\nLocalizing sources...")
    localizer = SourceLocalizer()
    sources = localizer.localize(network, features, pathways)

    print(f"Detected {len(sources)} sources: {sources}")

    # Calculate initial impacts
    print("\nCalculating impacts...")
    impact_model = ImpactModel(alpha=2.0)
    initial_impacts = impact_model.calculate_initial_impacts(network, features)

    # Generate transmission factors
    impact_model.generate_transmission_factors(network, seed=42)

    # Optimize resource allocation
    print("Optimizing resource allocation...")
    optimizer = ResourceOptimizer(impact_model)

    # Define critical nodes (e.g., nodes with high impact)
    critical_nodes = [node for node, impact in sorted(initial_impacts.items(), key=lambda x: x[1], reverse=True)[:10]]

    # Optimize allocation
    allocation = optimizer.optimize(
        network=network,
        pathways=pathways,
        initial_impacts=initial_impacts,
        critical_nodes=critical_nodes,
        max_impact=0.1
    )

    print(f"Allocated resources to {len(allocation)} nodes.")

    # Evaluate allocation
    evaluation = optimizer.evaluate_allocation(
        network=network,
        pathways=pathways,
        initial_impacts=initial_impacts,
        allocation=allocation,
        critical_nodes=critical_nodes
    )

    print("\nEvaluation results:")
    for metric, value in evaluation.items():
        print(f"{metric}: {value:.4f}")

    # Visualize network and pathways
    print("\nVisualizing results...")

    # Plot network
    fig1 = plot_network(network, title="Synthetic Network")
    plt.savefig("network.png")

    # Plot pathways
    fig2 = plot_pathways(network, pathways[:5], title="Detected Pathways")
    plt.savefig("pathways.png")

    # Plot time series
    fig3 = plot_time_series(time, signals, title="Time Series Data",
                           highlight_indices=[network.node_to_index(source) for source in sources])
    plt.savefig("time_series.png")

    print("Visualization saved to network.png, pathways.png, and time_series.png.")

    plt.show()


if __name__ == "__main__":
    main()
