"""
Run a sample experiment using the pre-generated synthetic dataset.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from experiments.synthetic_experiments import load_synthetic_dataset
from src.feature_extraction.stft import STFT
from src.pathway_detection.detector import PathwayDetector
from src.source_localization.localizer import SourceLocalizer
from src.intervention.impact_model import ImpactModel
from src.intervention.optimizer import ResourceOptimizer

# Create output directories
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Load the pre-generated dataset
print("Loading synthetic dataset...")
network, signals, time, true_sources, true_pathways = load_synthetic_dataset(
    network_type='ba',
    n_nodes=100,
    snr_db=10.0,
    delay_uncertainty=0.1,
    dataset_idx=0,
    data_dir='data/synthetic'
)

print(f"Dataset loaded successfully.")
print(f"Number of nodes: {len(network.get_nodes())}")
print(f"Number of edges: {len(network.get_edges())}")
print(f"Number of time points: {len(time)}")
print(f"Number of true sources: {len(true_sources)}")
print(f"Number of true pathways: {len(true_pathways)}")

# Extract features
print("\nExtracting features...")
stft = STFT(window_size=256, overlap=0.75)
event_freq = 0.1
features = stft.extract_features(
    np.array([signals[i] for i in range(len(network))]),
    freq=event_freq,
    amplitude_threshold=0.2
)

# Detect pathways
print("\nDetecting pathways...")
detector = PathwayDetector(
    delay_tolerance=0.5,
    phase_tolerance=np.pi/4,
    amplitude_threshold=0.2
)
detected_pathways = detector.detect(network, features, event_freq)
print(f"Detected {len(detected_pathways)} pathways.")

# Localize sources
print("\nLocalizing sources...")
localizer = SourceLocalizer()
detected_sources = localizer.localize(network, features, detected_pathways)
print(f"Detected sources: {detected_sources}")
print(f"True sources: {true_sources}")

# Calculate initial impacts
print("\nCalculating impacts...")
impact_model = ImpactModel(alpha=2.0)
initial_impacts = impact_model.calculate_initial_impacts(network, features)
impact_model.generate_transmission_factors(network, seed=42)

# Select critical nodes (10% of nodes)
n_critical = max(1, int(len(network.get_nodes()) * 0.1))
critical_nodes = [node for node, impact in sorted(initial_impacts.items(), key=lambda x: x[1], reverse=True)[:n_critical]]
print(f"Selected {len(critical_nodes)} critical nodes.")

# Optimize resource allocation
print("\nOptimizing resource allocation...")
optimizer = ResourceOptimizer(impact_model)
allocation = optimizer.optimize(
    network=network,
    pathways=detected_pathways,
    initial_impacts=initial_impacts,
    critical_nodes=critical_nodes,
    max_impact=0.1,
    budget=len(network.get_nodes()) * 0.1
)
print(f"Allocated resources to {len(allocation)} nodes.")

# Evaluate allocation
evaluation = optimizer.evaluate_allocation(
    network=network,
    pathways=detected_pathways,
    initial_impacts=initial_impacts,
    allocation=allocation,
    critical_nodes=critical_nodes
)

print("\nEvaluation results:")
for metric, value in evaluation.items():
    print(f"{metric}: {value:.4f}")

# Visualize time series data
print("\nVisualizing time series data...")
plt.figure(figsize=(12, 6))
for i in range(min(5, len(signals))):  # Plot first 5 signals
    plt.plot(time, signals[i], label=f"Node {i}")
plt.title("Time Series Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/figures/time_series.png", dpi=300, bbox_inches='tight')

print("\nExperiment completed successfully!")
print("Results saved to results/figures/time_series.png")
