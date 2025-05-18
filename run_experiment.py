"""
Run a full experiment on the synthetic dataset.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from src.network.graph import DynamicNetwork
from src.feature_extraction.stft import STFT
from src.pathway_detection.detector import PathwayDetector
from src.source_localization.localizer import SourceLocalizer
from src.intervention.impact_model import ImpactModel
from src.intervention.optimizer import ResourceOptimizer
from src.utils.visualization import plot_network, plot_pathways, plot_time_series

# Create output directories
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Load the dataset
print("Loading dataset...")
network_filename = 'data/synthetic/ba_networks/ba_100_10.0db_0.1delay_0.graphml'
data_filename = 'data/synthetic/ba_networks/ba_100_10.0db_0.1delay_0.pkl'

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

print(f"Dataset loaded successfully.")
print(f"Number of nodes: {len(network.get_nodes())}")
print(f"Number of edges: {len(network.get_edges())}")
print(f"Number of time points: {len(time)}")
print(f"Number of true sources: {len(true_sources)}")
print(f"Number of true pathways: {len(true_pathways)}")

# Step 1: Feature Extraction
print("\nStep 1: Feature Extraction")
stft = STFT(window_size=256, overlap=0.75)
event_freq = 0.1
features = stft.extract_features(
    np.array([signals[i] for i in range(len(network))]),
    freq=event_freq,
    amplitude_threshold=0.2
)

# Get active nodes
active_nodes = []
for i in features['amplitude']:
    if i in features['activation_time'] and features['activation_time'][i] is not None:
        active_nodes.append(i)

print(f"Number of active nodes: {len(active_nodes)}")

# Step 2: Pathway Detection
print("\nStep 2: Pathway Detection")
# Try different parameter settings
delay_tolerances = [0.1, 0.5, 1.0]
phase_tolerances = [np.pi/8, np.pi/4, np.pi/2]
amplitude_thresholds = [0.1, 0.2, 0.5]

best_detector = None
best_pathways = []
max_pathways = 0

for delay_tolerance in delay_tolerances:
    for phase_tolerance in phase_tolerances:
        for amplitude_threshold in amplitude_thresholds:
            detector = PathwayDetector(
                delay_tolerance=delay_tolerance,
                phase_tolerance=phase_tolerance,
                amplitude_threshold=amplitude_threshold
            )
            pathways = detector.detect(network, features, event_freq)
            
            print(f"  Parameters: delay_tolerance={delay_tolerance}, phase_tolerance={phase_tolerance}, amplitude_threshold={amplitude_threshold}")
            print(f"  Detected {len(pathways)} pathways.")
            
            if len(pathways) > max_pathways:
                max_pathways = len(pathways)
                best_detector = detector
                best_pathways = pathways

if best_detector is not None:
    print(f"\nBest parameters: delay_tolerance={best_detector.delay_tolerance}, phase_tolerance={best_detector.phase_tolerance}, amplitude_threshold={best_detector.amplitude_threshold}")
    print(f"Detected {len(best_pathways)} pathways.")
    
    # Print pathway details
    for i, pathway in enumerate(best_pathways[:5]):  # Show first 5 pathways
        print(f"Pathway {i+1}: {pathway}")
        print(f"  Source: {pathway.get_source()}")
        print(f"  Sink: {pathway.get_sink()}")
        print(f"  Length: {pathway.get_length()}")
        print(f"  Delays: {pathway.delays}")
else:
    print("No pathways detected with any parameter setting.")
    best_pathways = []

# Step 3: Source Localization
print("\nStep 3: Source Localization")
localizer = SourceLocalizer()
detected_sources = localizer.localize(network, features, best_pathways)
print(f"Detected sources: {detected_sources}")
print(f"True sources: {true_sources}")

# Calculate source localization accuracy
if detected_sources and true_sources:
    accuracy = 1.0 if any(source in true_sources for source in detected_sources) else 0.0
    print(f"Source localization accuracy: {accuracy}")

# Step 4: Intervention Optimization
print("\nStep 4: Intervention Optimization")
impact_model = ImpactModel(alpha=2.0)
initial_impacts = impact_model.calculate_initial_impacts(network, features)
impact_model.generate_transmission_factors(network, seed=42)

# Select critical nodes (10% of nodes)
n_critical = max(1, int(len(network.get_nodes()) * 0.1))
critical_nodes = [node for node, impact in sorted(initial_impacts.items(), key=lambda x: x[1], reverse=True)[:n_critical]]
print(f"Selected {len(critical_nodes)} critical nodes.")

# Optimize resource allocation
optimizer = ResourceOptimizer(impact_model)
allocation = optimizer.optimize(
    network=network,
    pathways=best_pathways,
    initial_impacts=initial_impacts,
    critical_nodes=critical_nodes,
    max_impact=0.1,
    budget=len(network.get_nodes()) * 0.1
)
print(f"Allocated resources to {len(allocation)} nodes.")

# Evaluate allocation
evaluation = optimizer.evaluate_allocation(
    network=network,
    pathways=best_pathways,
    initial_impacts=initial_impacts,
    allocation=allocation,
    critical_nodes=critical_nodes
)

print("\nEvaluation results:")
for metric, value in evaluation.items():
    print(f"{metric}: {value:.4f}")

# Step 5: Visualization
print("\nStep 5: Visualization")

# Visualize network with sources and critical nodes
plt.figure(figsize=(12, 10))
G = network.graph
pos = nx.spring_layout(G, seed=42)

# Draw all nodes and edges
nx.draw_networkx_nodes(
    G,
    pos=pos,
    node_size=50,
    node_color='lightgray',
    alpha=0.5
)
nx.draw_networkx_edges(
    G,
    pos=pos,
    width=0.5,
    alpha=0.3,
    arrows=True,
    arrowsize=10
)

# Highlight source nodes
if detected_sources:
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=detected_sources,
        node_size=100,
        node_color='red',
        alpha=1.0,
        label='Detected Sources'
    )

# Highlight critical nodes
if critical_nodes:
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=critical_nodes,
        node_size=80,
        node_color='blue',
        alpha=0.8,
        label='Critical Nodes'
    )

# Highlight nodes with allocated resources
if allocation:
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=list(allocation.keys()),
        node_size=80,
        node_color='green',
        alpha=0.8,
        label='Resource Allocation'
    )

# Draw pathways
colors = plt.cm.tab10.colors
for i, pathway in enumerate(best_pathways[:5]):  # Show first 5 pathways
    edges = pathway.get_edges()
    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=edges,
        width=2.0,
        alpha=0.8,
        edge_color=colors[i % len(colors)],
        arrows=True,
        arrowsize=15,
        label=f'Pathway {i+1}'
    )

plt.title("Network with Detected Pathways, Sources, and Resource Allocation")
plt.legend()
plt.axis('off')
plt.savefig("results/figures/network_analysis.png", dpi=300, bbox_inches='tight')

# Visualize time series data
plt.figure(figsize=(12, 6))

# Plot a few signals
for i in range(min(5, len(signals))):
    plt.plot(time, signals[i], alpha=0.5, linewidth=1, color='lightgray')

# Plot signals of detected sources
for source in detected_sources:
    source_idx = network.node_to_index(source)
    if source_idx in signals:
        plt.plot(time, signals[source_idx], label=f"Detected Source {source}", linewidth=2, color='red')

# Plot signals of critical nodes
for i, node in enumerate(critical_nodes[:3]):  # Show first 3 critical nodes
    node_idx = network.node_to_index(node)
    if node_idx in signals:
        plt.plot(time, signals[node_idx], label=f"Critical Node {node}", linewidth=1.5, color=colors[i % len(colors)])

plt.title("Time Series Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/figures/time_series_analysis.png", dpi=300, bbox_inches='tight')

print("\nExperiment completed successfully!")
print("Results saved to results/figures/")
