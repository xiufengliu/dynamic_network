"""
Visualize the synthetic dataset.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from src.network.graph import DynamicNetwork

# Create output directory
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
metadata = data['metadata']

print(f"Dataset loaded successfully.")
print(f"Number of nodes: {len(network.get_nodes())}")
print(f"Number of edges: {len(network.get_edges())}")
print(f"Number of time points: {len(time)}")
print(f"Number of true sources: {len(true_sources)}")
print(f"Number of true pathways: {len(true_pathways)}")
print(f"Metadata: {metadata}")

# Visualize network
print("\nVisualizing network...")
plt.figure(figsize=(10, 8))
G = network.graph
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(
    G,
    pos=pos,
    node_size=50,
    node_color='skyblue',
    edge_color='gray',
    alpha=0.8,
    with_labels=False
)

# Highlight source nodes
if true_sources:
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=true_sources,
        node_size=100,
        node_color='red',
        alpha=1.0
    )

plt.title("Network Visualization")
plt.axis('off')
plt.savefig("results/figures/network.png", dpi=300, bbox_inches='tight')

# Visualize time series data
print("\nVisualizing time series data...")
plt.figure(figsize=(12, 6))

# Plot a few signals
for i in range(min(5, len(signals))):
    plt.plot(time, signals[i], label=f"Node {i}")

# If there are source nodes, plot their signals
for source in true_sources:
    source_idx = network.node_to_index(source)
    if source_idx in signals:
        plt.plot(time, signals[source_idx], label=f"Source Node {source}", linewidth=2, color='red')

plt.title("Time Series Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/figures/time_series.png", dpi=300, bbox_inches='tight')

print("\nVisualization completed successfully!")
print("Results saved to results/figures/")
