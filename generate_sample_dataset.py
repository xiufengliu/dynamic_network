"""
Generate a sample synthetic dataset for testing.
"""

import os
import numpy as np
import pickle
from experiments.synthetic_experiments import generate_synthetic_data

# Create output directory
os.makedirs('data/synthetic/ba_networks', exist_ok=True)

# Generate a small synthetic dataset
print("Generating synthetic dataset...")
network, signals, time, true_sources, true_pathways = generate_synthetic_data(
    network_type='ba',
    n_nodes=100,
    n_samples=1000,
    event_freq=0.1,
    snr_db=10.0,
    delay_uncertainty=0.1,
    seed=42
)

# Save the dataset
print("Saving dataset...")
network_filename = 'data/synthetic/ba_networks/ba_100_10.0db_0.1delay_0.graphml'
data_filename = 'data/synthetic/ba_networks/ba_100_10.0db_0.1delay_0.pkl'

# Save network
network.save_to_file(network_filename)

# Save signals, time, true sources, true pathways, and metadata
with open(data_filename, 'wb') as f:
    pickle.dump({
        'signals': signals,
        'time': time,
        'true_sources': true_sources,
        'true_pathways': true_pathways,
        'metadata': {
            'network_type': 'ba',
            'n_nodes': 100,
            'snr_db': 10.0,
            'delay_uncertainty': 0.1,
            'seed': 42
        }
    }, f)

print(f"Dataset saved to {network_filename} and {data_filename}")
print(f"Number of nodes: {len(network.get_nodes())}")
print(f"Number of edges: {len(network.get_edges())}")
print(f"Number of time points: {len(time)}")
print(f"Number of true sources: {len(true_sources)}")
print(f"Number of true pathways: {len(true_pathways)}")
