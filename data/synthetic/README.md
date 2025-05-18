# Synthetic Datasets

This directory contains synthetic datasets for testing and evaluating the framework.

## Directory Structure

- `ba_networks/`: Barabási-Albert networks
- `er_networks/`: Erdős-Rényi networks
- `ws_networks/`: Watts-Strogatz networks
- `grid_networks/`: Grid networks

## Data Format

Each network is stored in GraphML format (`.graphml`), and the corresponding time-series data is stored in NumPy format (`.npy`).

## Generation

Synthetic data can be generated using the functions in `src/network/generators.py` and the example in `examples/synthetic_example.py`.
