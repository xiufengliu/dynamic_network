# Causal Pathway Inference and Optimized Intervention in Dynamic Networks

This project implements a methodological framework for modeling, detecting, localizing, and mitigating propagating events in dynamic networks. The framework provides a comprehensive solution for analyzing spatiotemporal dynamics of propagating phenomena in complex networked systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

The framework provides tools for:
- Representing dynamic networks as weighted directed graphs
- Extracting time-varying features from nodal time-series data
- Detecting event propagation pathways based on causality, delay consistency, and feature evolution
- Localizing event sources based on temporal precedence and pathway topology
- Optimizing resource allocation for event mitigation

## Installation

```bash
# Clone the repository
git clone https://github.com/username/dynamic_network.git
cd dynamic_network

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Download the datasets
python download_datasets.py
```

### Datasets

The project uses four real-world network datasets:

1. **email-Eu-core-temporal** (5.3MB): Email communication network from a European research institution
   - Included in the repository

2. **roadNet-CA** (84MB): Road network of California
   - Downloaded by the `download_datasets.py` script

3. **wiki-Talk** (64MB): Wikipedia talk (communication) network
   - Downloaded by the `download_datasets.py` script

4. **soc-redditHyperlinks-body** (305MB): Reddit hyperlinks network
   - Downloaded by the `download_datasets.py` script

The smaller dataset (email-Eu-core-temporal) is included directly in the repository, while the larger datasets are downloaded by the `download_datasets.py` script from their original sources.

## Features

- **Network Representation**: Flexible representation of dynamic networks as weighted directed graphs
- **Feature Extraction**: Advanced time-frequency analysis for extracting amplitude, phase, and other features
- **Pathway Detection**: Rigorous algorithm for detecting event propagation pathways based on causality and consistency
- **Source Localization**: Accurate localization of event sources based on temporal precedence
- **Optimized Intervention**: Resource allocation optimization for mitigating event propagation
- **Visualization**: Comprehensive visualization tools for networks, pathways, and time-series data

## Running Experiments

The project includes several scripts for running experiments. Here's a guide to using them:

### 1. Quick Start: Minimal Test

To quickly test if everything is working correctly, run the minimal test:

```bash
python run_minimal_test.py
```

This will:
- Run a test on a small synthetic network
- Test the core functionality (pathway detection, source localization, intervention)
- Take only a few seconds to complete

### 2. Running Experiments on a Single Dataset

To run experiments on a single dataset:

```bash
python run_single_dataset.py --dataset email --subgraph_size 50 --debug
```

Parameters:
- `--dataset`: Choose from `email`, `roadnet`, `wiki`, or `reddit`
- `--subgraph_size`: Number of nodes to include in the subgraph (smaller = faster)
- `--debug`: Enable detailed logging

Results will be saved to:
- `results/single_dataset/data/`: Raw results and summary CSV
- `results/single_dataset/figures/`: Visualizations

### 3. Running All Experiments

To run all experiments (this may take a long time):

```bash
python run_all_experiments.py --all
```

Options:
- `--all`: Run all experiments
- `--basic`: Run only basic experiments on real-world datasets
- `--baseline`: Run only baseline comparison
- `--scalability`: Run only scalability analysis
- `--sensitivity`: Run only parameter sensitivity analysis
- `--debug`: Enable detailed logging
- `--timeout`: Set timeout for each experiment in seconds (default: 3600)

Example for running only basic experiments with debugging:
```bash
python run_all_experiments.py --basic --debug
```

### 4. Step-by-Step Execution

For detailed analysis and debugging:

```bash
python run_step_by_step.py --dataset email --subgraph_size 20 --debug
```

This runs the experiment step by step with pauses between each step.

### 5. Checking Datasets and Implementation

Before running experiments, you can check if:
- Datasets exist and are in the correct format: `python check_datasets.py`
- Required classes and methods are implemented: `python check_implementation.py`

### 6. Experiment Results

All experiment results are saved in the `results/` directory:

- **Minimal Test Results**: `results/minimal_test/`
- **Single Dataset Results**:
  - Data: `results/single_dataset/data/`
  - Figures: `results/single_dataset/figures/`
- **All Experiments Results**:
  - Basic experiments: `results/real_world/`
  - Baseline comparison: `results/baseline/`
  - Scalability analysis: `results/scalability/`
  - Sensitivity analysis: `results/sensitivity/`

Each directory contains:
- Raw data in pickle format (`.pkl`)
- Summary statistics in CSV format (`.csv`)
- Visualizations in PNG format (`.png`)

### Script Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_minimal_test.py` | Quick test on synthetic data | First test to verify setup |
| `run_single_dataset.py` | Run on one dataset | Testing a specific dataset |
| `run_all_experiments.py` | Run all experiments | Full evaluation |
| `run_step_by_step.py` | Step-by-step execution | Debugging |
| `check_datasets.py` | Verify datasets | Before running experiments |
| `check_implementation.py` | Verify implementation | After code changes |
| `download_datasets.py` | Download large datasets | Initial setup |

### Code Example

For programmatic usage, here's a basic example:

```python
import numpy as np
from src.network.graph import DynamicNetwork
from src.feature_extraction.stft import STFT
from src.pathway_detection.detector import PathwayDetector
from src.source_localization.localizer import SourceLocalizer
from src.intervention.optimizer import ResourceOptimizer

# Create a network
network = DynamicNetwork()
network.load_from_file('data/synthetic/ba_networks/ba_1000.graphml')

# Load time-series data
time_series_data = np.load('data/synthetic/ba_networks/ba_1000_timeseries.npy')

# Extract features
stft = STFT(window_size=256, overlap=0.5)
features = stft.extract_features(time_series_data, freq=0.1)

# Detect pathways
detector = PathwayDetector(
    delay_tolerance=0.5,
    phase_tolerance=np.pi/4,
    amplitude_threshold=0.1
)
pathways = detector.detect(network, features, event_freq=0.1)

# Localize sources
localizer = SourceLocalizer()
sources = localizer.localize(network, features, pathways)

# Optimize intervention
optimizer = ResourceOptimizer()
allocation = optimizer.optimize(
    network=network,
    pathways=pathways,
    initial_impacts=initial_impacts,
    critical_nodes=critical_nodes,
    max_impact=0.1
)
```

## Troubleshooting

If you encounter issues running the experiments:

1. **Check the datasets**:
   ```bash
   python check_datasets.py
   ```
   Make sure all datasets are available and in the correct format.

2. **Check the implementation**:
   ```bash
   python check_implementation.py
   ```
   Verify that all required classes and methods are implemented correctly.

3. **Run with smaller subgraph size**:
   ```bash
   python run_single_dataset.py --dataset email --subgraph_size 20 --debug
   ```
   Using a smaller subgraph reduces memory usage and computation time.

4. **Enable debug logging**:
   Add the `--debug` flag to any script to get detailed logging information.

5. **Check log files**:
   - `minimal_test.log`
   - `single_dataset.log`
   - `run_real_world_experiments.log`

For more detailed debugging information, see [README_DEBUGGING.md](README_DEBUGGING.md).

## Project Structure

```
dynamic_network/
├── data/                           # Data directory
│   ├── synthetic/                  # Synthetic datasets
│   │   ├── ba_networks/            # Barabási-Albert networks
│   │   ├── er_networks/            # Erdős-Rényi networks
│   │   ├── ws_networks/            # Watts-Strogatz networks
│   │   └── grid_networks/          # Grid networks
│   └── real_world/                 # Real-world datasets
│       ├── power_grid/             # Power grid data
│       ├── twitter/                # Twitter MemeTracker data
│       └── enron/                  # Enron email network data
├── src/                            # Source code
│   ├── network/                    # Network representation and modeling
│   │   ├── graph.py                # Graph representation (G = (V, E, W))
│   │   └── generators.py           # Network generators (BA, ER, WS, Grid)
│   ├── feature_extraction/         # Time-series feature extraction
│   │   ├── stft.py                 # Short-Time Fourier Transform implementation
│   │   ├── amplitude.py            # Amplitude extraction
│   │   └── phase.py                # Phase extraction and unwrapping
│   ├── pathway_detection/          # Event propagation pathway detection
│   │   ├── definition.py           # Formal definition of pathways
│   │   ├── detector.py             # Pathway detection algorithm
│   │   └── validation.py           # Pathway validation and consistency checks
│   ├── source_localization/        # Source localization
│   │   ├── localizer.py            # Source localization algorithm
│   │   └── evaluation.py           # Evaluation metrics for localization
│   ├── intervention/               # Optimized resource allocation
│   │   ├── impact_model.py         # Event impact modeling
│   │   ├── optimizer.py            # Resource allocation optimization
│   │   └── greedy_heuristic.py     # Greedy heuristic implementation
│   └── utils/                      # Utility functions
│       ├── metrics.py              # Evaluation metrics
│       ├── visualization.py        # Visualization utilities
│       └── io.py                   # Input/output utilities
├── experiments/                    # Experimental evaluation
├── tests/                          # Unit tests
└── examples/                       # Usage examples
    └── synthetic_example.py        # Example with synthetic data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
@article{dynamic_network_2023,
  title={Causal Pathway Inference and Optimized Intervention in Dynamic Networks},
  author={Author, One and Author, Two and Author, Three},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023}
}
```

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
