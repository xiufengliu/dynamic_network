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

## Usage

### Synthetic Example

The repository includes a synthetic example that demonstrates the full workflow:

```bash
python examples/synthetic_example.py
```

This will:
1. Generate a synthetic network and event propagation
2. Extract features using STFT
3. Detect propagation pathways
4. Localize event sources
5. Optimize resource allocation for mitigation
6. Visualize the results

### Basic Usage Example

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
    initial_impacts=impact_model.calculate_initial_impacts(network, features),
    critical_nodes=critical_nodes,
    max_impact=0.1
)
```

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
