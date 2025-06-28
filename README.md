# CTGNN-PSI Framework

A PyTorch implementation of Causal Temporal Graph Neural Networks with Propagation Structure Inference for dynamic network analysis.

## Overview

CTGNN-PSI is a deep learning framework designed for causal inference and source localization in temporal networks. It combines graph neural networks with temporal modeling to identify causal pathways and optimize intervention strategies in dynamic systems.

## Key Features

- **Causal Temporal Graph Convolution (CTGC)**: Graph convolution layers that incorporate causal constraints
- **Variational Temporal Embeddings (VTE)**: Uncertainty-aware temporal representations
- **Neural Hawkes Process Integration**: Principled temporal dynamics modeling
- **Multi-task Learning**: Joint pathway detection and source localization
- **Scalable Implementation**: Efficient processing for large-scale networks

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- PyTorch Geometric >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- NetworkX >= 2.6.0
- Matplotlib >= 3.4.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0

### Setup

```bash
git clone https://github.com/yourusername/ctgnn-psi.git
cd ctgnn-psi
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import torch
from src.ctgnn_psi_framework import CTGNNPSIFramework

# Initialize the framework
model = CTGNNPSIFramework(
    input_dim=64,          # Node feature dimension
    hidden_dim=128,        # Hidden layer dimension
    num_layers=3,          # Number of GNN layers
    num_nodes=100,         # Number of nodes in the graph
    device='cuda'          # Device to run on
)

# Prepare your data
node_features = torch.randn(1, 100, 64)      # [batch, nodes, features]
edge_index = torch.randint(0, 100, (2, 200)) # [2, num_edges]
timestamps = torch.randn(200)                # Edge timestamps

# Forward pass
outputs = model(node_features, edge_index, timestamps)

# Extract results
pathway_predictions = outputs['pathway_logits']    # Pathway detection
source_predictions = outputs['source_logits']      # Source localization
uncertainty = outputs['uncertainty']               # Prediction uncertainty
```

### Training Example

```python
import torch.nn.functional as F
from torch.optim import Adam

# Initialize model and optimizer
model = CTGNNPSIFramework(input_dim=64, hidden_dim=128, num_layers=3, num_nodes=100)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(node_features, edge_index, timestamps)
    
    # Calculate losses
    pathway_loss = F.binary_cross_entropy_with_logits(
        outputs['pathway_logits'], pathway_targets
    )
    source_loss = F.binary_cross_entropy_with_logits(
        outputs['source_logits'], source_targets
    )
    
    # Total loss with uncertainty regularization
    total_loss = pathway_loss + source_loss + 0.1 * outputs['uncertainty'].mean()
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

## Project Structure

```
src/
├── ctgnn_psi_framework.py     # Main framework implementation
├── feature_extraction/        # Signal processing and feature extraction
│   ├── amplitude.py           # Amplitude-based features
│   ├── phase.py              # Phase-based features
│   └── stft.py               # Short-time Fourier transform
├── intervention/             # Intervention optimization algorithms
│   ├── greedy_heuristic.py   # Greedy intervention strategies
│   ├── impact_model.py       # Impact assessment models
│   └── optimizer.py          # Intervention optimization
├── learning/                # Core learning components
│   ├── gnn_model.py         # Graph neural network models
│   └── trainer.py           # Training utilities
├── network/                 # Network generation and utilities
│   ├── generators.py        # Synthetic network generators
│   └── graph.py             # Graph data structures
├── pathway_detection/       # Pathway detection algorithms
│   ├── definition.py        # Pathway definitions
│   ├── detector.py          # Detection algorithms
│   └── validation.py        # Validation methods
├── source_localization/     # Source localization methods
│   ├── evaluation.py        # Evaluation metrics
│   └── localizer.py         # Localization algorithms
└── utils/                   # Utility functions
    ├── io.py                # Input/output operations
    ├── metrics.py           # Evaluation metrics
    ├── real_world_loader.py # Real-world data loaders
    └── visualization.py     # Visualization tools

examples/
└── synthetic_example.py     # Complete usage example

tests/
├── test_network.py         # Network module tests
└── test_pathway_detector.py # Pathway detection tests
```

## Framework Components

### 1. Causal Temporal Graph Convolution (CTGC)

The CTGC layer incorporates causal constraints into graph convolution operations:

```python
from src.ctgnn_psi_framework import CTGCLayer

ctgc = CTGCLayer(input_dim=64, output_dim=128)
output = ctgc(x, edge_index, timestamps)
```

### 2. Variational Temporal Embeddings (VTE)

VTE provides uncertainty quantification for temporal network representations:

```python
from src.ctgnn_psi_framework import VTELayer

vte = VTELayer(input_dim=128, latent_dim=64)
embeddings, uncertainty = vte(temporal_features)
```

### 3. Neural Hawkes Process

Temporal dynamics modeling with theoretical guarantees:

```python
from src.ctgnn_psi_framework import NeuralHawkesLayer

hawkes = NeuralHawkesLayer(hidden_dim=128)
intensity = hawkes(embeddings, timestamps)
```

## Usage Examples

### Synthetic Data Generation

```python
from src.network.generators import generate_temporal_network

# Generate synthetic temporal network
graph_data = generate_temporal_network(
    num_nodes=100,
    num_timesteps=50,
    edge_probability=0.1,
    feature_dim=64
)
```

### Real-world Data Loading

```python
from src.utils.real_world_loader import load_temporal_network

# Load real-world temporal network data
data = load_temporal_network('path/to/dataset.txt')
```

### Evaluation

```python
from src.utils.metrics import evaluate_pathway_detection, evaluate_source_localization

# Evaluate pathway detection
pathway_metrics = evaluate_pathway_detection(
    predictions=pathway_predictions,
    ground_truth=pathway_labels
)

# Evaluate source localization  
source_metrics = evaluate_source_localization(
    predictions=source_predictions,
    ground_truth=source_labels
)
```

## Advanced Configuration

### Custom Model Configuration

```python
model = CTGNNPSIFramework(
    input_dim=128,
    hidden_dim=256,
    num_layers=4,
    num_nodes=500,
    dropout=0.2,
    activation='relu',
    normalization='batch',
    uncertainty_method='variational',
    device='cuda:0'
)
```

### Multi-GPU Training

```python
import torch.nn as nn

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model = model.cuda()
```

## Testing

Run the test suite to verify the installation:

```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- PyTorch Geometric team for graph neural network implementations
- Contributors to the temporal network analysis community
