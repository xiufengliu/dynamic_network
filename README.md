# CTGNN-PSI: Causal Temporal Graph Neural Networks with Propagation Structure Inference

A PyTorch implementation of CTGNN-PSI framework for causal inference and source localization in dynamic networks.

## Features

- **Causal Temporal Graph Convolution (CTGC)**: Incorporates causal constraints into temporal graph neural networks
- **Variational Temporal Embeddings (VTE)**: Provides uncertainty quantification for temporal network representations
- **Neural Hawkes Process**: Models temporal dynamics with theoretical guarantees
- **Multi-task Learning**: Simultaneous pathway detection and source localization
- **Scalable Architecture**: Efficient implementation for large-scale networks

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from src.ctgnn_psi_framework import CTGNNPSIFramework

# Initialize framework
model = CTGNNPSIFramework(
    input_dim=64,
    hidden_dim=128,
    num_layers=3,
    num_nodes=100,
    device='cuda'
)

# Forward pass
x = torch.randn(1, 100, 64)  # [batch, nodes, features]
edge_index = torch.randint(0, 100, (2, 200))  # [2, edges]
timestamps = torch.randn(200)  # edge timestamps

outputs = model(x, edge_index, timestamps)
```

## Project Structure

```
src/
├── ctgnn_psi_framework.py     # Main CTGNN-PSI implementation
├── feature_extraction/        # Signal processing modules
├── intervention/             # Intervention optimization
├── learning/                # GNN models and training
├── network/                 # Graph utilities
├── pathway_detection/       # Pathway detection algorithms
├── source_localization/     # Source localization methods
└── utils/                   # Utility functions

examples/
├── synthetic_example.py     # Basic usage example

tests/
├── test_network.py         # Network module tests
└── test_pathway_detector.py # Pathway detection tests
```

## Usage

### Basic Example

See `examples/synthetic_example.py` for a complete example of how to use the framework.

### Framework Components

1. **CTGC Layer**: Causal temporal graph convolution
2. **VTE Layer**: Variational temporal embeddings with uncertainty
3. **Neural Hawkes**: Temporal intensity modeling
4. **Multi-task Head**: Pathway detection and source localization

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- PyTorch Geometric >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- NetworkX >= 2.6.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ctgnn_psi_2025,
  title={CTGNN-PSI: Causal Temporal Graph Neural Networks with Propagation Structure Inference},
  author={[Authors]},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025}
}
```
