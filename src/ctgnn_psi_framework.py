"""
CTGNN-PSI: Causal-Temporal Graph Neural Networks with Probabilistic Source Inference
Implementation of the complete framework as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree
import math
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from scipy.stats import norm, dirichlet
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CTGNNConfig:
    """Configuration for CTGNN-PSI framework"""
    # Network parameters
    num_nodes: int = 100
    hidden_dim: int = 64
    latent_dim: int = 32
    num_temporal_scales: int = 3
    
    # CTGC parameters
    ctgc_layers: int = 3
    temporal_window: float = 1.0
    
    # VTE parameters
    lstm_hidden: int = 128
    attention_dim: int = 64
    beta_vae: float = 1.0
    
    # Hawkes process parameters
    hawkes_base_intensity: float = 0.1
    hawkes_decay: float = 1.0
    
    # Training parameters
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 32
    
    # Evaluation parameters
    uncertainty_threshold: float = 0.1
    credible_interval: float = 0.95


class CausalTemporalKernel(nn.Module):
    """Learnable causal-temporal kernel as defined in Equation (1)"""
    
    def __init__(self, num_scales: int, temporal_window: float):
        super().__init__()
        self.num_scales = num_scales
        self.temporal_window = temporal_window
        
        # Learnable parameters for each scale
        self.weights = nn.Parameter(torch.randn(num_scales))
        self.mu = nn.Parameter(torch.linspace(0, temporal_window, num_scales))
        self.sigma = nn.Parameter(torch.ones(num_scales) * temporal_window / num_scales)
        self.lambda_decay = nn.Parameter(torch.ones(num_scales))
    
    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Compute causal-temporal kernel values
        Args:
            delta_t: Time differences [batch_size, num_edges]
        Returns:
            kernel_values: Kernel values for each scale [batch_size, num_edges, num_scales]
        """
        batch_size, num_edges = delta_t.shape
        
        # Enforce causality: kernel = 0 for delta_t < 0
        causal_mask = (delta_t >= 0).float()
        
        # Expand dimensions for broadcasting
        delta_t_exp = delta_t.unsqueeze(-1)  # [batch_size, num_edges, 1]
        mu_exp = self.mu.view(1, 1, -1)      # [1, 1, num_scales]
        sigma_exp = self.sigma.view(1, 1, -1)  # [1, 1, num_scales]
        lambda_exp = self.lambda_decay.view(1, 1, -1)  # [1, 1, num_scales]
        weights_exp = self.weights.view(1, 1, -1)  # [1, 1, num_scales]
        
        # Gaussian component
        gaussian_term = torch.exp(-0.5 * ((delta_t_exp - mu_exp) / sigma_exp) ** 2)
        
        # Exponential decay component
        decay_term = torch.exp(-lambda_exp * delta_t_exp)
        
        # Combined kernel
        kernel_values = weights_exp * gaussian_term * decay_term
        
        # Apply causality mask
        causal_mask_exp = causal_mask.unsqueeze(-1)
        kernel_values = kernel_values * causal_mask_exp
        
        return kernel_values


class CTGCLayer(nn.Module):
    """Causal-Temporal Graph Convolution Layer"""
    
    def __init__(self, in_features: int, out_features: int, num_temporal_scales: int, temporal_window: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_temporal_scales = num_temporal_scales
        
        # Causal-temporal kernel
        self.temporal_kernel = CausalTemporalKernel(num_temporal_scales, temporal_window)
        
        # Learnable weights for each temporal scale
        self.scale_weights = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_temporal_scales)
        ])
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                delta_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CTGC layer
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            delta_t: Time differences [num_edges]
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        # Compute causal-temporal kernels
        delta_t_batch = delta_t.unsqueeze(0)  # [1, num_edges]
        kernel_values = self.temporal_kernel(delta_t_batch)  # [1, num_edges, num_scales]
        kernel_values = kernel_values.squeeze(0)  # [num_edges, num_scales]
        
        # Initialize output
        out = torch.zeros(num_nodes, self.out_features, device=x.device)
        
        # Apply convolution for each temporal scale
        for s in range(self.num_temporal_scales):
            # Get kernel values for this scale
            scale_kernels = kernel_values[:, s]  # [num_edges]
            
            # Apply linear transformation
            transformed_x = self.scale_weights[s](x)  # [num_nodes, out_features]
            
            # Aggregate messages with temporal kernels
            src_nodes = edge_index[0]  # Source nodes
            tgt_nodes = edge_index[1]  # Target nodes
            
            # Create weighted adjacency matrix for this scale
            weighted_adj = torch.zeros(num_nodes, num_nodes, device=x.device)
            weighted_adj[tgt_nodes, src_nodes] = scale_kernels
            
            # Apply graph convolution
            scale_out = torch.mm(weighted_adj, transformed_x)
            out += scale_out
        
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class VariationalTemporalEncoder(nn.Module):
    """Variational Temporal Encoder with BiLSTM and attention"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 lstm_hidden: int, attention_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lstm_hidden = lstm_hidden
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention_dim = attention_dim
        self.attention_w = nn.Linear(2 * lstm_hidden, attention_dim)
        self.attention_v = nn.Parameter(torch.randn(attention_dim))
        
        # Variational parameters
        self.fc_mu = nn.Linear(2 * lstm_hidden, latent_dim)
        self.fc_logvar = nn.Linear(2 * lstm_hidden, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def attention(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector
        Args:
            lstm_out: LSTM outputs [batch_size, seq_len, 2*lstm_hidden]
        Returns:
            context: Attention-weighted context [batch_size, 2*lstm_hidden]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        # Compute attention energies
        energy = torch.tanh(self.attention_w(lstm_out))  # [batch_size, seq_len, attention_dim]
        energy = torch.matmul(energy, self.attention_v)  # [batch_size, seq_len]
        
        # Compute attention weights
        attention_weights = F.softmax(energy, dim=1)  # [batch_size, seq_len]
        
        # Compute context vector
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, 2*lstm_hidden]
        
        return context, attention_weights
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of VTE
        Args:
            x: Input time series [batch_size, seq_len, input_dim]
            training: Whether in training mode
        Returns:
            z: Latent representation [batch_size, latent_dim]
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            kl_loss: KL divergence loss
        """
        batch_size, seq_len, _ = x.shape
        
        # BiLSTM encoding
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_len, 2*lstm_hidden]
        
        # Attention mechanism
        context, attention_weights = self.attention(lstm_out)  # [batch_size, 2*lstm_hidden]
        
        # Variational parameters
        mu = self.fc_mu(context)  # [batch_size, latent_dim]
        logvar = self.fc_logvar(context)  # [batch_size, latent_dim]
        
        # Reparameterization
        if training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()
        
        return z, mu, logvar, kl_loss


class NeuralHawkesProcess(nn.Module):
    """Neural Hawkes Process for source localization"""
    
    def __init__(self, num_nodes: int, hidden_dim: int, base_intensity: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.base_intensity = base_intensity
        
        # Node embeddings
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        
        # Base intensity network
        self.base_intensity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Influence function network
        self.influence_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),  # [e_i, e_j, time_diff]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Decay parameter network
        self.decay_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # [e_i, e_j]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    
    def compute_intensity(self, node_idx: int, query_time: float, 
                         event_history: List[Tuple[float, int]]) -> torch.Tensor:
        """
        Compute Hawkes process intensity for a node at query time
        Args:
            node_idx: Target node index
            query_time: Query time
            event_history: List of (time, node) events before query_time
        Returns:
            intensity: Hawkes process intensity
        """
        # Get node embedding
        node_emb = self.node_embeddings(torch.tensor(node_idx, device=self.node_embeddings.weight.device))
        
        # Base intensity
        base_intensity = self.base_intensity_net(node_emb)
        
        # Influence from previous events
        influence_intensity = torch.tensor(0.0, device=node_emb.device)
        
        for event_time, event_node in event_history:
            if event_time < query_time:
                # Get event node embedding
                event_emb = self.node_embeddings(torch.tensor(event_node, device=self.node_embeddings.weight.device))
                
                # Time difference
                time_diff = query_time - event_time
                time_diff_tensor = torch.tensor([time_diff], device=node_emb.device)
                
                # Compute influence
                influence_input = torch.cat([node_emb, event_emb, time_diff_tensor])
                influence = self.influence_net(influence_input)
                
                # Compute decay
                decay_input = torch.cat([node_emb, event_emb])
                decay = self.decay_net(decay_input)
                
                # Add influence with decay
                influence_intensity += influence * torch.exp(-decay * time_diff)
        
        total_intensity = base_intensity + influence_intensity
        return total_intensity
    
    def log_likelihood(self, events: List[Tuple[float, int]], time_window: float) -> torch.Tensor:
        """
        Compute log-likelihood of observed events
        Args:
            events: List of (time, node) events
            time_window: Total observation time window
        Returns:
            log_likelihood: Log-likelihood of events
        """
        log_likelihood = torch.tensor(0.0, device=self.node_embeddings.weight.device)
        
        # Log-likelihood of observed events
        for i, (event_time, event_node) in enumerate(events):
            event_history = events[:i]  # Events before current event
            intensity = self.compute_intensity(event_node, event_time, event_history)
            log_likelihood += torch.log(intensity + 1e-8)  # Add small epsilon for numerical stability
        
        # Integral term (compensator)
        # For simplicity, we approximate the integral using the base intensities
        for node_idx in range(self.num_nodes):
            node_emb = self.node_embeddings(torch.tensor(node_idx, device=self.node_embeddings.weight.device))
            base_intensity = self.base_intensity_net(node_emb)
            log_likelihood -= base_intensity * time_window
        
        return log_likelihood


class CTGNN_PSI(nn.Module):
    """Complete CTGNN-PSI Framework"""
    
    def __init__(self, config: CTGNNConfig):
        super().__init__()
        self.config = config
        
        # CTGC layers
        self.ctgc_layers = nn.ModuleList()
        for i in range(config.ctgc_layers):
            in_dim = config.hidden_dim if i > 0 else config.hidden_dim
            out_dim = config.hidden_dim
            self.ctgc_layers.append(
                CTGCLayer(in_dim, out_dim, config.num_temporal_scales, config.temporal_window)
            )
        
        # Input projection
        self.input_projection = nn.Linear(1, config.hidden_dim)  # For time series features
        
        # Variational Temporal Encoder
        self.vte = VariationalTemporalEncoder(
            config.hidden_dim, config.hidden_dim, config.latent_dim,
            config.lstm_hidden, config.attention_dim
        )
        
        # Neural Hawkes Process
        self.hawkes = NeuralHawkesProcess(
            config.num_nodes, config.hidden_dim, config.hawkes_base_intensity
        )
        
        # Pathway detection head
        self.pathway_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Source localization head
        self.source_head = nn.Sequential(
            nn.Linear(config.latent_dim + config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_nodes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                delta_t: torch.Tensor, time_series: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of complete CTGNN-PSI framework
        Args:
            node_features: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            delta_t: Time differences [num_edges]
            time_series: Time series data [batch_size, seq_len, 1]
        Returns:
            Dictionary containing all outputs
        """
        batch_size = time_series.size(0)
        num_nodes = node_features.size(0)
        
        # Project time series to hidden dimension
        projected_features = self.input_projection(time_series)  # [batch_size, seq_len, hidden_dim]
        
        # CTGC encoding
        x = self.input_projection(node_features)  # [num_nodes, hidden_dim]
        for ctgc_layer in self.ctgc_layers:
            x = ctgc_layer(x, edge_index, delta_t)
        
        # Variational Temporal Encoding
        z, mu, logvar, kl_loss = self.vte(projected_features, training=self.training)
        
        # Pathway detection
        pathway_probs = self.pathway_head(z)  # [batch_size, 1]
        
        # Source localization (combine temporal and structural features)
        # Expand node features for batch
        expanded_node_features = x.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, hidden_dim]
        expanded_z = z.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch_size, num_nodes, latent_dim]
        
        # Combine features
        combined_features = torch.cat([expanded_z, expanded_node_features], dim=-1)  # [batch_size, num_nodes, latent_dim + hidden_dim]
        
        # Source probabilities
        source_probs = self.source_head(combined_features.view(-1, combined_features.size(-1)))  # [batch_size * num_nodes, num_nodes]
        source_probs = source_probs.view(batch_size, num_nodes, num_nodes)  # [batch_size, num_nodes, num_nodes]
        source_probs = source_probs.mean(dim=1)  # [batch_size, num_nodes]
        
        return {
            'node_embeddings': x,
            'latent_representation': z,
            'mu': mu,
            'logvar': logvar,
            'kl_loss': kl_loss,
            'pathway_probs': pathway_probs,
            'source_probs': source_probs
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        losses = {}
        
        # Pathway detection loss
        if 'pathway_labels' in targets:
            pathway_loss = F.binary_cross_entropy(
                outputs['pathway_probs'].squeeze(), 
                targets['pathway_labels'].float()
            )
            losses['pathway_loss'] = pathway_loss
        
        # Source localization loss
        if 'source_labels' in targets:
            source_loss = F.cross_entropy(outputs['source_probs'], targets['source_labels'])
            losses['source_loss'] = source_loss
        
        # KL divergence loss
        losses['kl_loss'] = outputs['kl_loss'] * self.config.beta_vae
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


if __name__ == "__main__":
    # Test the implementation
    config = CTGNNConfig()
    model = CTGNN_PSI(config)
    
    # Create dummy data
    num_nodes = 100
    num_edges = 200
    seq_len = 50
    batch_size = 4
    
    # Node features (for simplicity, just random features)
    node_features = torch.randn(num_nodes, 1)
    
    # Edge indices
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Time differences (should be positive for causal relationships)
    delta_t = torch.rand(num_edges) * 2.0  # Random time differences
    
    # Time series data
    time_series = torch.randn(batch_size, seq_len, 1)
    
    # Forward pass
    outputs = model(node_features, edge_index, delta_t, time_series)
    
    print("Model output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    # Test loss computation
    targets = {
        'pathway_labels': torch.randint(0, 2, (batch_size,)),
        'source_labels': torch.randint(0, num_nodes, (batch_size,))
    }
    
    losses = model.compute_loss(outputs, targets)
    print("\nLoss values:")
    for key, value in losses.items():
        print(f"{key}: {value.item():.4f}")
