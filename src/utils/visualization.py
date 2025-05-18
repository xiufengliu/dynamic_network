"""
Visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union, Set
from ..network.graph import DynamicNetwork
from ..pathway_detection.definition import PropagationPathway


def plot_network(network: DynamicNetwork, node_colors: Optional[Dict[Union[int, str], str]] = None,
                node_sizes: Optional[Dict[Union[int, str], float]] = None,
                edge_colors: Optional[Dict[Tuple[Union[int, str], Union[int, str]], str]] = None,
                edge_widths: Optional[Dict[Tuple[Union[int, str], Union[int, str]], float]] = None,
                title: str = "Network Visualization", figsize: Tuple[int, int] = (10, 8),
                pos: Optional[Dict[Union[int, str], Tuple[float, float]]] = None) -> plt.Figure:
    """
    Plot a network.
    
    Args:
        network: The network to plot.
        node_colors: Dictionary mapping node IDs to colors.
        node_sizes: Dictionary mapping node IDs to sizes.
        edge_colors: Dictionary mapping edge tuples to colors.
        edge_widths: Dictionary mapping edge tuples to widths.
        title: Plot title.
        figsize: Figure size.
        pos: Dictionary mapping node IDs to positions.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the graph
    G = network.graph
    
    # Generate positions if not provided
    if pos is None:
        pos = nx.spring_layout(G)
    
    # Set default node colors
    if node_colors is None:
        node_colors = {node: 'skyblue' for node in G.nodes()}
    
    # Set default node sizes
    if node_sizes is None:
        node_sizes = {node: 300 for node in G.nodes()}
    
    # Set default edge colors
    if edge_colors is None:
        edge_colors = {(u, v): 'gray' for u, v in G.edges()}
    
    # Set default edge widths
    if edge_widths is None:
        edge_widths = {(u, v): 1.0 for u, v in G.edges()}
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[node_colors.get(node, 'skyblue') for node in G.nodes()],
        node_size=[node_sizes.get(node, 300) for node in G.nodes()],
        alpha=0.8,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[edge_colors.get((u, v), 'gray') for u, v in G.edges()],
        width=[edge_widths.get((u, v), 1.0) for u, v in G.edges()],
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_family='sans-serif',
        ax=ax
    )
    
    # Set title
    ax.set_title(title)
    
    # Remove axis
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_pathways(network: DynamicNetwork, pathways: List[PropagationPathway],
                 title: str = "Pathway Visualization", figsize: Tuple[int, int] = (10, 8),
                 pos: Optional[Dict[Union[int, str], Tuple[float, float]]] = None) -> plt.Figure:
    """
    Plot pathways on a network.
    
    Args:
        network: The network.
        pathways: List of pathways to plot.
        title: Plot title.
        figsize: Figure size.
        pos: Dictionary mapping node IDs to positions.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the graph
    G = network.graph
    
    # Generate positions if not provided
    if pos is None:
        pos = nx.spring_layout(G)
    
    # Draw all nodes and edges in gray
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color='lightgray',
        node_size=300,
        alpha=0.5,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='lightgray',
        width=1.0,
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_family='sans-serif',
        ax=ax
    )
    
    # Draw pathways with different colors
    colors = plt.cm.tab10.colors
    
    for i, pathway in enumerate(pathways):
        color = colors[i % len(colors)]
        
        # Draw pathway nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=pathway.nodes,
            node_color=color,
            node_size=500,
            alpha=0.8,
            ax=ax
        )
        
        # Draw pathway edges
        edges = pathway.get_edges()
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=color,
            width=2.0,
            alpha=0.8,
            arrows=True,
            arrowsize=15,
            ax=ax
        )
    
    # Set title
    ax.set_title(title)
    
    # Remove axis
    ax.axis('off')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=f'Pathway {i+1}')
                      for i in range(len(pathways))]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_time_series(time: np.ndarray, signals: Dict[int, np.ndarray],
                    title: str = "Time Series", figsize: Tuple[int, int] = (10, 6),
                    highlight_indices: Optional[List[int]] = None) -> plt.Figure:
    """
    Plot time series data.
    
    Args:
        time: Time array.
        signals: Dictionary mapping node indices to signal arrays.
        title: Plot title.
        figsize: Figure size.
        highlight_indices: List of node indices to highlight.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all signals in gray
    for idx, signal in signals.items():
        ax.plot(time, signal, color='lightgray', alpha=0.5, linewidth=1)
    
    # Highlight specific signals
    if highlight_indices is not None:
        colors = plt.cm.tab10.colors
        
        for i, idx in enumerate(highlight_indices):
            if idx in signals:
                color = colors[i % len(colors)]
                ax.plot(time, signals[idx], color=color, linewidth=2, label=f'Node {idx}')
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    
    # Add legend if there are highlighted signals
    if highlight_indices is not None and len(highlight_indices) > 0:
        ax.legend()
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_stft(time: np.ndarray, freq: np.ndarray, stft: np.ndarray,
             title: str = "STFT Magnitude", figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot STFT magnitude.
    
    Args:
        time: Time array.
        freq: Frequency array.
        stft: STFT magnitude array.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot STFT magnitude
    im = ax.pcolormesh(time, freq, np.abs(stft), shading='gouraud', cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='Magnitude')
    
    # Tight layout
    plt.tight_layout()
    
    return fig
