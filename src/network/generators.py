"""
Network generators for synthetic data.
"""

import networkx as nx
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .graph import DynamicNetwork


def generate_barabasi_albert_network(n: int, m: int, seed: Optional[int] = None, 
                                     delay_range: Tuple[float, float] = (1.0, 5.0)) -> DynamicNetwork:
    """
    Generate a Barabási-Albert network.
    
    Args:
        n: Number of nodes.
        m: Number of edges to attach from a new node to existing nodes.
        seed: Random seed.
        delay_range: Range of nominal propagation delays.
        
    Returns:
        A DynamicNetwork instance.
    """
    # Generate a Barabási-Albert graph
    ba_graph = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    
    # Convert to a directed graph
    di_graph = nx.DiGraph()
    for u, v in ba_graph.edges():
        di_graph.add_edge(u, v)
    
    # Create a DynamicNetwork
    network = DynamicNetwork()
    
    # Add nodes
    for node in di_graph.nodes():
        network.add_node(node)
    
    # Add edges with random delays
    rng = np.random.RandomState(seed)
    for u, v in di_graph.edges():
        delay = rng.uniform(delay_range[0], delay_range[1])
        network.add_edge(u, v, weight=delay)
    
    return network


def generate_erdos_renyi_network(n: int, p: float, seed: Optional[int] = None,
                                 delay_range: Tuple[float, float] = (1.0, 5.0)) -> DynamicNetwork:
    """
    Generate an Erdős-Rényi network.
    
    Args:
        n: Number of nodes.
        p: Probability of edge creation.
        seed: Random seed.
        delay_range: Range of nominal propagation delays.
        
    Returns:
        A DynamicNetwork instance.
    """
    # Generate an Erdős-Rényi graph
    er_graph = nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=True)
    
    # Create a DynamicNetwork
    network = DynamicNetwork()
    
    # Add nodes
    for node in er_graph.nodes():
        network.add_node(node)
    
    # Add edges with random delays
    rng = np.random.RandomState(seed)
    for u, v in er_graph.edges():
        delay = rng.uniform(delay_range[0], delay_range[1])
        network.add_edge(u, v, weight=delay)
    
    return network


def generate_watts_strogatz_network(n: int, k: int, p: float, seed: Optional[int] = None,
                                    delay_range: Tuple[float, float] = (1.0, 5.0)) -> DynamicNetwork:
    """
    Generate a Watts-Strogatz network.
    
    Args:
        n: Number of nodes.
        k: Each node is connected to k nearest neighbors in ring topology.
        p: Probability of rewiring each edge.
        seed: Random seed.
        delay_range: Range of nominal propagation delays.
        
    Returns:
        A DynamicNetwork instance.
    """
    # Generate a Watts-Strogatz graph
    ws_graph = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    
    # Convert to a directed graph
    di_graph = nx.DiGraph()
    for u, v in ws_graph.edges():
        di_graph.add_edge(u, v)
    
    # Create a DynamicNetwork
    network = DynamicNetwork()
    
    # Add nodes
    for node in di_graph.nodes():
        network.add_node(node)
    
    # Add edges with random delays
    rng = np.random.RandomState(seed)
    for u, v in di_graph.edges():
        delay = rng.uniform(delay_range[0], delay_range[1])
        network.add_edge(u, v, weight=delay)
    
    return network


def generate_grid_network(n: int, m: int, seed: Optional[int] = None,
                          delay_range: Tuple[float, float] = (1.0, 5.0)) -> DynamicNetwork:
    """
    Generate a grid network.
    
    Args:
        n: Number of rows.
        m: Number of columns.
        seed: Random seed.
        delay_range: Range of nominal propagation delays.
        
    Returns:
        A DynamicNetwork instance.
    """
    # Generate a grid graph
    grid_graph = nx.grid_2d_graph(n, m)
    
    # Convert to a directed graph
    di_graph = nx.DiGraph()
    for u, v in grid_graph.edges():
        di_graph.add_edge(u, v)
        di_graph.add_edge(v, u)  # Make it bidirectional
    
    # Create a DynamicNetwork
    network = DynamicNetwork()
    
    # Add nodes
    for node in di_graph.nodes():
        # Convert tuple node to string
        node_str = f"{node[0]}_{node[1]}"
        network.add_node(node_str)
    
    # Add edges with random delays
    rng = np.random.RandomState(seed)
    for u, v in di_graph.edges():
        # Convert tuple nodes to strings
        u_str = f"{u[0]}_{u[1]}"
        v_str = f"{v[0]}_{v[1]}"
        delay = rng.uniform(delay_range[0], delay_range[1])
        network.add_edge(u_str, v_str, weight=delay)
    
    return network


def generate_random_network(n: int, edge_prob: float, directed: bool = True, 
                            seed: Optional[int] = None,
                            delay_range: Tuple[float, float] = (1.0, 5.0),
                            **attr) -> DynamicNetwork:
    """
    Generate a random network with custom attributes.
    
    Args:
        n: Number of nodes.
        edge_prob: Probability of edge creation.
        directed: Whether the network is directed.
        seed: Random seed.
        delay_range: Range of nominal propagation delays.
        **attr: Additional attributes for nodes and edges.
        
    Returns:
        A DynamicNetwork instance.
    """
    # Create a DynamicNetwork
    network = DynamicNetwork()
    
    # Add nodes
    for i in range(n):
        network.add_node(i, **attr.get('node_attr', {}))
    
    # Add edges
    rng = np.random.RandomState(seed)
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < edge_prob:
                delay = rng.uniform(delay_range[0], delay_range[1])
                network.add_edge(i, j, weight=delay, **attr.get('edge_attr', {}))
                
                # If undirected, add the reverse edge
                if not directed and not network.graph.has_edge(j, i):
                    network.add_edge(j, i, weight=delay, **attr.get('edge_attr', {}))
    
    return network
