"""
Compare real-world datasets with synthetic networks.

This script compares the properties of real-world datasets with synthetic networks
to identify similarities and differences in their structural characteristics.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from src.network.graph import DynamicNetwork
from src.network.generators import (
    generate_barabasi_albert_network,
    generate_erdos_renyi_network,
    generate_watts_strogatz_network
)
from src.utils.real_world_loader import (
    load_roadnet_ca,
    load_wiki_talk,
    load_email_eu_core,
    load_reddit_hyperlinks
)

# Create output directory
os.makedirs('results/comparison', exist_ok=True)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

def generate_synthetic_networks(n_nodes=1000, seed=42):
    """
    Generate synthetic networks for comparison.
    
    Args:
        n_nodes: Number of nodes in each synthetic network
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of synthetic networks
    """
    print(f"Generating synthetic networks with {n_nodes} nodes...")
    
    networks = {}
    
    # Barabási-Albert network (scale-free)
    networks["ba"] = generate_barabasi_albert_network(
        n=n_nodes, 
        m=5,  # Each new node attaches to 5 existing nodes
        seed=seed
    )
    
    # Erdős-Rényi network (random)
    networks["er"] = generate_erdos_renyi_network(
        n=n_nodes,
        p=0.01,  # Probability of edge creation
        seed=seed
    )
    
    # Watts-Strogatz network (small-world)
    networks["ws"] = generate_watts_strogatz_network(
        n=n_nodes,
        k=10,  # Each node connected to k nearest neighbors
        p=0.1,  # Probability of rewiring
        seed=seed
    )
    
    return networks

def compute_network_metrics(network):
    """
    Compute various network metrics.
    
    Args:
        network: The network to analyze
        
    Returns:
        Dictionary of metrics
    """
    G = network.graph
    
    metrics = {
        "nodes": len(G),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / len(G),
        "max_degree": max(dict(G.degree()).values()),
        "min_degree": min(dict(G.degree()).values()),
    }
    
    # Compute degree assortativity (correlation between degrees of connected nodes)
    try:
        metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(G)
    except:
        metrics["degree_assortativity"] = None
    
    # Compute clustering coefficient (only for smaller networks)
    if len(G) < 10000:
        try:
            metrics["avg_clustering"] = nx.average_clustering(G)
        except:
            metrics["avg_clustering"] = None
    
    # Compute connected components
    if G.is_directed():
        try:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            metrics["largest_cc_percentage"] = len(largest_cc) / len(G) * 100
        except:
            metrics["largest_cc_percentage"] = None
    else:
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            metrics["largest_cc_percentage"] = len(largest_cc) / len(G) * 100
        except:
            metrics["largest_cc_percentage"] = None
    
    return metrics

def compare_degree_distributions(networks, output_file):
    """
    Compare degree distributions of different networks.
    
    Args:
        networks: Dictionary of networks to compare
        output_file: Path to save the comparison plot
    """
    print("Comparing degree distributions...")
    
    plt.figure(figsize=(12, 8))
    
    for name, network in networks.items():
        G = network.graph
        degrees = [d for _, d in G.degree()]
        
        # Create histogram
        hist, bin_edges = np.histogram(degrees, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot on log-log scale
        plt.loglog(bin_centers, hist, 'o-', label=name, alpha=0.7, markersize=4)
    
    plt.title("Degree Distribution Comparison")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Probability (log scale)")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def compare_metrics(metrics_dict, output_file):
    """
    Compare network metrics across different networks.
    
    Args:
        metrics_dict: Dictionary of network metrics
        output_file: Path to save the comparison plot
    """
    print("Comparing network metrics...")
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    
    # Select metrics to compare
    plot_metrics = ['density', 'avg_degree', 'degree_assortativity', 'avg_clustering', 'largest_cc_percentage']
    plot_df = metrics_df[plot_metrics].copy()
    
    # Create bar plots for each metric
    fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(plot_metrics):
        ax = axes[i]
        plot_df[metric].plot(kind='bar', ax=ax)
        ax.set_title(f"Comparison of {metric}")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(output_file.replace('.png', '.csv'))

def main():
    """Main function to compare real-world and synthetic networks."""
    print("Starting comparison of real-world and synthetic networks...")
    
    # Dictionary to store all networks
    networks = {}
    
    # Load real-world networks
    try:
        print("\nLoading real-world networks...")
        networks["roadNet-CA"] = load_roadnet_ca(ROADNET_PATH)
        networks["wiki-Talk"] = load_wiki_talk(WIKI_TALK_PATH)
        networks["email-Eu"] = load_email_eu_core(EMAIL_PATH)
        networks["reddit"] = load_reddit_hyperlinks(REDDIT_PATH)
    except Exception as e:
        print(f"Error loading real-world networks: {e}")
    
    # Generate synthetic networks
    synthetic_networks = generate_synthetic_networks(n_nodes=1000)
    
    # Add synthetic networks to the dictionary
    for name, network in synthetic_networks.items():
        networks[f"synthetic_{name}"] = network
    
    # Compute metrics for all networks
    metrics = {}
    for name, network in networks.items():
        print(f"Computing metrics for {name}...")
        metrics[name] = compute_network_metrics(network)
    
    # Compare degree distributions
    compare_degree_distributions(networks, "results/comparison/degree_distributions.png")
    
    # Compare network metrics
    compare_metrics(metrics, "results/comparison/network_metrics.png")
    
    print("\nComparison completed successfully!")
    print("Results saved to results/comparison/")

if __name__ == "__main__":
    main()
