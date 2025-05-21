"""
Visualize real-world datasets.

This script loads and visualizes the four real-world datasets:
1. roadNet-CA - Road network of California
2. wiki-Talk - Wikipedia talk page interactions
3. email-Eu-core-temporal - Email communications in a European research institution
4. soc-redditHyperlinks-body - Reddit hyperlinks between communities
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
from collections import Counter
from src.network.graph import DynamicNetwork
from src.utils.real_world_loader import (
    load_roadnet_ca,
    load_wiki_talk,
    load_email_eu_core,
    load_reddit_hyperlinks
)

# Create output directory
os.makedirs('results/figures/real_world', exist_ok=True)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

def visualize_network_sample(network, title, output_file, sample_size=1000, seed=42):
    """
    Visualize a sample of the network.
    
    Args:
        network: The network to visualize
        title: The title of the plot
        output_file: The output file path
        sample_size: Number of nodes to sample
        seed: Random seed for reproducibility
    """
    print(f"Visualizing {title}...")
    
    # Get a sample of nodes if the network is too large
    nodes = network.get_nodes()
    if len(nodes) > sample_size:
        random.seed(seed)
        sampled_nodes = random.sample(nodes, sample_size)
        subgraph = network.create_subgraph(sampled_nodes)
        G = subgraph.graph
        print(f"Sampled {sample_size} nodes from {len(nodes)} total nodes")
    else:
        G = network.graph
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Use different layout algorithms based on network size
    if len(G) < 500:
        pos = nx.spring_layout(G, seed=seed)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Draw the network
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=20,
        node_color='skyblue',
        edge_color='gray',
        alpha=0.7,
        with_labels=False,
        arrows=False
    )
    
    plt.title(f"{title}\n({len(network.get_nodes())} nodes, {len(network.get_edges())} edges)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_degree_distribution(network, title, output_file):
    """
    Visualize the degree distribution of the network.
    
    Args:
        network: The network to visualize
        title: The title of the plot
        output_file: The output file path
    """
    print(f"Visualizing degree distribution for {title}...")
    
    # Get degree distribution
    degrees = [d for _, d in network.graph.degree()]
    degree_counts = Counter(degrees)
    
    # Sort by degree
    x = sorted(degree_counts.keys())
    y = [degree_counts[d] for d in x]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot on log-log scale
    plt.loglog(x, y, 'o-', markersize=4, alpha=0.7)
    
    plt.title(f"Degree Distribution - {title}")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_temporal_data(network, title, output_file, has_timestamps=False):
    """
    Analyze temporal aspects of the network if available.
    
    Args:
        network: The network to analyze
        title: The title of the plot
        output_file: The output file path
        has_timestamps: Whether the network has timestamp data
    """
    if not has_timestamps:
        return
    
    print(f"Analyzing temporal patterns for {title}...")
    
    # Extract timestamps from edges
    timestamps = []
    for u, v, data in network.graph.edges(data=True):
        if 'timestamp' in data:
            timestamps.append(data['timestamp'])
    
    if not timestamps:
        return
    
    # Convert to pandas Series for easier analysis
    ts_series = pd.Series(timestamps)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of timestamps
    plt.hist(ts_series, bins=100, alpha=0.7)
    
    plt.title(f"Temporal Distribution - {title}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to load and visualize all datasets."""
    print("Starting visualization of real-world datasets...")
    
    # 1. roadNet-CA
    print("\n=== Processing roadNet-CA dataset ===")
    try:
        roadnet = load_roadnet_ca(ROADNET_PATH)
        print(f"Loaded roadNet-CA: {len(roadnet.get_nodes())} nodes, {len(roadnet.get_edges())} edges")
        
        visualize_network_sample(
            roadnet, 
            "California Road Network", 
            "results/figures/real_world/roadnet_ca_graph.png"
        )
        
        visualize_degree_distribution(
            roadnet,
            "California Road Network",
            "results/figures/real_world/roadnet_ca_degree_dist.png"
        )
    except Exception as e:
        print(f"Error processing roadNet-CA: {e}")
    
    # 2. wiki-Talk
    print("\n=== Processing wiki-Talk dataset ===")
    try:
        wiki_talk = load_wiki_talk(WIKI_TALK_PATH)
        print(f"Loaded wiki-Talk: {len(wiki_talk.get_nodes())} nodes, {len(wiki_talk.get_edges())} edges")
        
        visualize_network_sample(
            wiki_talk, 
            "Wikipedia Talk Network", 
            "results/figures/real_world/wiki_talk_graph.png"
        )
        
        visualize_degree_distribution(
            wiki_talk,
            "Wikipedia Talk Network",
            "results/figures/real_world/wiki_talk_degree_dist.png"
        )
    except Exception as e:
        print(f"Error processing wiki-Talk: {e}")
    
    # 3. email-Eu-core-temporal
    print("\n=== Processing email-Eu-core-temporal dataset ===")
    try:
        email = load_email_eu_core(EMAIL_PATH)
        print(f"Loaded email-Eu-core: {len(email.get_nodes())} nodes, {len(email.get_edges())} edges")
        
        visualize_network_sample(
            email, 
            "EU Research Institution Email Network", 
            "results/figures/real_world/email_eu_graph.png",
            sample_size=500
        )
        
        visualize_degree_distribution(
            email,
            "EU Research Institution Email Network",
            "results/figures/real_world/email_eu_degree_dist.png"
        )
        
        analyze_temporal_data(
            email,
            "EU Research Institution Email Network",
            "results/figures/real_world/email_eu_temporal.png",
            has_timestamps=True
        )
    except Exception as e:
        print(f"Error processing email-Eu-core: {e}")
    
    # 4. soc-redditHyperlinks-body
    print("\n=== Processing soc-redditHyperlinks-body dataset ===")
    try:
        reddit = load_reddit_hyperlinks(REDDIT_PATH)
        print(f"Loaded Reddit Hyperlinks: {len(reddit.get_nodes())} nodes, {len(reddit.get_edges())} edges")
        
        visualize_network_sample(
            reddit, 
            "Reddit Hyperlinks Network", 
            "results/figures/real_world/reddit_graph.png",
            sample_size=500
        )
        
        visualize_degree_distribution(
            reddit,
            "Reddit Hyperlinks Network",
            "results/figures/real_world/reddit_degree_dist.png"
        )
        
        analyze_temporal_data(
            reddit,
            "Reddit Hyperlinks Network",
            "results/figures/real_world/reddit_temporal.png",
            has_timestamps=True
        )
    except Exception as e:
        print(f"Error processing Reddit Hyperlinks: {e}")
    
    print("\nVisualization completed successfully!")
    print("Results saved to results/figures/real_world/")

if __name__ == "__main__":
    main()
