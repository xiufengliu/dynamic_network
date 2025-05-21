"""
Analyze properties of real-world datasets.

This script analyzes the network properties of the four real-world datasets:
1. roadNet-CA - Road network of California
2. wiki-Talk - Wikipedia talk page interactions
3. email-Eu-core-temporal - Email communications in a European research institution
4. soc-redditHyperlinks-body - Reddit hyperlinks between communities
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from src.network.graph import DynamicNetwork
from src.utils.real_world_loader import (
    load_roadnet_ca,
    load_wiki_talk,
    load_email_eu_core,
    load_reddit_hyperlinks
)

# Create output directory
os.makedirs('results/analysis/real_world', exist_ok=True)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

def compute_basic_stats(network, title):
    """
    Compute basic statistics of the network.
    
    Args:
        network: The network to analyze
        title: The title for the analysis
        
    Returns:
        A dictionary of statistics
    """
    print(f"Computing basic statistics for {title}...")
    G = network.graph
    
    stats = {
        "title": title,
        "nodes": len(G),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed": G.is_directed(),
        "avg_degree": sum(dict(G.degree()).values()) / len(G),
        "max_degree": max(dict(G.degree()).values()),
        "min_degree": min(dict(G.degree()).values()),
    }
    
    # Compute additional statistics for smaller networks
    if len(G) < 10000:
        try:
            # Compute connected components
            if G.is_directed():
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(G)
                stats["largest_wcc_size"] = len(largest_cc)
                stats["largest_wcc_percentage"] = len(largest_cc) / len(G) * 100
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                stats["num_connected_components"] = nx.number_connected_components(G)
                stats["largest_cc_size"] = len(largest_cc)
                stats["largest_cc_percentage"] = len(largest_cc) / len(G) * 100
            
            # Create subgraph of largest component for further analysis
            if G.is_directed():
                largest_cc_graph = G.subgraph(largest_cc).copy()
            else:
                largest_cc_graph = G.subgraph(largest_cc).copy()
            
            # Compute diameter and average path length for the largest component
            if len(largest_cc) < 5000:  # Only for reasonably sized components
                if G.is_directed():
                    stats["diameter"] = nx.diameter(largest_cc_graph.to_undirected())
                    stats["avg_path_length"] = nx.average_shortest_path_length(largest_cc_graph)
                else:
                    stats["diameter"] = nx.diameter(largest_cc_graph)
                    stats["avg_path_length"] = nx.average_shortest_path_length(largest_cc_graph)
        except Exception as e:
            print(f"  Warning: Could not compute some metrics: {e}")
    
    return stats

def analyze_degree_distribution(network, title, output_file):
    """
    Analyze the degree distribution of the network.
    
    Args:
        network: The network to analyze
        title: The title for the analysis
        output_file: The output file path
    """
    print(f"Analyzing degree distribution for {title}...")
    G = network.graph
    
    # Get degree distribution
    if G.is_directed():
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        # Create dataframe
        df = pd.DataFrame({
            'node': list(G.nodes()),
            'in_degree': [in_degrees[n] for n in G.nodes()],
            'out_degree': [out_degrees[n] for n in G.nodes()],
            'total_degree': [in_degrees[n] + out_degrees[n] for n in G.nodes()]
        })
    else:
        degrees = dict(G.degree())
        df = pd.DataFrame({
            'node': list(G.nodes()),
            'degree': [degrees[n] for n in G.nodes()]
        })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    return df

def analyze_temporal_patterns(network, title, output_file):
    """
    Analyze temporal patterns in the network if timestamps are available.
    
    Args:
        network: The network to analyze
        title: The title for the analysis
        output_file: The output file path
        
    Returns:
        A dictionary of temporal statistics or None if no timestamps
    """
    print(f"Analyzing temporal patterns for {title}...")
    G = network.graph
    
    # Check if edges have timestamp attribute
    has_timestamps = False
    for _, _, data in G.edges(data=True):
        if 'timestamp' in data:
            has_timestamps = True
            break
    
    if not has_timestamps:
        print(f"  No timestamps found in {title}")
        return None
    
    # Extract timestamps
    edge_times = []
    for u, v, data in G.edges(data=True):
        if 'timestamp' in data:
            edge_times.append((u, v, data['timestamp']))
    
    # Sort by timestamp
    edge_times.sort(key=lambda x: x[2])
    
    # Create dataframe
    df = pd.DataFrame(edge_times, columns=['source', 'target', 'timestamp'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Compute temporal statistics
    stats = {
        "title": title,
        "num_temporal_edges": len(df),
        "min_timestamp": df['timestamp'].min(),
        "max_timestamp": df['timestamp'].max(),
        "timespan": df['timestamp'].max() - df['timestamp'].min()
    }
    
    return stats

def main():
    """Main function to analyze all datasets."""
    print("Starting analysis of real-world datasets...")
    
    # Store all statistics
    all_stats = []
    
    # 1. roadNet-CA
    print("\n=== Analyzing roadNet-CA dataset ===")
    try:
        start_time = time.time()
        roadnet = load_roadnet_ca(ROADNET_PATH)
        load_time = time.time() - start_time
        print(f"Loaded roadNet-CA in {load_time:.2f} seconds: {len(roadnet.get_nodes())} nodes, {len(roadnet.get_edges())} edges")
        
        # Compute basic statistics
        stats = compute_basic_stats(roadnet, "California Road Network")
        stats["load_time"] = load_time
        all_stats.append(stats)
        
        # Analyze degree distribution
        analyze_degree_distribution(
            roadnet,
            "California Road Network",
            "results/analysis/real_world/roadnet_ca_degrees.csv"
        )
    except Exception as e:
        print(f"Error analyzing roadNet-CA: {e}")
    
    # 2. wiki-Talk
    print("\n=== Analyzing wiki-Talk dataset ===")
    try:
        start_time = time.time()
        wiki_talk = load_wiki_talk(WIKI_TALK_PATH)
        load_time = time.time() - start_time
        print(f"Loaded wiki-Talk in {load_time:.2f} seconds: {len(wiki_talk.get_nodes())} nodes, {len(wiki_talk.get_edges())} edges")
        
        # Compute basic statistics
        stats = compute_basic_stats(wiki_talk, "Wikipedia Talk Network")
        stats["load_time"] = load_time
        all_stats.append(stats)
        
        # Analyze degree distribution
        analyze_degree_distribution(
            wiki_talk,
            "Wikipedia Talk Network",
            "results/analysis/real_world/wiki_talk_degrees.csv"
        )
    except Exception as e:
        print(f"Error analyzing wiki-Talk: {e}")
    
    # 3. email-Eu-core-temporal
    print("\n=== Analyzing email-Eu-core-temporal dataset ===")
    try:
        start_time = time.time()
        email = load_email_eu_core(EMAIL_PATH)
        load_time = time.time() - start_time
        print(f"Loaded email-Eu-core in {load_time:.2f} seconds: {len(email.get_nodes())} nodes, {len(email.get_edges())} edges")
        
        # Compute basic statistics
        stats = compute_basic_stats(email, "EU Research Institution Email Network")
        stats["load_time"] = load_time
        all_stats.append(stats)
        
        # Analyze degree distribution
        analyze_degree_distribution(
            email,
            "EU Research Institution Email Network",
            "results/analysis/real_world/email_eu_degrees.csv"
        )
        
        # Analyze temporal patterns
        temporal_stats = analyze_temporal_patterns(
            email,
            "EU Research Institution Email Network",
            "results/analysis/real_world/email_eu_temporal.csv"
        )
        if temporal_stats:
            stats.update(temporal_stats)
    except Exception as e:
        print(f"Error analyzing email-Eu-core: {e}")
    
    # 4. soc-redditHyperlinks-body
    print("\n=== Analyzing soc-redditHyperlinks-body dataset ===")
    try:
        start_time = time.time()
        reddit = load_reddit_hyperlinks(REDDIT_PATH)
        load_time = time.time() - start_time
        print(f"Loaded Reddit Hyperlinks in {load_time:.2f} seconds: {len(reddit.get_nodes())} nodes, {len(reddit.get_edges())} edges")
        
        # Compute basic statistics
        stats = compute_basic_stats(reddit, "Reddit Hyperlinks Network")
        stats["load_time"] = load_time
        all_stats.append(stats)
        
        # Analyze degree distribution
        analyze_degree_distribution(
            reddit,
            "Reddit Hyperlinks Network",
            "results/analysis/real_world/reddit_degrees.csv"
        )
        
        # Analyze temporal patterns
        temporal_stats = analyze_temporal_patterns(
            reddit,
            "Reddit Hyperlinks Network",
            "results/analysis/real_world/reddit_temporal.csv"
        )
        if temporal_stats:
            stats.update(temporal_stats)
    except Exception as e:
        print(f"Error analyzing Reddit Hyperlinks: {e}")
    
    # Save all statistics to CSV
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv("results/analysis/real_world/network_statistics.csv", index=False)
    
    print("\nAnalysis completed successfully!")
    print("Results saved to results/analysis/real_world/")

if __name__ == "__main__":
    main()
