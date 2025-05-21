"""
Utilities for loading real-world datasets into DynamicNetwork format.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
from ..network.graph import DynamicNetwork


def load_roadnet_ca(file_path: str) -> DynamicNetwork:
    """
    Load the roadNet-CA dataset.
    
    Args:
        file_path: Path to the roadNet-CA.txt file
        
    Returns:
        A DynamicNetwork instance representing the road network
    """
    network = DynamicNetwork()
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines starting with #
    data_lines = [line.strip() for line in lines if not line.startswith('#')]
    
    # Process edges
    for line in data_lines:
        if line:
            source, target = map(int, line.split())
            
            # Add nodes if they don't exist
            if source not in network.graph:
                network.add_node(source)
            if target not in network.graph:
                network.add_node(target)
            
            # Add edges in both directions (undirected graph)
            network.add_edge(source, target, weight=1.0)
            network.add_edge(target, source, weight=1.0)
    
    return network


def load_wiki_talk(file_path: str) -> DynamicNetwork:
    """
    Load the wiki-Talk dataset.
    
    Args:
        file_path: Path to the wiki-Talk.txt file
        
    Returns:
        A DynamicNetwork instance representing the communication network
    """
    network = DynamicNetwork()
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines starting with #
    data_lines = [line.strip() for line in lines if not line.startswith('#')]
    
    # Process edges
    for line in data_lines:
        if line:
            parts = line.split()
            if len(parts) >= 2:
                source, target = map(int, parts[:2])
                
                # Add nodes if they don't exist
                if source not in network.graph:
                    network.add_node(source)
                if target not in network.graph:
                    network.add_node(target)
                
                # Add directed edge
                network.add_edge(source, target, weight=1.0)
    
    return network


def load_email_eu_core(file_path: str) -> DynamicNetwork:
    """
    Load the email-Eu-core-temporal dataset.
    
    Args:
        file_path: Path to the email-Eu-core-temporal.txt file
        
    Returns:
        A DynamicNetwork instance representing the email communication network
    """
    network = DynamicNetwork()
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines starting with #
    data_lines = [line.strip() for line in lines if not line.startswith('#')]
    
    # Process edges
    for line in data_lines:
        if line:
            parts = line.split()
            if len(parts) >= 3:
                source, target, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Add nodes if they don't exist
                if source not in network.graph:
                    network.add_node(source)
                if target not in network.graph:
                    network.add_node(target)
                
                # Add directed edge with timestamp as attribute
                network.add_edge(source, target, weight=1.0, timestamp=timestamp)
    
    return network


def load_reddit_hyperlinks(file_path: str) -> DynamicNetwork:
    """
    Load the soc-redditHyperlinks-body dataset.
    
    Args:
        file_path: Path to the soc-redditHyperlinks-body.tsv file
        
    Returns:
        A DynamicNetwork instance representing the Reddit hyperlink network
    """
    network = DynamicNetwork()
    
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Process each row
    for _, row in df.iterrows():
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        timestamp = pd.to_datetime(row['TIMESTAMP']).timestamp()
        sentiment = row['LINK_SENTIMENT']
        
        # Add nodes if they don't exist
        if source not in network.graph:
            network.add_node(source)
        if target not in network.graph:
            network.add_node(target)
        
        # Add directed edge with attributes
        network.add_edge(
            source, 
            target, 
            weight=1.0, 
            timestamp=timestamp,
            sentiment=sentiment
        )
    
    return network
