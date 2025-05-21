"""
Compare our method against baseline methods on real-world datasets.

This script implements the baseline methods described in the paper and compares them
against our proposed method on the four real-world datasets.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from src.network.graph import DynamicNetwork
from src.utils.real_world_loader import (
    load_roadnet_ca,
    load_wiki_talk,
    load_email_eu_core,
    load_reddit_hyperlinks
)
from src.feature_extraction.stft import STFT
from src.pathway_detection.detector import PathwayDetector
from src.source_localization.localizer import SourceLocalizer
from src.intervention.optimizer import ResourceOptimizer

# Create output directories
os.makedirs('results/baselines/real_world', exist_ok=True)
os.makedirs('results/baselines/real_world/figures', exist_ok=True)
os.makedirs('results/baselines/real_world/data', exist_ok=True)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

# Experiment parameters
STFT_WINDOW_SIZE = 256
STFT_OVERLAP = 0.5
DELAY_TOLERANCE = 0.5
PHASE_TOLERANCE = np.pi/4
AMPLITUDE_THRESHOLD = 0.1
EVENT_FREQ = 0.1  # Characteristic frequency for oscillatory events
NUM_SOURCES = 5  # Number of sources to simulate
NUM_CRITICAL_NODES = 20  # Number of critical nodes for intervention
MAX_IMPACT = 0.1  # Maximum permissible impact for critical nodes
NUM_RUNS = 5  # Number of experiment runs for statistical significance

# Baseline methods for pathway detection
class TemporalCausality:
    """Baseline method: Temporal Causality (TC)"""
    
    def __init__(self):
        self.name = "TC"
    
    def detect(self, network, features, event_freq):
        """
        Detect pathways using only Conditions 1 and 2 (link existence and causality).
        """
        G = network.graph
        nodes = list(features.keys())
        
        # Get activation times
        activation_times = {}
        for node in nodes:
            activation_times[node] = features[node]['activation_time']
        
        # Create candidate propagation graph
        candidate_edges = []
        for u in nodes:
            for v in nodes:
                if u != v and G.has_edge(u, v):
                    # Check causality
                    if activation_times[v] > activation_times[u]:
                        candidate_edges.append((u, v))
        
        # Build pathways using DFS
        pathways = []
        visited = set()
        
        def dfs(node, path):
            visited.add(node)
            path.append(node)
            
            # Find next nodes in the path
            next_nodes = []
            for u, v in candidate_edges:
                if u == node and v not in visited:
                    next_nodes.append(v)
            
            if not next_nodes:  # End of path
                pathways.append(path.copy())
            else:
                for next_node in next_nodes:
                    dfs(next_node, path)
            
            path.pop()  # Backtrack
            visited.remove(node)
        
        # Start DFS from potential sources (nodes with no incoming edges)
        potential_sources = set(nodes)
        for u, v in candidate_edges:
            potential_sources.discard(v)
        
        for source in potential_sources:
            dfs(source, [])
        
        return pathways

class TemporalCausalityDelayConsistency:
    """Baseline method: TC + Delay Consistency (TCDC)"""
    
    def __init__(self, delay_tolerance=0.5):
        self.name = "TCDC"
        self.delay_tolerance = delay_tolerance
    
    def detect(self, network, features, event_freq):
        """
        Detect pathways using Conditions 1, 2, and 3 (link existence, causality, and delay consistency).
        """
        G = network.graph
        nodes = list(features.keys())
        
        # Get activation times
        activation_times = {}
        for node in nodes:
            activation_times[node] = features[node]['activation_time']
        
        # Create candidate propagation graph
        candidate_edges = []
        for u in nodes:
            for v in nodes:
                if u != v and G.has_edge(u, v):
                    # Check causality
                    if activation_times[v] > activation_times[u]:
                        # Check delay consistency
                        measured_delay = activation_times[v] - activation_times[u]
                        nominal_delay = G[u][v].get('weight', 1.0)
                        
                        if abs(measured_delay - nominal_delay) <= self.delay_tolerance:
                            candidate_edges.append((u, v))
        
        # Build pathways using DFS (same as TC)
        pathways = []
        visited = set()
        
        def dfs(node, path):
            visited.add(node)
            path.append(node)
            
            # Find next nodes in the path
            next_nodes = []
            for u, v in candidate_edges:
                if u == node and v not in visited:
                    next_nodes.append(v)
            
            if not next_nodes:  # End of path
                pathways.append(path.copy())
            else:
                for next_node in next_nodes:
                    dfs(next_node, path)
            
            path.pop()  # Backtrack
            visited.remove(node)
        
        # Start DFS from potential sources (nodes with no incoming edges)
        potential_sources = set(nodes)
        for u, v in candidate_edges:
            potential_sources.discard(v)
        
        for source in potential_sources:
            dfs(source, [])
        
        return pathways

class NoPhase:
    """Baseline method: Ablated Framework (No-Phase)"""
    
    def __init__(self, delay_tolerance=0.5):
        self.name = "No-Phase"
        self.delay_tolerance = delay_tolerance
    
    def detect(self, network, features, event_freq):
        """
        Detect pathways using our full framework but omitting Condition 4 (phase consistency).
        """
        # This is essentially the same as TCDC but using our framework's implementation
        detector = PathwayDetector(
            delay_tolerance=self.delay_tolerance,
            phase_tolerance=float('inf'),  # Effectively ignore phase
            amplitude_threshold=AMPLITUDE_THRESHOLD
        )
        return detector.detect(network, features, event_freq)

# Baseline methods for source localization
class PropagationCentrality:
    """Baseline method: Propagation Centrality (PC)"""
    
    def __init__(self):
        self.name = "PC"
    
    def localize(self, network, features, pathways):
        """
        Localize sources using propagation centrality.
        """
        G = network.graph
        nodes = list(features.keys())
        
        # Get activation times
        activation_times = {}
        for node in nodes:
            activation_times[node] = features[node]['activation_time']
        
        # Calculate propagation centrality for each node
        pc_scores = {}
        for node in nodes:
            # Sum of inverse time differences to all other nodes
            score = 0
            for other in nodes:
                if other != node and activation_times[other] > activation_times[node]:
                    time_diff = activation_times[other] - activation_times[node]
                    score += 1 / time_diff
            pc_scores[node] = score
        
        # Sort nodes by PC score in descending order
        sorted_nodes = sorted(pc_scores.keys(), key=lambda x: pc_scores[x], reverse=True)
        
        return sorted_nodes

class EarliestActivator:
    """Baseline method: Earliest Activator (EA)"""
    
    def __init__(self):
        self.name = "EA"
    
    def localize(self, network, features, pathways):
        """
        Localize sources by selecting nodes with globally minimum activation time.
        """
        nodes = list(features.keys())
        
        # Get activation times
        activation_times = {}
        for node in nodes:
            activation_times[node] = features[node]['activation_time']
        
        # Sort nodes by activation time in ascending order
        sorted_nodes = sorted(activation_times.keys(), key=lambda x: activation_times[x])
        
        return sorted_nodes

class DegreeCentrality:
    """Baseline method: Degree Centrality (DC)"""
    
    def __init__(self):
        self.name = "DC"
    
    def localize(self, network, features, pathways):
        """
        Localize sources using degree centrality.
        """
        G = network.graph
        nodes = list(features.keys())
        
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G.subgraph(nodes))
        
        # Sort nodes by degree centrality in descending order
        sorted_nodes = sorted(degree_centrality.keys(), key=lambda x: degree_centrality[x], reverse=True)
        
        return sorted_nodes

class BetweennessCentrality:
    """Baseline method: Betweenness Centrality (BC)"""
    
    def __init__(self):
        self.name = "BC"
    
    def localize(self, network, features, pathways):
        """
        Localize sources using betweenness centrality.
        """
        G = network.graph
        nodes = list(features.keys())
        
        # Calculate betweenness centrality (may be slow for large networks)
        try:
            betweenness_centrality = nx.betweenness_centrality(G.subgraph(nodes))
            
            # Sort nodes by betweenness centrality in descending order
            sorted_nodes = sorted(betweenness_centrality.keys(), key=lambda x: betweenness_centrality[x], reverse=True)
        except:
            # Fallback to degree centrality if betweenness calculation fails
            print("Betweenness centrality calculation failed, falling back to degree centrality")
            dc = DegreeCentrality()
            sorted_nodes = dc.localize(network, features, pathways)
        
        return sorted_nodes

# Baseline methods for intervention
class RandomPlacement:
    """Baseline method: Random Placement (RP)"""
    
    def __init__(self):
        self.name = "RP"
    
    def optimize(self, network, pathways, initial_impacts, critical_nodes, max_impact):
        """
        Randomly place resources until all critical nodes are below max_impact.
        """
        G = network.graph
        nodes = list(G.nodes())
        
        # Randomly select nodes until all critical nodes are below max_impact
        allocation = []
        remaining_nodes = nodes.copy()
        
        # Simulate impact reduction (simplified model)
        current_impacts = initial_impacts.copy()
        
        while any(current_impacts.get(node, 0) > max_impact for node in critical_nodes) and remaining_nodes:
            # Randomly select a node
            node = np.random.choice(remaining_nodes)
            remaining_nodes.remove(node)
            allocation.append(node)
            
            # Reduce impact at this node and downstream nodes (simplified)
            current_impacts[node] *= 0.5  # 50% reduction
            
            # Propagate reduction downstream (simplified)
            for path in pathways:
                if node in path:
                    idx = path.index(node)
                    for downstream in path[idx+1:]:
                        current_impacts[downstream] *= 0.75  # 25% reduction for downstream nodes
        
        return allocation

class HighestImpactFirst:
    """Baseline method: Highest Impact First (HIF)"""
    
    def __init__(self):
        self.name = "HIF"
    
    def optimize(self, network, pathways, initial_impacts, critical_nodes, max_impact):
        """
        Place resources at nodes with highest impact first.
        """
        # Sort nodes by impact in descending order
        sorted_nodes = sorted(initial_impacts.keys(), key=lambda x: initial_impacts[x], reverse=True)
        
        # Select nodes until all critical nodes are below max_impact
        allocation = []
        current_impacts = initial_impacts.copy()
        
        for node in sorted_nodes:
            if any(current_impacts.get(crit_node, 0) > max_impact for crit_node in critical_nodes):
                allocation.append(node)
                
                # Reduce impact at this node and downstream nodes (simplified)
                current_impacts[node] *= 0.5  # 50% reduction
                
                # Propagate reduction downstream (simplified)
                for path in pathways:
                    if node in path:
                        idx = path.index(node)
                        for downstream in path[idx+1:]:
                            current_impacts[downstream] *= 0.75  # 25% reduction for downstream nodes
            else:
                break
        
        return allocation

class SourceProximity:
    """Baseline method: Source Proximity (SP)"""
    
    def __init__(self):
        self.name = "SP"
    
    def optimize(self, network, pathways, initial_impacts, critical_nodes, max_impact):
        """
        Place resources close to detected sources.
        """
        G = network.graph
        
        # Identify sources (first nodes in pathways)
        sources = []
        for path in pathways:
            if path and path[0] not in sources:
                sources.append(path[0])
        
        if not sources:
            # Fallback to highest impact if no sources
            hif = HighestImpactFirst()
            return hif.optimize(network, pathways, initial_impacts, critical_nodes, max_impact)
        
        # Calculate proximity to sources for each node
        proximity_scores = {}
        for node in G.nodes():
            min_distance = float('inf')
            for source in sources:
                try:
                    distance = nx.shortest_path_length(G, source=source, target=node)
                    min_distance = min(min_distance, distance)
                except nx.NetworkXNoPath:
                    continue
            
            if min_distance == float('inf'):
                proximity_scores[node] = 0
            else:
                proximity_scores[node] = 1 / (1 + min_distance)  # Higher score for closer nodes
        
        # Sort nodes by proximity score in descending order
        sorted_nodes = sorted(proximity_scores.keys(), key=lambda x: proximity_scores[x], reverse=True)
        
        # Select nodes until all critical nodes are below max_impact
        allocation = []
        current_impacts = initial_impacts.copy()
        
        for node in sorted_nodes:
            if any(current_impacts.get(crit_node, 0) > max_impact for crit_node in critical_nodes):
                allocation.append(node)
                
                # Reduce impact at this node and downstream nodes (simplified)
                current_impacts[node] *= 0.5  # 50% reduction
                
                # Propagate reduction downstream (simplified)
                for path in pathways:
                    if node in path:
                        idx = path.index(node)
                        for downstream in path[idx+1:]:
                            current_impacts[downstream] *= 0.75  # 25% reduction for downstream nodes
            else:
                break
        
        return allocation

def run_baseline_comparison(network_name, network, num_runs=5):
    """
    Run baseline comparison on a real-world network.
    
    Args:
        network_name: Name of the network
        network: The network to experiment on
        num_runs: Number of experiment runs
    
    Returns:
        results: Dictionary of experiment results
    """
    print(f"\n=== Running baseline comparison on {network_name} ===")
    
    # Initialize pathway detection methods
    pathway_methods = [
        PathwayDetector(delay_tolerance=DELAY_TOLERANCE, phase_tolerance=PHASE_TOLERANCE, amplitude_threshold=AMPLITUDE_THRESHOLD),  # Our method
        TemporalCausality(),
        TemporalCausalityDelayConsistency(delay_tolerance=DELAY_TOLERANCE),
        NoPhase(delay_tolerance=DELAY_TOLERANCE)
    ]
    
    # Initialize source localization methods
    source_methods = [
        SourceLocalizer(),  # Our method
        PropagationCentrality(),
        EarliestActivator(),
        DegreeCentrality(),
        BetweennessCentrality()
    ]
    
    # Initialize intervention methods
    intervention_methods = [
        ResourceOptimizer(),  # Our method
        RandomPlacement(),
        HighestImpactFirst(),
        SourceProximity()
    ]
    
    # Initialize results
    results = {
        'pathway_detection': {method.name if hasattr(method, 'name') else 'Our Method': [] for method in pathway_methods},
        'source_localization': {method.name if hasattr(method, 'name') else 'Our Method': [] for method in source_methods},
        'intervention': {method.name if hasattr(method, 'name') else 'Our Method': [] for method in intervention_methods}
    }
    
    # Get nodes for source selection
    nodes = list(network.graph.nodes())
    
    # Run multiple experiments
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # Select random sources
        sources = np.random.choice(nodes, size=NUM_SOURCES, replace=False)
        print(f"Selected sources: {sources}")
        
        # Simulate event propagation (same as in run_real_world_experiments.py)
        # ... (code for simulate_event_propagation function)
        
        # Extract features using STFT
        print("Extracting features using STFT...")
        stft = STFT(window_size=STFT_WINDOW_SIZE, overlap=STFT_OVERLAP)
        features = {}
        # ... (code for feature extraction)
        
        # Compare pathway detection methods
        print("Comparing pathway detection methods...")
        pathway_results = {}
        for method in pathway_methods:
            method_name = method.name if hasattr(method, 'name') else 'Our Method'
            print(f"  Running {method_name}...")
            
            detected_pathways = method.detect(network, features, EVENT_FREQ)
            
            # Evaluate against ground truth
            precision, recall, f1, pji = evaluate_pathway_detection(detected_pathways, true_pathways)
            results['pathway_detection'][method_name].append({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'pji': pji
            })
            
            # Save pathways for source localization
            pathway_results[method_name] = detected_pathways
        
        # Compare source localization methods
        print("Comparing source localization methods...")
        for method in source_methods:
            method_name = method.name if hasattr(method, 'name') else 'Our Method'
            print(f"  Running {method_name}...")
            
            # Use pathways from our method for all source localization methods
            detected_sources = method.localize(network, features, pathway_results['Our Method'])
            
            # Evaluate against ground truth
            sr1, sr3, sr5, mrts, ed = evaluate_source_localization(detected_sources, sources, network)
            results['source_localization'][method_name].append({
                'sr@1': sr1,
                'sr@3': sr3,
                'sr@5': sr5,
                'mrts': mrts,
                'ed': ed
            })
        
        # Compare intervention methods
        print("Comparing intervention methods...")
        # Select critical nodes
        critical_nodes = np.random.choice(nodes, size=NUM_CRITICAL_NODES, replace=False)
        
        # Calculate initial impacts
        initial_impacts = {}
        # ... (code for calculating initial impacts)
        
        for method in intervention_methods:
            method_name = method.name if hasattr(method, 'name') else 'Our Method'
            print(f"  Running {method_name}...")
            
            # Use pathways from our method for all intervention methods
            allocation = method.optimize(
                network=network,
                pathways=pathway_results['Our Method'],
                initial_impacts=initial_impacts,
                critical_nodes=critical_nodes,
                max_impact=MAX_IMPACT
            )
            
            # Evaluate intervention
            tir, cer, success = evaluate_intervention(allocation, initial_impacts, critical_nodes, MAX_IMPACT)
            results['intervention'][method_name].append({
                'tir': tir,
                'cer': cer,
                'success': success
            })
    
    return results

def main():
    """Main function to run baseline comparison on all real-world datasets."""
    print("Starting baseline comparison on real-world datasets...")
    
    # Dictionary to store results for all networks
    all_results = {}
    
    # Run comparison on each dataset
    # ... (code for running comparison on each dataset)
    
    # Save results
    save_results(all_results)
    
    # Generate comparative visualizations
    generate_comparative_visualizations(all_results)
    
    print("\nBaseline comparison completed successfully!")
    print("Results saved to results/baselines/real_world/")

if __name__ == "__main__":
    main()
