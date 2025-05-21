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

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run baseline comparison on real-world datasets.')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Set debug level
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("Debug mode enabled")

    # Dictionary to store results for all networks
    all_results = {}

    # Run comparison on each dataset
    try:
        # Email dataset (smallest)
        if os.path.exists(EMAIL_PATH):
            print(f"\nLoading email dataset from {EMAIL_PATH}...")
            email_network = load_email_eu_core(EMAIL_PATH)
            print(f"Loaded email network with {len(email_network.get_nodes())} nodes and {len(email_network.get_edges())} edges")

            # Create a smaller subgraph for testing
            subgraph_size = min(100, len(email_network.get_nodes()))
            subgraph_nodes = list(email_network.graph.nodes())[:subgraph_size]
            email_subgraph = email_network.create_subgraph(subgraph_nodes)
            print(f"Created subgraph with {len(email_subgraph.get_nodes())} nodes and {len(email_subgraph.get_edges())} edges")

            # Run comparison
            email_results = run_baseline_comparison("email", email_subgraph, num_runs=NUM_RUNS)
            all_results["email"] = email_results
        else:
            print(f"Email dataset not found at {EMAIL_PATH}")

        # Wiki-Talk dataset
        if os.path.exists(WIKI_TALK_PATH):
            print(f"\nLoading wiki-talk dataset from {WIKI_TALK_PATH}...")
            wiki_network = load_wiki_talk(WIKI_TALK_PATH)
            print(f"Loaded wiki-talk network with {len(wiki_network.get_nodes())} nodes and {len(wiki_network.get_edges())} edges")

            # Create a smaller subgraph for testing
            subgraph_size = min(100, len(wiki_network.get_nodes()))
            subgraph_nodes = list(wiki_network.graph.nodes())[:subgraph_size]
            wiki_subgraph = wiki_network.create_subgraph(subgraph_nodes)
            print(f"Created subgraph with {len(wiki_subgraph.get_nodes())} nodes and {len(wiki_subgraph.get_edges())} edges")

            # Run comparison
            wiki_results = run_baseline_comparison("wiki", wiki_subgraph, num_runs=NUM_RUNS)
            all_results["wiki"] = wiki_results
        else:
            print(f"Wiki-Talk dataset not found at {WIKI_TALK_PATH}")

        # RoadNet-CA dataset
        if os.path.exists(ROADNET_PATH):
            print(f"\nLoading roadnet dataset from {ROADNET_PATH}...")
            roadnet_network = load_roadnet_ca(ROADNET_PATH)
            print(f"Loaded roadnet network with {len(roadnet_network.get_nodes())} nodes and {len(roadnet_network.get_edges())} edges")

            # Create a smaller subgraph for testing
            subgraph_size = min(100, len(roadnet_network.get_nodes()))
            subgraph_nodes = list(roadnet_network.graph.nodes())[:subgraph_size]
            roadnet_subgraph = roadnet_network.create_subgraph(subgraph_nodes)
            print(f"Created subgraph with {len(roadnet_subgraph.get_nodes())} nodes and {len(roadnet_subgraph.get_edges())} edges")

            # Run comparison
            roadnet_results = run_baseline_comparison("roadnet", roadnet_subgraph, num_runs=NUM_RUNS)
            all_results["roadnet"] = roadnet_results
        else:
            print(f"RoadNet-CA dataset not found at {ROADNET_PATH}")

        # Reddit dataset
        if os.path.exists(REDDIT_PATH):
            print(f"\nLoading reddit dataset from {REDDIT_PATH}...")
            reddit_network = load_reddit_hyperlinks(REDDIT_PATH)
            print(f"Loaded reddit network with {len(reddit_network.get_nodes())} nodes and {len(reddit_network.get_edges())} edges")

            # Create a smaller subgraph for testing
            subgraph_size = min(100, len(reddit_network.get_nodes()))
            subgraph_nodes = list(reddit_network.graph.nodes())[:subgraph_size]
            reddit_subgraph = reddit_network.create_subgraph(subgraph_nodes)
            print(f"Created subgraph with {len(reddit_subgraph.get_nodes())} nodes and {len(reddit_subgraph.get_edges())} edges")

            # Run comparison
            reddit_results = run_baseline_comparison("reddit", reddit_subgraph, num_runs=NUM_RUNS)
            all_results["reddit"] = reddit_results
        else:
            print(f"Reddit dataset not found at {REDDIT_PATH}")
    except Exception as e:
        import traceback
        print(f"Error running baseline comparison: {str(e)}")
        print(traceback.format_exc())

    # Save results if we have any
    if all_results:
        # Save results
        save_results(all_results)

        # Generate comparative visualizations
        generate_comparative_visualizations(all_results)

        print("\nBaseline comparison completed successfully!")
        print("Results saved to results/baselines/real_world/")
    else:
        print("\nNo results to save. Make sure at least one dataset is available.")

def save_results(results):
    """Save experiment results to files."""
    # Create directories if they don't exist
    os.makedirs('results/baselines/real_world/data', exist_ok=True)
    os.makedirs('results/baselines/real_world/figures', exist_ok=True)

    # Save as pickle for later analysis
    import pickle
    with open('results/baselines/real_world/data/baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save summary as CSV
    summary = []
    for network_name, network_results in results.items():
        row = {'network': network_name}

        # Add pathway detection metrics
        for method, values in network_results['pathway_detection'].items():
            row[f'pathway_{method}_mean'] = np.mean(values)
            row[f'pathway_{method}_std'] = np.std(values)

        # Add source localization metrics
        for method, values in network_results['source_localization'].items():
            row[f'source_{method}_mean'] = np.mean(values)
            row[f'source_{method}_std'] = np.std(values)

        # Add intervention metrics
        for method, values in network_results['intervention'].items():
            row[f'intervention_{method}_mean'] = np.mean(values)
            row[f'intervention_{method}_std'] = np.std(values)

        summary.append(row)

    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/baselines/real_world/data/baseline_summary.csv', index=False)

def generate_comparative_visualizations(results):
    """Generate comparative visualizations of baseline comparison results."""
    if not results:
        print("No results to visualize.")
        return

    networks = list(results.keys())
    if not networks:
        print("No networks in results.")
        return

    # 1. Pathway Detection Performance Comparison
    plt.figure(figsize=(14, 10))

    # Get methods
    pathway_methods = list(results[networks[0]]['pathway_detection'].keys())

    # Calculate means and standard deviations
    method_means = {}
    method_stds = {}

    for method in pathway_methods:
        method_means[method] = [np.mean(results[net]['pathway_detection'][method]) for net in networks]
        method_stds[method] = [np.std(results[net]['pathway_detection'][method]) for net in networks]

    # Plot
    x = np.arange(len(networks))
    width = 0.8 / len(pathway_methods)

    for i, method in enumerate(pathway_methods):
        offset = (i - len(pathway_methods) / 2 + 0.5) * width
        plt.bar(x + offset, method_means[method], width, yerr=method_stds[method],
                label=method, capsize=5)

    plt.xlabel('Network')
    plt.ylabel('F1 Score')
    plt.title('Pathway Detection Performance Comparison')
    plt.xticks(x, networks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/baselines/real_world/figures/pathway_detection_comparison.png', dpi=300)
    plt.close()

    # 2. Source Localization Performance Comparison
    plt.figure(figsize=(14, 10))

    # Get methods
    source_methods = list(results[networks[0]]['source_localization'].keys())

    # Calculate means and standard deviations
    method_means = {}
    method_stds = {}

    for method in source_methods:
        method_means[method] = [np.mean(results[net]['source_localization'][method]) for net in networks]
        method_stds[method] = [np.std(results[net]['source_localization'][method]) for net in networks]

    # Plot
    x = np.arange(len(networks))
    width = 0.8 / len(source_methods)

    for i, method in enumerate(source_methods):
        offset = (i - len(source_methods) / 2 + 0.5) * width
        plt.bar(x + offset, method_means[method], width, yerr=method_stds[method],
                label=method, capsize=5)

    plt.xlabel('Network')
    plt.ylabel('Success Rate')
    plt.title('Source Localization Performance Comparison')
    plt.xticks(x, networks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/baselines/real_world/figures/source_localization_comparison.png', dpi=300)
    plt.close()

    # 3. Intervention Performance Comparison
    plt.figure(figsize=(14, 10))

    # Get methods
    intervention_methods = list(results[networks[0]]['intervention'].keys())

    # Calculate means and standard deviations
    method_means = {}
    method_stds = {}

    for method in intervention_methods:
        method_means[method] = [np.mean(results[net]['intervention'][method]) for net in networks]
        method_stds[method] = [np.std(results[net]['intervention'][method]) for net in networks]

    # Plot
    x = np.arange(len(networks))
    width = 0.8 / len(intervention_methods)

    for i, method in enumerate(intervention_methods):
        offset = (i - len(intervention_methods) / 2 + 0.5) * width
        plt.bar(x + offset, method_means[method], width, yerr=method_stds[method],
                label=method, capsize=5)

    plt.xlabel('Network')
    plt.ylabel('Effectiveness Score')
    plt.title('Intervention Performance Comparison')
    plt.xticks(x, networks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/baselines/real_world/figures/intervention_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
