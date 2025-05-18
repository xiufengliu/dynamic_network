"""
Pathway detection algorithm implementation.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Union
from ..network.graph import DynamicNetwork
from .definition import PropagationPathway


class PathwayDetector:
    """
    A class for detecting event propagation pathways.

    Attributes:
        delay_tolerance (float): Tolerance for delay consistency.
        phase_tolerance (float): Tolerance for phase consistency.
        amplitude_threshold (float): Threshold for amplitude significance.
        max_path_length (int): Maximum path length to consider.
    """

    def __init__(self, delay_tolerance: float = 0.5, phase_tolerance: float = np.pi/4,
                 amplitude_threshold: float = 0.1, max_path_length: int = 10):
        """
        Initialize the pathway detector.

        Args:
            delay_tolerance: Tolerance for delay consistency.
            phase_tolerance: Tolerance for phase consistency.
            amplitude_threshold: Threshold for amplitude significance.
            max_path_length: Maximum path length to consider.
        """
        self.delay_tolerance = delay_tolerance
        self.phase_tolerance = phase_tolerance
        self.amplitude_threshold = amplitude_threshold
        self.max_path_length = max_path_length

    def detect(self, network: DynamicNetwork, features: Dict[str, Dict[int, Dict[str, np.ndarray]]],
               event_freq: float) -> List[PropagationPathway]:
        """
        Detect event propagation pathways.

        Args:
            network: The network.
            features: Dictionary with features extracted from time-series data.
            event_freq: Characteristic frequency of the event.

        Returns:
            List of detected pathways.
        """
        # Get active nodes
        active_nodes = self._get_active_nodes(features)

        # Get activation times
        activation_times = self._get_activation_times(features, active_nodes)

        # Get phases at activation
        activation_phases = self._get_activation_phases(features, active_nodes)

        # Get amplitudes at activation
        activation_amplitudes = self._get_activation_amplitudes(features, active_nodes)

        # Construct candidate propagation graph
        candidate_graph = self._construct_candidate_graph(
            network, active_nodes, activation_times, activation_phases, event_freq
        )

        # Find source candidates
        source_candidates = self._find_source_candidates(candidate_graph, activation_times)

        # Extract pathways using DFS
        pathways = self._extract_pathways(
            candidate_graph, source_candidates, activation_times,
            activation_phases, activation_amplitudes, event_freq
        )

        return pathways

    def _get_active_nodes(self, features: Dict[str, Dict[int, Dict[str, np.ndarray]]]) -> List[int]:
        """
        Get active nodes from features.

        Args:
            features: Dictionary with features.

        Returns:
            List of active node indices.
        """
        active_nodes = []

        for i in features['amplitude']:
            amplitude = features['amplitude'][i]
            if np.any(amplitude > self.amplitude_threshold):
                active_nodes.append(i)

        return active_nodes

    def _get_activation_times(self, features: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                              active_nodes: List[int]) -> Dict[int, float]:
        """
        Get activation times for active nodes.

        Args:
            features: Dictionary with features.
            active_nodes: List of active node indices.

        Returns:
            Dictionary mapping node indices to activation times.
        """
        activation_times = {}
        times = features['times']

        for i in active_nodes:
            amplitude = features['amplitude'][i]
            is_active = amplitude > self.amplitude_threshold

            if np.any(is_active):
                activation_idx = np.argmax(is_active)
                activation_times[i] = times[activation_idx]

        return activation_times

    def _get_activation_phases(self, features: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                               active_nodes: List[int]) -> Dict[int, float]:
        """
        Get phases at activation for active nodes.

        Args:
            features: Dictionary with features.
            active_nodes: List of active node indices.

        Returns:
            Dictionary mapping node indices to activation phases.
        """
        activation_phases = {}
        times = features['times']

        for i in active_nodes:
            if i in features['activation_time'] and features['activation_time'][i] is not None:
                activation_time = features['activation_time'][i]
                activation_idx = np.argmin(np.abs(times - activation_time))

                if i in features['phase']:
                    phase = features['phase'][i]
                    activation_phases[i] = phase[activation_idx]

        return activation_phases

    def _get_activation_amplitudes(self, features: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                                  active_nodes: List[int]) -> Dict[int, float]:
        """
        Get amplitudes at activation for active nodes.

        Args:
            features: Dictionary with features.
            active_nodes: List of active node indices.

        Returns:
            Dictionary mapping node indices to activation amplitudes.
        """
        activation_amplitudes = {}
        times = features['times']

        for i in active_nodes:
            if i in features['activation_time'] and features['activation_time'][i] is not None:
                activation_time = features['activation_time'][i]
                activation_idx = np.argmin(np.abs(times - activation_time))

                if i in features['amplitude']:
                    amplitude = features['amplitude'][i]
                    activation_amplitudes[i] = amplitude[activation_idx]

        return activation_amplitudes

    def _construct_candidate_graph(self, network: DynamicNetwork, active_nodes: List[int],
                                  activation_times: Dict[int, float], activation_phases: Dict[int, float],
                                  event_freq: float) -> nx.DiGraph:
        """
        Construct a candidate propagation graph.

        Args:
            network: The network.
            active_nodes: List of active node indices.
            activation_times: Dictionary mapping node indices to activation times.
            activation_phases: Dictionary mapping node indices to activation phases.
            event_freq: Characteristic frequency of the event.

        Returns:
            Candidate propagation graph.
        """
        candidate_graph = nx.DiGraph()

        # Add active nodes to the candidate graph
        for i in active_nodes:
            node_id = network.index_to_node(i)
            candidate_graph.add_node(node_id, index=i, activation_time=activation_times.get(i),
                                    activation_phase=activation_phases.get(i))

        # Add edges that satisfy the conditions
        for i in active_nodes:
            for j in active_nodes:
                if i != j:
                    node_i = network.index_to_node(i)
                    node_j = network.index_to_node(j)

                    # Check if there is an edge in the original network
                    if network.graph.has_edge(node_i, node_j):
                        # Check causality
                        measured_delay = activation_times[j] - activation_times[i]

                        if measured_delay > 0:
                            # Check delay consistency
                            nominal_delay = network.get_nominal_delay(node_i, node_j)
                            delay_consistent = abs(measured_delay - nominal_delay) <= self.delay_tolerance

                            # Check phase consistency if phases are available
                            phase_consistent = True
                            if i in activation_phases and j in activation_phases:
                                phase_i = activation_phases[i]
                                phase_j = activation_phases[j]
                                expected_phase_shift = 2 * np.pi * event_freq * measured_delay
                                actual_phase_diff = (phase_j - phase_i) % (2 * np.pi)
                                phase_consistent = abs(actual_phase_diff - expected_phase_shift) <= self.phase_tolerance

                            # Add edge if all conditions are met
                            if delay_consistent and phase_consistent:
                                candidate_graph.add_edge(node_i, node_j, measured_delay=measured_delay,
                                                       nominal_delay=nominal_delay)

        return candidate_graph

    def _find_source_candidates(self, candidate_graph: nx.DiGraph,
                               activation_times: Dict[int, float]) -> List[str]:
        """
        Find source candidates based on earliest activation times.

        Args:
            candidate_graph: Candidate propagation graph.
            activation_times: Dictionary mapping node indices to activation times.

        Returns:
            List of source candidate node IDs.
        """
        source_candidates = []

        # Find nodes with no incoming edges
        for node in candidate_graph.nodes():
            if candidate_graph.in_degree(node) == 0:
                source_candidates.append(node)

        # If no nodes with no incoming edges, use nodes with earliest activation times
        if not source_candidates:
            min_time = float('inf')
            min_nodes = []

            for node in candidate_graph.nodes():
                idx = candidate_graph.nodes[node]['index']
                if idx in activation_times:
                    time = activation_times[idx]
                    if time < min_time:
                        min_time = time
                        min_nodes = [node]
                    elif time == min_time:
                        min_nodes.append(node)

            source_candidates = min_nodes

        return source_candidates

    def _extract_pathways(self, candidate_graph: nx.DiGraph, source_candidates: List[str],
                         activation_times: Dict[int, float], activation_phases: Dict[int, float],
                         activation_amplitudes: Dict[int, float], event_freq: float) -> List[PropagationPathway]:
        """
        Extract pathways using DFS.

        Args:
            candidate_graph: Candidate propagation graph.
            source_candidates: List of source candidate node IDs.
            activation_times: Dictionary mapping node indices to activation times.
            activation_phases: Dictionary mapping node indices to activation phases.
            activation_amplitudes: Dictionary mapping node indices to activation amplitudes.
            event_freq: Characteristic frequency of the event.

        Returns:
            List of detected pathways.
        """
        pathways = []

        for source in source_candidates:
            # Find all paths from the source to any other node
            paths = []
            for target in candidate_graph.nodes():
                if target != source:  # Skip self-loops
                    try:
                        # Find paths from source to this target
                        for path in nx.all_simple_paths(candidate_graph, source=source, target=target,
                                                      cutoff=self.max_path_length):
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        # No path exists, continue to next target
                        continue

            # Process all found paths
            for path in paths:
                if len(path) > 1:  # Ensure the path has at least two nodes
                    pathway = PropagationPathway(path, event_freq)

                    # Add delays, phases, and amplitudes
                    for i in range(len(path)-1):
                        source_node = path[i]
                        target_node = path[i+1]

                        # Get indices
                        source_idx = candidate_graph.nodes[source_node]['index']
                        target_idx = candidate_graph.nodes[target_node]['index']

                        # Add delay
                        delay = activation_times[target_idx] - activation_times[source_idx]
                        pathway.delays.append(delay)

                        # Add phases
                        if source_idx in activation_phases:
                            pathway.phases.append(activation_phases[source_idx])
                        if i == len(path)-2 and target_idx in activation_phases:
                            pathway.phases.append(activation_phases[target_idx])

                        # Add amplitudes
                        if source_idx in activation_amplitudes:
                            pathway.amplitudes.append(activation_amplitudes[source_idx])
                        if i == len(path)-2 and target_idx in activation_amplitudes:
                            pathway.amplitudes.append(activation_amplitudes[target_idx])

                        # Add activation times
                        if source_idx in activation_times:
                            pathway.activation_times.append(activation_times[source_idx])
                        if i == len(path)-2 and target_idx in activation_times:
                            pathway.activation_times.append(activation_times[target_idx])

                    pathways.append(pathway)

        return pathways
