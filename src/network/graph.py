"""
Graph representation for dynamic networks.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set


class DynamicNetwork:
    """
    A class representing a dynamic network as a weighted directed graph G = (V, E, W).

    Attributes:
        graph (nx.DiGraph): The underlying directed graph.
        node_mapping (Dict): Mapping between node IDs and indices.
    """

    def __init__(self):
        """Initialize an empty dynamic network."""
        self.graph = nx.DiGraph()
        self.node_mapping = {}  # Maps node IDs to indices
        self.reverse_mapping = {}  # Maps indices to node IDs

    def add_node(self, node_id: Union[int, str], **attr) -> None:
        """
        Add a node to the network.

        Args:
            node_id: The ID of the node.
            **attr: Additional node attributes.
        """
        self.graph.add_node(node_id, **attr)
        if node_id not in self.node_mapping:
            idx = len(self.node_mapping)
            self.node_mapping[node_id] = idx
            self.reverse_mapping[idx] = node_id

    def add_edge(self, source: Union[int, str], target: Union[int, str],
                 weight: float = 1.0, **attr) -> None:
        """
        Add a directed edge to the network.

        Args:
            source: The source node ID.
            target: The target node ID.
            weight: The edge weight (propagation delay).
            **attr: Additional edge attributes.
        """
        # Ensure nodes exist
        if source not in self.graph:
            self.add_node(source)
        if target not in self.graph:
            self.add_node(target)

        # Add the edge with the weight as the nominal propagation delay
        self.graph.add_edge(source, target, weight=weight, nominal_delay=weight, **attr)

    def get_nodes(self) -> List[Union[int, str]]:
        """
        Get all nodes in the network.

        Returns:
            List of node IDs.
        """
        return list(self.graph.nodes())

    def get_edges(self) -> List[Tuple[Union[int, str], Union[int, str], Dict]]:
        """
        Get all edges in the network.

        Returns:
            List of tuples (source, target, attributes).
        """
        return [(u, v, self.graph[u][v]) for u, v in self.graph.edges()]

    def get_neighbors(self, node_id: Union[int, str]) -> List[Union[int, str]]:
        """
        Get the neighbors of a node.

        Args:
            node_id: The ID of the node.

        Returns:
            List of neighbor node IDs.
        """
        return list(self.graph.successors(node_id))

    def get_predecessors(self, node_id: Union[int, str]) -> List[Union[int, str]]:
        """
        Get the predecessors of a node.

        Args:
            node_id: The ID of the node.

        Returns:
            List of predecessor node IDs.
        """
        return list(self.graph.predecessors(node_id))

    def get_edge_weight(self, source: Union[int, str], target: Union[int, str]) -> float:
        """
        Get the weight of an edge.

        Args:
            source: The source node ID.
            target: The target node ID.

        Returns:
            The edge weight.
        """
        return self.graph[source][target]['weight']

    def get_nominal_delay(self, source: Union[int, str], target: Union[int, str]) -> float:
        """
        Get the nominal propagation delay of an edge.

        Args:
            source: The source node ID.
            target: The target node ID.

        Returns:
            The nominal propagation delay.
        """
        return self.graph[source][target]['nominal_delay']

    def node_to_index(self, node_id: Union[int, str]) -> int:
        """
        Convert a node ID to its index.

        Args:
            node_id: The ID of the node.

        Returns:
            The index of the node.
        """
        return self.node_mapping[node_id]

    def index_to_node(self, index: int) -> Union[int, str]:
        """
        Convert a node index to its ID.

        Args:
            index: The index of the node.

        Returns:
            The ID of the node.
        """
        return self.reverse_mapping[index]

    def save_to_file(self, filename: str) -> None:
        """
        Save the network to a file.

        Args:
            filename: The name of the file.
        """
        nx.write_graphml(self.graph, filename)

    def load_from_file(self, filename: str) -> None:
        """
        Load the network from a file.

        Args:
            filename: The name of the file.
        """
        self.graph = nx.read_graphml(filename)

        # Convert node IDs back to their original types if possible
        # GraphML stores everything as strings, so we need to convert back
        old_graph = self.graph
        self.graph = nx.DiGraph()

        # Convert node IDs and add nodes
        for node_id in old_graph.nodes():
            # Try to convert string to int if it's a number
            try:
                if node_id.isdigit():
                    converted_id = int(node_id)
                else:
                    converted_id = node_id
            except (AttributeError, ValueError):
                converted_id = node_id

            # Add node with attributes
            self.graph.add_node(converted_id, **old_graph.nodes[node_id])

        # Add edges with converted IDs
        for u, v, data in old_graph.edges(data=True):
            # Convert source and target IDs
            try:
                if u.isdigit():
                    u_converted = int(u)
                else:
                    u_converted = u
            except (AttributeError, ValueError):
                u_converted = u

            try:
                if v.isdigit():
                    v_converted = int(v)
                else:
                    v_converted = v
            except (AttributeError, ValueError):
                v_converted = v

            # Add edge with attributes
            self.graph.add_edge(u_converted, v_converted, **data)

        # Rebuild the node mapping
        self.node_mapping = {}
        self.reverse_mapping = {}
        for i, node_id in enumerate(self.graph.nodes()):
            self.node_mapping[node_id] = i
            self.reverse_mapping[i] = node_id

        # Ensure all edges have a nominal_delay attribute
        for u, v, data in self.graph.edges(data=True):
            if 'weight' not in data:
                data['weight'] = 1.0
            if 'nominal_delay' not in data:
                data['nominal_delay'] = data['weight']

    def create_subgraph(self, nodes: List[Union[int, str]]) -> 'DynamicNetwork':
        """
        Create a subgraph containing only the specified nodes.

        Args:
            nodes: List of node IDs to include in the subgraph.

        Returns:
            A new DynamicNetwork instance representing the subgraph.
        """
        subgraph = DynamicNetwork()

        # Add nodes
        for node in nodes:
            if node in self.graph:
                attrs = self.graph.nodes[node].copy()
                subgraph.add_node(node, **attrs)

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            if u in nodes and v in nodes:
                # Create a copy of the data to avoid modifying the original
                edge_data = data.copy()

                # Remove weight and nominal_delay to avoid duplicates
                weight = edge_data.pop('weight', 1.0)
                if 'nominal_delay' in edge_data:
                    del edge_data['nominal_delay']

                subgraph.add_edge(u, v, weight=weight, **edge_data)

        return subgraph

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the network.

        Returns:
            A numpy array representing the adjacency matrix.
        """
        return nx.to_numpy_array(self.graph, nodelist=list(self.reverse_mapping.values()))

    def get_delay_matrix(self) -> np.ndarray:
        """
        Get the delay matrix of the network.

        Returns:
            A numpy array representing the delay matrix.
        """
        adj_matrix = self.get_adjacency_matrix()
        delay_matrix = np.zeros_like(adj_matrix)

        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] > 0:
                    u = self.index_to_node(i)
                    v = self.index_to_node(j)
                    delay_matrix[i, j] = self.get_nominal_delay(u, v)

        return delay_matrix

    def __len__(self) -> int:
        """
        Get the number of nodes in the network.

        Returns:
            The number of nodes.
        """
        return len(self.graph)
