"""
Tests for the network module.
"""

import unittest
import numpy as np
import networkx as nx
import tempfile
import os

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network.graph import DynamicNetwork
from src.network.generators import (
    generate_barabasi_albert_network,
    generate_erdos_renyi_network,
    generate_watts_strogatz_network,
    generate_grid_network
)


class TestDynamicNetwork(unittest.TestCase):
    """Test cases for the DynamicNetwork class."""

    def setUp(self):
        """Set up test fixtures."""
        self.network = DynamicNetwork()

        # Add nodes
        self.network.add_node(1)
        self.network.add_node(2)
        self.network.add_node(3)

        # Add edges
        self.network.add_edge(1, 2, weight=2.0)
        self.network.add_edge(2, 3, weight=3.0)

    def test_add_node(self):
        """Test adding a node."""
        self.network.add_node(4)
        self.assertIn(4, self.network.graph)
        self.assertEqual(self.network.node_to_index(4), 3)

    def test_add_edge(self):
        """Test adding an edge."""
        self.network.add_edge(1, 3, weight=4.0)
        self.assertTrue(self.network.graph.has_edge(1, 3))
        self.assertEqual(self.network.get_edge_weight(1, 3), 4.0)

    def test_get_nodes(self):
        """Test getting nodes."""
        nodes = self.network.get_nodes()
        self.assertEqual(set(nodes), {1, 2, 3})

    def test_get_edges(self):
        """Test getting edges."""
        edges = self.network.get_edges()
        self.assertEqual(len(edges), 2)
        self.assertIn((1, 2, {'weight': 2.0, 'nominal_delay': 2.0}), edges)
        self.assertIn((2, 3, {'weight': 3.0, 'nominal_delay': 3.0}), edges)

    def test_get_neighbors(self):
        """Test getting neighbors."""
        neighbors = self.network.get_neighbors(1)
        self.assertEqual(neighbors, [2])

    def test_get_predecessors(self):
        """Test getting predecessors."""
        predecessors = self.network.get_predecessors(3)
        self.assertEqual(predecessors, [2])

    def test_get_edge_weight(self):
        """Test getting edge weight."""
        weight = self.network.get_edge_weight(1, 2)
        self.assertEqual(weight, 2.0)

    def test_get_nominal_delay(self):
        """Test getting nominal delay."""
        delay = self.network.get_nominal_delay(2, 3)
        self.assertEqual(delay, 3.0)

    def test_node_to_index(self):
        """Test converting node to index."""
        index = self.network.node_to_index(2)
        self.assertEqual(index, 1)

    def test_index_to_node(self):
        """Test converting index to node."""
        node = self.network.index_to_node(0)
        self.assertEqual(node, 1)

    def test_save_load(self):
        """Test saving and loading a network."""
        with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as tmp:
            filename = tmp.name

        try:
            # Save the network
            self.network.save_to_file(filename)

            # Load the network
            loaded_network = DynamicNetwork()
            loaded_network.load_from_file(filename)

            # Check if the loaded network is the same
            self.assertEqual(set(loaded_network.get_nodes()), set(self.network.get_nodes()))
            self.assertEqual(len(loaded_network.get_edges()), len(self.network.get_edges()))
        finally:
            # Clean up
            os.unlink(filename)

    def test_create_subgraph(self):
        """Test creating a subgraph."""
        subgraph = self.network.create_subgraph([1, 2])
        self.assertEqual(set(subgraph.get_nodes()), {1, 2})
        self.assertEqual(len(subgraph.get_edges()), 1)

    def test_get_adjacency_matrix(self):
        """Test getting adjacency matrix."""
        adj_matrix = self.network.get_adjacency_matrix()
        self.assertEqual(adj_matrix.shape, (3, 3))
        self.assertEqual(adj_matrix[0, 1], 2.0)
        self.assertEqual(adj_matrix[1, 2], 3.0)

    def test_get_delay_matrix(self):
        """Test getting delay matrix."""
        delay_matrix = self.network.get_delay_matrix()
        self.assertEqual(delay_matrix.shape, (3, 3))
        self.assertEqual(delay_matrix[0, 1], 2.0)
        self.assertEqual(delay_matrix[1, 2], 3.0)

    def test_len(self):
        """Test getting the number of nodes."""
        self.assertEqual(len(self.network), 3)


class TestNetworkGenerators(unittest.TestCase):
    """Test cases for the network generators."""

    def test_generate_barabasi_albert_network(self):
        """Test generating a Barabási-Albert network."""
        network = generate_barabasi_albert_network(n=100, m=2, seed=42)
        self.assertEqual(len(network), 100)
        self.assertGreater(len(network.get_edges()), 0)

    def test_generate_erdos_renyi_network(self):
        """Test generating an Erdős-Rényi network."""
        network = generate_erdos_renyi_network(n=100, p=0.05, seed=42)
        self.assertEqual(len(network), 100)
        self.assertGreater(len(network.get_edges()), 0)

    def test_generate_watts_strogatz_network(self):
        """Test generating a Watts-Strogatz network."""
        network = generate_watts_strogatz_network(n=100, k=4, p=0.1, seed=42)
        self.assertEqual(len(network), 100)
        self.assertGreater(len(network.get_edges()), 0)

    def test_generate_grid_network(self):
        """Test generating a grid network."""
        network = generate_grid_network(n=10, m=10, seed=42)
        self.assertEqual(len(network), 100)
        self.assertGreater(len(network.get_edges()), 0)


if __name__ == '__main__':
    unittest.main()
