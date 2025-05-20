import unittest
import numpy as np
import networkx as nx
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from network.graph import DynamicNetwork
from pathway_detection.detector import PathwayDetector
from pathway_detection.definition import PropagationPathway

class TestPathwayDetector(unittest.TestCase):
    def setUp(self):
        """Set up common test resources."""
        self.detector = PathwayDetector(
            delay_tolerance=0.5,
            phase_tolerance=np.pi / 4,
            amplitude_threshold=0.1,
            max_path_length=5
        )

        # Create a simple DynamicNetwork
        self.network = DynamicNetwork()
        self.network.add_node(0)
        self.network.add_node(1)
        self.network.add_node(2)
        self.network.add_node(3)

        # Add edges with weights (nominal delays)
        # Note: DynamicNetwork uses node IDs directly, not necessarily 0-indexed if created differently.
        # Here, node 0 is index 0, node 1 is index 1, etc. because we add them in order.
        self.network.add_edge(0, 1, weight=1.0) # Nominal delay 0 -> 1 is 1.0
        self.network.add_edge(1, 2, weight=1.0) # Nominal delay 1 -> 2 is 1.0
        self.network.add_edge(0, 2, weight=2.0) # Nominal delay 0 -> 2 is 2.0
        # Add a reverse edge to test path directionality
        self.network.add_edge(2, 0, weight=0.5)


    def test_detect_no_active_nodes(self):
        """Test pathway detection when no nodes are active."""
        features = {
            'amplitude': {
                0: np.array([0.05, 0.05]), 1: np.array([0.05, 0.05]),
                2: np.array([0.05, 0.05]), 3: np.array([0.05, 0.05])
            },
            'times': np.array([0.0, 1.0]),
            'phase': {
                0: np.array([0.0, 0.0]), 1: np.array([0.0, 0.0]),
                2: np.array([0.0, 0.0]), 3: np.array([0.0, 0.0])
            }
            # 'activation_time' is not needed here as it's derived internally
        }
        pathways = self.detector.detect(self.network, features, event_freq=0.1)
        self.assertEqual(len(pathways), 0)

    def test_detect_no_pathways_found(self):
        """Test pathway detection where nodes are active but no valid pathways form."""
        # Scenario 1: Delay mismatch
        # Edge 0->1 has nominal_delay 1.0. Detector delay_tolerance is 0.5
        # Activation times: Node 0 at 0.0, Node 1 at 1.8. Measured delay = 1.8.
        # |1.8 - 1.0| = 0.8, which is > 0.5 (delay_tolerance)
        features_delay_mismatch = {
            'amplitude': {
                0: np.array([0.2, 0.05]),  # Node 0 active at time 0.0
                1: np.array([0.05, 0.2]),  # Node 1 active at time 1.0 (but times array is [0.0, 1.8])
                                         # This needs to be fixed. times array should reflect actual time points.
                2: np.array([0.05, 0.05]),
                3: np.array([0.05, 0.05])
            },
            'times': np.array([0.0, 1.8]), # Times array for indexing
             # 'activation_time' will be derived: {0: 0.0, 1: 1.8}
            'phase': { # Phases don't matter much if delay is inconsistent
                0: np.array([0.0, 0.0]),
                1: np.array([0.0, 0.0])
            }
        }
        # Correcting features for activation times based on 'amplitude' and 'times'
        # Node 0 active at features['times'][0] = 0.0
        # Node 1 active at features['times'][1] = 1.8
        pathways_delay = self.detector.detect(self.network, features_delay_mismatch, event_freq=0.1)
        self.assertEqual(len(pathways_delay), 0, "Failed: Delay mismatch should result in no pathways.")

        # Scenario 2: Phase mismatch
        # Edge 0->1, nominal_delay 1.0.
        # Node 0 active at 0.0, Node 1 active at 1.0. Measured delay = 1.0.
        # |1.0 - 1.0| = 0.0 <= 0.5 (delay_tolerance) -> delay is consistent.
        # Phase: Node 0 phase at 0.0 is 0.1. Node 1 phase at 1.0 is np.pi.
        # Event freq = 0.1. Expected phase shift = 2 * np.pi * event_freq * measured_delay
        # = 2 * np.pi * 0.1 * 1.0 = 0.2 * np.pi.
        # Actual phase diff = (np.pi - 0.1) % (2 * np.pi) approx (3.14159 - 0.1) = 3.04159
        # Phase tolerance is np.pi/4 = 0.785
        # |3.04159 - 0.2 * np.pi| = |3.04159 - 0.6283| = 2.413 > 0.785 -> phase is inconsistent.
        features_phase_mismatch = {
            'times': np.array([0.0, 1.0, 2.0]), # Times array
            'amplitude': {
                0: np.array([0.2, 0.05, 0.05]), # Node 0 active at t=0.0
                1: np.array([0.05, 0.2, 0.05]), # Node 1 active at t=1.0
                2: np.array([0.05, 0.05, 0.05]),
                3: np.array([0.05, 0.05, 0.05])
            },
            # 'activation_time' will be {0: 0.0, 1: 1.0}
            'phase': {
                0: np.array([0.1, 0.1, 0.1]), # Phase for node 0 at t=0.0 is 0.1
                1: np.array([0.0, np.pi, 0.0]) # Phase for node 1 at t=1.0 is np.pi
            }
        }
        pathways_phase = self.detector.detect(self.network, features_phase_mismatch, event_freq=0.1)
        self.assertEqual(len(pathways_phase), 0, "Failed: Phase mismatch should result in no pathways.")


    def test_detect_simple_pathway(self):
        """Test detection of a simple valid pathway (e.g., 0 -> 1)."""
        # Edge 0->1, nominal_delay 1.0. Detector delay_tolerance 0.5.
        # Node 0 active at t=0.0, Node 1 active at t=1.0.
        # Measured_delay = 1.0. |1.0 - 1.0| = 0.0 <= 0.5. Delay is consistent.
        # Phases: Node 0 at t=0.0 has phase 0.1.
        # Node 1 at t=1.0 has phase (0.1 + 2 * np.pi * 0.1 * 1.0) % (2*np.pi)
        # event_freq = 0.1, measured_delay = 1.0
        # Expected phase shift = 2 * np.pi * 0.1 * 1.0 = 0.2 * np.pi.
        # Phase for node 1 at t=1.0 should be around 0.1 + 0.2 * np.pi.
        # Phase tolerance is np.pi/4 = 0.785.
        
        event_freq = 0.1
        delay_0_1 = 1.0 # This is also the nominal delay for edge 0->1
        
        phase_node0_t0 = 0.1
        # Expected phase for node 1 at t=1.0 for consistency
        phase_node1_t1 = (phase_node0_t0 + 2 * np.pi * event_freq * delay_0_1) % (2 * np.pi)

        features = {
            'times': np.array([0.0, 1.0, 2.0, 3.0]),
            'amplitude': {
                0: np.array([0.2, 0.05, 0.05, 0.05]),  # Node 0 active at t=0.0
                1: np.array([0.05, 0.2, 0.05, 0.05]),  # Node 1 active at t=1.0
                2: np.array([0.05, 0.05, 0.05, 0.05]), # Node 2 not significantly active or active later
                3: np.array([0.05, 0.05, 0.05, 0.05])
            },
            # 'activation_time' derived: {0: 0.0, 1: 1.0}
            'phase': {
                0: np.array([phase_node0_t0, 0.0, 0.0, 0.0]), # Phase for node 0 at t=0.0
                1: np.array([0.0, phase_node1_t1, 0.0, 0.0]), # Phase for node 1 at t=1.0
                2: np.array([0.0, 0.0, 0.0, 0.0]),
                3: np.array([0.0, 0.0, 0.0, 0.0])
            }
        }

        pathways = self.detector.detect(self.network, features, event_freq=event_freq)
        
        self.assertEqual(len(pathways), 1, f"Expected 1 pathway, got {len(pathways)}")
        
        # Node IDs in DynamicNetwork are the ones given (0, 1, 2, 3)
        # PathwayDetector._construct_candidate_graph uses network.index_to_node(idx)
        # but if nodes are 0,1,2,3 and added in order, index_to_node(i) = i.
        # Let's assume node IDs are directly used for pathways for simplicity if consistent.
        # The Pathway object stores node IDs.
        
        # Check if the nodes in the pathway are [0, 1]
        # The current implementation of PathwayDetector returns node objects from the original graph.
        # If nodes are simple integers, it's fine. If they are complex objects, comparison needs care.
        # Given self.network.add_node(0), etc., the node itself is the integer.
        self.assertEqual(pathways[0].nodes, [0, 1])

        # Optional: Check delay
        self.assertAlmostEqual(pathways[0].delays[0], delay_0_1, places=5)
        
        # Optional: Check phases (first node's phase, second node's phase)
        # Activation phases are stored for each node in the pathway.
        # For path 0->1, pathway.phases should contain [phase_node0_t0, phase_node1_t1]
        self.assertAlmostEqual(pathways[0].phases[0], phase_node0_t0, places=5)
        self.assertAlmostEqual(pathways[0].phases[1], phase_node1_t1, places=5)
        
        # Optional: Check amplitudes
        # Amplitudes at activation: node 0 at t=0 (0.2), node 1 at t=1 (0.2)
        self.assertAlmostEqual(pathways[0].amplitudes[0], 0.2, places=5)
        self.assertAlmostEqual(pathways[0].amplitudes[1], 0.2, places=5)

        # Optional: Check activation times
        self.assertAlmostEqual(pathways[0].activation_times[0], 0.0, places=5)
        self.assertAlmostEqual(pathways[0].activation_times[1], 1.0, places=5)

    def test_detect_pathway_with_intermediate_node(self):
        """Test detection of a pathway 0 -> 1 -> 2."""
        self.detector = PathwayDetector(
            delay_tolerance=0.5,
            phase_tolerance=np.pi / 4,
            amplitude_threshold=0.1,
            max_path_length=5
        )
        # Network: 0 --(1.0)--> 1 --(1.0)--> 2
        
        event_freq = 0.1
        delay_0_1 = 1.0 # Nominal delay for 0->1
        delay_1_2 = 1.0 # Nominal delay for 1->2

        # Activation times
        act_time_0 = 0.0
        act_time_1 = act_time_0 + delay_0_1 # 1.0
        act_time_2 = act_time_1 + delay_1_2 # 2.0

        # Phases
        phase_0_t0 = 0.1
        phase_1_t1 = (phase_0_t0 + 2 * np.pi * event_freq * delay_0_1) % (2 * np.pi)
        phase_2_t2 = (phase_1_t1 + 2 * np.pi * event_freq * delay_1_2) % (2 * np.pi)

        features = {
            'times': np.array([0.0, 1.0, 2.0, 3.0]),
            'amplitude': {
                0: np.array([0.2, 0.05, 0.05, 0.05]),  # Node 0 active at t=0.0
                1: np.array([0.05, 0.2, 0.05, 0.05]),  # Node 1 active at t=1.0
                2: np.array([0.05, 0.05, 0.2, 0.05]),  # Node 2 active at t=2.0
                3: np.array([0.05, 0.05, 0.05, 0.05])
            },
            'phase': {
                0: np.array([phase_0_t0, 0.0, 0.0, 0.0]),
                1: np.array([0.0, phase_1_t1, 0.0, 0.0]),
                2: np.array([0.0, 0.0, phase_2_t2, 0.0]),
                3: np.array([0.0, 0.0, 0.0, 0.0])
            }
        }

        pathways = self.detector.detect(self.network, features, event_freq=event_freq)
        
        self.assertEqual(len(pathways), 1, f"Expected 1 pathway (0->1->2), got {len(pathways)}")
        
        detected_path = pathways[0]
        self.assertEqual(detected_path.nodes, [0, 1, 2])

        # Check delays in the pathway object: [delay_0_1, delay_1_2]
        self.assertEqual(len(detected_path.delays), 2)
        self.assertAlmostEqual(detected_path.delays[0], delay_0_1, places=5)
        self.assertAlmostEqual(detected_path.delays[1], delay_1_2, places=5)

        # Check phases: [phase_0_t0, phase_1_t1, phase_2_t2]
        self.assertEqual(len(detected_path.phases), 3)
        self.assertAlmostEqual(detected_path.phases[0], phase_0_t0, places=5)
        self.assertAlmostEqual(detected_path.phases[1], phase_1_t1, places=5)
        self.assertAlmostEqual(detected_path.phases[2], phase_2_t2, places=5)

        # Check amplitudes: [amp_0_t0, amp_1_t1, amp_2_t2]
        self.assertEqual(len(detected_path.amplitudes), 3)
        self.assertAlmostEqual(detected_path.amplitudes[0], 0.2, places=5) # From features['amplitude'][0]
        self.assertAlmostEqual(detected_path.amplitudes[1], 0.2, places=5) # From features['amplitude'][1]
        self.assertAlmostEqual(detected_path.amplitudes[2], 0.2, places=5) # From features['amplitude'][2]

        # Check activation times: [act_time_0, act_time_1, act_time_2]
        self.assertEqual(len(detected_path.activation_times), 3)
        self.assertAlmostEqual(detected_path.activation_times[0], act_time_0, places=5)
        self.assertAlmostEqual(detected_path.activation_times[1], act_time_1, places=5)
        self.assertAlmostEqual(detected_path.activation_times[2], act_time_2, places=5)

    def test_detect_multiple_pathways(self):
        """Test detection of multiple pathways, including branching."""
        # Network: 0 --(1.0)--> 1 --(1.0)--> 2
        #          0 --(2.0)--> 2 (direct path)
        # Expect two pathways: 0->1->2 and 0->2
        
        event_freq = 0.1
        # Pathway 0->1->2
        delay_0_1 = 1.0
        delay_1_2 = 1.0
        act_time_0 = 0.0
        act_time_1 = act_time_0 + delay_0_1 # 1.0
        act_time_2_path1 = act_time_1 + delay_1_2 # 2.0
        
        phase_0_t0 = 0.1
        phase_1_t1 = (phase_0_t0 + 2 * np.pi * event_freq * delay_0_1) % (2 * np.pi)
        phase_2_t2_path1 = (phase_1_t1 + 2 * np.pi * event_freq * delay_1_2) % (2 * np.pi)

        # Pathway 0->2 (direct)
        delay_0_2_direct = 2.0 # This is the nominal delay of edge 0->2
        # Node 2's activation for this path should be consistent with this delay from node 0
        # For simplicity, we assume node 2's activation properties (time, phase) are primarily
        # determined by one path if multiple paths converge, or the test setup ensures
        # it's compatible with both if we want both detected.
        # The current detector logic might create two versions of node 2 if activation times differ
        # or just one if they are the same. Here, act_time_2_path1 (2.0) matches delay_0_2_direct from t=0.
        phase_2_t2_direct = (phase_0_t0 + 2 * np.pi * event_freq * delay_0_2_direct) % (2*np.pi)

        # We need to ensure that phase_2_t2_path1 is very close to phase_2_t2_direct
        # phase_2_t2_path1 = ( (phase_0_t0 + 2*pi*f*d01) + 2*pi*f*d12 ) % (2*pi)
        #                  = ( phase_0_t0 + 2*pi*f*(d01+d12) ) % (2*pi)
        # phase_2_t2_direct = ( phase_0_t0 + 2*pi*f*d02_direct ) % (2*pi)
        # If d01+d12 = d02_direct (i.e. 1.0+1.0 = 2.0), then the phases will be consistent.

        features = {
            'times': np.array([0.0, 1.0, 2.0, 3.0]),
            'amplitude': {
                0: np.array([0.2, 0.05, 0.05, 0.05]),  # Node 0 active at t=0.0
                1: np.array([0.05, 0.2, 0.05, 0.05]),  # Node 1 active at t=1.0
                2: np.array([0.05, 0.05, 0.2, 0.05]),  # Node 2 active at t=2.0
                3: np.array([0.05, 0.05, 0.05, 0.05])
            },
            'phase': { # Phases set up for node 2 to be consistent with both paths
                0: np.array([phase_0_t0, 0.0, 0.0, 0.0]),
                1: np.array([0.0, phase_1_t1, 0.0, 0.0]),
                2: np.array([0.0, 0.0, phase_2_t2_path1, 0.0]), # or phase_2_t2_direct, should be same
                3: np.array([0.0, 0.0, 0.0, 0.0])
            }
        }

        pathways = self.detector.detect(self.network, features, event_freq=event_freq)
        
        self.assertEqual(len(pathways), 2, f"Expected 2 pathways, got {len(pathways)}")

        path_repr = sorted([p.nodes for p in pathways]) # Sort to make assertion order-independent
        expected_paths = sorted([[0, 1, 2], [0, 2]])
        self.assertEqual(path_repr, expected_paths)

    def test_max_path_length(self):
        """Test that pathways longer than max_path_length are not found."""
        # Setup: 0 -> 1 -> 2 -> 3. max_path_length is 2 (i.e., 3 nodes).
        # Path 0->1->2 should be found. Path 0->1->2->3 should NOT if max_path_length=2 for detector.
        # max_path_length in detector is 5 by default setup. Let's set it to 2 for this test.
        # A path length is number of edges. So path 0-1-2 has length 2.
        # nx.all_simple_paths cutoff is number of nodes. cutoff=N means paths up to N nodes.
        # If max_path_length = 2 (edges), this means cutoff = 3 (nodes).
        
        self.detector.max_path_length = 2 # Max 2 edges, i.e., 3 nodes.
        
        self.network = DynamicNetwork() # New network for this specific test
        self.network.add_nodes_from([0,1,2,3])
        self.network.add_edge(0,1,weight=1.0)
        self.network.add_edge(1,2,weight=1.0)
        self.network.add_edge(2,3,weight=1.0)

        event_freq = 0.1
        delay = 1.0
        
        act_time_0 = 0.0
        act_time_1 = act_time_0 + delay # 1.0
        act_time_2 = act_time_1 + delay # 2.0
        act_time_3 = act_time_2 + delay # 3.0

        phase_0 = 0.1
        phase_1 = (phase_0 + 2 * np.pi * event_freq * delay) % (2*np.pi)
        phase_2 = (phase_1 + 2 * np.pi * event_freq * delay) % (2*np.pi)
        phase_3 = (phase_2 + 2 * np.pi * event_freq * delay) % (2*np.pi)

        features = {
            'times': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            'amplitude': {
                0: np.array([0.2, 0.05, 0.05, 0.05, 0.05]),
                1: np.array([0.05, 0.2, 0.05, 0.05, 0.05]),
                2: np.array([0.05, 0.05, 0.2, 0.05, 0.05]),
                3: np.array([0.05, 0.05, 0.05, 0.2, 0.05]),
            },
            'phase': {
                0: np.array([phase_0,0,0,0,0]),
                1: np.array([0,phase_1,0,0,0]),
                2: np.array([0,0,phase_2,0,0]),
                3: np.array([0,0,0,phase_3,0]),
            }
        }
        pathways = self.detector.detect(self.network, features, event_freq=event_freq)
        
        # Expected: Path 0->1->2 (length 2 edges, 3 nodes) should be found.
        # Path 0->1 (length 1 edge, 2 nodes) should be found.
        # Path 1->2 (length 1 edge, 2 nodes) should be found.
        # Path 2->3 (length 1 edge, 2 nodes) should be found.
        # Path 1->2->3 (length 2 edges, 3 nodes) should be found.
        # Path 0->1->2->3 (length 3 edges, 4 nodes) should NOT be found.
        
        # The detector's _extract_pathways iterates through sources, then targets, then nx.all_simple_paths.
        # It will find [0,1], [0,1,2], [1,2], [1,2,3], [2,3]
        # It should not find [0,1,2,3] because its length (3 edges) > self.detector.max_path_length (2 edges).
        
        found_path_nodes = sorted([p.nodes for p in pathways])
        expected_paths_found = sorted([
            [0,1], [1,2], [2,3], # length 1 paths
            [0,1,2], [1,2,3]    # length 2 paths
        ])
        
        self.assertEqual(len(found_path_nodes), len(expected_paths_found), 
                         f"Expected {len(expected_paths_found)} pathways, got {len(found_path_nodes)}. Paths: {found_path_nodes}")
        self.assertEqual(found_path_nodes, expected_paths_found)


if __name__ == '__main__':
    unittest.main()
