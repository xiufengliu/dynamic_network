"""
Formal definition of event propagation pathways.
"""

from typing import List, Tuple, Dict, Optional, Union, Set
import numpy as np


class PropagationPathway:
    """
    A class representing an event propagation pathway.
    
    Attributes:
        nodes (List): Ordered sequence of nodes in the pathway.
        event_freq (float): Characteristic frequency of the event.
        delays (List[float]): Measured propagation delays between consecutive nodes.
        phases (List[float]): Phases at each node.
    """
    
    def __init__(self, nodes: List[Union[int, str]], event_freq: float):
        """
        Initialize a propagation pathway.
        
        Args:
            nodes: Ordered sequence of nodes in the pathway.
            event_freq: Characteristic frequency of the event.
        """
        self.nodes = nodes
        self.event_freq = event_freq
        self.delays = []  # Measured propagation delays between consecutive nodes
        self.phases = []  # Phases at each node
        self.amplitudes = []  # Amplitudes at each node
        self.activation_times = []  # Activation times at each node
    
    def add_node(self, node: Union[int, str], delay: Optional[float] = None, 
                 phase: Optional[float] = None, amplitude: Optional[float] = None,
                 activation_time: Optional[float] = None) -> None:
        """
        Add a node to the pathway.
        
        Args:
            node: Node to add.
            delay: Measured propagation delay from the previous node.
            phase: Phase at the node.
            amplitude: Amplitude at the node.
            activation_time: Activation time at the node.
        """
        self.nodes.append(node)
        
        if delay is not None:
            self.delays.append(delay)
        
        if phase is not None:
            self.phases.append(phase)
        
        if amplitude is not None:
            self.amplitudes.append(amplitude)
        
        if activation_time is not None:
            self.activation_times.append(activation_time)
    
    def get_source(self) -> Union[int, str]:
        """
        Get the source node of the pathway.
        
        Returns:
            Source node.
        """
        return self.nodes[0]
    
    def get_sink(self) -> Union[int, str]:
        """
        Get the sink node of the pathway.
        
        Returns:
            Sink node.
        """
        return self.nodes[-1]
    
    def get_length(self) -> int:
        """
        Get the length of the pathway (number of nodes).
        
        Returns:
            Length of the pathway.
        """
        return len(self.nodes)
    
    def get_total_delay(self) -> float:
        """
        Get the total propagation delay along the pathway.
        
        Returns:
            Total delay.
        """
        return sum(self.delays)
    
    def get_edges(self) -> List[Tuple[Union[int, str], Union[int, str]]]:
        """
        Get the edges in the pathway.
        
        Returns:
            List of edges (source, target).
        """
        return [(self.nodes[i], self.nodes[i+1]) for i in range(len(self.nodes)-1)]
    
    def contains_node(self, node: Union[int, str]) -> bool:
        """
        Check if the pathway contains a node.
        
        Args:
            node: Node to check.
            
        Returns:
            True if the pathway contains the node, False otherwise.
        """
        return node in self.nodes
    
    def contains_edge(self, source: Union[int, str], target: Union[int, str]) -> bool:
        """
        Check if the pathway contains an edge.
        
        Args:
            source: Source node.
            target: Target node.
            
        Returns:
            True if the pathway contains the edge, False otherwise.
        """
        for i in range(len(self.nodes)-1):
            if self.nodes[i] == source and self.nodes[i+1] == target:
                return True
        return False
    
    def get_subpathway(self, start_idx: int, end_idx: int) -> 'PropagationPathway':
        """
        Get a subpathway from the current pathway.
        
        Args:
            start_idx: Start index.
            end_idx: End index.
            
        Returns:
            Subpathway.
        """
        subpathway = PropagationPathway(self.nodes[start_idx:end_idx+1], self.event_freq)
        
        if len(self.delays) > 0:
            subpathway.delays = self.delays[start_idx:end_idx]
        
        if len(self.phases) > 0:
            subpathway.phases = self.phases[start_idx:end_idx+1]
        
        if len(self.amplitudes) > 0:
            subpathway.amplitudes = self.amplitudes[start_idx:end_idx+1]
        
        if len(self.activation_times) > 0:
            subpathway.activation_times = self.activation_times[start_idx:end_idx+1]
        
        return subpathway
    
    def __str__(self) -> str:
        """
        Get a string representation of the pathway.
        
        Returns:
            String representation.
        """
        path_str = " -> ".join(str(node) for node in self.nodes)
        return f"Pathway: {path_str}"
    
    def __repr__(self) -> str:
        """
        Get a string representation of the pathway.
        
        Returns:
            String representation.
        """
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two pathways are equal.
        
        Args:
            other: Other pathway.
            
        Returns:
            True if the pathways are equal, False otherwise.
        """
        if not isinstance(other, PropagationPathway):
            return False
        
        return (self.nodes == other.nodes and 
                self.event_freq == other.event_freq and 
                self.delays == other.delays and 
                self.phases == other.phases and 
                self.amplitudes == other.amplitudes and 
                self.activation_times == other.activation_times)
    
    def __hash__(self) -> int:
        """
        Get the hash of the pathway.
        
        Returns:
            Hash value.
        """
        return hash((tuple(self.nodes), self.event_freq))
