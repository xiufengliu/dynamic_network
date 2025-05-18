"""
Pathway validation and consistency checks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ..network.graph import DynamicNetwork
from .definition import PropagationPathway


def check_link_existence(pathway: PropagationPathway, network: DynamicNetwork) -> bool:
    """
    Check if all links in the pathway exist in the network.
    
    Args:
        pathway: The pathway to check.
        network: The network.
        
    Returns:
        True if all links exist, False otherwise.
    """
    for i in range(len(pathway.nodes) - 1):
        source = pathway.nodes[i]
        target = pathway.nodes[i + 1]
        
        if not network.graph.has_edge(source, target):
            return False
    
    return True


def check_causality(pathway: PropagationPathway) -> bool:
    """
    Check if the pathway satisfies causality (positive delays).
    
    Args:
        pathway: The pathway to check.
        
    Returns:
        True if causality is satisfied, False otherwise.
    """
    return all(delay > 0 for delay in pathway.delays)


def check_delay_consistency(pathway: PropagationPathway, network: DynamicNetwork, 
                           tolerance: float = 0.5) -> bool:
    """
    Check if the pathway satisfies delay consistency.
    
    Args:
        pathway: The pathway to check.
        network: The network.
        tolerance: Tolerance for delay consistency.
        
    Returns:
        True if delay consistency is satisfied, False otherwise.
    """
    for i in range(len(pathway.nodes) - 1):
        source = pathway.nodes[i]
        target = pathway.nodes[i + 1]
        measured_delay = pathway.delays[i]
        nominal_delay = network.get_nominal_delay(source, target)
        
        if abs(measured_delay - nominal_delay) > tolerance:
            return False
    
    return True


def check_phase_consistency(pathway: PropagationPathway, tolerance: float = np.pi/4) -> bool:
    """
    Check if the pathway satisfies phase consistency.
    
    Args:
        pathway: The pathway to check.
        tolerance: Tolerance for phase consistency.
        
    Returns:
        True if phase consistency is satisfied, False otherwise.
    """
    if len(pathway.phases) < 2 or len(pathway.delays) < 1:
        return True  # Not enough information to check
    
    for i in range(len(pathway.nodes) - 1):
        if i < len(pathway.phases) - 1 and i < len(pathway.delays):
            source_phase = pathway.phases[i]
            target_phase = pathway.phases[i + 1]
            delay = pathway.delays[i]
            
            expected_shift = 2 * np.pi * pathway.event_freq * delay
            actual_diff = (target_phase - source_phase) % (2 * np.pi)
            
            if abs(actual_diff - expected_shift) > tolerance:
                return False
    
    return True


def validate_pathway(pathway: PropagationPathway, network: DynamicNetwork, 
                    delay_tolerance: float = 0.5, phase_tolerance: float = np.pi/4) -> bool:
    """
    Validate a pathway against all criteria.
    
    Args:
        pathway: The pathway to validate.
        network: The network.
        delay_tolerance: Tolerance for delay consistency.
        phase_tolerance: Tolerance for phase consistency.
        
    Returns:
        True if the pathway is valid, False otherwise.
    """
    # Check link existence
    if not check_link_existence(pathway, network):
        return False
    
    # Check causality
    if not check_causality(pathway):
        return False
    
    # Check delay consistency
    if not check_delay_consistency(pathway, network, delay_tolerance):
        return False
    
    # Check phase consistency
    if not check_phase_consistency(pathway, phase_tolerance):
        return False
    
    return True


def filter_valid_pathways(pathways: List[PropagationPathway], network: DynamicNetwork,
                         delay_tolerance: float = 0.5, phase_tolerance: float = np.pi/4) -> List[PropagationPathway]:
    """
    Filter out invalid pathways.
    
    Args:
        pathways: List of pathways to filter.
        network: The network.
        delay_tolerance: Tolerance for delay consistency.
        phase_tolerance: Tolerance for phase consistency.
        
    Returns:
        List of valid pathways.
    """
    valid_pathways = []
    
    for pathway in pathways:
        if validate_pathway(pathway, network, delay_tolerance, phase_tolerance):
            valid_pathways.append(pathway)
    
    return valid_pathways
