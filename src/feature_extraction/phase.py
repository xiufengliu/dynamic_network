"""
Phase extraction and unwrapping from time-series data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase to ensure continuity.
    
    Args:
        phase: Array of phase values.
        
    Returns:
        Unwrapped phase.
    """
    return np.unwrap(phase)


class PhaseExtractor:
    """
    A class for extracting phase features from time-series data.
    
    Attributes:
        tolerance (float): Phase consistency tolerance.
    """
    
    def __init__(self, tolerance: float = np.pi/4):
        """
        Initialize the phase extractor.
        
        Args:
            tolerance: Phase consistency tolerance.
        """
        self.tolerance = tolerance
    
    def extract(self, stft_features: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                active_nodes: List[int]) -> Dict[int, Dict[str, Union[np.ndarray, float]]]:
        """
        Extract phase features from STFT features.
        
        Args:
            stft_features: Dictionary with STFT features.
            active_nodes: List of active node indices.
            
        Returns:
            Dictionary with phase features for each active signal.
        """
        phase_features = {}
        
        for i in active_nodes:
            if i in stft_features['phase']:
                phase = stft_features['phase'][i]
                unwrapped_phase = stft_features['unwrapped_phase'][i]
                times = stft_features['times']
                
                # Get activation time
                activation_time = stft_features['activation_time'][i]
                
                # Get phase at activation time
                activation_phase = None
                if activation_time is not None:
                    activation_idx = np.argmin(np.abs(times - activation_time))
                    activation_phase = phase[activation_idx]
                    activation_unwrapped_phase = unwrapped_phase[activation_idx]
                
                # Store features
                phase_features[i] = {
                    'phase': phase,
                    'unwrapped_phase': unwrapped_phase,
                    'activation_phase': activation_phase,
                    'activation_unwrapped_phase': activation_unwrapped_phase
                }
        
        return phase_features
    
    def check_phase_consistency(self, phase1: float, phase2: float, 
                                delay: float, freq: float) -> bool:
        """
        Check if two phases are consistent with the given delay and frequency.
        
        Args:
            phase1: Phase at the source node.
            phase2: Phase at the target node.
            delay: Propagation delay.
            freq: Frequency of the event.
            
        Returns:
            True if phases are consistent, False otherwise.
        """
        # Calculate expected phase shift
        expected_shift = 2 * np.pi * freq * delay
        
        # Calculate actual phase difference
        actual_diff = (phase2 - phase1) % (2 * np.pi)
        
        # Check if the difference is within tolerance
        return abs(actual_diff - expected_shift) <= self.tolerance
    
    def calculate_phase_difference(self, phase1: float, phase2: float) -> float:
        """
        Calculate the phase difference between two phases.
        
        Args:
            phase1: First phase.
            phase2: Second phase.
            
        Returns:
            Phase difference in radians.
        """
        return (phase2 - phase1) % (2 * np.pi)
    
    def get_activation_phases(self, phase_features: Dict[int, Dict[str, Union[np.ndarray, float]]]) -> Dict[int, float]:
        """
        Get the activation phases of all active nodes.
        
        Args:
            phase_features: Dictionary with phase features.
            
        Returns:
            Dictionary mapping node indices to activation phases.
        """
        activation_phases = {}
        
        for i in phase_features:
            if phase_features[i]['activation_phase'] is not None:
                activation_phases[i] = phase_features[i]['activation_phase']
        
        return activation_phases
    
    def get_activation_unwrapped_phases(self, phase_features: Dict[int, Dict[str, Union[np.ndarray, float]]]) -> Dict[int, float]:
        """
        Get the activation unwrapped phases of all active nodes.
        
        Args:
            phase_features: Dictionary with phase features.
            
        Returns:
            Dictionary mapping node indices to activation unwrapped phases.
        """
        activation_unwrapped_phases = {}
        
        for i in phase_features:
            if phase_features[i]['activation_unwrapped_phase'] is not None:
                activation_unwrapped_phases[i] = phase_features[i]['activation_unwrapped_phase']
        
        return activation_unwrapped_phases
