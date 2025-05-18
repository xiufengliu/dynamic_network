"""
Amplitude extraction from time-series data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class AmplitudeExtractor:
    """
    A class for extracting amplitude features from time-series data.
    
    Attributes:
        threshold (float): Amplitude significance threshold.
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Initialize the amplitude extractor.
        
        Args:
            threshold: Amplitude significance threshold.
        """
        self.threshold = threshold
    
    def extract(self, stft_features: Dict[str, Dict[int, Dict[str, np.ndarray]]]) -> Dict[int, Dict[str, Union[np.ndarray, float, bool]]]:
        """
        Extract amplitude features from STFT features.
        
        Args:
            stft_features: Dictionary with STFT features.
            
        Returns:
            Dictionary with amplitude features for each signal.
        """
        amplitude_features = {}
        
        for i in stft_features['amplitude']:
            amplitude = stft_features['amplitude'][i]
            times = stft_features['times']
            is_active = amplitude > self.threshold
            
            # Find activation time (time when amplitude first exceeds threshold)
            activation_time = None
            if np.any(is_active):
                activation_idx = np.argmax(is_active)
                activation_time = times[activation_idx]
            
            # Find peak amplitude and its time
            peak_amplitude = np.max(amplitude) if len(amplitude) > 0 else 0.0
            peak_time = None
            if peak_amplitude > 0:
                peak_idx = np.argmax(amplitude)
                peak_time = times[peak_idx]
            
            # Calculate energy
            energy = np.sum(amplitude ** 2)
            
            # Store features
            amplitude_features[i] = {
                'amplitude': amplitude,
                'is_active': is_active,
                'activation_time': activation_time,
                'peak_amplitude': peak_amplitude,
                'peak_time': peak_time,
                'energy': energy
            }
        
        return amplitude_features
    
    def get_active_nodes(self, amplitude_features: Dict[int, Dict[str, Union[np.ndarray, float, bool]]]) -> List[int]:
        """
        Get the list of active nodes.
        
        Args:
            amplitude_features: Dictionary with amplitude features.
            
        Returns:
            List of active node indices.
        """
        active_nodes = []
        
        for i in amplitude_features:
            if amplitude_features[i]['activation_time'] is not None:
                active_nodes.append(i)
        
        return active_nodes
    
    def get_activation_times(self, amplitude_features: Dict[int, Dict[str, Union[np.ndarray, float, bool]]]) -> Dict[int, float]:
        """
        Get the activation times of all active nodes.
        
        Args:
            amplitude_features: Dictionary with amplitude features.
            
        Returns:
            Dictionary mapping node indices to activation times.
        """
        activation_times = {}
        
        for i in amplitude_features:
            if amplitude_features[i]['activation_time'] is not None:
                activation_times[i] = amplitude_features[i]['activation_time']
        
        return activation_times
