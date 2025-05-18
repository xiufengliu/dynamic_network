"""
Short-Time Fourier Transform implementation.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, List, Optional, Union


class STFT:
    """
    A class for performing Short-Time Fourier Transform on time-series data.
    
    Attributes:
        window_size (int): Size of the window function.
        overlap (float): Overlap between consecutive windows (0.0 to 1.0).
        window_func (str): Window function to use.
    """
    
    def __init__(self, window_size: int = 256, overlap: float = 0.5, 
                 window_func: str = 'hann'):
        """
        Initialize the STFT processor.
        
        Args:
            window_size: Size of the window function.
            overlap: Overlap between consecutive windows (0.0 to 1.0).
            window_func: Window function to use.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.window_func = window_func
        
        # Calculate the hop length
        self.hop_length = int(window_size * (1 - overlap))
    
    def compute(self, signal_data: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the STFT of a signal.
        
        Args:
            signal_data: The input signal (1D array).
            fs: Sampling frequency.
            
        Returns:
            Tuple of (frequencies, times, STFT coefficients).
        """
        # Compute the STFT
        frequencies, times, stft_coeffs = signal.stft(
            signal_data,
            fs=fs,
            window=self.window_func,
            nperseg=self.window_size,
            noverlap=int(self.window_size * self.overlap),
            return_onesided=True,
            boundary='zeros',
            padded=True
        )
        
        return frequencies, times, stft_coeffs
    
    def extract_features(self, signals: np.ndarray, freq: float, fs: float = 1.0, 
                         amplitude_threshold: float = 0.0) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
        """
        Extract amplitude and phase features from multiple signals at a specific frequency.
        
        Args:
            signals: Array of signals, shape (n_signals, n_samples).
            freq: Frequency of interest.
            fs: Sampling frequency.
            amplitude_threshold: Threshold for amplitude significance.
            
        Returns:
            Dictionary with features for each signal.
        """
        n_signals = signals.shape[0]
        features = {
            'amplitude': {},
            'phase': {},
            'unwrapped_phase': {},
            'activation_time': {},
            'is_active': {}
        }
        
        for i in range(n_signals):
            # Compute STFT
            frequencies, times, stft_coeffs = self.compute(signals[i], fs)
            
            # Find the closest frequency bin
            freq_idx = np.argmin(np.abs(frequencies - freq))
            
            # Extract amplitude and phase
            amplitude = np.abs(stft_coeffs[freq_idx])
            phase = np.angle(stft_coeffs[freq_idx])
            unwrapped_phase = np.unwrap(phase)
            
            # Determine if the signal is active (amplitude > threshold)
            is_active = amplitude > amplitude_threshold
            
            # Find activation time (time when amplitude first exceeds threshold)
            activation_time = None
            if np.any(is_active):
                activation_idx = np.argmax(is_active)
                activation_time = times[activation_idx]
            
            # Store features
            features['amplitude'][i] = amplitude
            features['phase'][i] = phase
            features['unwrapped_phase'][i] = unwrapped_phase
            features['activation_time'][i] = activation_time
            features['is_active'][i] = is_active
            
        # Add times and frequencies to the features
        features['times'] = times
        features['frequencies'] = frequencies
        features['freq_idx'] = freq_idx
        
        return features
    
    def inverse(self, stft_coeffs: np.ndarray, fs: float = 1.0) -> np.ndarray:
        """
        Compute the inverse STFT.
        
        Args:
            stft_coeffs: STFT coefficients.
            fs: Sampling frequency.
            
        Returns:
            Reconstructed signal.
        """
        # Compute the inverse STFT
        _, reconstructed = signal.istft(
            stft_coeffs,
            fs=fs,
            window=self.window_func,
            nperseg=self.window_size,
            noverlap=int(self.window_size * self.overlap),
            boundary=True
        )
        
        return reconstructed
