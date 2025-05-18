"""
Causal Pathway Inference and Optimized Intervention in Dynamic Networks.
"""

from .network import DynamicNetwork
from .feature_extraction import STFT, AmplitudeExtractor, PhaseExtractor, unwrap_phase
from .pathway_detection import PropagationPathway, PathwayDetector
from .source_localization import SourceLocalizer
from .intervention import ImpactModel, ResourceOptimizer, GreedyHeuristic

__version__ = '0.1.0'
