"""
Check implementation of required classes and methods.

This script checks if all required classes and methods are implemented correctly.
"""

import os
import sys
import inspect
import importlib
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_module_exists(module_name):
    """Check if a module exists."""
    try:
        module = importlib.import_module(module_name)
        logger.info(f"✓ Module exists: {module_name}")
        return module
    except ImportError as e:
        logger.error(f"✗ Module does not exist: {module_name}")
        logger.error(f"  Error: {str(e)}")
        return None

def check_class_exists(module, class_name):
    """Check if a class exists in a module."""
    if module is None:
        return None
    
    try:
        cls = getattr(module, class_name)
        logger.info(f"✓ Class exists: {class_name}")
        return cls
    except AttributeError:
        logger.error(f"✗ Class does not exist: {class_name}")
        return None

def check_method_exists(cls, method_name):
    """Check if a method exists in a class."""
    if cls is None:
        return False
    
    try:
        method = getattr(cls, method_name)
        logger.info(f"✓ Method exists: {method_name}")
        return True
    except AttributeError:
        logger.error(f"✗ Method does not exist: {method_name}")
        return False

def check_dynamic_network():
    """Check the DynamicNetwork class."""
    logger.info("\n=== Checking DynamicNetwork class ===")
    
    # Check module
    module = check_module_exists("src.network.graph")
    if module is None:
        return False
    
    # Check class
    cls = check_class_exists(module, "DynamicNetwork")
    if cls is None:
        return False
    
    # Check methods
    methods_ok = True
    for method_name in ["add_node", "add_edge", "get_nodes", "get_edges", "create_subgraph"]:
        if not check_method_exists(cls, method_name):
            methods_ok = False
    
    # Try to create an instance
    try:
        network = cls()
        logger.info("✓ Successfully created DynamicNetwork instance")
        
        # Try basic operations
        network.add_node(1)
        network.add_node(2)
        network.add_edge(1, 2, weight=1.0)
        
        nodes = network.get_nodes()
        edges = network.get_edges()
        
        logger.info(f"✓ Basic operations successful. Nodes: {nodes}, Edges: {edges}")
        
        # Try creating a subgraph
        subgraph = network.create_subgraph([1])
        logger.info(f"✓ Subgraph creation successful. Nodes: {subgraph.get_nodes()}")
        
        return methods_ok
    except Exception as e:
        logger.error(f"✗ Error creating or using DynamicNetwork instance: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_stft():
    """Check the STFT class."""
    logger.info("\n=== Checking STFT class ===")
    
    # Check module
    module = check_module_exists("src.feature_extraction.stft")
    if module is None:
        return False
    
    # Check class
    cls = check_class_exists(module, "STFT")
    if cls is None:
        return False
    
    # Check methods
    methods_ok = True
    for method_name in ["extract_features"]:
        if not check_method_exists(cls, method_name):
            methods_ok = False
    
    # Try to create an instance
    try:
        stft = cls(window_size=256, overlap=0.5)
        logger.info("✓ Successfully created STFT instance")
        
        # Try to import numpy for testing
        import numpy as np
        
        # Create a test signal
        fs = 100
        duration = 1
        t = np.linspace(0, duration, int(fs * duration))
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        
        # Try to extract features
        features = stft.extract_features(signal, 10)
        logger.info(f"✓ Feature extraction successful. Features: {features.keys() if isinstance(features, dict) else 'not a dict'}")
        
        return methods_ok
    except Exception as e:
        logger.error(f"✗ Error creating or using STFT instance: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_pathway_detector():
    """Check the PathwayDetector class."""
    logger.info("\n=== Checking PathwayDetector class ===")
    
    # Check module
    module = check_module_exists("src.pathway_detection.detector")
    if module is None:
        return False
    
    # Check class
    cls = check_class_exists(module, "PathwayDetector")
    if cls is None:
        return False
    
    # Check methods
    methods_ok = True
    for method_name in ["detect"]:
        if not check_method_exists(cls, method_name):
            methods_ok = False
    
    # Try to create an instance
    try:
        detector = cls(delay_tolerance=0.5, phase_tolerance=0.5, amplitude_threshold=0.1)
        logger.info("✓ Successfully created PathwayDetector instance")
        return methods_ok
    except Exception as e:
        logger.error(f"✗ Error creating PathwayDetector instance: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_source_localizer():
    """Check the SourceLocalizer class."""
    logger.info("\n=== Checking SourceLocalizer class ===")
    
    # Check module
    module = check_module_exists("src.source_localization.localizer")
    if module is None:
        return False
    
    # Check class
    cls = check_class_exists(module, "SourceLocalizer")
    if cls is None:
        return False
    
    # Check methods
    methods_ok = True
    for method_name in ["localize"]:
        if not check_method_exists(cls, method_name):
            methods_ok = False
    
    # Try to create an instance
    try:
        localizer = cls()
        logger.info("✓ Successfully created SourceLocalizer instance")
        return methods_ok
    except Exception as e:
        logger.error(f"✗ Error creating SourceLocalizer instance: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_resource_optimizer():
    """Check the ResourceOptimizer class."""
    logger.info("\n=== Checking ResourceOptimizer class ===")
    
    # Check module
    module = check_module_exists("src.intervention.optimizer")
    if module is None:
        return False
    
    # Check class
    cls = check_class_exists(module, "ResourceOptimizer")
    if cls is None:
        return False
    
    # Check methods
    methods_ok = True
    for method_name in ["optimize"]:
        if not check_method_exists(cls, method_name):
            methods_ok = False
    
    # Try to create an instance
    try:
        optimizer = cls()
        logger.info("✓ Successfully created ResourceOptimizer instance")
        return methods_ok
    except Exception as e:
        logger.error(f"✗ Error creating ResourceOptimizer instance: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_real_world_loader():
    """Check the real-world loader functions."""
    logger.info("\n=== Checking real-world loader functions ===")
    
    # Check module
    module = check_module_exists("src.utils.real_world_loader")
    if module is None:
        return False
    
    # Check functions
    functions_ok = True
    for function_name in ["load_roadnet_ca", "load_wiki_talk", "load_email_eu_core", "load_reddit_hyperlinks"]:
        try:
            func = getattr(module, function_name)
            logger.info(f"✓ Function exists: {function_name}")
        except AttributeError:
            logger.error(f"✗ Function does not exist: {function_name}")
            functions_ok = False
    
    return functions_ok

def main():
    """Main function to check all implementations."""
    logger.info("Checking implementation of required classes and methods...")
    
    # Check each component
    dynamic_network_ok = check_dynamic_network()
    stft_ok = check_stft()
    pathway_detector_ok = check_pathway_detector()
    source_localizer_ok = check_source_localizer()
    resource_optimizer_ok = check_resource_optimizer()
    real_world_loader_ok = check_real_world_loader()
    
    # Print summary
    logger.info("\n=== Implementation Check Summary ===")
    logger.info(f"DynamicNetwork: {'OK' if dynamic_network_ok else 'FAILED'}")
    logger.info(f"STFT: {'OK' if stft_ok else 'FAILED'}")
    logger.info(f"PathwayDetector: {'OK' if pathway_detector_ok else 'FAILED'}")
    logger.info(f"SourceLocalizer: {'OK' if source_localizer_ok else 'FAILED'}")
    logger.info(f"ResourceOptimizer: {'OK' if resource_optimizer_ok else 'FAILED'}")
    logger.info(f"Real-world loader: {'OK' if real_world_loader_ok else 'FAILED'}")
    
    # Return success if all components are OK
    return all([
        dynamic_network_ok,
        stft_ok,
        pathway_detector_ok,
        source_localizer_ok,
        resource_optimizer_ok,
        real_world_loader_ok
    ])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
