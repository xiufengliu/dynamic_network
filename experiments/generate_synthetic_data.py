"""
Generate synthetic datasets for experiments.

This script generates synthetic datasets for all network types and configurations
mentioned in the paper, and saves them to the data directory.
"""

import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import argparse
from typing import Dict, List, Tuple, Optional, Union, Set

# Add the parent directory to the path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.network.graph import DynamicNetwork
from src.network.generators import (
    generate_barabasi_albert_network,
    generate_erdos_renyi_network,
    generate_watts_strogatz_network,
    generate_grid_network
)
from src.pathway_detection.definition import PropagationPathway

# Import the synthetic data generation function from the main experiments script
from synthetic_experiments import generate_synthetic_data


def generate_and_save_datasets(
    network_types: List[str] = ['ba', 'er', 'ws', 'grid'],
    n_nodes_list: List[int] = [100, 500, 1000, 2000],
    snr_db_list: List[float] = [5, 10, 20],
    delay_uncertainty_list: List[float] = [0.05, 0.1, 0.2],
    n_samples: int = 1000,
    event_freq: float = 0.1,
    n_datasets_per_config: int = 5,
    base_seed: int = 42,
    output_dir: str = '../data/synthetic'
) -> None:
    """
    Generate and save synthetic datasets for all configurations.
    
    Args:
        network_types: List of network types to generate.
        n_nodes_list: List of network sizes to generate.
        snr_db_list: List of SNR values to use.
        delay_uncertainty_list: List of delay uncertainty values to use.
        n_samples: Number of time samples.
        event_freq: Characteristic frequency of the event.
        n_datasets_per_config: Number of datasets to generate per configuration.
        base_seed: Base random seed.
        output_dir: Directory to save datasets.
    """
    # Create output directories
    for network_type in network_types:
        os.makedirs(os.path.join(output_dir, f'{network_type}_networks'), exist_ok=True)
    
    # Generate datasets for all configurations
    total_configs = len(network_types) * len(n_nodes_list) * len(snr_db_list) * len(delay_uncertainty_list) * n_datasets_per_config
    
    with tqdm(total=total_configs, desc="Generating datasets") as pbar:
        for network_type in network_types:
            for n_nodes in n_nodes_list:
                for snr_db in snr_db_list:
                    for delay_uncertainty in delay_uncertainty_list:
                        for dataset_idx in range(n_datasets_per_config):
                            # Set seed for reproducibility
                            seed = base_seed + dataset_idx
                            
                            # Generate synthetic data
                            network, signals, time, true_sources, true_pathways = generate_synthetic_data(
                                network_type=network_type,
                                n_nodes=n_nodes,
                                n_samples=n_samples,
                                event_freq=event_freq,
                                snr_db=snr_db,
                                delay_uncertainty=delay_uncertainty,
                                seed=seed
                            )
                            
                            # Create dataset metadata
                            metadata = {
                                'network_type': network_type,
                                'n_nodes': n_nodes,
                                'n_samples': n_samples,
                                'event_freq': event_freq,
                                'snr_db': snr_db,
                                'delay_uncertainty': delay_uncertainty,
                                'seed': seed,
                                'true_sources': true_sources,
                                'n_true_pathways': len(true_pathways)
                            }
                            
                            # Create filename
                            filename_base = f"{network_type}_{n_nodes}_{snr_db}db_{delay_uncertainty}delay_{dataset_idx}"
                            network_filename = os.path.join(output_dir, f'{network_type}_networks', f"{filename_base}.graphml")
                            data_filename = os.path.join(output_dir, f'{network_type}_networks', f"{filename_base}.pkl")
                            
                            # Save network
                            network.save_to_file(network_filename)
                            
                            # Save signals, time, true sources, true pathways, and metadata
                            with open(data_filename, 'wb') as f:
                                pickle.dump({
                                    'signals': signals,
                                    'time': time,
                                    'true_sources': true_sources,
                                    'true_pathways': true_pathways,
                                    'metadata': metadata
                                }, f)
                            
                            pbar.update(1)
    
    print(f"Generated {total_configs} datasets in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic datasets for experiments.')
    
    # Dataset parameters
    parser.add_argument('--network_types', type=str, nargs='+', default=['ba', 'er', 'ws', 'grid'],
                        help='Network types to generate')
    parser.add_argument('--n_nodes_list', type=int, nargs='+', default=[100, 500, 1000],
                        help='Network sizes to generate')
    parser.add_argument('--snr_db_list', type=float, nargs='+', default=[5, 10, 20],
                        help='SNR values to use')
    parser.add_argument('--delay_uncertainty_list', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Delay uncertainty values to use')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of time samples')
    parser.add_argument('--event_freq', type=float, default=0.1,
                        help='Characteristic frequency of the event')
    parser.add_argument('--n_datasets_per_config', type=int, default=5,
                        help='Number of datasets to generate per configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output_dir', type=str, default='../data/synthetic',
                        help='Directory to save datasets')
    
    args = parser.parse_args()
    
    # Print generation configuration
    print("\nDataset Generation Configuration:")
    print(f"Network types: {args.network_types}")
    print(f"Network sizes: {args.n_nodes_list}")
    print(f"SNR values: {args.snr_db_list}")
    print(f"Delay uncertainty values: {args.delay_uncertainty_list}")
    print(f"Number of time samples: {args.n_samples}")
    print(f"Event frequency: {args.event_freq}")
    print(f"Datasets per configuration: {args.n_datasets_per_config}")
    print(f"Base seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Generate and save datasets
    generate_and_save_datasets(
        network_types=args.network_types,
        n_nodes_list=args.n_nodes_list,
        snr_db_list=args.snr_db_list,
        delay_uncertainty_list=args.delay_uncertainty_list,
        n_samples=args.n_samples,
        event_freq=args.event_freq,
        n_datasets_per_config=args.n_datasets_per_config,
        base_seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
