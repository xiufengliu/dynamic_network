"""
Run experiments on a single dataset with minimal parameters.

This script runs the experiments on a single dataset with minimal parameters
to test if the full experiment works.
"""

import os
import sys
import time
import logging
import traceback
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils.real_world_loader import (
    load_roadnet_ca,
    load_wiki_talk,
    load_email_eu_core,
    load_reddit_hyperlinks
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('single_dataset.log')
    ]
)
logger = logging.getLogger(__name__)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

def main():
    """Main function to run experiments on a single dataset."""
    parser = argparse.ArgumentParser(description='Run experiments on a single dataset.')
    parser.add_argument('--dataset', type=str, choices=['roadnet', 'wiki', 'email', 'reddit'],
                        default='email', help='Dataset to use')
    parser.add_argument('--subgraph_size', type=int, default=50, help='Size of subgraph to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting experiments on a single dataset...")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Subgraph size: {args.subgraph_size}")

    # Create output directories
    os.makedirs('results/single_dataset', exist_ok=True)

    try:
        # Load dataset
        logger.info(f"Loading {args.dataset} dataset...")

        if args.dataset == 'roadnet':
            path = ROADNET_PATH
            loader = load_roadnet_ca
        elif args.dataset == 'wiki':
            path = WIKI_TALK_PATH
            loader = load_wiki_talk
        elif args.dataset == 'email':
            path = EMAIL_PATH
            loader = load_email_eu_core
        elif args.dataset == 'reddit':
            path = REDDIT_PATH
            loader = load_reddit_hyperlinks
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            return False

        # Check if file exists
        if not os.path.exists(path):
            logger.error(f"Dataset file not found: {path}")
            return False

        # Load dataset
        start_time = time.time()
        network = loader(path)
        load_time = time.time() - start_time
        logger.info(f"Loaded {args.dataset} in {load_time:.2f}s: {len(network.get_nodes())} nodes, {len(network.get_edges())} edges")

        # Create subgraph
        logger.info(f"Creating subgraph with {args.subgraph_size} nodes...")
        subgraph_size = min(args.subgraph_size, len(network.get_nodes()))
        subgraph_nodes = list(network.graph.nodes())[:subgraph_size]

        start_time = time.time()
        subgraph = network.create_subgraph(subgraph_nodes)
        subgraph_time = time.time() - start_time
        logger.info(f"Created subgraph in {subgraph_time:.2f}s with {len(subgraph.get_nodes())} nodes, {len(subgraph.get_edges())} edges")

        # Run the experiment
        logger.info("Running experiment...")

        # Import the run_experiment function and save_results function from run_real_world_experiments.py
        from run_real_world_experiments import run_experiment, save_results, generate_comparative_visualizations

        # Run the experiment with minimal parameters
        results = run_experiment(
            network_name=args.dataset,
            network=subgraph,
            num_runs=1  # Just one run for testing
        )

        # Save the results
        logger.info("Saving results...")

        # Create a dictionary with the dataset name as the key
        all_results = {args.dataset: results}

        # Save the results
        try:
            # Create output directories
            os.makedirs('results/single_dataset/data', exist_ok=True)
            os.makedirs('results/single_dataset/figures', exist_ok=True)

            # Save as pickle for later analysis
            import pickle
            with open(f'results/single_dataset/data/{args.dataset}_results.pkl', 'wb') as f:
                pickle.dump(all_results, f)

            # Save summary as CSV
            summary = []
            for network_name, network_results in all_results.items():
                row = {'network': network_name}

                # Add pathway detection metrics
                for metric, values in network_results['pathway_detection'].items():
                    row[f'pathway_{metric}_mean'] = np.mean(values)
                    row[f'pathway_{metric}_std'] = np.std(values)

                # Add source localization metrics
                for metric, values in network_results['source_localization'].items():
                    row[f'source_{metric}_mean'] = np.mean(values)
                    row[f'source_{metric}_std'] = np.std(values)

                # Add intervention metrics
                for metric, values in network_results['intervention'].items():
                    row[f'intervention_{metric}_mean'] = np.mean(values)
                    row[f'intervention_{metric}_std'] = np.std(values)

                # Add runtime metrics
                for metric, values in network_results['runtime'].items():
                    row[f'runtime_{metric}_mean'] = np.mean(values)
                    row[f'runtime_{metric}_std'] = np.std(values)

                summary.append(row)

            # Convert to DataFrame and save
            import pandas as pd
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(f'results/single_dataset/data/{args.dataset}_summary.csv', index=False)

            logger.info(f"Results saved to results/single_dataset/data/{args.dataset}_results.pkl")
            logger.info(f"Summary saved to results/single_dataset/data/{args.dataset}_summary.csv")

            # Generate visualizations
            logger.info("Generating visualizations...")

            # Create a custom visualization function that saves to the single_dataset directory
            def generate_single_dataset_visualizations(results):
                """Generate visualizations for a single dataset."""
                # 1. Pathway Detection Performance
                plt.figure(figsize=(12, 8))
                networks = list(results.keys())

                # F1 scores
                f1_means = [np.mean(results[net]['pathway_detection']['f1_score']) for net in networks]
                f1_stds = [np.std(results[net]['pathway_detection']['f1_score']) for net in networks]

                x = np.arange(len(networks))
                width = 0.35

                plt.bar(x, f1_means, width, yerr=f1_stds, label='F1 Score', capsize=5)

                plt.xlabel('Network')
                plt.ylabel('Score')
                plt.title('Pathway Detection Performance')
                plt.xticks(x, networks, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/single_dataset/figures/{args.dataset}_pathway_detection.png', dpi=300)
                plt.close()

                # 2. Source Localization Performance
                plt.figure(figsize=(12, 8))

                # Success rates
                sr1_means = [np.mean(results[net]['source_localization']['sr@1']) for net in networks]
                sr3_means = [np.mean(results[net]['source_localization']['sr@3']) for net in networks]
                sr5_means = [np.mean(results[net]['source_localization']['sr@5']) for net in networks]

                sr1_stds = [np.std(results[net]['source_localization']['sr@1']) for net in networks]
                sr3_stds = [np.std(results[net]['source_localization']['sr@3']) for net in networks]
                sr5_stds = [np.std(results[net]['source_localization']['sr@5']) for net in networks]

                x = np.arange(len(networks))
                width = 0.25

                plt.bar(x - width, sr1_means, width, yerr=sr1_stds, label='SR@1', capsize=5)
                plt.bar(x, sr3_means, width, yerr=sr3_stds, label='SR@3', capsize=5)
                plt.bar(x + width, sr5_means, width, yerr=sr5_stds, label='SR@5', capsize=5)

                plt.xlabel('Network')
                plt.ylabel('Success Rate')
                plt.title('Source Localization Performance')
                plt.xticks(x, networks, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/single_dataset/figures/{args.dataset}_source_localization.png', dpi=300)
                plt.close()

                # 3. Intervention Performance
                plt.figure(figsize=(12, 8))

                # TIR and CER
                tir_means = [np.mean(results[net]['intervention']['tir']) for net in networks]
                cer_means = [np.mean(results[net]['intervention']['cer']) for net in networks]

                tir_stds = [np.std(results[net]['intervention']['tir']) for net in networks]
                cer_stds = [np.std(results[net]['intervention']['cer']) for net in networks]

                x = np.arange(len(networks))
                width = 0.35

                plt.bar(x - width/2, tir_means, width, yerr=tir_stds, label='TIR', capsize=5)
                plt.bar(x + width/2, cer_means, width, yerr=cer_stds, label='CER', capsize=5)

                plt.xlabel('Network')
                plt.ylabel('Score')
                plt.title('Intervention Performance')
                plt.xticks(x, networks, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/single_dataset/figures/{args.dataset}_intervention.png', dpi=300)
                plt.close()

                # 4. Runtime Comparison
                plt.figure(figsize=(12, 8))

                # Runtimes
                pathway_times = [np.mean(results[net]['runtime']['pathway_detection']) for net in networks]
                source_times = [np.mean(results[net]['runtime']['source_localization']) for net in networks]
                intervention_times = [np.mean(results[net]['runtime']['intervention']) for net in networks]

                pathway_stds = [np.std(results[net]['runtime']['pathway_detection']) for net in networks]
                source_stds = [np.std(results[net]['runtime']['source_localization']) for net in networks]
                intervention_stds = [np.std(results[net]['runtime']['intervention']) for net in networks]

                x = np.arange(len(networks))
                width = 0.25

                plt.bar(x - width, pathway_times, width, yerr=pathway_stds, label='Pathway Detection', capsize=5)
                plt.bar(x, source_times, width, yerr=source_stds, label='Source Localization', capsize=5)
                plt.bar(x + width, intervention_times, width, yerr=intervention_stds, label='Intervention', capsize=5)

                plt.xlabel('Network')
                plt.ylabel('Runtime (seconds)')
                plt.title('Algorithm Runtime Comparison')
                plt.xticks(x, networks, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'results/single_dataset/figures/{args.dataset}_runtime.png', dpi=300)
                plt.close()

            # Call the custom visualization function
            generate_single_dataset_visualizations(all_results)
            logger.info("Visualizations saved to results/single_dataset/figures/")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            logger.error(traceback.format_exc())

        logger.info("Experiment completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
