"""
Run all synthetic data experiments.

This script runs all the experiments for synthetic data:
1. Pathway Detection
2. Source Localization
3. Intervention Optimization
4. Parameter Sensitivity Analysis
5. Scalability Analysis
"""

import os
import time
import argparse
from typing import List

# Create output directories
os.makedirs('results/synthetic', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


def run_experiment(experiment_name: str, args: argparse.Namespace) -> None:
    """
    Run a specific experiment.
    
    Args:
        experiment_name: Name of the experiment to run.
        args: Command-line arguments.
    """
    print(f"\n{'=' * 80}")
    print(f"Running {experiment_name} experiment...")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    if experiment_name == 'pathway_detection':
        from synthetic_experiments import run_pathway_detection_experiment, plot_pathway_detection_results
        
        results = run_pathway_detection_experiment(
            network_types=args.network_types,
            n_nodes_list=args.n_nodes_list,
            snr_db_list=args.snr_db_list,
            delay_uncertainty_list=args.delay_uncertainty_list,
            n_runs=args.n_runs,
            base_seed=args.seed
        )
        
        # Save results
        results.to_csv('results/synthetic/pathway_detection_results.csv', index=False)
        
        # Plot results
        summary = plot_pathway_detection_results(results)
        
    elif experiment_name == 'source_localization':
        from source_localization_experiments import run_source_localization_experiment, plot_source_localization_results
        
        results = run_source_localization_experiment(
            network_types=args.network_types,
            n_nodes_list=args.n_nodes_list,
            snr_db_list=args.snr_db_list,
            sparse_observation_ratios=args.sparse_observation_ratios,
            n_runs=args.n_runs,
            base_seed=args.seed
        )
        
        # Save results
        results.to_csv('results/synthetic/source_localization_results.csv', index=False)
        
        # Plot results
        summary = plot_source_localization_results(results)
        
    elif experiment_name == 'intervention':
        from intervention_experiments import run_intervention_experiment, plot_intervention_results
        
        results = run_intervention_experiment(
            network_types=args.network_types,
            n_nodes_list=args.n_nodes_list,
            snr_db_list=[10],  # Fixed SNR for simplicity
            budget_ratios=args.budget_ratios,
            critical_node_ratios=args.critical_node_ratios,
            n_runs=args.n_runs,
            base_seed=args.seed
        )
        
        # Save results
        results.to_csv('results/synthetic/intervention_results.csv', index=False)
        
        # Plot results
        summary = plot_intervention_results(results)
        
    elif experiment_name == 'sensitivity':
        from sensitivity_analysis import run_sensitivity_analysis, plot_sensitivity_results
        
        results = run_sensitivity_analysis(
            delay_tolerance_values=args.delay_tolerance_values,
            phase_tolerance_values=args.phase_tolerance_values,
            amplitude_threshold_values=args.amplitude_threshold_values,
            window_size_values=args.window_size_values,
            n_runs=args.n_runs,
            base_seed=args.seed
        )
        
        # Save results
        results.to_csv('results/synthetic/sensitivity_analysis_results.csv', index=False)
        
        # Plot results
        summary = plot_sensitivity_results(results)
        
    elif experiment_name == 'scalability':
        from scalability_analysis import (
            run_scalability_analysis_network_size,
            run_scalability_analysis_signal_length,
            plot_scalability_results
        )
        
        # Run scalability analysis for network size
        network_results = run_scalability_analysis_network_size(
            n_nodes_list=args.n_nodes_list,
            n_runs=args.n_runs,
            base_seed=args.seed
        )
        
        # Run scalability analysis for signal length
        signal_results = run_scalability_analysis_signal_length(
            n_samples_list=args.n_samples_list,
            n_runs=args.n_runs,
            base_seed=args.seed
        )
        
        # Save results
        network_results.to_csv('results/synthetic/network_size_scalability_results.csv', index=False)
        signal_results.to_csv('results/synthetic/signal_length_scalability_results.csv', index=False)
        
        # Plot results
        network_summary, signal_summary = plot_scalability_results(network_results, signal_results)
    
    end_time = time.time()
    print(f"\n{experiment_name} experiment completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to results/synthetic/{experiment_name}_results.csv")
    print(f"Plots saved to results/figures/")


def main():
    parser = argparse.ArgumentParser(description='Run synthetic data experiments.')
    
    # General parameters
    parser.add_argument('--experiments', type=str, nargs='+', default=['pathway_detection', 'source_localization', 'intervention', 'sensitivity', 'scalability'],
                        help='Experiments to run')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs per configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    
    # Network parameters
    parser.add_argument('--network_types', type=str, nargs='+', default=['ba', 'er', 'ws', 'grid'],
                        help='Network types to test')
    parser.add_argument('--n_nodes_list', type=int, nargs='+', default=[100, 500, 1000],
                        help='Network sizes to test')
    
    # Signal parameters
    parser.add_argument('--n_samples_list', type=int, nargs='+', default=[500, 1000, 2000, 5000],
                        help='Signal lengths to test')
    parser.add_argument('--snr_db_list', type=float, nargs='+', default=[5, 10, 20],
                        help='SNR values to test')
    
    # Pathway detection parameters
    parser.add_argument('--delay_uncertainty_list', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Delay uncertainty values to test')
    
    # Source localization parameters
    parser.add_argument('--sparse_observation_ratios', type=float, nargs='+', default=[1.0, 0.75, 0.5],
                        help='Observation ratios to test')
    
    # Intervention parameters
    parser.add_argument('--budget_ratios', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Budget ratios to test')
    parser.add_argument('--critical_node_ratios', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Critical node ratios to test')
    
    # Sensitivity analysis parameters
    parser.add_argument('--delay_tolerance_values', type=float, nargs='+', default=[0.1, 0.2, 0.5, 1.0, 2.0],
                        help='Delay tolerance values to test')
    parser.add_argument('--phase_tolerance_values', type=float, nargs='+', default=[0.19634954, 0.39269908, 0.78539816, 1.57079633, 3.14159265],
                        help='Phase tolerance values to test (π/16, π/8, π/4, π/2, π)')
    parser.add_argument('--amplitude_threshold_values', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.5, 1.0],
                        help='Amplitude threshold values to test')
    parser.add_argument('--window_size_values', type=int, nargs='+', default=[64, 128, 256, 512, 1024],
                        help='Window size values to test')
    
    args = parser.parse_args()
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"Experiments: {args.experiments}")
    print(f"Number of runs: {args.n_runs}")
    print(f"Base seed: {args.seed}")
    print(f"Network types: {args.network_types}")
    print(f"Network sizes: {args.n_nodes_list}")
    print(f"SNR values: {args.snr_db_list}")
    print()
    
    # Run selected experiments
    for experiment in args.experiments:
        run_experiment(experiment, args)
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
