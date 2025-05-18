# Experimental Evaluation

This directory contains scripts for the experimental evaluation of the framework, as described in Section 4 of the paper.

## Experiments

The experiments are organized into the following scripts:

1. **synthetic_experiments.py**: Main experiments for pathway detection on synthetic data.
2. **source_localization_experiments.py**: Source localization experiments on synthetic data.
3. **intervention_experiments.py**: Intervention optimization experiments on synthetic data.
4. **sensitivity_analysis.py**: Parameter sensitivity analysis.
5. **scalability_analysis.py**: Scalability analysis for network size and signal length.
6. **run_all_experiments.py**: Script to run all experiments.

## Running the Experiments

You can run all experiments using the `run_all_experiments.py` script:

```bash
python run_all_experiments.py
```

Or run specific experiments:

```bash
python run_all_experiments.py --experiments pathway_detection source_localization
```

## Command-line Arguments

The `run_all_experiments.py` script accepts the following command-line arguments:

### General Parameters
- `--experiments`: Experiments to run (default: all)
- `--n_runs`: Number of runs per configuration (default: 5)
- `--seed`: Base random seed (default: 42)

### Network Parameters
- `--network_types`: Network types to test (default: ba, er, ws, grid)
- `--n_nodes_list`: Network sizes to test (default: 100, 500, 1000)

### Signal Parameters
- `--n_samples_list`: Signal lengths to test (default: 500, 1000, 2000, 5000)
- `--snr_db_list`: SNR values to test (default: 5, 10, 20)

### Pathway Detection Parameters
- `--delay_uncertainty_list`: Delay uncertainty values to test (default: 0.05, 0.1, 0.2)

### Source Localization Parameters
- `--sparse_observation_ratios`: Observation ratios to test (default: 1.0, 0.75, 0.5)

### Intervention Parameters
- `--budget_ratios`: Budget ratios to test (default: 0.05, 0.1, 0.2)
- `--critical_node_ratios`: Critical node ratios to test (default: 0.05, 0.1, 0.2)

### Sensitivity Analysis Parameters
- `--delay_tolerance_values`: Delay tolerance values to test (default: 0.1, 0.2, 0.5, 1.0, 2.0)
- `--phase_tolerance_values`: Phase tolerance values to test (default: π/16, π/8, π/4, π/2, π)
- `--amplitude_threshold_values`: Amplitude threshold values to test (default: 0.05, 0.1, 0.2, 0.5, 1.0)
- `--window_size_values`: Window size values to test (default: 64, 128, 256, 512, 1024)

## Example

To run the pathway detection experiment with reduced parameters for faster execution:

```bash
python run_all_experiments.py --experiments pathway_detection --n_runs 3 --n_nodes_list 100 500 --snr_db_list 10
```

## Results

The results are saved to the following directories:
- `results/synthetic/`: CSV files with raw results
- `results/figures/`: Plots and visualizations

## Baseline Methods

The experiments include the following baseline methods:

### For Pathway Detection
- **Temporal Causality (TC)**: Detects pathways using only temporal causality.
- **TC + Delay Consistency (TCDC)**: Extends TC with delay consistency.
- **Ablated Framework (No-Phase)**: Our full framework omitting phase consistency.

### For Source Localization
- **Propagation Centrality (PC)**: Ranks nodes based on their centrality in the propagation.
- **Earliest Activator (EA)**: Selects node(s) with globally minimum activation time.
- **Degree Centrality (DC)**: Ranks nodes based on their degree centrality.
- **Betweenness Centrality (BC)**: Ranks nodes based on their betweenness centrality.

### For Optimized Intervention
- **Random Placement (RP)**: Randomly allocates resources to nodes.
- **Highest Impact First (HIF)**: Allocates resources to nodes with the highest initial impact.
- **Source Proximity (SP)**: Allocates resources to nodes close to the source.
