# Real-World Network Experiments

This directory contains scripts for running experiments on real-world network datasets.

## Datasets

The experiments use four real-world network datasets:

1. **roadNet-CA** - Road network of California
   - Path: `data/real_world/roadNet-CA.txt`
   - Undirected, static network
   - 1,965,206 nodes, 2,766,607 edges

2. **wiki-Talk** - Wikipedia talk page interactions
   - Path: `data/real_world/wiki-Talk.txt`
   - Directed, temporal communication network
   - 2,394,385 nodes, 5,021,410 edges

3. **email-Eu-core-temporal** - Email communications in a European research institution
   - Path: `data/real_world/email-Eu-core-temporal.txt`
   - Directed, temporal, community-annotated communication network
   - 986 nodes, 332,334 edges

4. **soc-redditHyperlinks-body** - Reddit hyperlinks between communities
   - Path: `data/real_world/soc-redditHyperlinks-body.tsv`
   - Directed, signed, temporal, attributed hyperlink network
   - 55,863 nodes, 858,490 edges

## Scripts

### Diagnostic Scripts

These scripts help diagnose issues with the datasets and implementation:

- `check_datasets.py` - Check if the datasets exist and are in the correct format
- `check_implementation.py` - Check if all required classes and methods are implemented correctly
- `run_minimal_test.py` - Run a minimal test on a small subset of the data
- `run_step_by_step.py` - Run the experiments step by step, with pauses between each step

### Experiment Scripts

These scripts run the full experiments:

- `run_real_world_experiments.py` - Run experiments on real-world datasets
- `run_baseline_comparison.py` - Compare our method against baseline methods
- `analyze_scalability.py` - Analyze the scalability of our methods
- `analyze_parameter_sensitivity.py` - Analyze the parameter sensitivity of our methods
- `run_all_experiments.py` - Run all experiments

## Running the Experiments

### Checking the Environment

Before running the experiments, check if the datasets and implementation are correct:

```bash
# Check if the datasets exist and are in the correct format
python check_datasets.py

# Check if all required classes and methods are implemented correctly
python check_implementation.py
```

### Running a Minimal Test

To quickly test if the implementation works, run a minimal test:

```bash
# Run a minimal test on a small subset of the data
python run_minimal_test.py
```

### Running Step by Step

To debug the experiments, run them step by step:

```bash
# Run the experiments step by step on the email dataset (smallest)
python run_step_by_step.py --dataset email --subgraph_size 100

# Run with debug logging
python run_step_by_step.py --dataset email --subgraph_size 100 --debug

# Run without pausing between steps
python run_step_by_step.py --dataset email --subgraph_size 100 --no_pause
```

### Running the Full Experiments

To run the full experiments:

```bash
# Run experiments on a specific dataset with debug logging
python run_real_world_experiments.py --dataset email --debug

# Run experiments with a smaller subgraph size
python run_real_world_experiments.py --dataset roadnet --subgraph_size 500

# Run experiments with fewer runs
python run_real_world_experiments.py --dataset wiki --runs 3
```

To run all experiments:

```bash
# Run all experiments
python run_all_experiments.py --all

# Run all experiments with debug logging
python run_all_experiments.py --all --debug

# Run specific experiments
python run_all_experiments.py --basic
python run_all_experiments.py --baseline
python run_all_experiments.py --scalability
python run_all_experiments.py --sensitivity
```

## Debugging

If the experiments hang or fail, try the following:

1. **Check the datasets**:
   ```bash
   python check_datasets.py
   ```

2. **Check the implementation**:
   ```bash
   python check_implementation.py
   ```

3. **Run a minimal test**:
   ```bash
   python run_minimal_test.py
   ```

4. **Run step by step with debug logging**:
   ```bash
   python run_step_by_step.py --dataset email --subgraph_size 50 --debug
   ```

5. **Run a specific dataset with debug logging and smaller subgraph**:
   ```bash
   python run_real_world_experiments.py --dataset email --subgraph_size 50 --runs 1 --debug
   ```

6. **Check the log files**:
   - `run_real_world_experiments.log`
   - `minimal_test.log`
   - `step_by_step.log`

## Common Issues and Solutions

### Hanging during simulation

If the script hangs during simulation, it might be due to:

1. **Large network size**: Try reducing the subgraph size:
   ```bash
   python run_real_world_experiments.py --dataset roadnet --subgraph_size 100
   ```

2. **Infinite loop in propagation**: Run step by step to identify the issue:
   ```bash
   python run_step_by_step.py --dataset email --debug
   ```

### Memory issues

If you encounter memory issues:

1. **Reduce subgraph size**:
   ```bash
   python run_real_world_experiments.py --dataset roadnet --subgraph_size 100
   ```

2. **Run on smaller datasets first**:
   ```bash
   python run_real_world_experiments.py --dataset email
   ```

### Runtime issues

If the experiments take too long:

1. **Reduce number of runs**:
   ```bash
   python run_real_world_experiments.py --dataset wiki --runs 1
   ```

2. **Run with timeout**:
   ```bash
   python run_all_experiments.py --basic --timeout 600
   ```

## Results

The results of the experiments are saved in the following directories:

- `results/experiments/real_world/` - Basic experiment results
- `results/baselines/real_world/` - Baseline comparison results
- `results/scalability/` - Scalability analysis results
- `results/sensitivity/` - Parameter sensitivity analysis results

Each directory contains:
- `data/` - Raw data in CSV and pickle format
- `figures/` - Visualizations in PNG format
