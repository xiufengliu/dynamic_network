# Debugging Guide for Real-World Network Experiments

This guide provides instructions for debugging issues with the real-world network experiments.

## Common Issues and Solutions

### 1. Script Hangs or Takes Too Long

If the script hangs or takes too long to run, try the following:

1. **Run with smaller subgraph size**:
   ```bash
   python run_real_world_experiments.py --dataset email --subgraph_size 20 --runs 1
   ```

2. **Run with debug logging**:
   ```bash
   python run_real_world_experiments.py --dataset email --subgraph_size 20 --runs 1 --debug
   ```

3. **Run step by step**:
   ```bash
   python run_step_by_step.py --dataset email --subgraph_size 20 --debug
   ```

4. **Run minimal test**:
   ```bash
   python run_minimal_test.py
   ```

### 2. Format Errors with STFT Features

If you encounter errors related to the STFT feature extraction, such as:
- `IndexError: tuple index out of range`
- `KeyError: 'amplitude'`

The issue is likely with the format of the features. The STFT implementation expects a specific format. Try running:

```bash
python run_single_dataset.py --dataset email --subgraph_size 20 --debug
```

### 3. Memory Issues

If you encounter memory issues with large datasets:

1. **Reduce subgraph size**:
   ```bash
   python run_real_world_experiments.py --dataset roadnet --subgraph_size 100
   ```

2. **Run on smaller datasets first**:
   ```bash
   python run_real_world_experiments.py --dataset email
   ```

### 4. Errors in Source Localization or Intervention

If you encounter errors in source localization or intervention optimization:

1. **Check if pathways are detected**:
   The source localization and intervention optimization depend on the detected pathways. If no pathways are detected, the script will use fallback methods.

2. **Run with debug logging**:
   ```bash
   python run_real_world_experiments.py --dataset email --subgraph_size 20 --runs 1 --debug
   ```

## Diagnostic Scripts

We've created several diagnostic scripts to help debug issues:

### 1. `check_datasets.py`

This script checks if the datasets exist and are in the correct format:

```bash
python check_datasets.py
```

### 2. `check_implementation.py`

This script checks if all required classes and methods are implemented correctly:

```bash
python check_implementation.py
```

### 3. `run_minimal_test.py`

This script runs a minimal test on a small synthetic network and the email dataset:

```bash
python run_minimal_test.py
```

### 4. `run_step_by_step.py`

This script runs the experiments step by step, with pauses between each step:

```bash
python run_step_by_step.py --dataset email --subgraph_size 20 --debug
```

### 5. `run_single_dataset.py`

This script runs the experiments on a single dataset with minimal parameters:

```bash
python run_single_dataset.py --dataset email --subgraph_size 20 --debug
```

## Log Files

The scripts create log files that can be useful for debugging:

- `run_real_world_experiments.log`
- `minimal_test.log`
- `step_by_step.log`
- `single_dataset.log`

## Specific Fixes for Known Issues

### 1. STFT Feature Format

The STFT implementation expects the input signal to be a 2D array with shape (n_signals, n_samples), but in our scripts, we're passing a 1D array. The fix is to reshape the signal:

```python
# Reshape the signal to match the expected input format (n_signals, n_samples)
signal_reshaped = np.array([signal])
extracted_features = stft.extract_features(signal_reshaped, EVENT_FREQ)
```

### 2. Source Localization with No Pathways

If no pathways are detected, the source localization will fail. The fix is to use a fallback method:

```python
# If no pathways were detected, use activation times to localize sources
if not detected_pathways:
    # Get active nodes
    active_nodes = list(features['amplitude'].keys())
    
    # Get activation times
    activation_times = features['activation_time']
    
    # Find nodes with minimum activation time
    min_time = min(activation_times.values())
    detected_sources = [node for node, time in activation_times.items() if time == min_time]
else:
    # Use pathways for source localization
    detected_sources = localizer.localize(network, features, detected_pathways)
```

### 3. NumPy Array Issues in Intervention Optimization

If you encounter errors like `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`, the issue is likely with NumPy arrays. The fix is to convert arrays to lists:

```python
# Convert to a list to avoid NumPy array issues
critical_nodes = critical_nodes.tolist()
```

## Running the Full Experiments

Once you've fixed the issues, you can run the full experiments:

```bash
python run_all_experiments.py --all
```

Or run specific experiments:

```bash
python run_all_experiments.py --basic
python run_all_experiments.py --baseline
python run_all_experiments.py --scalability
python run_all_experiments.py --sensitivity
```

You can also add the `--debug` flag for more detailed logging:

```bash
python run_all_experiments.py --all --debug
```

And set a timeout for each experiment:

```bash
python run_all_experiments.py --all --timeout 1800
```
