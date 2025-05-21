"""
Run all experiments on real-world datasets.

This script runs all the experiments for the paper:
1. Basic experiments on real-world datasets
2. Baseline comparison
3. Scalability analysis
4. Parameter sensitivity analysis
"""

import os
import time
import argparse
import subprocess
import sys
import traceback

def run_command(command, description, timeout=None):
    """Run a command and print its output."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        # Print output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)  # Ensure output is flushed immediately

        # Wait for process to complete with timeout
        return_code = process.wait(timeout=timeout)
        end_time = time.time()

        print(f"\n{'=' * 80}")
        print(f"Finished: {description}")
        print(f"Return code: {return_code}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"{'=' * 80}\n")

        return return_code

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\n{'=' * 80}")
        print(f"ERROR: {description} timed out after {timeout} seconds")
        print(f"{'=' * 80}\n")
        return -1
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ERROR: {description} failed with exception: {str(e)}")
        traceback.print_exc()
        print(f"{'=' * 80}\n")
        return -1

def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description='Run experiments on real-world datasets.')
    parser.add_argument('--basic', action='store_true', help='Run basic experiments')
    parser.add_argument('--baseline', action='store_true', help='Run baseline comparison')
    parser.add_argument('--scalability', action='store_true', help='Run scalability analysis')
    parser.add_argument('--sensitivity', action='store_true', help='Run parameter sensitivity analysis')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--debug', action='store_true', help='Run with debug flags')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds for each experiment (default: 3600)')

    args = parser.parse_args()

    # If no arguments are provided, show help
    if not any([args.basic, args.baseline, args.scalability, args.sensitivity, args.all]):
        parser.print_help()
        return

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Add debug flag if specified
    debug_flag = "--debug" if args.debug else ""

    # Run experiments
    results = {}

    if args.all or args.basic:
        # Start with a minimal test to check if everything works
        results['minimal_test'] = run_command('python run_minimal_test.py', 'Minimal test', timeout=60)

        # Run the real-world experiments with small subgraph size and debug flag
        cmd = f'python run_real_world_experiments.py --dataset email --subgraph_size 50 --runs 1 {debug_flag}'
        results['basic_small'] = run_command(cmd, 'Basic experiments on email dataset (small)', timeout=300)

        # If successful, run the full experiments
        cmd = f'python run_real_world_experiments.py {debug_flag}'
        results['basic'] = run_command(cmd, 'Basic experiments on real-world datasets', timeout=args.timeout)

    if args.all or args.baseline:
        cmd = f'python run_baseline_comparison.py {debug_flag}'
        results['baseline'] = run_command(cmd, 'Baseline comparison', timeout=args.timeout)

    if args.all or args.scalability:
        cmd = f'python analyze_scalability.py {debug_flag}'
        results['scalability'] = run_command(cmd, 'Scalability analysis', timeout=args.timeout)

    if args.all or args.sensitivity:
        cmd = f'python analyze_parameter_sensitivity.py {debug_flag}'
        results['sensitivity'] = run_command(cmd, 'Parameter sensitivity analysis', timeout=args.timeout)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    for name, code in results.items():
        status = "SUCCESS" if code == 0 else "FAILED"
        print(f"{name}: {status} (return code: {code})")
    print("=" * 80)

    # Return non-zero if any experiment failed
    if any(code != 0 for code in results.values()):
        print("\nSome experiments failed!")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
