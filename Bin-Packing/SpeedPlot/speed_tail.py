import random
import numpy as np
import time  # For timing
import os
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for headless environments
import matplotlib.pyplot as plt

# --- Optimal Bin Packing (Backtracking) ---
min_bins_solution_global = float('inf')
best_bin_config_global = []


def solve_bin_packing_optimal_recursive(items_to_pack, bin_capacity, current_bins_items,
                                        current_bins_remaining_capacity):
    global min_bins_solution_global, best_bin_config_global
    if not items_to_pack:
        num_bins_used = len(current_bins_items)
        if num_bins_used < min_bins_solution_global:
            min_bins_solution_global = num_bins_used
            best_bin_config_global = [list(b) for b in current_bins_items]
        return
    if len(current_bins_items) >= min_bins_solution_global:
        return

    item_to_place = items_to_pack[0]
    remaining_items_after_this = items_to_pack[1:]

    for i in range(len(current_bins_items)):
        if item_to_place <= current_bins_remaining_capacity[i]:
            current_bins_items[i].append(item_to_place)
            current_bins_remaining_capacity[i] -= item_to_place
            solve_bin_packing_optimal_recursive(remaining_items_after_this, bin_capacity, current_bins_items,
                                                current_bins_remaining_capacity)
            current_bins_remaining_capacity[i] += item_to_place
            current_bins_items[i].pop()

    # Try placing in a new bin
    if len(current_bins_items) + 1 < min_bins_solution_global:
        current_bins_items.append([item_to_place])
        current_bins_remaining_capacity.append(bin_capacity - item_to_place)
        solve_bin_packing_optimal_recursive(remaining_items_after_this, bin_capacity, current_bins_items,
                                            current_bins_remaining_capacity)
        current_bins_remaining_capacity.pop()
        current_bins_items.pop()


def optimal_bin_packing(items, bin_capacity):
    global min_bins_solution_global, best_bin_config_global
    min_bins_solution_global = float('inf')
    best_bin_config_global = []
    valid_items = sorted([item for item in items if 0 < item <= bin_capacity], reverse=True)
    if not valid_items:
        return []
    solve_bin_packing_optimal_recursive(valid_items, bin_capacity, [], [])
    return best_bin_config_global


# --- Data Generation for Optimal Algorithm ---
def generate_optimal_dataset(items_per_sample, min_item_size, max_item_size, bin_capacity):
    current_items = [random.randint(min_item_size, max_item_size) for _ in range(items_per_sample)]
    return list(current_items), bin_capacity


# --- Worker Function for a Single Run ---
def run_single(_):
    """
    Generates one dataset, runs optimal_bin_packing, and returns the duration.
    The underscore argument is just a placeholder to allow imap over a range.
    """
    ITEMS_PER_SAMPLE_OPTIMAL = 30  # Must match main settings
    MIN_ITEM_SIZE_GLOBAL = 1
    MAX_ITEM_SIZE_GLOBAL = 100
    BIN_CAPACITY_GLOBAL = 100

    items, bin_capacity = generate_optimal_dataset(
        ITEMS_PER_SAMPLE_OPTIMAL,
        MIN_ITEM_SIZE_GLOBAL,
        MAX_ITEM_SIZE_GLOBAL,
        BIN_CAPACITY_GLOBAL
    )

    start_time = time.perf_counter()
    optimal_bin_packing(items, bin_capacity)
    end_time = time.perf_counter()
    return end_time - start_time


# --- Main testing function ---
def test_algorithm_runtimes():
    num_runs = 10000  # Total number of runs
    cpu_count = os.cpu_count() or 1

    print(f"--- Starting {num_runs} Runs for Tail-Latency Measurement (using {cpu_count} cores) ---")
    print("ITEMS_PER_SAMPLE_OPTIMAL: 30")
    print(f"Total runs for optimal_bin_packing: {num_runs}\n")

    exact_times = []

    # Create a pool of worker processes equal to the number of CPU cores
    with Pool(processes=cpu_count) as pool:
        # Use imap to lazily evaluate durations and wrap with tqdm for progress
        for duration in tqdm(pool.imap(run_single, range(num_runs)), total=num_runs, desc="Running optimal_bin_packing"):
            exact_times.append(duration)

    # Convert to NumPy array and sort
    exact_times = np.array(exact_times)
    sorted_times = np.sort(exact_times)

    # --- Figure 1: All sorted runtimes ---
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_times, linewidth=1)
    plt.xlabel("Run Index (sorted)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Sorted Runtimes for optimal_bin_packing over 10000 Runs")
    plt.tight_layout()
    plt.savefig("sorted_runtimes.png", dpi=300)
    plt.close()

    # --- Figure 2: Tail (last 1%) of runs ---
    cutoff = int(0.99 * num_runs)  # drop the fastest 99%
    tail_times = sorted_times[cutoff:]

    plt.figure(figsize=(12, 6))
    plt.plot(tail_times, linewidth=1)
    plt.xlabel("Run Index (sorted, tail 1%)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Tail 1% Runtimes (excluding fastest 99%)")
    plt.tight_layout()
    plt.savefig("tail_runtimes.png", dpi=300)
    plt.close()

    # Summary
    p99 = sorted_times[int(0.99 * num_runs)]
    print(f"99th percentile runtime: {p99:.6f} seconds")
    print(f"Max runtime: {sorted_times[-1]:.6f} seconds")
    print("\nPlots saved as 'sorted_runtimes.png' and 'tail_runtimes.png'")


if __name__ == "__main__":
    test_algorithm_runtimes()

