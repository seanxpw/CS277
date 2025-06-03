import random
import numpy as np
import itertools  # Required for combinations in the exact algorithm
import time  # For timing

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
    if len(current_bins_items) >= min_bins_solution_global: return

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
    if not valid_items: return []
    solve_bin_packing_optimal_recursive(valid_items, bin_capacity, [], [])
    return best_bin_config_global


# --- First Fit Algorithm (Approximate) ---
def first_fit(items, bin_capacity):
    """
    Packs items into bins using the First Fit algorithm.

    Args:
        items (list of int): Sizes of items to pack.
        bin_capacity (int): Capacity of each bin.

    Returns:
        int: Number of bins used. (Still useful for direct comparison)
        list of list of int: The actual bins with items.
    """
    if bin_capacity <= 0:
        return 0, []

    bins_content = []
    bin_remaining_capacity = []
    valid_items = [item for item in items if 0 < item <= bin_capacity]

    for item_size in valid_items:
        placed = False
        for i in range(len(bins_content)):
            if item_size <= bin_remaining_capacity[i]:
                bins_content[i].append(item_size)
                bin_remaining_capacity[i] -= item_size
                placed = True
                break
        if not placed:
            bins_content.append([item_size])
            bin_remaining_capacity.append(bin_capacity - item_size)

    return len(bins_content), bins_content


# --- Data Generation for Optimal Algorithm ---
def generate_optimal_dataset(items_per_sample, min_item_size, max_item_size, bin_capacity):
    current_items = [random.randint(min_item_size, max_item_size) for _ in range(items_per_sample)]
    return list(current_items), bin_capacity


# --- Main testing function ---
def test_algorithm_runtimes():
    # --- Common Configuration Parameters ---
    num_runs = 6 # 1 initial run + 5 for averaging
    # These are the single source of truth for configuring a run of this script.
    BIN_CAPACITY_GLOBAL = 100
    # MIN_ITEM_SIZE_GLOBAL must be > 0 for the calculation of label_matrix_cols.
    MIN_ITEM_SIZE_GLOBAL = 1
    # MAX_ITEM_SIZE_GLOBAL should be <= BIN_CAPACITY_GLOBAL for items to be packable.
    MAX_ITEM_SIZE_GLOBAL = 100

    if MIN_ITEM_SIZE_GLOBAL <= 0:
        raise ValueError("MIN_ITEM_SIZE_GLOBAL must be greater than 0.")

    # --- Dataset 2: Optimal (Backtracking/Exact) ---
    ITEMS_PER_SAMPLE_OPTIMAL = 30  # CRITICAL: Keep very small (e.g., 5-7 for reasonable time)

    exact_times = []
    greedy_times = []

    print(f"--- Starting Runtime Comparison ---")
    print(f"ITEMS_PER_SAMPLE_OPTIMAL: {ITEMS_PER_SAMPLE_OPTIMAL}")
    print(f"Total runs for each algorithm: {num_runs}\n")

    for i in range(num_runs):
        print(f"--- Run {i + 1}/{num_runs} ---")

        # Generate a new random graph for each run
        items, bin_capacity = generate_optimal_dataset(ITEMS_PER_SAMPLE_OPTIMAL, MIN_ITEM_SIZE_GLOBAL,
                                                       MAX_ITEM_SIZE_GLOBAL, BIN_CAPACITY_GLOBAL)
        print(f"Running binpack with {len(items)} items and {bin_capacity} capacity.")

        # Time exact_vertex_cover_bruteforce
        start_time = time.perf_counter()
        exact_cover = optimal_bin_packing(items, bin_capacity)
        end_time = time.perf_counter()
        duration_exact = end_time - start_time
        exact_times.append(duration_exact)
        print(f"Exact Brute-Force: Time = {duration_exact:.6f}s, Bins needed = {len(exact_cover)}")

        # Time greedy_vertex_cover_2approx
        start_time = time.perf_counter()
        greedy_cover = first_fit(items, bin_capacity)
        end_time = time.perf_counter()
        duration_greedy = end_time - start_time
        greedy_times.append(duration_greedy)
        print(f"Greedy First-Fit: Time = {duration_greedy:.6f}s, Bins needed = {len(greedy_cover)}")
        print("-" * 20)

    # Calculate averages, discarding the first run
    if num_runs > 1:
        avg_exact_time = np.mean(exact_times[1:])
        avg_greedy_time = np.mean(greedy_times[1:])

        print(f"\n--- Results (Average of last {num_runs - 1} runs) ---")
        print(f"Average Exact Brute-Force Time: {avg_exact_time:.6f}s")
        print(f"Average Greedy First-Fit Time:   {avg_greedy_time:.6f}s")
    elif num_runs == 1:
        print(f"\n--- Results (Single Run) ---")
        print(f"Exact Brute-Force Time: {exact_times[0]:.6f}s")
        print(f"Greedy First-Fit Time:   {greedy_times[0]:.6f}s")
    else:
        print("\nNo runs performed.")

    print(f"\nRaw times for Exact Algorithm: {[f'{t:.6f}' for t in exact_times]}")
    print(f"Raw times for Greedy Algorithm: {[f'{t:.6f}' for t in greedy_times]}")


if __name__ == "__main__":
    test_algorithm_runtimes()
