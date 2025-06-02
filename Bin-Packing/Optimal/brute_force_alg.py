import random
import numpy as np
import os
import datetime
import itertools # Not directly used in this backtracking version, but often relevant

# --- Helper function to convert bin configuration to a fixed-size matrix ---
def convert_bins_to_fixed_matrix(bins_content, label_matrix_rows, label_matrix_cols):
    """
    Converts a list of bins (each bin is a list of item sizes) into a fixed-size 2D NumPy matrix.

    Args:
        bins_content (list of list of int): The actual bins with items.
        label_matrix_rows (int): The first dimension of the output matrix (max number of bins).
        label_matrix_cols (int): The second dimension of the output matrix (max items per bin).

    Returns:
        np.ndarray: A 2D matrix representing the bin packing, padded with zeros.
    """
    label_matrix = np.zeros((label_matrix_rows, label_matrix_cols), dtype=np.int16)
    for bin_idx, current_bin_items in enumerate(bins_content):
        if bin_idx >= label_matrix_rows:
            # This means the algorithm used more bins than we allocated for in the label matrix.
            # Data for bins beyond label_matrix_rows will be truncated.
            # print(f"Warning: Actual number of bins ({len(bins_content)}) exceeds "
            #       f"label matrix dimension ({label_matrix_rows}). Truncating bins.")
            break
        for item_idx, item_size in enumerate(current_bin_items):
            if item_idx >= label_matrix_cols:
                # This means a bin contains more items than we allocated slots for.
                # Data for items beyond label_matrix_cols in this bin will be truncated.
                # print(f"Warning: Bin {bin_idx+1} has {len(current_bin_items)} items, "
                #       f"exceeds label matrix dimension ({label_matrix_cols}). Truncating items in this bin.")
                break
            label_matrix[bin_idx, item_idx] = item_size
    return label_matrix


# --- Optimal Bin Packing (Backtracking) ---
min_bins_solution_global = float('inf')
best_bin_config_global = []

def solve_bin_packing_optimal_recursive(items_to_pack, bin_capacity, current_bins_items, current_bins_remaining_capacity):
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
            solve_bin_packing_optimal_recursive(remaining_items_after_this, bin_capacity, current_bins_items, current_bins_remaining_capacity)
            current_bins_remaining_capacity[i] += item_to_place
            current_bins_items[i].pop()

    if len(current_bins_items) + 1 < min_bins_solution_global:
        current_bins_items.append([item_to_place])
        current_bins_remaining_capacity.append(bin_capacity - item_to_place)
        solve_bin_packing_optimal_recursive(remaining_items_after_this, bin_capacity, current_bins_items, current_bins_remaining_capacity)
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

# --- Data Generation for Optimal Algorithm ---
def generate_optimal_dataset(num_samples, items_per_sample, min_item_size, max_item_size, bin_capacity):
    # Calculate label matrix dimensions based on current parameters
    label_matrix_rows = bin_capacity
    label_matrix_cols = bin_capacity // min_item_size if min_item_size > 0 else bin_capacity

    print(f"\nGenerating dataset for Optimal Bin Packing (Backtracking)...")
    print(f"WARNING: This is VERY SLOW. Use small num_samples and items_per_sample.")
    print(f"Parameters: num_samples={num_samples}, items_per_sample={items_per_sample}, bin_capacity={bin_capacity}")
    print(f"Calculated label matrix dimensions per sample: {label_matrix_rows} bins x {label_matrix_cols} items/bin")

    all_inputs = []
    all_labels_matrix = []

    for i in range(num_samples):
        current_items = [random.randint(min_item_size, max_item_size) for _ in range(items_per_sample)]
        input_vector = np.array(current_items, dtype=np.int16)
        all_inputs.append(input_vector)

        print(f"  Processing sample {i+1}/{num_samples} for optimal solution (items: {items_per_sample})...")
        optimal_bin_configuration = optimal_bin_packing(list(current_items), bin_capacity)
        label_matrix = convert_bins_to_fixed_matrix(optimal_bin_configuration, label_matrix_rows, label_matrix_cols)
        all_labels_matrix.append(label_matrix)
        print(f"    Optimal bins for sample {i+1}: {len(optimal_bin_configuration)} bins. Matrix generated.")

    input_dataset = np.array(all_inputs)
    label_dataset = np.array(all_labels_matrix, dtype=np.int16) # This will be a 3D array

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bin_packing_optimal_dataset_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)

    input_npy_path = os.path.join(output_dir, "input.npy")
    label_npy_path = os.path.join(output_dir, "label.npy")

    np.save(input_npy_path, input_dataset)
    np.save(label_npy_path, label_dataset)

    print(f"Optimal dataset saved successfully in {output_dir}:")
    print(f"  Inputs shape: {input_dataset.shape}")
    print(f"  Labels shape: {label_dataset.shape} (num_samples, calculated_max_bins, calculated_max_items_per_bin)")

    readme_content = f"""Bin Packing Dataset (Optimal Algorithm - Backtracking)
- input.npy: Item sizes for each sample. Shape: {input_dataset.shape} (num_samples, items_per_sample)
             Data type: int16
- label.npy: Optimal bin packing configuration for each sample.
             Shape: {label_dataset.shape} (num_samples, {label_matrix_rows}, {label_matrix_cols})
             Data type: int16.
             Interpretation: label[sample_idx, bin_idx, item_slot_idx] = item_size. Zeros indicate empty slots/bins.
             The number of bins used for a sample can be found by counting non-empty rows.

Dataset Generation Parameters:
- Number of samples: {num_samples}
- Items per sample: {items_per_sample}
- Min item size: {min_item_size}
- Max item size: {max_item_size}
- Bin capacity: {bin_capacity}
- Algorithm: Backtracking to find the optimal bin configuration.
- Label matrix dimensions: {label_matrix_rows} (max bins, calculated as bin_capacity) x {label_matrix_cols} (max items per bin, calculated as bin_capacity // min_item_size)
"""
    with open(os.path.join(output_dir, "dataset_readme.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Readme file created in {output_dir}")
    return input_dataset, label_dataset

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Common Configuration Parameters ---
    # These are the single source of truth for configuring a run of this script.
    BIN_CAPACITY_GLOBAL = 100
    # MIN_ITEM_SIZE_GLOBAL must be > 0 for the calculation of label_matrix_cols.
    MIN_ITEM_SIZE_GLOBAL = 1 
    # MAX_ITEM_SIZE_GLOBAL should be <= BIN_CAPACITY_GLOBAL for items to be packable.
    MAX_ITEM_SIZE_GLOBAL = 100 
    
    if MIN_ITEM_SIZE_GLOBAL <= 0:
        raise ValueError("MIN_ITEM_SIZE_GLOBAL must be greater than 0.")

    # --- Dataset 2: Optimal (Backtracking/Exact) ---
    NUM_SAMPLES_OPTIMAL = 1000
    ITEMS_PER_SAMPLE_OPTIMAL = 30 # CRITICAL: Keep very small (e.g., 5-7 for reasonable time)
    
    print("Starting Optimal dataset generation...")
    print("WARNING: Optimal dataset generation is EXTREMELY SLOW for 'ITEMS_PER_SAMPLE_OPTIMAL' > 7-8.")
    print(f"Current ITEMS_PER_SAMPLE_OPTIMAL = {ITEMS_PER_SAMPLE_OPTIMAL}")
    
    generate_optimal_dataset(
        num_samples=NUM_SAMPLES_OPTIMAL,
        items_per_sample=ITEMS_PER_SAMPLE_OPTIMAL, 
        min_item_size=MIN_ITEM_SIZE_GLOBAL,
        max_item_size=MAX_ITEM_SIZE_GLOBAL, 
        bin_capacity=BIN_CAPACITY_GLOBAL
    )

    print("\nAll dataset generation tasks complete.")
    print(f"Reminder: Label matrices dimensions are dynamically calculated based on "
          f"bin_capacity ({BIN_CAPACITY_GLOBAL}) and min_item_size ({MIN_ITEM_SIZE_GLOBAL}).")
    print("If actual bins/items exceed these calculated dimensions, data is truncated in the label matrix.")