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

# --- Data Generation for First Fit ---
def generate_ff_dataset(num_samples, items_per_sample, min_item_size, max_item_size, bin_capacity):
    # Calculate label matrix dimensions based on current parameters
    label_matrix_rows = bin_capacity 
    label_matrix_cols = bin_capacity // min_item_size if min_item_size > 0 else bin_capacity # Avoid division by zero

    print(f"\nGenerating dataset for First Fit (Approximate)...")
    print(f"Parameters: num_samples={num_samples}, items_per_sample={items_per_sample}, bin_capacity={bin_capacity}")
    print(f"Calculated label matrix dimensions per sample: {label_matrix_rows} bins x {label_matrix_cols} items/bin")

    all_inputs = []
    all_labels_matrix = [] # Stores the 2D matrices

    for i in range(num_samples):
        current_items = [random.randint(min_item_size, max_item_size) for _ in range(items_per_sample)]
        input_vector = np.array(current_items, dtype=np.int16) 
        all_inputs.append(input_vector)

        _, bins_config_ff = first_fit(current_items, bin_capacity)
        label_matrix = convert_bins_to_fixed_matrix(bins_config_ff, label_matrix_rows, label_matrix_cols)
        all_labels_matrix.append(label_matrix)

        if (i + 1) % (num_samples // 10 if num_samples >= 10 else 1) == 0:
            print(f"  Generated sample {i+1}/{num_samples}")

    input_dataset = np.array(all_inputs)
    label_dataset = np.array(all_labels_matrix, dtype=np.int16) # This will be a 3D array

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bin_packing_ff_dataset_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)

    input_npy_path = os.path.join(output_dir, "input.npy")
    label_npy_path = os.path.join(output_dir, "label.npy")

    np.save(input_npy_path, input_dataset)
    np.save(label_npy_path, label_dataset)

    print(f"First Fit dataset saved successfully in {output_dir}:")
    print(f"  Inputs shape: {input_dataset.shape}")
    print(f"  Labels shape: {label_dataset.shape} (num_samples, calculated_max_bins, calculated_max_items_per_bin)")
    
    readme_content = f"""Bin Packing Dataset (First Fit Algorithm)
- input.npy: Item sizes for each sample. Shape: {input_dataset.shape} (num_samples, items_per_sample)
             Data type: int16
- label.npy: Bin packing configuration for each sample using First Fit.
             Shape: {label_dataset.shape} (num_samples, {label_matrix_rows}, {label_matrix_cols})
             Data type: int16.
             Interpretation: label[sample_idx, bin_idx, item_slot_idx] = item_size. Zeros indicate empty slots/bins.
             The number of bins used for a sample can be found by counting non-empty rows (rows with at least one non-zero item).

Dataset Generation Parameters:
- Number of samples: {num_samples}
- Items per sample: {items_per_sample}
- Min item size: {min_item_size}
- Max item size: {max_item_size}
- Bin capacity: {bin_capacity}
- Label matrix dimensions: {label_matrix_rows} (max bins, calculated as bin_capacity) x {label_matrix_cols} (max items per bin, calculated as bin_capacity // min_item_size)
"""
    with open(os.path.join(output_dir, "dataset_readme.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Readme file created in {output_dir}")
    return input_dataset, label_dataset

# --- Optimal Bin Packing (Backtracking) ---
min_bins_solution_global = float('inf')
best_bin_config_global = []


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

    # --- Dataset 1: First Fit (Approximate) ---
    NUM_SAMPLES_FF = 1000 # Can be larger for FF
    ITEMS_PER_SAMPLE_FF = 200 
    
    print("Starting First Fit dataset generation...")
    generate_ff_dataset(
        num_samples=NUM_SAMPLES_FF,
        items_per_sample=ITEMS_PER_SAMPLE_FF,
        min_item_size=MIN_ITEM_SIZE_GLOBAL,
        max_item_size=MAX_ITEM_SIZE_GLOBAL,
        bin_capacity=BIN_CAPACITY_GLOBAL
    )

