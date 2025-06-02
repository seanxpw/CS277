import random
import numpy as np
import os
import datetime
import itertools # For optimal algorithm (though current backtracking doesn't directly use it)

# --- Item Generation Functions ---

def generate_items_uniform(num_items, min_val, max_val, **kwargs): # Added **kwargs for consistent signature
    """Generates items with sizes from a uniform distribution."""
    return [random.randint(min_val, max_val) for _ in range(num_items)]

def generate_items_zipf_low(num_items, min_val, max_val, zipf_param=2.0, **kwargs):
    """Generates items skewed towards min_val using Zipf."""
    items = []
    for _ in range(num_items):
        # np.random.zipf(a) generates integers k >= 1 with P(k) ~ 1/k^a
        # We want to map these so that k=1 (most probable for a>1) corresponds to min_val
        raw_zipf_val = np.random.zipf(zipf_param)
        item = min_val + raw_zipf_val - 1 # if raw_zipf_val=1, item=min_val
        item = min(item, max_val) # Clip to max_val
        # Ensure it's not less than min_val (already handled by starting at min_val)
        items.append(int(item))
    return items

def generate_items_zipf_high(num_items, min_val, max_val, zipf_param=2.0, **kwargs):
    """Generates items skewed towards max_val using Zipf."""
    items = []
    for _ in range(num_items):
        raw_zipf_val = np.random.zipf(zipf_param)
        # We want raw_zipf_val=1 to map to max_val
        item = max_val - (raw_zipf_val - 1) # if raw_zipf_val=1, item=max_val
        item = max(item, min_val) # Clip to min_val
        items.append(int(item))
    return items

def generate_items_bimodal(num_items, min_val, max_val, capacity, **kwargs): # Added capacity as it might be relevant
    """Generates items from a bimodal distribution (small or large items)."""
    items = []
    # Define 'small' and 'large' ranges
    # Let small items be in the first quarter of the [min_val, max_val] range
    # Let large items be in the last quarter of the [min_val, max_val] range
    span = max_val - min_val
    if span < 3: # If range is too small, behave like uniform
        return generate_items_uniform(num_items, min_val, max_val)

    low_peak_end = min_val + span // 4
    high_peak_start = max_val - span // 4
    
    # Ensure peaks don't cross and are valid
    if low_peak_end < min_val : low_peak_end = min_val
    if high_peak_start > max_val : high_peak_start = max_val
    if low_peak_end >= high_peak_start: # If ranges overlap or are invalid, use simpler split
        low_peak_end = min_val + span // 2
        high_peak_start = low_peak_end + 1 if low_peak_end < max_val else max_val


    for _ in range(num_items):
        if random.random() < 0.5: # 50% small items
            item = random.randint(min_val, low_peak_end)
        else: # 50% large items
            item = random.randint(high_peak_start, max_val)
        items.append(item)
    return items


# --- First Fit Algorithm (Approximate) ---
def first_fit(items, bin_capacity):
    if bin_capacity <= 0:
        return 0 # Number of bins
    # bins_content = [] # We don't need to store content if only count is needed for label
    bin_remaining_capacity = []
    num_bins_used = 0
    valid_items = [item for item in items if 0 < item <= bin_capacity]

    for item_size in valid_items:
        placed = False
        for i in range(num_bins_used): # Iterate only over currently used bins
            if item_size <= bin_remaining_capacity[i]:
                # bins_content[i].append(item_size) # Not needed for count
                bin_remaining_capacity[i] -= item_size
                placed = True
                break
        if not placed:
            num_bins_used += 1
            # bins_content.append([item_size]) # Not needed for count
            bin_remaining_capacity.append(bin_capacity - item_size)
    return num_bins_used


# --- Optimal Bin Packing (Backtracking) ---
min_bins_solution_global = float('inf')
# best_bin_config_global is no longer needed if we only return the count

def solve_bin_packing_optimal_recursive(items_to_pack, bin_capacity, current_bins_items_len, current_bins_remaining_capacity): # Pass len instead of items
    global min_bins_solution_global

    if not items_to_pack:
        # num_bins_used = len(current_bins_items) # Now use current_bins_items_len
        if current_bins_items_len < min_bins_solution_global:
            min_bins_solution_global = current_bins_items_len
            # No need to store best_bin_config_global
        return

    # Pruning 1
    if current_bins_items_len >= min_bins_solution_global:
        return

    item_to_place = items_to_pack[0]
    remaining_items_after_this = items_to_pack[1:]

    # Option 1: Try existing bins
    for i in range(current_bins_items_len): # Use current_bins_items_len
        if item_to_place <= current_bins_remaining_capacity[i]:
            # current_bins_items[i].append(item_to_place) # Not tracking content
            current_bins_remaining_capacity[i] -= item_to_place
            
            solve_bin_packing_optimal_recursive(remaining_items_after_this, bin_capacity, 
                                                current_bins_items_len, # Length doesn't change
                                                current_bins_remaining_capacity)
            
            current_bins_remaining_capacity[i] += item_to_place # Backtrack
            # current_bins_items[i].pop() # Not tracking content

    # Option 2: New bin
    # Pruning 2
    if current_bins_items_len + 1 < min_bins_solution_global:
        # current_bins_items.append([item_to_place]) # Not tracking content
        current_bins_remaining_capacity.append(bin_capacity - item_to_place) # Add new capacity
        
        solve_bin_packing_optimal_recursive(remaining_items_after_this, bin_capacity, 
                                            current_bins_items_len + 1, # Length increases
                                            current_bins_remaining_capacity)
        
        current_bins_remaining_capacity.pop() # Backtrack: remove capacity of new bin
        # current_bins_items.pop() # Not tracking content


def optimal_bin_packing(items, bin_capacity):
    global min_bins_solution_global
    min_bins_solution_global = float('inf') # Reset for each new problem instance
    
    valid_items = sorted([item for item in items if 0 < item <= bin_capacity], reverse=True)
    if not valid_items:
        return 0 

    solve_bin_packing_optimal_recursive(valid_items, bin_capacity, 0, []) # Initial: 0 bins, empty capacities list
    
    return min_bins_solution_global if min_bins_solution_global != float('inf') else len(valid_items) # Fallback for safety


# --- Generic Dataset Generation Function ---
def generate_dataset(
    dataset_name_prefix, 
    algorithm_func, # e.g., first_fit or optimal_bin_packing
    num_total_samples, 
    items_per_sample, 
    min_item_size, 
    max_item_size, 
    bin_capacity
    ):

    print(f"\nGenerating dataset for {dataset_name_prefix}...")
    print(f"Total Samples: {num_total_samples}, Items/Sample: {items_per_sample}, Bin Capacity: {bin_capacity}")
    if algorithm_func == optimal_bin_packing:
        print("WARNING: Optimal algorithm is VERY SLOW. Ensure items_per_sample is small.")

    all_inputs = []
    all_labels = []

    dist_generators = [
        ("Uniform", generate_items_uniform),
        ("Zipf_Low (more small items)", generate_items_zipf_low),
        ("Zipf_High (more large items)", generate_items_zipf_high),
        ("Bimodal (small and large items)", generate_items_bimodal)
    ]
    
    num_dist_types = len(dist_generators)
    samples_per_dist_type = num_total_samples // num_dist_types
    remaining_samples = num_total_samples % num_dist_types # Distribute any remainder

    sample_counter = 0
    for i_dist, (dist_name, generator_func) in enumerate(dist_generators):
        current_samples_for_this_dist = samples_per_dist_type + (1 if i_dist < remaining_samples else 0)
        if current_samples_for_this_dist == 0:
            continue

        print(f"  Generating {current_samples_for_this_dist} samples using {dist_name} distribution...")
        for i in range(current_samples_for_this_dist):
            # For bimodal, pass capacity. For others, it will be ignored by **kwargs
            current_items = generator_func(items_per_sample, min_item_size, max_item_size, capacity=bin_capacity)
            
            input_vector = np.array(current_items, dtype=np.int16)
            all_inputs.append(input_vector)

            num_bins = algorithm_func(current_items, bin_capacity)
            all_labels.append(num_bins)
            
            sample_counter += 1
            if sample_counter % (num_total_samples // 20 if num_total_samples >=20 else 1) == 0 or sample_counter == num_total_samples :
                 print(f"    Processed sample {sample_counter}/{num_total_samples} (Current dist: {i+1}/{current_samples_for_this_dist})")
            if algorithm_func == optimal_bin_packing and items_per_sample > 7: # Progress for slow optimal
                 if i % (current_samples_for_this_dist // 5 if current_samples_for_this_dist >=5 else 1) == 0 : # More frequent for optimal
                      print(f"      Optimal sample {i+1}/{current_samples_for_this_dist} done for {dist_name}.")


    input_dataset = np.array(all_inputs)
    label_dataset = np.array(all_labels, dtype=np.int16)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"bin_packing_{dataset_name_prefix.lower().replace(' ', '_')}_dataset_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)

    input_npy_path = os.path.join(output_dir, "input.npy")
    label_npy_path = os.path.join(output_dir, "label.npy")

    np.save(input_npy_path, input_dataset)
    np.save(label_npy_path, label_dataset)

    print(f"{dataset_name_prefix} dataset saved successfully in {output_dir}:")
    print(f"  Inputs shape: {input_dataset.shape}")
    print(f"  Labels shape: {label_dataset.shape} (number of bins used)")

    readme_content = f"""Bin Packing Dataset ({dataset_name_prefix} Algorithm)
- input.npy: Item sizes for each sample. Shape: {input_dataset.shape} (num_samples, items_per_sample)
             Data type: int16. Items generated from a mix of distributions (Uniform, Zipf Low, Zipf High, Bimodal).
- label.npy: Number of bins used by the {dataset_name_prefix} algorithm for each sample. Shape: {label_dataset.shape}
             Data type: int16

Dataset Generation Parameters:
- Total Number of samples: {num_total_samples} (distributed among 4 item generation strategies)
- Items per sample: {items_per_sample}
- Min item size: {min_item_size}
- Max item size: {max_item_size}
- Bin capacity: {bin_capacity}
"""
    with open(os.path.join(output_dir, "dataset_readme.txt"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Readme file created in {output_dir}")
    return input_dataset, label_dataset

# --- Main Execution Block ---
if __name__ == "__main__":
    BIN_CAPACITY_GLOBAL = 100
    MIN_ITEM_SIZE_GLOBAL = 1 
    MAX_ITEM_SIZE_GLOBAL = 100 
    
    if MIN_ITEM_SIZE_GLOBAL <= 0:
        raise ValueError("MIN_ITEM_SIZE_GLOBAL must be greater than 0.")

    TOTAL_SAMPLES = 2000 # Total samples, will be divided among 4 distributions

    # --- Dataset 1: First Fit (Approximate) ---
    ITEMS_PER_SAMPLE_FF = 10 # User specified 30 items in their example for optimal, let's keep FF similar or larger
    
    generate_dataset(
        dataset_name_prefix=f"FirstFit_size={ITEMS_PER_SAMPLE_FF}",
        algorithm_func=first_fit,
        num_total_samples=TOTAL_SAMPLES,
        items_per_sample=ITEMS_PER_SAMPLE_FF,
        min_item_size=MIN_ITEM_SIZE_GLOBAL,
        max_item_size=MAX_ITEM_SIZE_GLOBAL,
        bin_capacity=BIN_CAPACITY_GLOBAL
    )

    # --- Dataset 2: Optimal (Backtracking/Exact) ---
    ITEMS_PER_SAMPLE_OPTIMAL = ITEMS_PER_SAMPLE_FF # CRITICAL: Keep very small (e.g., 6-8 for reasonable time)
                                 # For 30 items, this would be EXTREMELY slow.
                                 # User reported 30 was fast; if that was a bug and now it's slow, this number is key.
    
    generate_dataset(
        dataset_name_prefix=f"Optimal_size={ITEMS_PER_SAMPLE_OPTIMAL}",
        algorithm_func=optimal_bin_packing,
        num_total_samples=TOTAL_SAMPLES, # Generate fewer samples for optimal due to time
                                              # e.g. if TOTAL_SAMPLES=2000, then 200 optimal samples
        items_per_sample=ITEMS_PER_SAMPLE_OPTIMAL, 
        min_item_size=MIN_ITEM_SIZE_GLOBAL,
        max_item_size=MAX_ITEM_SIZE_GLOBAL, 
        bin_capacity=BIN_CAPACITY_GLOBAL
    )

    print("\nAll dataset generation tasks complete.")