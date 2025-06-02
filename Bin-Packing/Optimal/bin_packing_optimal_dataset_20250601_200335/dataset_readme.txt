Bin Packing Dataset (Optimal Algorithm - Backtracking)
- input.npy: Item sizes for each sample. Shape: (1000, 30) (num_samples, items_per_sample)
             Data type: int16
- label.npy: Optimal bin packing configuration for each sample.
             Shape: (1000, 100, 100) (num_samples, 100, 100)
             Data type: int16.
             Interpretation: label[sample_idx, bin_idx, item_slot_idx] = item_size. Zeros indicate empty slots/bins.
             The number of bins used for a sample can be found by counting non-empty rows.

Dataset Generation Parameters:
- Number of samples: 1000
- Items per sample: 30
- Min item size: 1
- Max item size: 100
- Bin capacity: 100
- Algorithm: Backtracking to find the optimal bin configuration.
- Label matrix dimensions: 100 (max bins, calculated as bin_capacity) x 100 (max items per bin, calculated as bin_capacity // min_item_size)
