Bin Packing Dataset (First Fit Algorithm)
- input.npy: Item sizes for each sample. Shape: (1000, 200) (num_samples, items_per_sample)
             Data type: int16
- label.npy: Bin packing configuration for each sample using First Fit.
             Shape: (1000, 100, 100) (num_samples, 100, 100)
             Data type: int16.
             Interpretation: label[sample_idx, bin_idx, item_slot_idx] = item_size. Zeros indicate empty slots/bins.
             The number of bins used for a sample can be found by counting non-empty rows (rows with at least one non-zero item).

Dataset Generation Parameters:
- Number of samples: 1000
- Items per sample: 200
- Min item size: 1
- Max item size: 100
- Bin capacity: 100
- Label matrix dimensions: 100 (max bins, calculated as bin_capacity) x 100 (max items per bin, calculated as bin_capacity // min_item_size)
