# Bin Packing Datasets for Approximation and Optimal Solutions 📦

This document describes two datasets generated for the Bin Packing Problem (BPP). The goal of the BPP is to pack a set of items of given sizes into the minimum number of bins, each with a fixed capacity.

These datasets are designed for tasks such as training machine learning models to predict bin packing configurations or to learn the behavior of specific packing algorithms.

## Datasets Overview 📊

Two distinct datasets are provided:

1.  **`Bin Packing FF (First Fit) Dataset`**: Generated using the **First Fit** heuristic algorithm. This is an approximate but fast algorithm.
    * **Location**: `Bin-Packing/Approx/bin_packing_ff_dataset_20250601_195502/`
2.  **`Bin Packing Optimal Dataset`**: Generated using a **backtracking algorithm** that finds the **optimal (minimum) number of bins**. This algorithm is exact but computationally intensive, hence typically used for smaller problem instances.
    * **Location**: `Bin-Packing/Optimal/bin_packing_optimal_dataset_20250601_200335/`

Both datasets share a common structure for their input and label files, detailed below.

Some details can be found at `Bin-Packing/Approx/bin_packing_ff_dataset_20250601_195502/dataset_readme.txt` and `/home/csgrads/xwang605/CS277/Bin-Packing/Optimal/bin_packing_optimal_dataset_20250601_200335/dataset_readme.txt`
---

## File Structure and Format 📁

Each dataset directory contains two NumPy binary files (`.npy`):

* `input.npy`: Contains the input data (item sizes for each sample).
* `label.npy`: Contains the corresponding labels (bin packing configurations).

### 1. `input.npy` (Input Data)

* **Description**: This file stores the item sizes for each problem instance (sample).
* **Data Type**: `int16`
* **Shape**: `(num_samples, items_per_sample)`
    * `num_samples`: The total number of problem instances in the dataset.
    * `items_per_sample`: The number of items in each problem instance. This value differs between the FF and Optimal datasets due to the computational demands of the optimal solver.
        * For FF Dataset: `(1000, 200)` (1000 samples, each with 200 items)
        * For Optimal Dataset: `(1000, 30)` (1000 samples, each with 30 items)
* **Content**: Each row `input[i]` is a 1D array representing the sizes of items for the `i`-th sample.
    * Example: `input[0] = [item1_size, item2_size, ..., itemN_size]`

### 2. `label.npy` (Label Data - Bin Configuration)

* **Description**:  This file stores the bin packing configuration for each corresponding input sample, as determined by either the First Fit algorithm or the Optimal (backtracking) algorithm. **for each sample, the result is represented by a fixed size 2D matrix. each row represents a bin, with in a row (bin) the elements are the weight. At first all elements are 0 meaning no items filled in any bins, with the progress of bin-packing, some 0s will be repalced by weights. You can calculated the number of non-zero rows to determine how many bins are used.**
* **Data Type**: `int16`
* **Shape**: `(num_samples, max_bins, max_items_per_bin)`
    * For both FF and Optimal Datasets: `(1000, 100, 100)`
        * `num_samples`: The total number of problem instances (1000).
        * `max_bins`: The maximum number of bins that can be represented in the label matrix (fixed at 100, calculated as `bin_capacity`).
        * `max_items_per_bin`: The maximum number of items that can be represented within a single bin in the label matrix (fixed at 100, calculated as `bin_capacity // min_item_size`).
* **Content**: Each `label[i]` is a 2D matrix representing the bin packing for the `i`-th sample.
    * `label[i, j, k] = item_size`: The `k`-th item (slot) in the `j`-th bin for the `i`-th sample has size `item_size`.
    * A value of `0` indicates an empty item slot or an entirely unused bin (if all slots in a bin row are 0).
* **Interpretation**:
    * To find the items in the first bin of the first sample: `label[0, 0, :]`. Non-zero values are the item sizes.
    * **Number of Bins Used**: For a given sample `s`, the number of bins actually used can be determined by finding the highest bin index `j` for which `label[s, j, :]` contains at least one non-zero item. Alternatively, count the number of rows `label[s, j, :]` that are not all zeros up to the actual number of bins used by the algorithm (which is generally much less than `max_bins`).

---

## Dataset Generation Parameters ⚙️

These parameters were used to generate the datasets and are crucial for understanding their characteristics:

Adjust them in `main`

**Common Parameters for both datasets:**

* **Min Item Size**: 1
* **Max Item Size**: 100
* **Bin Capacity**: 100
* **Label Matrix Dimensions (Calculated)**:
    * `max_bins` (rows): 100 (derived from `bin_capacity`)
    * `max_items_per_bin` (columns): 100 (derived from `bin_capacity // min_item_size`)

**Specific Parameters:**

* **First Fit (FF) Dataset**:
    * Number of Samples: 1000
    * Items per Sample: 200
    * Algorithm: First Fit heuristic.
* **Optimal Dataset**:
    * Number of Samples: 1000
    * Items per Sample: 30 (kept small due to the high computational cost of the exact algorithm).
    * Algorithm: Backtracking to find the optimal (minimum) number of bins.

**Important Notes on Label Matrix Truncation**:

* The `label.npy` matrix has fixed dimensions (`100x100` for bins and items-per-bin).
* If an algorithm (theoretically) uses more than 100 bins for a sample, or if a single bin contains more than 100 items, the representation in `label.npy` will be truncated.
    * Given `Bin Capacity = 100` and `Min Item Size = 1`, a single bin cannot physically hold more than 100 items of `min_item_size`. So, the `max_items_per_bin = 100` dimension is generally sufficient.
    * The number of bins used is typically much less than 100, especially for the `Optimal Dataset` with only 30 items per sample (max 30 bins). For the `FF Dataset` with 200 items, it's also unlikely to exceed 100 bins with a capacity of 100 unless item sizes are pathologically small and numerous (which is not the case here, as total items are 200).

---
