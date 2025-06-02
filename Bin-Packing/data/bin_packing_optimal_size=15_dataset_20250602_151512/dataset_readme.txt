Bin Packing Dataset (Optimal_size=15 Algorithm)
- input.npy: Item sizes for each sample. Shape: (2000, 15) (num_samples, items_per_sample)
             Data type: int16. Items generated from a mix of distributions (Uniform, Zipf Low, Zipf High, Bimodal).
- label.npy: Number of bins used by the Optimal_size=15 algorithm for each sample. Shape: (2000,)
             Data type: int16

Dataset Generation Parameters:
- Total Number of samples: 2000 (distributed among 4 item generation strategies)
- Items per sample: 15
- Min item size: 1
- Max item size: 100
- Bin capacity: 100
