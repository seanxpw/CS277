import random
import numpy as np
import os
import datetime # For timestamp
import math
import itertools # Required for combinations in the exact algorithm
from BruteForceAlgorithm import exact_vertex_cover_bruteforce

# --- Parameters for Dataset Generation ---
# Keep fixed_num_vertices VERY SMALL for the exact algorithm (e.g., 8-12)
# For N=10, 2^10=1024. For N=15, 2^15=32768. For N=20, 2^20 > 1 million combinations to check.
fixed_num_vertices = 20    # CRITICAL: Keep this small!
num_samples = 200           # Number of graph samples to generate
edge_probability = 0.1   # Probability for an edge to exist


# --- Graph Generation Function (from your previous code) ---
def generate_gnp_random_graph(num_vertices, p_edge):
    """
    Generates a G(n,p) random graph.
    Returns the number of vertices and a list of unique edges.
    Vertices are 0-indexed.
    """
    edges = set() # Use a set to ensure unique edges directly
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices): # Avoid self-loops and duplicate edges by construction
            if random.random() < p_edge:
                edges.add(tuple(sorted((i, j)))) # Store edges in a canonical form
    return num_vertices, list(edges)


# --- Function to print a single small example using the EXACT algorithm ---
def print_single_sample_example_exact(example_num_vertices=6, example_edge_probability=0.4):
    """
    Generates and prints a single small example of a graph, its adjacency matrix,
    and the OPTIMAL vertex cover vector obtained by the brute-force algorithm.
    """
    print("\n--- Generating a Single Small Example (Exact Algorithm) ---")
    if example_num_vertices > 15: # Warning for larger N with brute force
        print(f"WARNING: example_num_vertices ({example_num_vertices}) is large for brute-force. This might take a long time.")
    print(f"Number of vertices for example: {example_num_vertices}")
    print(f"Edge probability for example: {example_edge_probability}\n")

    # Generate graph
    num_v_actual, edges = generate_gnp_random_graph(example_num_vertices, example_edge_probability)

    print(f"Generated Graph ({num_v_actual} vertices):")
    if not edges:
        print("No edges were generated.")
    else:
        print("Edges (0-indexed):")
        for edge in sorted(list(edges)): # Print sorted edges for consistency
            print(f"  {edge}")
    print("-" * 30)

    # 1. Create Adjacency Matrix
    adj_matrix = np.zeros((example_num_vertices, example_num_vertices), dtype=np.int8)
    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    print("Input: Adjacency Matrix (X):")
    print(adj_matrix)
    print("-" * 30)

    # 2. Apply Exact Algorithm
    print("Calculating exact vertex cover (this may take time for larger graphs)...")
    cover_set = exact_vertex_cover_bruteforce(example_num_vertices, edges)
    print("Exact calculation complete.")

    # 3. Create Cover Vector
    cover_vector = np.zeros(example_num_vertices, dtype=np.int8)
    for vertex_in_cover in cover_set:
        if vertex_in_cover < example_num_vertices:
            cover_vector[vertex_in_cover] = 1
    
    print("Output: Optimal Vertex Cover (Set):")
    print(sorted(list(cover_set))) 
    print(f"Size of optimal cover: {len(cover_set)}")
    print("\nOutput: Optimal Vertex Cover Vector (Y - Label for DL):")
    print(cover_vector)
    print("Total number of vertices marked in optimal cover:", cover_vector.sum())
    print("--- End of Single Small Example (Exact Algorithm) ---\n")


# --- Main data generation function for EXACT algorithm ---
def main_generate_consolidated_dataset_exact():
    """
    Generates a dataset using the EXACT (brute-force) vertex cover algorithm.
    Saves into two NumPy arrays:
    - input.npy: Contains all adjacency matrices.
    - label.npy: Contains all OPTIMAL cover vectors.
    
    WARNING: This is computationally very expensive. Use with small num_vertices.
    """


    print(f"WARNING: Generating dataset with EXACT algorithm for {fixed_num_vertices} vertices.")
    print("This can be very time-consuming per sample.")
    # if fixed_num_vertices > 12:
    #     proceed = input(f"Number of vertices is {fixed_num_vertices}, which is high for brute-force. Proceed? (yes/no): ")
    #     if proceed.lower() != 'yes':
    #         print("Dataset generation aborted by user.")
    #         return
            
    # --- Output Directory ---
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"vertex_cover_btute_force_{timestamp_str}"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved in directory: {output_dir}")
    except OSError as e:
        print(f"Error creating directory: {e}")
        return

    print(f"Generating {num_samples} samples for the exact dataset...")
    print(f"Each graph will have {fixed_num_vertices} vertices.")
    
    all_inputs = []
    all_labels = []

    for i in range(num_samples):
        sample_idx = i + 1
        print(f"Generating sample {sample_idx}/{num_samples} (N={fixed_num_vertices})...")
        
        _, edges = generate_gnp_random_graph(fixed_num_vertices, edge_probability) 
        
        adj_matrix = np.zeros((fixed_num_vertices, fixed_num_vertices), dtype=np.int8)
        for u, v in edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
        
        # Get OPTIMAL Cover Vector using the brute-force algorithm
        print(f"  Calculating exact cover for sample {sample_idx}...")
        cover_set = exact_vertex_cover_bruteforce(fixed_num_vertices, edges)
        print(f"  Exact cover found for sample {sample_idx}, size: {len(cover_set)}")
        
        cover_vector = np.zeros(fixed_num_vertices, dtype=np.int8)
        for vertex_in_cover in cover_set:
            if vertex_in_cover < fixed_num_vertices:
                cover_vector[vertex_in_cover] = 1
        
        all_inputs.append(adj_matrix)
        all_labels.append(cover_vector)
    
    input_dataset = np.array(all_inputs, dtype=np.int8)
    label_dataset = np.array(all_labels, dtype=np.int8)

    print(f"\nInput dataset shape: {input_dataset.shape}")
    print(f"Label dataset shape: {label_dataset.shape}")

    input_npy_path = os.path.join(output_dir, "input.npy")
    label_npy_path = os.path.join(output_dir, "label.npy")

    try:
        np.save(input_npy_path, input_dataset)
        np.save(label_npy_path, label_dataset)
        print(f"\nConsolidated EXACT dataset saved successfully:")
        print(f"Inputs: {input_npy_path}")
        print(f"Labels: {label_npy_path}")
    except IOError as e:
        print(f"Error saving consolidated dataset: {e}")
    
    readme_content = f"""Dataset generated by Python script using an EXACT brute-force algorithm for Vertex Cover.
Data type for inputs (adjacency matrices): numpy.int8
Shape for inputs: {input_dataset.shape} (num_samples, num_vertices, num_vertices)

Data type for labels (OPTIMAL cover vectors): numpy.int8
Shape for labels: {label_dataset.shape} (num_samples, num_vertices)

Generation details:
- Number of samples: {num_samples}
- Number of vertices per graph: {fixed_num_vertices}
- Edge probability (G(n,p) model): {edge_probability}
- Algorithm for vertex cover: Exact brute-force algorithm (guarantees minimum vertex cover).
"""
    readme_path = os.path.join(output_dir, "dataset_readme.txt")
    try:
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Readme file created: {readme_path}")
    except IOError as e:
        print(f"Error creating readme file: {e}")

    print(f"\nExact dataset generation complete.")


if __name__ == "__main__":
    # 1. Print a single small example to show the exact algorithm's output
    # Adjust parameters as needed, keep example_num_vertices small
    print_single_sample_example_exact(example_num_vertices=7, example_edge_probability=0.5)
    
    # 2. Proceed to generate the full dataset using the EXACT algorithm
    # Ensure fixed_num_vertices in main_generate_consolidated_dataset_exact() is small!
    # You can comment this out if you only want to see the example.
    main_generate_consolidated_dataset_exact()
