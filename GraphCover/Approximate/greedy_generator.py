import random
import numpy as np
import os
import datetime # For timestamp
import math
from greedy_algorithm import greedy_vertex_cover_2approx

# --- Parameters for Dataset Generation ---
fixed_num_vertices = 1000     # Fixed number of vertices
num_samples = 5           # Number of graph samples to generate
edge_probability = 0.1   # Probability for an edge to exist

# --- (generate_gnp_random_graph and greedy_vertex_cover_2approx functions are the same) ---
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



# --- Function to print a single sample (remains the same) ---
def print_single_sample_example(example_num_vertices=10, example_edge_probability=0.3):
    """
    Generates and prints a single small example of a graph, its adjacency matrix,
    and the vertex cover vector obtained by the greedy algorithm.
    """
    print("\n--- Generating a Single Small Example ---")
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

    # 2. Apply Greedy Algorithm
    cover_set = greedy_vertex_cover_2approx(example_num_vertices, edges)

    # 3. Create Cover Vector
    cover_vector = np.zeros(example_num_vertices, dtype=np.int8)
    for vertex_in_cover in cover_set:
        if vertex_in_cover < example_num_vertices:
            cover_vector[vertex_in_cover] = 1
    
    print("Output: Greedy Vertex Cover (Set S):")
    print(sorted(list(cover_set))) # Print the set of vertices in the cover
    print("\nOutput: Vertex Cover Vector (Y - Label for DL):")
    print(cover_vector)
    print("Total number of vertices marked in cover:", cover_vector.sum())
    print("--- End of Single Small Example ---\n")


# --- Modified main data generation function ---
def main_generate_consolidated_dataset():
    """
    Generates a dataset and saves it into two NumPy arrays:
    - input.npy: Contains all adjacency matrices. Shape: (num_samples, num_vertices, num_vertices)
    - label.npy: Contains all cover vectors. Shape: (num_samples, num_vertices)
    """


    # --- Output Directory ---
    # Create a base directory for the consolidated dataset, possibly with a timestamp
    # Or, you can use a fixed name like "generated_dataset"
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"vertex_cover_greedy_{timestamp_str}"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved in directory: {output_dir}")
    except OSError as e:
        print(f"Error creating directory: {e}")
        return

    print(f"Generating {num_samples} samples for the consolidated dataset...")
    print(f"Each graph will have {fixed_num_vertices} vertices.")
    
    all_inputs = []
    all_labels = []

    for i in range(num_samples):
        sample_idx = i + 1
        
        # Generate graph data
        _, edges = generate_gnp_random_graph(fixed_num_vertices, edge_probability) 
        
        # Create Adjacency Matrix
        adj_matrix = np.zeros((fixed_num_vertices, fixed_num_vertices), dtype=np.int8)
        for u, v in edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
        
        # Get Cover Vector using the approximate algorithm
        cover_set = greedy_vertex_cover_2approx(fixed_num_vertices, edges)
        cover_vector = np.zeros(fixed_num_vertices, dtype=np.int8)
        for vertex_in_cover in cover_set:
            if vertex_in_cover < fixed_num_vertices: # Should always be true if generated correctly
                cover_vector[vertex_in_cover] = 1
        
        all_inputs.append(adj_matrix)
        all_labels.append(cover_vector)

        # Progress update
        if sample_idx % (num_samples // 20 if num_samples >= 20 else 1) == 0 or sample_idx == num_samples:
            print(f"Generated sample {sample_idx}/{num_samples}...")
    
    # Convert lists of arrays to single large NumPy arrays
    # Input shape: (num_samples, fixed_num_vertices, fixed_num_vertices)
    # Label shape: (num_samples, fixed_num_vertices)
    input_dataset = np.array(all_inputs, dtype=np.int8)
    label_dataset = np.array(all_labels, dtype=np.int8)

    print(f"\nInput dataset shape: {input_dataset.shape}")
    print(f"Label dataset shape: {label_dataset.shape}")

    # Define file paths for the consolidated files
    input_npy_path = os.path.join(output_dir, "input.npy")
    label_npy_path = os.path.join(output_dir, "label.npy")

    try:
        np.save(input_npy_path, input_dataset)
        np.save(label_npy_path, label_dataset)
        print(f"\nConsolidated dataset saved successfully:")
        print(f"Inputs: {input_npy_path}")
        print(f"Labels: {label_npy_path}")
    except IOError as e:
        print(f"Error saving consolidated dataset: {e}")
    
    # Create a Readme file as suggested in the user's instructions
    readme_content = f"""Dataset generated by Python script.
Data type for inputs (adjacency matrices): numpy.int8
Shape for inputs: {input_dataset.shape} (num_samples, num_vertices, num_vertices)

Data type for labels (cover vectors): numpy.int8
Shape for labels: {label_dataset.shape} (num_samples, num_vertices)

Generation details:
- Number of samples: {num_samples}
- Number of vertices per graph: {fixed_num_vertices}
- Edge probability (G(n,p) model): {edge_probability}
- Algorithm for vertex cover: 2-approximation greedy algorithm.
"""
    readme_path = os.path.join(output_dir, "dataset_readme.txt")
    try:
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Readme file created: {readme_path}")
    except IOError as e:
        print(f"Error creating readme file: {e}")

    print(f"\nDataset generation complete.")


if __name__ == "__main__":
    # 1. Print a single small example to show the format
    # You can adjust parameters or comment this out if not needed for every run.
    print_single_sample_example(example_num_vertices=10, example_edge_probability=0.3)
    
    # 2. Proceed to generate the full consolidated dataset
    # The old function main_generate_for_dl_structured_timestamped() is replaced by this one.
    main_generate_consolidated_dataset()
