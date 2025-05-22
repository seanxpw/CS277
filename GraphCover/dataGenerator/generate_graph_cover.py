import random
import numpy as np
import os
import datetime # For timestamp
import math

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

def greedy_vertex_cover_2approx(num_vertices, edges):
    """
    Implements the 2-approximation greedy algorithm for Vertex Cover.
    Iterates through edges; if an edge is 'uncovered', add both its endpoints to the cover.
    Args:
        num_vertices (int): Number of vertices.
        edges (list of tuples): List of unique edges, e.g., [(0,1), (1,2)].
    Returns:
        set: The set of vertices in the cover.
    """
    vertex_cover_set = set()
    
    # The order of edges can affect the specific set S, but it's still a 2-approx.
    # We iterate through the provided list of edges.
    for u, v in edges:
        # If this edge is not yet covered by a vertex already in the cover set
        if not (u in vertex_cover_set or v in vertex_cover_set):
            vertex_cover_set.add(u)
            vertex_cover_set.add(v)
            
    return vertex_cover_set

# --- Function to print a single sample ---
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
    print("total numner of vertex marked",cover_vector.sum())
    print("--- End of Single Small Example ---\n")


# --- Main data generation function (remains the same) ---
def main_generate_for_dl_structured_timestamped():
    # --- Parameters for Dataset Generation ---
    fixed_num_vertices = 100    # Fixed number of vertices as requested
    num_samples = 5         # Number of graph samples to generate in this run
    edge_probability = 0.05   # Probability for an edge to exist

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"vertex_cover_dataset_{timestamp_str}"
    inputs_dir_path = os.path.join(base_output_dir, "inputs")
    labels_dir_path = os.path.join(base_output_dir, "labels")

    try:
        os.makedirs(inputs_dir_path, exist_ok=True)
        os.makedirs(labels_dir_path, exist_ok=True)
        print(f"Output will be saved in base directory: {base_output_dir}")
    except OSError as e:
        print(f"Error creating directories: {e}")
        return

    print(f"Generating {num_samples} samples for the main dataset...")
    print(f"Each graph will have {fixed_num_vertices} vertices.")
    
    zfill_width = len(str(num_samples))

    for i in range(num_samples):
        sample_idx = i + 1
        num_v_actual, edges = generate_gnp_random_graph(fixed_num_vertices, edge_probability) 
        adj_matrix = np.zeros((fixed_num_vertices, fixed_num_vertices), dtype=np.int8)
        for u, v in edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
        cover_set = greedy_vertex_cover_2approx(fixed_num_vertices, edges)
        cover_vector = np.zeros(fixed_num_vertices, dtype=np.int8)
        for vertex_in_cover in cover_set:
            if vertex_in_cover < fixed_num_vertices:
                cover_vector[vertex_in_cover] = 1
        
        sample_id_str = str(sample_idx).zfill(zfill_width)
        adj_matrix_filename = os.path.join(inputs_dir_path, f"sample_{sample_id_str}_adj.npy")
        cover_vector_filename = os.path.join(labels_dir_path, f"sample_{sample_id_str}_cover.npy")

        try:
            np.save(adj_matrix_filename, adj_matrix)
            np.save(cover_vector_filename, cover_vector)
        except IOError as e:
            print(f"Error saving sample {sample_id_str}: {e}")
            continue 

        if sample_idx % (num_samples // 20 if num_samples >= 20 else 1) == 0 or sample_idx == num_samples :
            print(f"Generated and saved main dataset sample {sample_idx}/{num_samples}...")
    
    print(f"\nMain dataset generation complete.")
    print(f"Input adjacency matrices saved in: {inputs_dir_path}")
    print(f"Output cover vectors saved in: {labels_dir_path}")


if __name__ == "__main__":
    # 1. Print a single small example to show the format
    print_single_sample_example(example_num_vertices=100, example_edge_probability=0.05 )
    
    # 2. Proceed to generate the full dataset as requested
    # You can comment out the line below if you ONLY want to see the example printout for now.
    main_generate_for_dl_structured_timestamped()
    
    # If you want to run both, uncomment the line above.
    # If you want to only run the main generation, comment out print_single_sample_example()
    # and uncomment main_generate_for_dl_structured_timestamped().

    # For now, let's assume you want to see the example and then generate a small dataset.
    # To generate a small dataset for quick testing along with the example:
    # Create a temporary main function call with fewer samples for testing if needed.
    # print("\nIf you want to generate the full dataset, please uncomment")
    # print("`main_generate_for_dl_structured_timestamped()` in the `if __name__ == \"__main__\":` block")
    # print("and run the script again.")