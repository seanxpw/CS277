import random
import numpy as np
import itertools # Required for combinations in the exact algorithm
import time # For timing

# --- Exact Vertex Cover Algorithm (Brute-Force) ---
def exact_vertex_cover_bruteforce(num_vertices, edges):
    """
    Finds a minimum vertex cover using a brute-force approach.
    Iterates through all possible subsets of vertices by increasing size.
    
    Args:
        num_vertices (int): Number of vertices.
        edges (list of tuples): List of unique edges, e.g., [(0,1), (1,2)].
        
    Returns:
        set: The set of vertices in a minimum vertex cover.
    """
    nodes = list(range(num_vertices))

    if not edges: # If there are no edges, the empty set is the minimum vertex cover.
        return set()

    for k in range(num_vertices + 1): # Iterate from cover size 0 up to num_vertices (size 0 if no edges)
                                      # Changed from range(1,...) to range(...) to handle k=0 for graph with no edges
        for candidate_subset_tuple in itertools.combinations(nodes, k):
            candidate_subset = set(candidate_subset_tuple)
            is_cover = True
            # Check if this candidate_subset covers all edges
            for u, v in edges:
                if u not in candidate_subset and v not in candidate_subset:
                    is_cover = False
                    break # This candidate is not a cover
            
            if is_cover:
                # Since we are iterating k (size of subset) in increasing order,
                # the first valid cover found will be of minimum size.
                return candidate_subset
                
    return set(nodes) # Should not be reached if graph has edges due to k=num_vertices case
                      # or handled by k=0 if no edges.

# --- Greedy 2-Approximation Algorithm for Vertex Cover ---
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
    
    # Create a copy of edges to modify if needed, or ensure iteration order doesn't matter for logic
    # For this algorithm, the order of edges can affect the specific set S, 
    # but it's still a 2-approximation. We iterate through the provided list of edges.
    current_edges = list(edges) # Iterate over a copy or ensure no modification

    for u, v in current_edges: # Iterating through the original list of edges
        # If this edge is not yet covered by a vertex already in the cover set
        if not (u in vertex_cover_set or v in vertex_cover_set):
            vertex_cover_set.add(u)
            vertex_cover_set.add(v)
            
    return vertex_cover_set

# --- Graph Generation ---
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

# --- Main testing function ---
def test_algorithm_runtimes():
    num_vertices_test = 20
    num_runs = 6 # 1 initial run + 5 for averaging
    edge_probability_test = 0.1 # Probability for an edge to exist, adjust as needed

    exact_times = []
    greedy_times = []

    print(f"--- Starting Runtime Comparison ---")
    print(f"Number of vertices: {num_vertices_test}")
    print(f"Edge probability: {edge_probability_test}")
    print(f"Total runs for each algorithm: {num_runs}\n")

    for i in range(num_runs):
        print(f"--- Run {i+1}/{num_runs} ---")
        
        # Generate a new random graph for each run
        actual_num_v, edges = generate_gnp_random_graph(num_vertices_test, edge_probability_test)
        print(f"Generated graph with {actual_num_v} vertices and {len(edges)} edges.")

        if not edges and actual_num_v > 0:
            print("Graph has no edges. Both algorithms should be very fast.")
        elif actual_num_v == 0:
            print("Graph has no vertices. Skipping timing for this run.")
            # Optionally add placeholder times or skip appending if this case is problematic
            exact_times.append(0.0)
            greedy_times.append(0.0)
            continue


        # Time exact_vertex_cover_bruteforce
        start_time = time.perf_counter()
        exact_cover = exact_vertex_cover_bruteforce(actual_num_v, edges)
        end_time = time.perf_counter()
        duration_exact = end_time - start_time
        exact_times.append(duration_exact)
        print(f"Exact Brute-Force: Time = {duration_exact:.6f}s, Cover size = {len(exact_cover)}")

        # Time greedy_vertex_cover_2approx
        start_time = time.perf_counter()
        greedy_cover = greedy_vertex_cover_2approx(actual_num_v, edges)
        end_time = time.perf_counter()
        duration_greedy = end_time - start_time
        greedy_times.append(duration_greedy)
        print(f"Greedy 2-Approx: Time = {duration_greedy:.6f}s, Cover size = {len(greedy_cover)}")
        print("-" * 20)

    # Calculate averages, discarding the first run
    if num_runs > 1:
        avg_exact_time = np.mean(exact_times[1:])
        avg_greedy_time = np.mean(greedy_times[1:])
        
        print(f"\n--- Results (Average of last {num_runs-1} runs) ---")
        print(f"Average Exact Brute-Force Time: {avg_exact_time:.6f}s")
        print(f"Average Greedy 2-Approx Time:   {avg_greedy_time:.6f}s")
    elif num_runs == 1:
        print(f"\n--- Results (Single Run) ---")
        print(f"Exact Brute-Force Time: {exact_times[0]:.6f}s")
        print(f"Greedy 2-Approx Time:   {greedy_times[0]:.6f}s")
    else:
        print("\nNo runs performed.")
        
    print(f"\nRaw times for Exact Algorithm: {[f'{t:.6f}' for t in exact_times]}")
    print(f"Raw times for Greedy Algorithm: {[f'{t:.6f}' for t in greedy_times]}")

if __name__ == "__main__":
    test_algorithm_runtimes()