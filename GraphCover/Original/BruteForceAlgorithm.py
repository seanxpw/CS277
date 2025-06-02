
import random
import numpy as np
import os
import datetime # For timestamp
import math
import itertools # Required for combinations in the exact algorithm
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

    for k in range(1, num_vertices + 1): # Iterate from cover size 1 up to num_vertices
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
                
    return set(nodes) # Should ideally be caught by k=num_vertices if graph is not empty and has edges.
                      # Or if graph has no edges, handled at the start.
