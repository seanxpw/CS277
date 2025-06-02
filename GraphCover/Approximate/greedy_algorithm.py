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
