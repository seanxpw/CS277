import random
import numpy as np
import os
import itertools # Required for combinations in the exact algorithm
import time
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting (saving to file)
import matplotlib.pyplot as plt

# --- Algorithm Implementations (remain the same) ---

def generate_gnp_random_graph(num_vertices, p_edge):
    edges = set()
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < p_edge:
                edges.add(tuple(sorted((i, j))))
    return num_vertices, list(edges)

def greedy_vertex_cover_2approx(num_vertices, edges):
    vertex_cover_set = set()
    # The order of edges can affect the specific set S, but it's still a 2-approx.
    # For timing consistency, we iterate through the provided list of edges as is.
    for u, v in edges:
        if not (u in vertex_cover_set or v in vertex_cover_set):
            vertex_cover_set.add(u)
            vertex_cover_set.add(v)
    return vertex_cover_set

def exact_vertex_cover_bruteforce(num_vertices, edges):
    nodes = list(range(num_vertices))
    if not edges: # If there are no edges, the empty set is the minimum vertex cover.
        return set()

    for k in range(num_vertices + 1): # k from 0 (empty set) to num_vertices (all nodes)
        for candidate_subset_tuple in itertools.combinations(nodes, k):
            candidate_subset = set(candidate_subset_tuple)
            is_cover = True
            if not candidate_subset and edges: # Optimization: empty set can't cover if edges exist
                is_cover = False
            else:
                for u, v in edges:
                    if u not in candidate_subset and v not in candidate_subset:
                        is_cover = False
                        break # This candidate is not a cover for this edge
            
            if is_cover:
                return candidate_subset
    return set(nodes) # Fallback, though a cover should always be found.

# --- Benchmarking Function ---
def run_benchmark_configured(p_edge_val):
    results = []
    n_current = 5
    
    MAX_SINGLE_RUN_TIME_SECONDS = 120.0  # 2 minutes
    NUM_RUNS_PER_N = 3 # Number of times to run each algorithm for averaging

    testing_exact = True
    testing_greedy = True

    print(f"Starting benchmark: p_edge = {p_edge_val}, {NUM_RUNS_PER_N} runs per (n, algorithm).")
    print(f"Max single run time for any algorithm: {MAX_SINGLE_RUN_TIME_SECONDS}s")

    while testing_greedy:
        if n_current > 5000 and not testing_exact:
             print(f"Stopping greedy test as n ({n_current}) has grown very large.")
             break
        
        print(f"\nTesting with n = {n_current} vertices...")
        actual_num_vertices, edges = generate_gnp_random_graph(n_current, p_edge_val)
        num_edges = len(edges)
        print(f"Generated graph with {actual_num_vertices} vertices and {num_edges} edges.")

        time_greedy_final = None
        time_exact_final = None

        # --- Greedy Algorithm Test ---
        if testing_greedy:
            current_greedy_times = []
            greedy_hit_timeout_this_n = False
            print(f"  Testing Greedy Algorithm ({NUM_RUNS_PER_N} runs):")
            for run_idx in range(NUM_RUNS_PER_N):
                try:
                    start_g = time.perf_counter()
                    greedy_vertex_cover_2approx(actual_num_vertices, edges)
                    end_g = time.perf_counter()
                    run_duration_g = end_g - start_g
                    print(f"    Greedy run {run_idx + 1}/{NUM_RUNS_PER_N}: {run_duration_g:.6f}s")

                    if run_duration_g > MAX_SINGLE_RUN_TIME_SECONDS:
                        print(f"    Greedy run {run_idx + 1}/{NUM_RUNS_PER_N} EXCEEDED time limit.")
                        time_greedy_final = float('inf')
                        testing_greedy = False # Stop all future greedy tests
                        greedy_hit_timeout_this_n = True
                        break 
                    current_greedy_times.append(run_duration_g)
                except Exception as e_g:
                    print(f"    Error in greedy run {run_idx + 1}/{NUM_RUNS_PER_N}: {e_g}")
                    time_greedy_final = float('inf')
                    testing_greedy = False
                    greedy_hit_timeout_this_n = True
                    break
            
            if not greedy_hit_timeout_this_n and current_greedy_times:
                time_greedy_final = sum(current_greedy_times) / len(current_greedy_times)
                print(f"  Greedy time for n={n_current} (avg of {len(current_greedy_times)} runs): {time_greedy_final:.6f}s")
            elif greedy_hit_timeout_this_n:
                 print(f"  Greedy algorithm recorded as timed out for n={n_current}.")
        
        # --- Exact Algorithm Test ---
        if testing_exact:
            current_exact_times = []
            exact_hit_timeout_this_n = False
            print(f"  Testing Exact Algorithm ({NUM_RUNS_PER_N} runs):")
            if n_current > 20: # Heuristic warning
                print(f"    Note: n={n_current} is large for brute-force. Expect long runtimes or timeout.")

            for run_idx in range(NUM_RUNS_PER_N):
                try:
                    start_e = time.perf_counter()
                    exact_vertex_cover_bruteforce(actual_num_vertices, edges)
                    end_e = time.perf_counter()
                    run_duration_e = end_e - start_e
                    print(f"    Exact run {run_idx + 1}/{NUM_RUNS_PER_N}: {run_duration_e:.6f}s")

                    if run_duration_e > MAX_SINGLE_RUN_TIME_SECONDS:
                        print(f"    Exact run {run_idx + 1}/{NUM_RUNS_PER_N} EXCEEDED time limit.")
                        time_exact_final = float('inf')
                        testing_exact = False # Stop all future exact tests
                        exact_hit_timeout_this_n = True
                        break
                    current_exact_times.append(run_duration_e)
                except KeyboardInterrupt:
                    print(f"    Exact run {run_idx + 1}/{NUM_RUNS_PER_N} manually interrupted.")
                    time_exact_final = float('inf')
                    testing_exact = False
                    exact_hit_timeout_this_n = True
                    break
                except Exception as e_e:
                    print(f"    Error in exact run {run_idx + 1}/{NUM_RUNS_PER_N}: {e_e}")
                    time_exact_final = float('inf')
                    testing_exact = False
                    exact_hit_timeout_this_n = True
                    break

            if not exact_hit_timeout_this_n and current_exact_times:
                time_exact_final = sum(current_exact_times) / len(current_exact_times)
                print(f"  Exact time for n={n_current} (avg of {len(current_exact_times)} runs): {time_exact_final:.6f}s")
            elif exact_hit_timeout_this_n:
                print(f"  Exact algorithm recorded as timed out for n={n_current}.")

        results.append({
            "n": n_current,
            "greedy_time": time_greedy_final,
            "exact_time": time_exact_final,
            "num_edges": num_edges
        })
        
        if not testing_greedy: # If greedy timed out or errored, stop all benchmarking
            break

        if testing_exact:
            n_current += 1
        else: # Exact is done, speed up n for greedy
            if n_current < 30: n_current += 2
            elif n_current < 50: n_current += 5
            elif n_current < 200: n_current += 20
            elif n_current < 1000: n_current += 100
            elif n_current < 2000: n_current += 200
            else: n_current += 500
                 
    return results

# --- Plotting Function ---
def plot_all_results(results, p_edge_val):
    if not results:
        print("No results to plot.")
        return

    n_values_all = np.array([r['n'] for r in results])
    greedy_times_all = np.array([r['greedy_time'] for r in results])
    exact_times_all = np.array([r['exact_time'] for r in results])

    epsilon = 1e-7 # Smallest time to plot on log scale if actual time is 0 or too small

    def prepare_plot_data(n_vals, t_vals):
        # Filter out None and inf before trying to plot
        valid_indices = np.where((t_vals != None) & (t_vals != np.inf) & (~np.isnan(t_vals)))
        plot_n = n_vals[valid_indices]
        # Ensure times are positive for log scale
        plot_t = np.maximum(t_vals[valid_indices].astype(float), epsilon) 
        return plot_n, plot_t

    plot_n_greedy, plot_t_greedy = prepare_plot_data(n_values_all, greedy_times_all)
    plot_n_exact, plot_t_exact = prepare_plot_data(n_values_all, exact_times_all)

    # Determine common minimum n for x-axis start if data exists
    min_n_for_plot = float('inf')
    if len(plot_n_greedy) > 0: min_n_for_plot = min(min_n_for_plot, plot_n_greedy[0])
    if len(plot_n_exact) > 0: min_n_for_plot = min(min_n_for_plot, plot_n_exact[0])
    if min_n_for_plot == float('inf'): min_n_for_plot = 0 # Default if no valid data

    # --- Plot 1: Exact Algorithm Focus ---
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    if len(plot_n_greedy) > 0:
        ax1.plot(plot_n_greedy, plot_t_greedy, marker='o', linestyle='-', label='Greedy Algorithm', zorder=2)
    if len(plot_n_exact) > 0:
        ax1.plot(plot_n_exact, plot_t_exact, marker='x', linestyle='--', label='Brute-Force Exact Algorithm', zorder=1)
    
    ax1.set_xlabel('Number of Vertices (n)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title(f'Performance (Exact Focus) vs. Graph Size (p_edge={p_edge_val})')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.7)
    ax1.set_yscale('log')
    if len(plot_n_exact) > 0:
        ax1.set_xlim(left=min_n_for_plot - 1 if min_n_for_plot > 0 else 0, right=max(plot_n_exact) + 2 if len(plot_n_exact) > 0 else n_current) # n_current could be last n
    else: # If no exact data, use greedy range or default
        ax1.set_xlim(left=min_n_for_plot -1 if min_n_for_plot > 0 else 0)

    plot_filename_exact = "performance_exact_focus.png"
    try:
        fig1.savefig(plot_filename_exact)
        print(f"\nPlot saved as {plot_filename_exact}")
    except Exception as e:
        print(f"Error saving exact focus plot: {e}")
    plt.close(fig1)

    # --- Plot 2: Greedy Algorithm Focus ---
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    if len(plot_n_greedy) > 0:
        ax2.plot(plot_n_greedy, plot_t_greedy, marker='o', linestyle='-', label='Greedy Algorithm', zorder=2)
    if len(plot_n_exact) > 0: # Exact data might be shorter
        ax2.plot(plot_n_exact, plot_t_exact, marker='x', linestyle='--', label='Brute-Force Exact Algorithm', zorder=1)

    ax2.set_xlabel('Number of Vertices (n)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title(f'Performance (Greedy Focus) vs. Graph Size (p_edge={p_edge_val})')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.7)
    ax2.set_yscale('log')
    if len(plot_n_greedy) > 0: # Full range for greedy
         ax2.set_xlim(left=min_n_for_plot - 1 if min_n_for_plot > 0 else 0) 
         # Let matplotlib decide the upper x_limit based on greedy data unless very specific need

    plot_filename_greedy = "performance_greedy_focus.png"
    try:
        fig2.savefig(plot_filename_greedy)
        print(f"Plot saved as {plot_filename_greedy}")
    except Exception as e:
        print(f"Error saving greedy focus plot: {e}")
    plt.close(fig2)

# --- Main Execution Block ---
if __name__ == "__main__":
    P_EDGE_BENCHMARK = 0.4
    benchmark_data = run_benchmark_configured(P_EDGE_BENCHMARK)
    
    if benchmark_data:
        plot_all_results(benchmark_data, P_EDGE_BENCHMARK)
    else:
        print("Benchmark did not produce data. No plot generated.")
    
    print("\nBenchmark complete.")