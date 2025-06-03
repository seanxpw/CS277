import os
import random
import time
import numpy as np
from PIL import Image # Pillow for image manipulation
import matplotlib.pyplot as plt

# Try to import CuPy, if not available, set a flag
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy found. GPU tests will be performed.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. GPU tests will be skipped. Only CPU performance will be measured.")

def get_random_images(folder_path, num_images=5):
    """Selects a specified number of random image paths from a folder."""
    all_images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(valid_extensions):
            all_images.append(os.path.join(folder_path, fname))

    if not all_images:
        return []

    if len(all_images) < num_images:
        print(f"Warning: Number of images in folder ({len(all_images)}) is less than requested ({num_images}). Using all available images.")
        return all_images
    return random.sample(all_images, num_images)

def preprocess_image(image_path, target_size):
    """Loads an image, converts it to grayscale, and resizes it."""
    try:
        img = Image.open(image_path).convert('L') # 'L' for grayscale
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        return np.array(img_resized, dtype=np.float32)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def run_fft_cpu(image_data):
    """Performs FFT on CPU and returns the execution time."""
    start_time = time.perf_counter()
    fft_result_cpu = np.fft.fft2(image_data)
    np.fft.fftshift(fft_result_cpu) # Common practice to shift zero-frequency component to center
    end_time = time.perf_counter()
    return end_time - start_time

def run_fft_gpu(image_data_np):
    """Performs FFT on GPU and returns the execution time."""
    if not CUPY_AVAILABLE:
        return float('nan') # Return Not a Number if CuPy is not available

    image_data_gpu = cp.asarray(image_data_np) # Transfer NumPy array to GPU
    start_time = time.perf_counter()
    fft_result_gpu = cp.fft.fft2(image_data_gpu)
    cp.fft.fftshift(fft_result_gpu)
    cp.cuda.Stream.null.synchronize() # Wait for GPU operations to complete
    end_time = time.perf_counter()
    return end_time - start_time

def main(folder_path):
    """Main function to run the FFT comparison benchmark."""
    resize_params = [32, 64, 128,256,512,1024]
    num_total_images_to_select = 6 # Total images to pick, first one for warm-up per size

    if num_total_images_to_select <= 1:
        print("Error: 'num_total_images_to_select' must be greater than 1 to allow for warm-up and timed runs.")
        return

    # Get random images
    image_paths = get_random_images(folder_path, num_total_images_to_select)
    if not image_paths:
        print(f"Error: No images found in the specified folder: {folder_path}. Please check the path.")
        return
    
    actual_num_images = len(image_paths)
    if actual_num_images < num_total_images_to_select :
        print(f"Warning: Using {actual_num_images} images as fewer were available than requested ({num_total_images_to_select}).")
        if actual_num_images <= 1:
            print("Error: Not enough images available for warm-up and timed runs. At least 2 images are needed.")
            return
            
    num_timed_images_per_size = actual_num_images - 1


    print(f"Will use {actual_num_images} images for testing:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")
    print(f"For each size, the first image will be a warm-up run and its time will be discarded.")
    print(f"Timings will be based on the subsequent {num_timed_images_per_size} image(s) per size.")
    print("-" * 30)

    results_cpu = {size: 0.0 for size in resize_params}
    results_gpu = {size: 0.0 for size in resize_params}
    images_timed_count = {size: 0 for size in resize_params} # Count of images actually timed (excluding warm-up)

    for size in resize_params:
        print(f"Starting tests for image size: {size}x{size}")
        total_time_cpu_for_size = 0.0
        total_time_gpu_for_size = 0.0
        timed_images_for_this_size = 0

        for i, image_path in enumerate(image_paths):
            print(f"  Processing image: {os.path.basename(image_path)} (Size: {size}x{size})")
            image_data = preprocess_image(image_path, size)

            if image_data is None:
                print(f"    Skipping image {os.path.basename(image_path)} due to preprocessing failure.")
                continue

            # CPU FFT
            time_cpu = run_fft_cpu(image_data)
            # GPU FFT
            time_gpu = run_fft_gpu(image_data) # Will be NaN if CuPy not available

            if i == 0: # This is the warm-up run for this size
                print(f"    Warm-up run: CPU FFT time: {time_cpu:.6f} s, GPU FFT time: {(f'{time_gpu:.6f}' if CUPY_AVAILABLE else 'N/A')} s. (This run is not timed for totals)")
            else: # These are the timed runs
                total_time_cpu_for_size += time_cpu
                print(f"    Timed CPU FFT time: {time_cpu:.6f} s")
                if CUPY_AVAILABLE:
                    if not np.isnan(time_gpu):
                        total_time_gpu_for_size += time_gpu
                        print(f"    Timed GPU FFT time: {time_gpu:.6f} s")
                    else:
                         print(f"    Timed GPU FFT: Skipped (CuPy error during run).")
                timed_images_for_this_size += 1
        
        if timed_images_for_this_size > 0:
            results_cpu[size] = total_time_cpu_for_size
            results_gpu[size] = total_time_gpu_for_size if CUPY_AVAILABLE and timed_images_for_this_size >0 else float('nan')
            images_timed_count[size] = timed_images_for_this_size
            gpu_time_str = f"{results_gpu[size]:.6f}" if CUPY_AVAILABLE and not np.isnan(results_gpu[size]) else "N/A"
            print(f"Size {size}x{size} completed. Total CPU time: {results_cpu[size]:.6f} s, Total GPU time: {gpu_time_str} s (based on {timed_images_for_this_size} timed images)")
        else:
            print(f"Size {size}x{size}: No images were successfully timed (after warm-up).")
        print("-" * 30)

    # --- Results Summary and Plotting ---
    print("\n--- Test Results Summary ---")
    print(f"{'Size':<10} | {'Total CPU Time (s)':<22} | {'Total GPU Time (s)':<22} | {'Timed Images':<15}")
    print("-" * 75)
    for size in resize_params:
        cpu_time_str = f"{results_cpu[size]:.6f}" if images_timed_count[size] > 0 else "N/A"
        
        if CUPY_AVAILABLE and images_timed_count[size] > 0 and not np.isnan(results_gpu[size]):
            gpu_time_str = f"{results_gpu[size]:.6f}"
        elif not CUPY_AVAILABLE and images_timed_count[size] > 0:
             gpu_time_str = "N/A (CuPy not found)"
        else: # No timed images or CuPy error
            gpu_time_str = "N/A"

        print(f"{size}x{size:<6} | {cpu_time_str:<22} | {gpu_time_str:<22} | {images_timed_count[size]:<15}")

    # Plotting
    labels = [f"{s}x{s}" for s in resize_params]
    
    # Filter out sizes with no timed images for plotting
    plot_labels = []
    cpu_times_plot = []
    gpu_times_plot = []

    for size in resize_params:
        if images_timed_count[size] > 0:
            plot_labels.append(f"{size}x{size}")
            cpu_times_plot.append(results_cpu[size])
            if CUPY_AVAILABLE and not np.isnan(results_gpu[size]):
                gpu_times_plot.append(results_gpu[size])
            else:
                # Add a placeholder if GPU data is missing for a size that has CPU data, to keep alignment
                # Or, decide how to handle partially missing data in plot. For line plot, better to have corresponding points.
                # For simplicity, if GPU data is consistently NaN or unavailable, gpu_times_plot might be empty.
                # Matplotlib handles plotting an empty list gracefully (it just doesn't plot that line).
                pass # Let the check below handle it

    x = np.arange(len(plot_labels)) # X-axis label positions based on successfully timed sizes

    if not plot_labels:
        print("\nNo data available for plotting (all image processing might have failed or no images were timed).")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    if cpu_times_plot:
        ax.plot(x, cpu_times_plot, marker='o', linestyle='-', label='CPU Total Time')

    # Only plot GPU if CuPy was available AND we have GPU times
    valid_gpu_times_for_plot = [t for t in results_gpu.values() if not np.isnan(t) and t != 0.0] # Check if any valid GPU times exist
    
    # We need to reconstruct gpu_times_plot aligned with plot_labels
    gpu_plot_data_for_graph = []
    if CUPY_AVAILABLE:
        for size_label in plot_labels: # Iterate based on sizes that actually have data
            size_val = int(size_label.split('x')[0])
            if images_timed_count[size_val] > 0 and not np.isnan(results_gpu[size_val]):
                 gpu_plot_data_for_graph.append(results_gpu[size_val])
            else:
                 # If a CPU point exists for this label but GPU doesn't, we might add NaN or handle it.
                 # For a simple line plot, it's often better if both lines have points for the same x-values.
                 # However, if GPU failed for a specific size, that point will be missing.
                 # Let's ensure gpu_plot_data_for_graph has same length as cpu_times_plot if CUPY_AVAILABLE by adding NaNs
                 gpu_plot_data_for_graph.append(np.nan) # Matplotlib can skip NaN for line plots


    if CUPY_AVAILABLE and any(not np.isnan(val) for val in gpu_plot_data_for_graph): # check if there's any non-NaN GPU data to plot
        ax.plot(x, gpu_plot_data_for_graph, marker='x', linestyle='--', label='GPU Total Time')
    elif CUPY_AVAILABLE and not any(images_timed_count.values()):
         print("\nNote: GPU data is missing or insufficient for plotting (perhaps all image processing failed).")
    elif not CUPY_AVAILABLE and any(images_timed_count.values()):
        print("\nNote: CuPy not available. GPU data is not included in the plot.")


    ax.set_ylabel('Total Execution Time (seconds)')
    ax.set_xlabel('Image Size')
    ax.set_title(f'CPU vs GPU FFT Execution Time (Warm-up Discarded, Timed: {num_timed_images_per_size} images/size)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("fft_performance_comparison_en.png")
    print("\nPlot saved as fft_performance_comparison_en.png")
    # plt.show()

if __name__ == "__main__":
    # Please replace this with the path to your image folder
    image_folder_path = "/home/sean/Omost/outputs_png" # Example: "C:/Users/YourUser/Pictures/TestSet"

    # Create a test folder and some random images if it doesn't exist
    if not os.path.exists(image_folder_path):
        print(f"Test folder '{image_folder_path}' does not exist. Creating it...")
        os.makedirs(image_folder_path)
        try:
            # Create 10 random images for testing
            for i in range(10): 
                img_array_dummy = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
                img_dummy = Image.fromarray(img_array_dummy)
                img_dummy.save(os.path.join(image_folder_path, f"random_image_{i+1}.png"))
            print(f"Created 10 random images in '{image_folder_path}' for testing.")
        except Exception as e:
            print(f"Failed to create dummy test images: {e}")
            print("Please ensure you have write permissions or manually place some images in the folder.")

    if not os.listdir(image_folder_path):
        print(f"Error: The folder '{image_folder_path}' is empty. Please add some image files to it.")
        print("Exiting program.")
    else:
        main(image_folder_path)