import os
import random
import time
import numpy as np
from PIL import Image # Pillow for image manipulation
import matplotlib.pyplot as plt
from scipy import ndimage as ndi # For CPU Sobel

# Try to import CuPy and its ndimage equivalent, if not available, set a flag
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndi
    CUPY_AVAILABLE = True
    print("CuPy and cupyx.scipy.ndimage found. GPU tests will be performed.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy or cupyx.scipy.ndimage not found. GPU tests will be skipped. Only CPU performance will be measured.")

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
        return np.array(img_resized, dtype=np.float32) / 255.0 # Normalize for Sobel
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def run_edge_detection_cpu(image_data):
    """Performs Sobel edge detection on CPU and returns the execution time."""
    start_time = time.perf_counter()
    sx = ndi.sobel(image_data, axis=0, mode='reflect')
    sy = ndi.sobel(image_data, axis=1, mode='reflect')
    sobel_magnitude = np.hypot(sx, sy)
    end_time = time.perf_counter()
    return end_time - start_time

def run_edge_detection_gpu(image_data_np):
    """Performs Sobel edge detection on GPU and returns the execution time."""
    if not CUPY_AVAILABLE:
        return float('nan')

    image_data_gpu = cp.asarray(image_data_np) # Transfer NumPy array to GPU
    start_time = time.perf_counter()
    sx_gpu = cupy_ndi.sobel(image_data_gpu, axis=0, mode='reflect')
    sy_gpu = cupy_ndi.sobel(image_data_gpu, axis=1, mode='reflect')
    sobel_magnitude_gpu = cp.hypot(sx_gpu, sy_gpu)
    cp.cuda.Stream.null.synchronize() # Wait for GPU operations to complete
    end_time = time.perf_counter()
    return end_time - start_time

def main(folder_path):
    """Main function to run the Edge Detection (Sobel) comparison benchmark."""
    resize_params = [32, 64, 128, 256, 512, 1024] # Using your updated resize parameters
    num_total_images_to_select = 5 # Total images to pick, first one for warm-up per size

    if num_total_images_to_select <= 1:
        print("Error: 'num_total_images_to_select' must be greater than 1 to allow for warm-up and timed runs.")
        return

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

    print(f"Will use {actual_num_images} images for testing edge detection:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")
    print(f"For each size, the first image will be a warm-up run and its time will be discarded.")
    print(f"Timings will be based on the subsequent {num_timed_images_per_size} image(s) per size.")
    print("-" * 30)

    results_cpu = {size: 0.0 for size in resize_params}
    results_gpu = {size: 0.0 for size in resize_params}
    images_timed_count = {size: 0 for size in resize_params}

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

            time_cpu = run_edge_detection_cpu(image_data)
            time_gpu = run_edge_detection_gpu(image_data) 

            if i == 0: # Warm-up run
                gpu_warmup_time_str = f'{time_gpu:.6f}' if CUPY_AVAILABLE and not np.isnan(time_gpu) else 'N/A'
                print(f"    Warm-up run: CPU Edge Detection time: {time_cpu:.6f} s, GPU Edge Detection time: {gpu_warmup_time_str} s. (This run is not timed for totals)")
            else: # Timed runs
                total_time_cpu_for_size += time_cpu
                print(f"    Timed CPU Edge Detection time: {time_cpu:.6f} s")
                if CUPY_AVAILABLE:
                    if not np.isnan(time_gpu):
                        total_time_gpu_for_size += time_gpu
                        print(f"    Timed GPU Edge Detection time: {time_gpu:.6f} s")
                    else:
                        print(f"    Timed GPU Edge Detection: Skipped (CuPy error or N/A).")
                timed_images_for_this_size += 1
        
        if timed_images_for_this_size > 0:
            results_cpu[size] = total_time_cpu_for_size
            results_gpu[size] = total_time_gpu_for_size if CUPY_AVAILABLE and timed_images_for_this_size > 0 else float('nan')
            images_timed_count[size] = timed_images_for_this_size
            gpu_total_time_str = f"{results_gpu[size]:.6f}" if CUPY_AVAILABLE and not np.isnan(results_gpu[size]) else "N/A"
            print(f"Size {size}x{size} completed. Total CPU time: {results_cpu[size]:.6f} s, Total GPU time: {gpu_total_time_str} s (based on {timed_images_for_this_size} timed images)")
        else:
            print(f"Size {size}x{size}: No images were successfully timed (after warm-up).")
        print("-" * 30)

    print("\n--- Edge Detection Test Results Summary ---")
    print(f"{'Size':<10} | {'Total CPU Time (s)':<22} | {'Total GPU Time (s)':<22} | {'Timed Images':<15}")
    print("-" * 75)
    for size in resize_params:
        cpu_time_str = f"{results_cpu[size]:.6f}" if images_timed_count[size] > 0 else "N/A"
        
        if CUPY_AVAILABLE and images_timed_count[size] > 0 and not np.isnan(results_gpu[size]):
            gpu_time_str = f"{results_gpu[size]:.6f}"
        elif not CUPY_AVAILABLE and images_timed_count[size] > 0:
            gpu_time_str = "N/A (CuPy not found)"
        else:
            gpu_time_str = "N/A"

        print(f"{size}x{size:<6} | {cpu_time_str:<22} | {gpu_time_str:<22} | {images_timed_count[size]:<15}")

    plot_labels = []
    cpu_times_plot = []
    gpu_times_plot_final = [] # Renamed to avoid conflict in outer scope if script is run multiple times in a session

    for size in resize_params:
        if images_timed_count[size] > 0:
            plot_labels.append(f"{size}x{size}")
            cpu_times_plot.append(results_cpu[size])
            if CUPY_AVAILABLE and not np.isnan(results_gpu[size]):
                gpu_times_plot_final.append(results_gpu[size])
            else:
                # If GPU data is missing for a size that has CPU data, add NaN for plotting
                gpu_times_plot_final.append(np.nan)


    x = np.arange(len(plot_labels)) 

    if not plot_labels:
        print("\nNo data available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    if cpu_times_plot:
        ax.plot(x, cpu_times_plot, marker='o', linestyle='-', label='CPU Total Time (Sobel)')

    # Check if there's any valid (non-NaN) GPU data to plot
    if CUPY_AVAILABLE and any(not np.isnan(val) for val in gpu_times_plot_final):
        ax.plot(x, gpu_times_plot_final, marker='x', linestyle='--', label='GPU Total Time (Sobel)')
    elif CUPY_AVAILABLE: # CuPy available, but all GPU times might be NaN or no timed images
        print("\nNote: GPU data available but might be all NaN or no images timed for GPU successfully.")
    elif not CUPY_AVAILABLE and any(images_timed_count.values()): # No CuPy, but CPU data exists
        print("\nNote: CuPy not available. GPU data is not included in the plot.")

    ax.set_ylabel('Total Execution Time (seconds)')
    ax.set_xlabel('Image Size')
    ax.set_title(f'CPU vs GPU Sobel Edge Detection Time (Warm-up Discarded, Timed: {num_timed_images_per_size} images/size)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("edge_detection_performance_comparison_en.png")
    print("\nPlot saved as edge_detection_performance_comparison_en.png")
    plt.show() # Added to display the plot

if __name__ == "__main__":
    image_folder_path = "/home/sean/Omost/outputs_png" # Using the path from your script
    # If you want to use a generic test path:
    # image_folder_path = "./test_images_edge_detection"


    # Create a test folder and some random images if it doesn't exist (optional, if using a generic path)
    # if image_folder_path == "./test_images_edge_detection" and not os.path.exists(image_folder_path):
    #     print(f"Test folder '{image_folder_path}' does not exist. Creating it...")
    #     os.makedirs(image_folder_path)
    #     try:
    #         for i in range(10): 
    #             img_array_dummy = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    #             img_dummy = Image.fromarray(img_array_dummy)
    #             img_dummy.save(os.path.join(image_folder_path, f"random_image_edge_{i+1}.png"))
    #         print(f"Created 10 random images in '{image_folder_path}' for testing.")
    #     except Exception as e:
    #         print(f"Failed to create dummy test images: {e}")
    #         print("Please ensure you have write permissions or manually place some images in the folder.")

    if not os.path.exists(image_folder_path):
        print(f"Error: The folder '{image_folder_path}' does not exist. Please create it and add image files.")
        print("Exiting program.")
    elif not os.listdir(image_folder_path):
        print(f"Error: The folder '{image_folder_path}' is empty. Please add some image files to it.")
        print("Exiting program.")
    else:
        main(image_folder_path)