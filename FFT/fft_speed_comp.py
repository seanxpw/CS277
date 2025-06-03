import os
import random
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F # Required by UNetCorrected's UpCorrected

# --- UNet Model Definition ---
# (假设这个 UNet 模型定义在一个名为 unet_model.py 的文件中，或者你可以直接粘贴到这里)
# 为了这个例子的完整性，我将 UNetCorrected 的定义包含进来。
# 如果你已经有 unet_model.py，可以改为 from unet_model import UNetCorrected

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class UpCorrected(nn.Module):
    """Upscaling then double conv - corrected channel logic"""
    def __init__(self, in_ch_up, in_ch_skip, out_ch_conv, bilinear=True):
        super().__init__()
        self.out_ch_conv = out_ch_conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch_up + in_ch_skip, out_ch_conv) # Input channels = upsampled + skip
        else:
            # ConvTranspose such that its output channels match out_ch_conv for the block
            # but its input channels match in_ch_up.
            # Often, ConvT is designed to output in_ch_up // 2 or out_ch_conv directly.
            # Let's assume ConvT outputs out_ch_conv directly for simplicity here if it's not halving.
            # More common: ConvTranspose2d(in_ch_up, out_ch_conv, kernel_size=2, stride=2)
            # This makes features from upsampling path have `out_ch_conv` channels.
            self.up = nn.ConvTranspose2d(in_ch_up, out_ch_conv, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_ch_conv + in_ch_skip, out_ch_conv)

    def forward(self, x_up, x_skip): # x_up from lower layer, x_skip from encoder
        x_up = self.up(x_up)
        diffY = x_skip.size()[2] - x_up.size()[2]
        diffX = x_skip.size()[3] - x_up.size()[3]
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetCorrected(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):
        super(UNetCorrected, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) 

        self.up1 = UpCorrected(1024, 512, 512, bilinear)
        self.up2 = UpCorrected(512, 256, 256, bilinear)
        self.up3 = UpCorrected(256, 128, 128, bilinear)
        self.up4 = UpCorrected(128, 64, 64, bilinear)
        
        self.outc = OutConv(64, n_channels_out)
        # For latency test, final activation might not be strictly necessary,
        # but if the model was designed with it, keep it.
        # If the output is meant to be FFT magnitude (non-negative), Tanh might not be ideal
        # unless target is normalized to [-1,1]. Sigmoid for [0,1] or linear output.
        # For now, let's assume linear output for latency, or keep Tanh if it was part of your design.
        # self.final_activation = nn.Tanh() # Or nn.Sigmoid() or None

    def forward(self, x):
        x1_skip = self.inc(x)
        x2_skip = self.down1(x1_skip)
        x3_skip = self.down2(x2_skip)
        x4_skip = self.down3(x3_skip)
        x_bottle = self.down4(x4_skip)
        
        x = self.up1(x_bottle, x4_skip)
        x = self.up2(x, x3_skip)
        x = self.up3(x, x2_skip)
        x = self.up4(x, x1_skip)
        
        logits = self.outc(x)
        # if hasattr(self, 'final_activation') and self.final_activation is not None:
        #     return self.final_activation(logits)
        return logits # Linear output for latency test seems fine
# --- End of UNet Model Definition ---


# --- CUPY SETUP ---
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy found. GPU FFT tests (with CuPy) will be performed.")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found. GPU FFT tests (with CuPy) will be skipped.")

# --- PYTORCH DEVICE SETUP ---
PT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch will use device: {PT_DEVICE} for NN latency tests.")


def get_random_images(folder_path, num_images=5):
    all_images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(valid_extensions):
            all_images.append(os.path.join(folder_path, fname))
    if not all_images: return []
    if len(all_images) < num_images:
        print(f"Warning: Images in folder ({len(all_images)}) < requested ({num_images}). Using all.")
        return all_images
    return random.sample(all_images, num_images)

def preprocess_image(image_path, target_size):
    try:
        img = Image.open(image_path).convert('L') # Grayscale
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        return np.array(img_resized, dtype=np.float32) # For FFT and NN input
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def run_fft_cpu(image_data):
    start_time = time.perf_counter()
    _ = np.fft.fft2(image_data) # Result not needed for latency
    # np.fft.fftshift(fft_result_cpu)
    end_time = time.perf_counter()
    return end_time - start_time

def run_fft_gpu(image_data_np):
    if not CUPY_AVAILABLE:
        return float('nan')
    image_data_gpu = cp.asarray(image_data_np)
    start_time = time.perf_counter()
    _ = cp.fft.fft2(image_data_gpu) # Result not needed
    # cp.fft.fftshift(fft_result_gpu)
    cp.cuda.Stream.null.synchronize()
    end_time = time.perf_counter()
    return end_time - start_time

def run_nn_inference_pytorch(model, image_data_np, device):
    """Performs NN inference using PyTorch and returns execution time."""
    # Input for PyTorch model: [batch_size, channels, height, width]
    # Grayscale, so 1 channel. Batch size 1 for latency testing.
    # image_data_np is HxW float32
    input_tensor = torch.from_numpy(image_data_np).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        start_time = time.perf_counter()
        _ = model(input_tensor) # We don't need the output for latency
        if device.type == 'cuda':
            torch.cuda.synchronize() # Important for accurate GPU timing
        end_time = time.perf_counter()
    return end_time - start_time


def main(image_folder_path):
    resize_params = [32, 64, 128, 256, 512, 1024]
    num_total_images_to_select = 6 

    if num_total_images_to_select <= 1:
        print("Error: 'num_total_images_to_select' must be > 1 for warm-up and timed runs.")
        return

    image_paths = get_random_images(image_folder_path, num_total_images_to_select)
    if not image_paths:
        print(f"Error: No images found in {image_folder_path}.")
        return
    
    actual_num_images = len(image_paths)
    if actual_num_images < 2:
        print(f"Error: Need at least 2 images for testing, found {actual_num_images}.")
        return
    num_timed_images_per_size = actual_num_images - 1

    print(f"Will use {actual_num_images} images. First image per size is warm-up.")
    print(f"Latency results based on {num_timed_images_per_size} timed image(s) per size.")
    print("-" * 30)

    results_latency = {
        size: {"cpu_latency": 0.0, "gpu_latency": 0.0, "nn_latency": 0.0, "timed_images": 0}
        for size in resize_params
    }

    for size in resize_params:
        print(f"Starting latency tests for image size: {size}x{size}")
        
        # Initialize a new PyTorch model for each size (with random weights)
        # Input is grayscale (1 channel), output is also 1 channel (e.g., FFT magnitude like)
        # The UNetCorrected architecture adapts to HxW of the input tensor.
        nn_model = UNetCorrected(n_channels_in=1, n_channels_out=1).to(PT_DEVICE)
        print(f"  Initialized PyTorch UNetCorrected model for {size}x{size} on {PT_DEVICE}.")

        current_size_data = {"cpu_latency": 0.0, "gpu_latency": 0.0, "nn_latency": 0.0, "timed_images": 0}

        for i, image_path in enumerate(image_paths):
            image_data = preprocess_image(image_path, size)
            if image_data is None:
                continue

            latency_cpu = run_fft_cpu(image_data)
            latency_gpu = run_fft_gpu(image_data)
            latency_nn = run_nn_inference_pytorch(nn_model, image_data, PT_DEVICE)

            if i == 0: # Warm-up run
                print(f"    Warm-up: CPU: {latency_cpu:.6f}s; "
                      f"GPU (CuPy): {(f'{latency_gpu:.6f}s' if not np.isnan(latency_gpu) else 'N/A')}; "
                      f"NN (PyTorch-{PT_DEVICE}): {latency_nn:.6f}s")
            else: # Timed runs
                current_size_data["cpu_latency"] += latency_cpu
                if not np.isnan(latency_gpu):
                    current_size_data["gpu_latency"] += latency_gpu
                current_size_data["nn_latency"] += latency_nn # nn_latency should always be a number
                current_size_data["timed_images"] += 1
        
        if current_size_data["timed_images"] > 0:
            results_latency[size] = current_size_data
            print(f"  Size {size}x{size} completed. Timed {current_size_data['timed_images']} images.")
        else:
             print(f"  Size {size}x{size}: No images were successfully timed for latency.")
        print("-" * 30)

    # --- Results Summary, CSV, and Plotting ---
    df_data = []
    plot_labels = []
    cpu_latency_plot, gpu_latency_plot, nn_latency_plot = [], [], []

    print("\n--- Latency Test Results Summary (Average per image) ---")
    header = f"{'Size':<10} | {'CPU FFT (s)':<15} | {'GPU CuPy FFT (s)':<18} | {'NN PyTorch Inf (s)':<20} | {'Timed Imgs':<10}"
    print(header)
    print("-" * len(header))

    for size_key in resize_params:
        res = results_latency[size_key]
        n_timed = res["timed_images"]
        avg_cpu_lat, avg_gpu_lat, avg_nn_lat = float('nan'), float('nan'), float('nan')

        if n_timed > 0:
            avg_cpu_lat = res["cpu_latency"] / n_timed
            avg_gpu_lat = (res["gpu_latency"] / n_timed) if res["gpu_latency"] > 0 and not np.isnan(res["gpu_latency"]) else float('nan')
            avg_nn_lat = (res["nn_latency"] / n_timed) if res["nn_latency"] > 0 and not np.isnan(res["nn_latency"]) else float('nan')

            df_data.append({
                "ImageSize": f"{size_key}x{size_key}", 
                "AvgCPU_FFT_Latency_s": avg_cpu_lat,
                "AvgGPU_CuPy_FFT_Latency_s": avg_gpu_lat, 
                "AvgNN_PyTorch_Inf_Latency_s": avg_nn_lat,
                "TimedImages": n_timed
            })
            print(f"{size_key}x{size_key:<6} | {avg_cpu_lat:<15.6f} | "
                  f"{avg_gpu_lat:<18.6f} | {avg_nn_lat:<20.6f} | {n_timed:<10}")

            plot_labels.append(f"{size_key}x{size_key}")
            cpu_latency_plot.append(avg_cpu_lat)
            gpu_latency_plot.append(avg_gpu_lat)
            nn_latency_plot.append(avg_nn_lat)
        else:
            plot_labels.append(f"{size_key}x{size_key}")
            cpu_latency_plot.append(float('nan'))
            gpu_latency_plot.append(float('nan'))
            nn_latency_plot.append(float('nan'))

    if df_data:
        df = pd.DataFrame(df_data)
        csv_filename = "fft_latency_comparison_pytorch.csv"
        df.to_csv(csv_filename, index=False, float_format='%.6g')
        print(f"\nLatency results saved to {csv_filename}")

    if not any(plot_labels):
        print("\nNo data processed for plotting.")
        return

    x_plot = np.arange(len(plot_labels))
    
    fig1, ax1 = plt.subplots(figsize=(13, 7))
    if any(not np.isnan(t) for t in cpu_latency_plot):
      ax1.plot(x_plot, cpu_latency_plot, marker='o', linestyle='-', label='CPU FFT Latency')
    if any(not np.isnan(t) for t in gpu_latency_plot) and CUPY_AVAILABLE:
      ax1.plot(x_plot, gpu_latency_plot, marker='x', linestyle='--', label='GPU FFT Latency (CuPy)')
    if any(not np.isnan(t) for t in nn_latency_plot):
      ax1.plot(x_plot, nn_latency_plot, marker='s', linestyle=':', label=f'NN Inf. Latency (PyTorch-{PT_DEVICE.type})')
    
    ax1.set_ylabel('Average Latency (seconds) - Log Scale')
    ax1.set_xlabel('Image Size')
    ax1.set_title(f'Method Latency Comparison (Warm-up, Timed: {num_timed_images_per_size} images/size)')
    ax1.set_xticks(x_plot); ax1.set_xticklabels(plot_labels, rotation=45, ha="right")
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.7); ax1.set_yscale('log')
    plt.tight_layout(); plt.savefig("fft_latency_comparison_pytorch.png")
    print("\nLatency comparison plot saved as fft_latency_comparison_pytorch.png")
    # plt.show()


if __name__ == "__main__":
    base_image_folder = "/home/sean/Omost/outputs_png" # 替换为你的图片文件夹路径
    
    # 检查图像文件夹是否存在
    if not os.path.exists(base_image_folder) or not os.listdir(base_image_folder):
        print(f"Warning: Image folder '{base_image_folder}' not found or empty. Creating dummy images in './dummy_images_latency_pt'.")
        base_image_folder = "./dummy_images_latency_pt"
        os.makedirs(base_image_folder, exist_ok=True)
        if not os.listdir(base_image_folder): 
            try:
                # 使用 num_total_images_to_select (如果已定义) 或默认值来创建足够的图像
                num_images_to_create = globals().get('num_total_images_to_select', 6)
                for i in range(num_images_to_create): 
                    img_array_dummy = np.random.randint(0, 256, size=(1024, 1024, 3), dtype=np.uint8) # 创建足够大的图像
                    img_dummy = Image.fromarray(img_array_dummy)
                    img_dummy.save(os.path.join(base_image_folder, f"random_image_pt_latency_test_{i+1}.png"))
                print(f"Created dummy images in '{base_image_folder}' for testing.")
            except Exception as e:
                print(f"Failed to create dummy test images: {e}")
    
    if not os.path.exists(base_image_folder) or not os.listdir(base_image_folder):
        print(f"Error: Image folder '{base_image_folder}' is empty or non-existent, and dummy creation failed. Exiting.")
    else:
        main(base_image_folder)