import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2 # OpenCV for image manipulation and FFT

class ImageDatasetNoLabels(Dataset):
    def __init__(self, root_dir, base_transform=None, fft_post_transform=None):
        """
        Initializes the dataset.

        Args:
            root_dir (string): The root directory containing all image files.
                               Example structure:
                               root_dir/
                               ├── image1.jpg
                               ├── image2.png
                               └── ...
            base_transform (callable, optional): Base transforms to apply to the raw PIL Image.
                                                 FFT will be performed on the output of this transform.
                                                 This should typically include Resize, Crop, ToTensor, Normalize.
            fft_post_transform (callable, optional): Transforms to apply to the FFT magnitude spectrum
                                                     after it's calculated and converted to a tensor.
        """
        self.root_dir = root_dir
        self.base_transform = base_transform
        self.fft_post_transform = fft_post_transform
        self.image_paths = []

# --- RECURSIVE IMAGE COLLECTION ---
        for dirpath, _, filenames in os.walk(root_dir):
            for img_name in filenames:
                img_path = os.path.join(dirpath, img_name)
                if self._is_image_file(img_name): # We don't need os.path.isfile here as os.walk gives us filenames
                    self.image_paths.append(img_path)

    def _is_image_file(self, filename):
        """Checks if a given filename corresponds to a common image file type."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image by index, applies base transformations, computes its FFT,
        and returns both the processed base image tensor and the processed FFT tensor.
        """
        img_path = self.image_paths[idx]

        # 1. Load the original image using PIL
        original_image_pil = Image.open(img_path).convert('RGB')

        # 2. Apply base_transform to get the "processed" image Tensor
        # This tensor will be the input for the raw image branch of your network.
        if self.base_transform:
            processed_base_image_tensor = self.base_transform(original_image_pil)
        else:
            # If no base_transform is provided, at least convert to a tensor
            processed_base_image_tensor = transforms.ToTensor()(original_image_pil)

        # 3. Convert the processed_base_image_tensor back to a NumPy array for FFT
        # Ensure the tensor is on CPU and permute to H, W, C format for OpenCV/NumPy
        image_for_fft_np = processed_base_image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Scale pixels back to 0-255 range and convert to uint8.
        # This is generally robust for FFT processing using OpenCV/NumPy.
        image_for_fft_np_scaled = (image_for_fft_np * 255).astype(np.uint8) 
        
        # Convert to grayscale if it's an RGB image
        if image_for_fft_np_scaled.shape[2] == 3:
            image_gray_for_fft = cv2.cvtColor(image_for_fft_np_scaled, cv2.COLOR_RGB2GRAY)
        else: # Already grayscale (1 channel)
            image_gray_for_fft = image_for_fft_np_scaled.squeeze() # Remove channel dim if present

        # 4. Perform 2D Fast Fourier Transform (FFT)
        f_transform = np.fft.fft2(image_gray_for_fft)
        # Shift the zero-frequency component to the center of the spectrum
        f_shift = np.fft.fftshift(f_transform)

        # Calculate the magnitude spectrum (logarithmic scale for better visualization)
        # Add a small constant (1e-8) to avoid log(0)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

        # 5. Post-process the FFT result (normalize, convert to tensor)
        # Normalize the magnitude spectrum to [0, 1] range (float32) for consistency
        magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        
        # Convert to PyTorch tensor and add a channel dimension: [1, H, W]
        fft_tensor = torch.from_numpy(magnitude_spectrum_normalized).unsqueeze(0) 

        # Apply any specified post-FFT transforms
        if self.fft_post_transform:
            processed_fft_tensor = self.fft_post_transform(fft_tensor)
        else:
            processed_fft_tensor = fft_tensor # Return as is if no post-transform

        # Return both the base image tensor and the FFT image tensor
        return processed_base_image_tensor, processed_fft_tensor


if __name__ == "__main__":
    # 1. Prepare a dummy dataset directory (no subfolders, just images directly)
    dataset_root = "my_images_no_labels"
    os.makedirs(dataset_root, exist_ok=True)
    Image.new('RGB', (60, 30), color = 'red').save(os.path.join(dataset_root, 'img1.jpg'))
    Image.new('RGB', (60, 30), color = 'green').save(os.path.join(dataset_root, 'img2.png'))
    Image.new('RGB', (60, 30), color = 'blue').save(os.path.join(dataset_root, 'img3.jpeg'))
    Image.new('RGB', (60, 30), color = 'yellow').save(os.path.join(dataset_root, 'img4.bmp'))

    # 2. Define a BASE TRANSFORM that all images will go through first.
    # This transform should include resizing/cropping to a consistent size (e.g., 224x224).
    base_image_transforms = transforms.Compose([
        transforms.Resize(256),              # Resize shortest side to 256 pixels
        transforms.CenterCrop(224),          # Crop to 224x224 from the center
        transforms.ToTensor(),               # Convert to Tensor (scales to [0.0, 1.0], C, H, W)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize (ImageNet stats)
    ])

    # 3. Define FFT result POST-PROCESSING transforms.
    # Note: Mean and std here are placeholders. You should calculate them
    # based on your actual FFT data's statistics for proper normalization.
    fft_post_transforms = transforms.Compose([
        # If your network expects 3 channels for FFT input, uncomment this:
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Placeholder: Replace with actual FFT mean/std
    ])

    # 4. Create the dataset instance
    image_dataset = ImageDatasetNoLabels(
        root_dir=dataset_root,
        base_transform=base_image_transforms,
        fft_post_transform=fft_post_transforms
    )

    print(f"Number of images in the dataset: {len(image_dataset)}")

    # 5. Create the DataLoader instance
    batch_size = 2
    # For inference or just reading, shuffle can be False, num_workers can be adjusted
    image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 

    # 6. Iterate through the DataLoader and inspect the loaded data
    print("\n--- Image Data Loading Example (No Labels) ---")
    for i, (base_images, fft_images) in enumerate(image_loader):
        print(f"Batch {i+1}:")
        print(f"  Base Image Tensor Shape: {base_images.shape}") # Expected: torch.Size([batch_size, 3, 224, 224])
        print(f"  FFT Image Tensor Shape: {fft_images.shape}")   # Expected: torch.Size([batch_size, 1, 224, 224])
        
        # Verify that FFT image dimensions match the base image's H, W
        assert fft_images.shape[2] == base_images.shape[2] and \
               fft_images.shape[3] == base_images.shape[3], \
               "FFT image dimensions do not match base image dimensions!"
        if i == 0: # Print only for the first batch
            break

    # Clean up the dummy data directory
    import shutil
    shutil.rmtree(dataset_root)
    print(f"\nCleaned up dummy dataset directory: {dataset_root}")