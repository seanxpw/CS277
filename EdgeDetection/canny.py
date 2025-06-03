import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2 # OpenCV for image manipulation and Canny edge detection
from pathlib import Path # For robust path handling

class CannyEdgeDetection(Dataset):
    def __init__(self, root_dir,
                 base_transform = transforms.Compose([
                    transforms.Resize(256),              # Resize shortest side to 256 pixels
                    transforms.CenterCrop(224),          # Crop to 224x224 from the center
                    transforms.ToTensor(),               # Convert to Tensor (scales to [0.0, 1.0], C, H, W)
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize (ImageNet stats)
                 ]),
                 edge_params=(100, 200), # Default Canny thresholds: (low_threshold, high_threshold)
                 edge_post_transform=transforms.Compose([
                    # Edge maps are typically single-channel binary (0/1 or 0/255).
                    # Normalize it to fit your model's input expectations.
                    # If your network expects 3 channels, uncomment this:
                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize(mean=[0.5], std=[0.5]) # Placeholder: Replace with actual edge map mean/std
                 ])):
        """
        Initializes the dataset to load all images recursively from a directory
        and apply base transformations and Canny edge detection.

        Args:
            root_dir (string): The root directory containing all image files.
                               It can contain subdirectories (recursive scan).
                               Example structure:
                               root_dir/
                               ├── image1.jpg
                               ├── subdir/
                               │   └── image2.png
                               └── ...
            base_transform (callable, optional): Base transforms to apply to the raw PIL Image.
                                                 Edge detection will be performed on the output of this transform.
                                                 This should typically include Resize, Crop, ToTensor, Normalize.
            edge_params (tuple): A tuple (low_threshold, high_threshold) for Canny edge detector.
            edge_post_transform (callable, optional): Transforms to apply to the Canny edge map
                                                      after it's calculated and converted to a tensor.
        """
        self.root_dir = Path(root_dir) # Use Path for easier handling
        self.base_transform = base_transform
        self.edge_params = edge_params
        self.edge_post_transform = edge_post_transform
        self.image_paths = []

        # --- RECURSIVE IMAGE COLLECTION ---
        # Collect all image file paths recursively from the root directory
        for dirpath, _, filenames in os.walk(self.root_dir):
            for img_name in filenames:
                img_path = os.path.join(dirpath, img_name)
                if self._is_image_file(img_name):
                    self.image_paths.append(img_path)
        # --- END RECURSIVE IMAGE COLLECTION ---
        
        # Optionally sort paths for consistent ordering, though not strictly needed without splitting
        self.image_paths.sort()

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in directory: {self.root_dir}. "
                "Please ensure the directory contains valid image files."
            )

    def _is_image_file(self, filename):
        """Checks if a given filename corresponds to a common image file type."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image by index, applies base transformations, computes its Canny edge map,
        and returns both the processed base image tensor and the processed edge map tensor.
        """
        # Ensure idx is a standard integer if it comes as a PyTorch tensor (e.g., from DataLoader)
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.image_paths[idx]

        # 1. Load the original image using PIL
        original_image_pil = Image.open(img_path).convert('RGB')

        # 2. Apply base_transform to get the "processed" image Tensor
        # This tensor will be the input for the raw image branch of your network.
        # It also determines the size for edge detection.
        if self.base_transform:
            processed_base_image_tensor = self.base_transform(original_image_pil)
        else:
            # If no base_transform is provided, at least convert to a tensor
            processed_base_image_tensor = transforms.ToTensor()(original_image_pil)

        # 3. Convert the processed_base_image_tensor back to a NumPy array for Canny
        # Ensure the tensor is on CPU and permute to H, W, C format for OpenCV/NumPy
        image_for_edge_np = processed_base_image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Scale pixels back to 0-255 range and convert to uint8.
        # Canny typically expects uint8 images.
        image_for_edge_np_scaled = (image_for_edge_np * 255).astype(np.uint8) 
        
        # Convert to grayscale for Canny edge detection
        if image_for_edge_np_scaled.shape[2] == 3: # If it's an RGB image
            image_gray_for_edge = cv2.cvtColor(image_for_edge_np_scaled, cv2.COLOR_RGB2GRAY)
        else: # Already grayscale (1 channel)
            image_gray_for_edge = image_for_edge_np_scaled.squeeze() # Remove channel dim if present

        # 4. Perform Canny Edge Detection
        low_threshold, high_threshold = self.edge_params
        edge_map_np = cv2.Canny(image_gray_for_edge, low_threshold, high_threshold)

        # 5. Post-process the Edge Map (normalize, convert to tensor)
        # Canny output is typically 0 or 255 (uint8). Normalize to [0, 1] for consistency.
        edge_map_normalized = edge_map_np.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor and add a channel dimension: [1, H, W]
        edge_tensor = torch.from_numpy(edge_map_normalized).unsqueeze(0) 

        # Apply any specified post-edge transforms
        if self.edge_post_transform:
            processed_edge_tensor = self.edge_post_transform(edge_tensor)
        else:
            processed_edge_tensor = edge_tensor # Return as is if no post-transform

        # Return both the base image tensor and the edge map tensor
        return processed_base_image_tensor, processed_edge_tensor


if __name__ == "__main__":
    # 1. Prepare a dummy dataset directory with some images for demonstration
    # This directory can contain subfolders as well.
    dataset_root = "my_images_no_split"
    os.makedirs(os.path.join(dataset_root, "subdir_a"), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, "subdir_b"), exist_ok=True)

    # Create 5 dummy images, some in subdirectories
    Image.new('RGB', (80, 80), color = 'red').save(os.path.join(dataset_root, 'img_root.jpg'))
    Image.new('RGB', (90, 90), color = 'green').save(os.path.join(dataset_root, 'subdir_a', 'img_sub_a1.png'))
    Image.new('RGB', (70, 70), color = 'blue').save(os.path.join(dataset_root, 'subdir_a', 'img_sub_a2.jpeg'))
    # Create a simple image with a clear edge for better visualization of Canny
    dummy_img_with_edge = Image.new('RGB', (100, 100), color=(0,0,0))
    for x in range(100):
        for y in range(100):
            if x > 50:
                dummy_img_with_edge.putpixel((x, y), (255, 255, 255)) # White half
            else:
                dummy_img_with_edge.putpixel((x, y), (0, 0, 0)) # Black half
    dummy_img_with_edge.save(os.path.join(dataset_root, 'subdir_b', 'img_sub_b1_edge.png'))
    Image.new('RGB', (110, 110), color = 'yellow').save(os.path.join(dataset_root, 'subdir_b', 'img_sub_b2.bmp'))


    # 2. Define a BASE TRANSFORM (using the default values in __init__ for this example)
    # base_image_transforms = transforms.Compose([...]) # No need to define explicitly if using defaults

    # 3. Define EDGE MAP POST-PROCESSING transforms (using the default values in __init__ for this example)
    # edge_post_transforms = transforms.Compose([...]) # No need to define explicitly if using defaults

    # 4. Create the dataset instance. It will use the default transforms.
    # You can also pass custom ones if needed:
    # image_dataset = ImageDatasetNoLabels(
    #     root_dir=dataset_root,
    #     base_transform=my_custom_base_transforms,
    #     edge_params=(30, 90), # Custom Canny thresholds
    #     edge_post_transform=my_custom_edge_post_transforms
    # )
    image_dataset = CannyEdgeDetection(root_dir=dataset_root)

    print(f"Number of images in the dataset: {len(image_dataset)}") # Expected: 5

    # 5. Create the DataLoader instance
    batch_size = 2
    # For inference or just reading, shuffle can be False, num_workers can be adjusted
    image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 

    # 6. Iterate through the DataLoader and inspect the loaded data
    print("\n--- Image Data Loading Example (No Labels, No Split) ---")
    for i, (base_images, edge_maps) in enumerate(image_loader):
        print(f"Batch {i+1}:")
        print(f"  Base Image Tensor Shape: {base_images.shape}")  # e.g., torch.Size([batch_size, 3, 224, 224])
        print(f"  Edge Map Tensor Shape: {edge_maps.shape}")      # e.g., torch.Size([batch_size, 1, 224, 224])
        
        # Verify that edge map dimensions match the base image's H, W
        assert edge_maps.shape[2] == base_images.shape[2] and \
               edge_maps.shape[3] == base_images.shape[3], \
               "Edge map dimensions do not match base image dimensions!"
        if i == 0:
            # For demonstration, you could optionally visualize one of the edge maps
            # import matplotlib.pyplot as plt
            # plt.imshow(edge_maps[0, 0].cpu().numpy(), cmap='gray')
            # plt.title("Example Canny Edge Map")
            # plt.show()
            pass # Keep output clean for automated checks
        if i >= 1: # Print only first two batches to avoid too much output
            break

    # Clean up the dummy data directory
    import shutil
    shutil.rmtree(dataset_root)
    print(f"\nCleaned up dummy dataset directory: {dataset_root}")