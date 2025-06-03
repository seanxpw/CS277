import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2 # OpenCV for image manipulation and Canny edge detection
from pathlib import Path # For robust path handling

class CannyEdgeDetectionFixed(Dataset): # Renamed for clarity
    def __init__(self, root_dir,
                 base_transform_spatial = transforms.Compose([
                     transforms.Resize(256),      # Resize shortest side to 256 pixels
                     transforms.CenterCrop(224)   # Crop to 224x224 from the center
                 ]),
                 base_transform_tensor = transforms.Compose([
                     transforms.ToTensor(),       # Convert to Tensor (scales to [0.0, 1.0], C, H, W)
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
        Initializes the dataset with fixed view consistency for Canny edge detection.

        Args:
            root_dir (string): The root directory containing all image files.
            base_transform_spatial (callable, optional): Spatial transforms (Resize, Crop)
                                                         applied to the PIL image first.
            base_transform_tensor (callable, optional): Transforms (ToTensor, Normalize)
                                                        applied after spatial transforms for the base image.
            edge_params (tuple): A tuple (low_threshold, high_threshold) for Canny edge detector.
            edge_post_transform (callable, optional): Transforms to apply to the Canny edge map
                                                      after it's calculated and converted to a tensor.
        """
        self.root_dir = Path(root_dir) # Use Path for easier handling
        self.base_transform_spatial = base_transform_spatial
        self.base_transform_tensor = base_transform_tensor
        self.edge_params = edge_params
        self.edge_post_transform = edge_post_transform
        self.image_paths = []

        for dirpath, _, filenames in os.walk(self.root_dir):
            for img_name in filenames:
                img_path = os.path.join(dirpath, img_name)
                if self._is_image_file(img_name):
                    self.image_paths.append(img_path)
        
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
        Retrieves an image by index, applies base transformations, computes its Canny edge map
        (ensuring view consistency), and returns both the processed base image tensor
        and the processed edge map tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.image_paths[idx]
        original_image_pil = Image.open(img_path).convert('RGB')

        # 1. Apply common spatial transformations to the original PIL image
        if self.base_transform_spatial:
            pil_after_spatial_transforms = self.base_transform_spatial(original_image_pil)
        else:
            pil_after_spatial_transforms = original_image_pil # Should ideally have spatial transforms

        # 2. Path A: Create processed_base_image_tensor
        # Apply ToTensor and Normalize to the spatially transformed PIL image
        if self.base_transform_tensor:
            processed_base_image_tensor = self.base_transform_tensor(pil_after_spatial_transforms)
        else:
            # Fallback: if no tensor/norm transform, just convert the spatially transformed PIL to tensor
            processed_base_image_tensor = transforms.ToTensor()(pil_after_spatial_transforms)


        # 3. Path B: Create image for Canny edge detection from the *same* spatially transformed PIL image
        # Convert the spatially transformed PIL image to a NumPy array (HxWxC, uint8, [0-255])
        image_for_edge_np_uint8 = np.array(pil_after_spatial_transforms)
        
        # Convert to grayscale for Canny edge detection
        if image_for_edge_np_uint8.ndim == 3 and image_for_edge_np_uint8.shape[2] == 3: # If it's an RGB image
            image_gray_for_edge = cv2.cvtColor(image_for_edge_np_uint8, cv2.COLOR_RGB2GRAY)
        elif image_for_edge_np_uint8.ndim == 2: # Already grayscale
            image_gray_for_edge = image_for_edge_np_uint8
        else: # Should not happen if .convert('RGB') was used and spatial transforms are typical
            raise ValueError(f"Image {img_path} after spatial transform resulted in unexpected shape for grayscale conversion: {image_for_edge_np_uint8.shape}")


        # 4. Perform Canny Edge Detection
        low_threshold, high_threshold = self.edge_params
        edge_map_np = cv2.Canny(image_gray_for_edge, low_threshold, high_threshold) # Output is uint8, 0 or 255

        # 5. Post-process the Edge Map
        # Normalize Canny output (0 or 255) to [0.0, 1.0] float for consistency before further transforms.
        edge_map_normalized_float = edge_map_np.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor and add a channel dimension: [1, H, W]
        edge_tensor = torch.from_numpy(edge_map_normalized_float).unsqueeze(0) 

        # Apply any specified post-edge transforms
        if self.edge_post_transform:
            processed_edge_tensor = self.edge_post_transform(edge_tensor)
        else:
            processed_edge_tensor = edge_tensor 

        return processed_base_image_tensor, processed_edge_tensor


if __name__ == "__main__":
    dataset_root = "my_images_canny_fixed"
    os.makedirs(os.path.join(dataset_root, "subdir_a"), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, "subdir_b"), exist_ok=True)

    # Create dummy images
    Image.new('RGB', (300, 400), color = 'red').save(os.path.join(dataset_root, 'img_root.jpg'))
    Image.new('RGB', (400, 300), color = 'green').save(os.path.join(dataset_root, 'subdir_a', 'img_sub_a1.png'))
    
    dummy_img_with_edge_orig_size = (280,320) # Original size different from crop
    dummy_img_with_edge = Image.new('RGB', dummy_img_with_edge_orig_size, color=(0,0,0))
    # Create a white rectangle in the middle for a clear edge
    rect_w, rect_h = dummy_img_with_edge_orig_size[0]//2, dummy_img_with_edge_orig_size[1]//2
    start_x, start_y = dummy_img_with_edge_orig_size[0]//4, dummy_img_with_edge_orig_size[1]//4
    for x_offset in range(rect_w):
        for y_offset in range(rect_h):
            dummy_img_with_edge.putpixel((start_x + x_offset, start_y + y_offset), (255,255,255))
    dummy_img_with_edge.save(os.path.join(dataset_root, 'subdir_b', 'img_clear_edge.png'))
    
    print(f"Created dummy images in {dataset_root}")

    # Define transform components as per the class's new __init__ structure
    spatial_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Example post-transform for Canny edge map (which is now [0,1] float before this)
    edge_post_transforms = transforms.Compose([
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # If model expects 3 channels
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1] if needed
    ])

    # Create the dataset instance
    canny_dataset_fixed = CannyEdgeDetectionFixed(
        root_dir=dataset_root,
        base_transform_spatial=spatial_transforms,
        base_transform_tensor=tensor_transforms,
        edge_params=(50, 150), # Example Canny thresholds
        edge_post_transform=edge_post_transforms
    )

    print(f"Number of images in the fixed Canny dataset: {len(canny_dataset_fixed)}")

    image_loader = DataLoader(canny_dataset_fixed, batch_size=1, shuffle=False, num_workers=0) 

    print("\n--- Fixed Canny Edge Detection Data Loading (Consistent View) ---")
    for i, (base_images, edge_maps) in enumerate(image_loader):
        print(f"Batch {i+1}:")
        print(f"  Base Image Tensor Shape: {base_images.shape}")
        print(f"  Edge Map Tensor Shape:   {edge_maps.shape}")
        
        assert edge_maps.shape[2] == base_images.shape[2] and \
               edge_maps.shape[3] == base_images.shape[3], \
               "Edge map dimensions do not match base image dimensions!"

        if i == 2: # Visualize the image with the clear edge (index 2 in sorted list)
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                
                # Visualize Base Image (approx original appearance)
                plt.subplot(1, 2, 1)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                base_img_vis = base_images[0].permute(1,2,0).cpu().numpy() * std + mean
                base_img_vis = np.clip(base_img_vis, 0, 1)
                plt.imshow(base_img_vis)
                plt.title(f"Base Image (Batch {i+1})")
                plt.axis('off')

                # Visualize Processed Canny Edge Map
                plt.subplot(1, 2, 2)
                # Inverse the example Normalize from edge_post_transform for display if it was applied
                # edge_map_vis_tensor = edge_maps[0,0].cpu().numpy() * 0.5 + 0.5 # If Normalize(0.5,0.5) was used
                # If no such Normalize or to see the [0,1] map before it:
                # We need to get the map before edge_post_transform for raw [0,1] Canny output.
                # For simplicity, let's assume edge_post_transform might make it [-1,1]
                # and try to scale it back to a viewable range.
                edge_map_display = edge_maps[0,0].cpu().numpy()
                if edge_post_transforms: # If post-processing was applied, try to un-normalize for display
                     # This depends on what edge_post_transform does.
                     # If it was Normalize(0.5, 0.5), then:
                     edge_map_display = edge_map_display * 0.5 + 0.5
                edge_map_display = np.clip(edge_map_display,0,1)

                plt.imshow(edge_map_display, cmap='gray')
                plt.title(f"Processed Canny Edge Map (Batch {i+1})")
                plt.axis('off')
                
                plt.savefig("canny_dataset_fixed_sample.png")
                print("Saved sample visualization to canny_dataset_fixed_sample.png")
                plt.close()
            except ImportError:
                print("Matplotlib not found. Skipping visualization.")
            except Exception as e:
                print(f"Error during visualization: {e}")
        
        if i >= 2: # Process a few samples for demo
            break

    # Clean up
    import shutil
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
        print(f"\nCleaned up dummy dataset directory: {dataset_root}")