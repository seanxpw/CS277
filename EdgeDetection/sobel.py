import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2 # OpenCV for image manipulation and Sobel edge detection
from pathlib import Path # For robust path handling

class SobelEdgeDetection(Dataset):
    def __init__(self, root_dir,
                 base_transform_spatial = transforms.Compose([ # 空间变换部分
                     transforms.Resize(256),
                     transforms.CenterCrop(224)
                 ]),
                 base_transform_tensor = transforms.Compose([ # 张量和归一化变换部分
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]),
                 edge_post_transform=transforms.Compose([
                     transforms.Normalize(mean=[0.5], std=[0.5])
                 ])):
        """
        Initializes the dataset.

        Args:
            root_dir (string): Root directory containing images.
            base_transform_spatial (callable, optional): Spatial transforms (Resize, Crop)
                                                         applied to the PIL image first.
            base_transform_tensor (callable, optional): Transforms (ToTensor, Normalize)
                                                        applied after spatial transforms for the base image.
            edge_post_transform (callable, optional): Transforms for the Sobel edge map tensor.
        """
        self.root_dir = Path(root_dir)
        self.base_transform_spatial = base_transform_spatial
        self.base_transform_tensor = base_transform_tensor
        self.edge_post_transform = edge_post_transform
        self.image_paths = []
        self.sobel_ksize = 3

        for dirpath, _, filenames in os.walk(self.root_dir):
            for img_name in filenames:
                img_path = os.path.join(dirpath, img_name)
                if self._is_image_file(img_name):
                    self.image_paths.append(img_path)
        
        self.image_paths.sort()
        if not self.image_paths:
            raise RuntimeError(f"No images found in directory: {self.root_dir}.")

    def _is_image_file(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.image_paths[idx]
        original_image_pil = Image.open(img_path).convert('RGB')

        # 1. Apply common spatial transformations
        if self.base_transform_spatial:
            pil_after_spatial_transforms = self.base_transform_spatial(original_image_pil)
        else:
            # If no spatial transform, use original (though typically you'd want one for consistent sizing)
            pil_after_spatial_transforms = original_image_pil


        # 2. Path A: Create processed_base_image_tensor
        # Apply ToTensor and Normalize to the spatially transformed PIL image
        if self.base_transform_tensor:
            processed_base_image_tensor = self.base_transform_tensor(pil_after_spatial_transforms)
        else: # Fallback if only spatial transforms were given (less common for base image)
            processed_base_image_tensor = transforms.ToTensor()(pil_after_spatial_transforms)


        # 3. Path B: Create image for Sobel from the *same* spatially transformed PIL image
        image_for_edge_np_uint8 = np.array(pil_after_spatial_transforms) # HxWxC, uint8, [0-255]

        # Convert to grayscale for Sobel
        if image_for_edge_np_uint8.ndim == 3 and image_for_edge_np_uint8.shape[2] == 3:
            image_gray_for_edge = cv2.cvtColor(image_for_edge_np_uint8, cv2.COLOR_RGB2GRAY)
        elif image_for_edge_np_uint8.ndim == 2:
             image_gray_for_edge = image_for_edge_np_uint8
        else:
            raise ValueError(f"Image {img_path} converted to unexpected shape for grayscale: {image_for_edge_np_uint8.shape}")

        # Perform Sobel Edge Detection
        sobel_x_np = cv2.Sobel(image_gray_for_edge, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobel_y_np = cv2.Sobel(image_gray_for_edge, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        edge_map_np = cv2.magnitude(sobel_x_np, sobel_y_np)

        # Normalize the Sobel magnitude map to [0.0, 1.0]
        edge_map_normalized = np.zeros_like(edge_map_np, dtype=np.float32)
        cv2.normalize(edge_map_np, edge_map_normalized, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Convert edge map to PyTorch tensor and add a channel dimension: [1, H, W]
        edge_tensor = torch.from_numpy(edge_map_normalized).unsqueeze(0) 

        # Apply any specified post-edge transforms
        if self.edge_post_transform:
            processed_edge_tensor = self.edge_post_transform(edge_tensor)
        else:
            processed_edge_tensor = edge_tensor

        return processed_base_image_tensor, processed_edge_tensor

if __name__ == "__main__":
    dataset_root = "my_images_sobel_consistent_view"
    os.makedirs(os.path.join(dataset_root, "subdir_a"), exist_ok=True)

    # Create some dummy images of varying sizes
    Image.new('RGB', (600, 400), color = 'red').save(os.path.join(dataset_root, 'img_root1.jpg'))
    Image.new('RGB', (400, 500), color = 'green').save(os.path.join(dataset_root, 'subdir_a', 'img_sub_a1.png'))
    
    # Image with a clear edge, original size different from crop size
    dummy_img_with_edge_orig_size = (300,300)
    dummy_img_with_edge = Image.new('RGB', dummy_img_with_edge_orig_size, color=(0,0,0))
    draw_rect = Image.new('RGB', (dummy_img_with_edge_orig_size[0]//2, dummy_img_with_edge_orig_size[1]), color=(255,255,255))
    dummy_img_with_edge.paste(draw_rect, (dummy_img_with_edge_orig_size[0]//4, 0))
    dummy_img_with_edge.save(os.path.join(dataset_root, 'img_with_clear_edge.png'))
    
    print(f"Created dummy images in {dataset_root}")

    # Define the transform components based on your previous default
    # These are now explicitly separated
    spatial_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    tensor_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    edge_post_proc_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]) # Example post-transform for edge map
    ])

    # Create the dataset instance
    sobel_dataset = SobelEdgeDetection(
        root_dir=dataset_root,
        base_transform_spatial=spatial_transforms,
        base_transform_tensor=tensor_transforms,
        edge_post_transform=edge_post_proc_transforms
    )
    print(f"Number of images in the Sobel dataset: {len(sobel_dataset)}")

    image_loader = DataLoader(sobel_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("\n--- Sobel Edge Detection Data Loading (Consistent View) ---")
    for i, (base_images, edge_maps) in enumerate(image_loader):
        print(f"Batch {i+1}:")
        print(f"  Base Image Tensor Shape: {base_images.shape}")
        print(f"  Edge Map Tensor Shape: {edge_maps.shape}")
        
        assert edge_maps.shape[2] == base_images.shape[2] and \
               edge_maps.shape[3] == base_images.shape[3], \
               "Edge map dimensions do not match base image dimensions!"

        if i == 2: # Visualize the image with the clear edge (index 2 in sorted list)
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                base_img_vis = base_images[0].permute(1,2,0).cpu().numpy() * std + mean
                base_img_vis = np.clip(base_img_vis, 0, 1)
                plt.imshow(base_img_vis)
                plt.title(f"Base Image (Batch {i+1})")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                edge_map_vis_tensor = edge_maps[0, 0].cpu().numpy() * 0.5 + 0.5 # Inverse example normalize
                plt.imshow(edge_map_vis_tensor, cmap='gray')
                plt.title(f"Processed Sobel Edge Map (Batch {i+1})")
                plt.axis('off')
                
                plt.savefig("sobel_dataset_consistent_sample.png")
                print("Saved sample visualization to sobel_dataset_consistent_sample.png")
                plt.close()
            except ImportError:
                print("Matplotlib not found. Skipping visualization.")
            except Exception as e:
                print(f"Error during visualization: {e}")
        
        if i >= 2: # Process a few samples
            break

    import shutil
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
        print(f"\nCleaned up dummy dataset directory: {dataset_root}")