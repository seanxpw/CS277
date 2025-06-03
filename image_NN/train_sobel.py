# train_sobel.py

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model.unet import UNet
from data.sobel_dataset import SobelEdgeDetection

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
spatial_transforms = T.Resize((256, 256))    # Resize PIL
tensor_transforms = T.ToTensor()             # PIL to [0,1] tensor
edge_post_proc_transforms = None             # 可选：归一化等

# Dataset & Dataloader
dataset_root = "dataset"
sobel_dataset = SobelEdgeDetection(
    root_dir=dataset_root,
    base_transform_spatial=spatial_transforms,
    base_transform_tensor=tensor_transforms,
    edge_post_transform=edge_post_proc_transforms
)
dataloader = DataLoader(sobel_dataset, batch_size=4, shuffle=True)

# --- Model ---
model = UNet(in_channels=3, out_channels=1).to(device)  # RGB输入，边缘图输出
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# Optionally: Save model
torch.save(model.state_dict(), "unet_sobel.pth")
