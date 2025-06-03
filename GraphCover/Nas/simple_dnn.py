import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from cs277_dataset import CS277Dataset


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        # First conv: maybe downsample spatially if stride=2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv: always stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels != out_channels or we downsample, build a 1×1 conv for the “skip” path
        if downsample or (in_channels != out_channels):
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class GraphCoverModel(pl.LightningModule):
    def __init__(self, hidden_dims=(128, 64), lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        #
        # 1) First, convert the 20×20 adjacency into a 1‐channel “image”:
        #    → input shape: [B, 1, 20, 20]
        #
        # 2) ResidualBlock(in_channels=1,  out_channels=16, downsample=False)
        # 3) ResidualBlock(16, 32, downsample=True)  ← this halves spatial dims (20→10)
        # 4) ResidualBlock(32, 64, downsample=True)  ← (10→5)
        #
        # After that, we have a [B, 64, 5, 5] tensor. We do a global pool over 5×5 → [B, 64, 1,1],
        # then flatten → [B, 64], then a Linear(64→20) → Sigmoid.
        #

        self.layer1 = ResidualBlock(in_channels=1, out_channels=16, downsample=False)  # keeps 20×20
        self.layer2 = ResidualBlock(in_channels=16, out_channels=32, downsample=True)  # → 10×10
        self.layer3 = ResidualBlock(in_channels=32, out_channels=64, downsample=True)  # → 5×5

        # Global average pool from 5×5 → 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B,64,5,5] → [B,64,1,1]

        # Final linear + sigmoid
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 64,1,1] → [B,64]
            nn.Linear(64, 20),  # → [B,20]
            nn.Sigmoid()  # final per‐vertex probability
        )

        # We used Sigmoid above, so use BCELoss:
        self.criterion = nn.BCELoss()

    def forward(self, adj_mat: torch.Tensor) -> torch.Tensor:
        # adj_mat: [B, 1, 20, 20]
        x = adj_mat  # [B, 1, 20, 20]

        x = self.layer1(x)  # [B,16,20,20]
        x = self.layer2(x)  # [B,32,10,10]
        x = self.layer3(x)  # [B,64, 5, 5]

        x = self.global_pool(x)  # [B,64,1,1]
        x = self.classifier(x)  # [B,20], in [0,1] due to Sigmoid
        return x

    def training_step(self, batch, batch_idx):
        adj, label = batch
        # adj: [batch,20,20], label: [batch,20]
        preds = self.forward(adj)  # [batch,20]
        loss = self.criterion(preds, label.float())
        # Log to TensorBoard (or other logger)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        adj, label = batch
        preds = self.forward(adj)
        loss = self.criterion(preds, label.float())

        # You could also compute e.g. accuracy or F1—here’s a simple “fraction correct”:
        pred_labels = (preds >= 0.5).float()
        acc = (pred_labels == label).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.val_outputs = {
            'preds': preds,
            'pred_labels': pred_labels,
            'labels': label,
        }


    def on_validation_epoch_end(self):
        """
        Called once per validation epoch with the list “outputs” from every batch’s validation_step.
        We’ll pick the very first batch’s preds/labels and print them (for example).
        """
        # outputs is a list of dicts, one per batch. Grab the first batch:
        first_batch = self.val_outputs
        preds = first_batch["preds"]  # Tensor of shape [batch_size, 20]
        pred_labels = first_batch["pred_labels"]  # Tensor of shape [batch_size, 20]
        labels = first_batch["labels"]  # Tensor of shape [batch_size, 20]

        # Let’s just print the first example in that batch:
        example_preds = preds[0]  # shape [20]
        example_pred_labels = pred_labels[0]  # shape [20]
        example_labels = labels[0]  # shape [20]

        # Convert label to integer (0/1) and preds to floats
        # (Lightning’s `self.print` works in Lightning’s logger/console)
        self.print(f"\n--- VALIDATION SAMPLE (epoch={self.current_epoch}) ---")
        self.print("Expected (label[0]):      ", example_labels.int().tolist())
        self.print("Predicted (preds[0]):     ", list(map(int,example_pred_labels.tolist())))
        # Round the model output to 3 decimals for readability:
        self.print("Model output (sigmoid):  ", example_preds.tolist())
        self.print("--- end sample ---\n")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


from typing import Optional, Tuple


class CS277DataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_cls,  # class handle for CS277Dataset
            data_args: dict,  # kwargs to pass into CS277Dataset(...)
            batch_size: int = 32,
            num_workers: int = 4,
            val_split: float = 0.2,
            seed: int = 42
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.data_args = data_args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # Instantiate the full dataset
        full = self.dataset_cls(**self.data_args)
        dataset_length = len(full)
        val_len = int(self.val_split * dataset_length)
        train_len = dataset_length - val_len

        # deterministically split
        self.train_dataset, self.val_dataset = random_split(
            full,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    # 1) Instantiate the DataModule
    data_args = {
        # fill in whatever CS277Dataset needs, e.g. paths, transforms, etc.
        # assume CS277Dataset(root="...", transform=None, ...)
        "root_dir": "/app/GraphCover/Original/vertex_cover_btute_force_20250602_113052",
        "train": True,
    }
    dm = CS277DataModule(
        dataset_cls=CS277Dataset,
        data_args=data_args,
        batch_size=64,
        num_workers=4,
        val_split=0.2,
        seed=42
    )

    # 2) Instantiate the LightningModule
    model = GraphCoverModel(
        hidden_dims=(128, 64),
        lr=1e-3
    )

    # 3) Create a Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        # gpus=1 if torch.cuda.is_available() else 0,
        # progress_bar_refresh_rate=20,
    )

    # 4) Fit
    trainer.fit(model, datamodule=dm)

    # Optionally, test or save the checkpoint afterwards:
    # trainer.test(model, datamodule=dm)
    # trainer.save_checkpoint("graph_cover.ckpt")
