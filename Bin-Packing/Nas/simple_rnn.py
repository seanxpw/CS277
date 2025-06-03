import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pathlib import Path

from cs277_dataset_1d import CS277Dataset


class AccumRNNModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        lr: float = 1e-3,
        seq_len: int = 30
    ):
        """
        RNNModel that processes each sample as a sequence of length `seq_len`.
        On each time step t, it produces a hidden vector h_t ∈ R^{hidden_size},
        then a little linear head projects h_t → a scalar y_t. We sum all y_t over
        t=1..seq_len to get one final scalar per sample.

        Args:
            input_size: size of each input “feature” at each time-step (here =1, since we feed one float at a time)
            hidden_size: RNN hidden dimension
            num_layers: number of stacked RNN layers
            lr: learning rate for Adam
            seq_len: how many time steps (must be 30 for CS277Dataset.size=30)
        """
        super().__init__()
        self.save_hyperparameters()

        # A vanilla RNN (you could also swap in nn.GRU or nn.LSTM here).
        #   input_size=1 (we feed each of the 30 elements as a scalar),
        #   hidden_size=hidden_size, num_layers=num_layers.
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )

        # At each step we map hidden→scalar, then sum all scalars over t=1..30
        self.step2scalar = nn.Linear(hidden_size, 1, bias=True)

        # We'll sum the 30 little scalars to a single prediction
        # Loss is MSE against a single label ∈ ℝ
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, seq_len], where seq_len=30.
        We first unsqueeze(-1) → [B, seq_len, 1], feed into RNN → hiddens [B, seq_len, H].
        Then we project each time‐step hidden→scalar, sum over time‐steps, and return [B,1].
        """
        # x: [B, 30]
        B, L = x.shape
        # Reshape to [B, 30, 1]
        x_seq = x.unsqueeze(-1)

        # hidden_states: [B, 30, H], _ = final hidden (we don't need it separately)
        hidden_states, _ = self.rnn(x_seq)  # batch_first=True

        # Project each h_t → a scalar y_t: shape becomes [B, 30, 1]
        y_t = self.step2scalar(hidden_states)

        # Sum over time dimension: [B, 30, 1] → [B, 1]
        y_sum = y_t.sum(dim=1)  # [B, 1]

        return y_sum  # final scalar per sample

    def training_step(self, batch, batch_idx):
        """
        batch: (inputs, labels), where inputs: [B, 30], labels: [B] or [B, 1].
        """
        inputs, labels = batch
        # Ensure labels shape is [B,1]
        labels = labels.view(-1, 1).float()

        preds = self.forward(inputs)  # [B,1]
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.view(-1, 1).float()

        preds = self.forward(inputs)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # For illustration, we’ll just save the first‐batch preds/labels
        self.val_outputs = {
            "preds": preds.detach(),
            "labels": labels.detach()
        }

    def on_validation_epoch_end(self):
        """
        Print out the first few predictions vs. labels once per validation epoch.
        """
        outs = self.val_outputs
        preds = outs["preds"]   # [B,1]
        labels = outs["labels"] # [B,1]

        # Print the first 4 samples in that batch
        sample_preds = preds[:4].view(-1).tolist()
        sample_labels = labels[:4].view(-1).tolist()

        self.print(f"\n--- VALIDATION SAMPLE (epoch={self.current_epoch}) ---")
        self.print("Expected (labels):   ", sample_labels)
        self.print("Predicted (scalars): ", sample_preds)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



class CS277RNNDataModule(pl.LightningDataModule):
    """
    Exactly the same as CS277DataModule, since CS277Dataset already returns
    1D‐vectors of length 30. We only need to drop the “channel” dimension that
    the CNN version expected.
    """
    def __init__(
        self,
        dataset_cls,       # CS277Dataset
        data_args: dict,   # e.g. {"root_dir": "...", "train": True}
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.data_args = data_args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.problem_size = None

    def setup(self, stage=None):
        # Instantiate the full dataset
        full = self.dataset_cls(**self.data_args)
        # The “size” attribute (e.g. 30) is still available on the dataset
        self.problem_size = full.size

        total_len = len(full)
        val_len = int(self.val_split * total_len)
        train_len = total_len - val_len

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

    def size(self):
        return self.problem_size


def eval_rnn(model: AccumRNNModel, root_dir: str):
    """
    A quick benchmarking / accuracy check, similar to the original `eval(...)`.
    Now each label is scalar, so we round the single‐scalar output.
    """
    import torch.utils.benchmark as benchmark

    model = model.to("cuda")
    dataset = CS277Dataset(root_dir=root_dir, train=False)  # label now assumed to be a single‐scalar per sample
    inputs = torch.Tensor(dataset.input).to("cuda").repeat_interleave(5, dim=0)  # shape [N, 30]
    model = model.to_torchscript(method="trace", example_inputs=inputs)

    model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))

    # warmup
    model(inputs)

    t0 = benchmark.Timer(
        stmt="model(inputs)",
        description=f"batch size = {inputs.shape[0]}",
        globals={"model": model, "inputs": inputs},
    )
    t1 = benchmark.Timer(
        stmt="model(inputs[0:1])",
        description="batch size = 1",
        globals={"model": model, "inputs": inputs},
    )
    print(t0.timeit(100))
    print(t1.timeit(100))

    # Compare predictions vs. “true” integers
    labels = torch.Tensor(dataset.label).to("cuda").view(-1, 1).repeat_interleave(5, dim=0)
    preds = model(inputs).clamp(0.0, 1.0)  # [N,1]
    preds_rounded = torch.round(preds)  # round to nearest integer
    labels_rounded = torch.round(labels)

    accuracy = (preds_rounded == labels_rounded).float().mean().item()
    print("accuracy (all samples):", accuracy)


if __name__ == "__main__":
    # 1) DataModule arguments
    data_args = {
        "root_dir": "/app/Bin-Packing/data/bin_packing_optimal_size=30_dataset_20250602_151119",
        "train": True,
    }

    # 2) If a checkpoint already exists, just load & eval
    if Path("bin-packing-rnn.ckpt").exists():
        print("Checkpoint already exists. Skipping training.")
        rnn_model = AccumRNNModel.load_from_checkpoint("bin-packing-rnn.ckpt")
        eval_rnn(rnn_model, data_args["root_dir"])
        exit(0)

    # 3) Build the DataModule
    dm = CS277RNNDataModule(
        dataset_cls=CS277Dataset,
        data_args=data_args,
        batch_size=64,
        num_workers=4,
        val_split=0.2,
        seed=42,
    )

    # 4) Build the RNN LightningModule
    model = AccumRNNModel(
        input_size=1,
        hidden_size=64,
        num_layers=1,
        lr=1e-5,
        seq_len=30
    )

    # 5) Trainer and fit
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint("bin-packing-rnn.ckpt")
