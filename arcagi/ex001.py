# Try a fixed-weights UNET model for a single example, repeated for many layers in
# residual blocks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Any, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Fix imports to avoid "arcagi" prefix
from data_loader import load_parquet_data, repeat_and_permute, one_hot_to_categorical
from unet import ResidualRepeatedFixedWeightsUNet


class SimpleCNN(nn.Module):
    """A simple CNN that can handle different input and output sizes."""

    def __init__(self, in_channels: int = 11, out_channels: int = 11) -> None:
        super().__init__()

        self.model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Third convolutional block
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Final convolutional layer to map to output channels
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SingleExampleUNetLightning(pl.LightningModule):
    def __init__(
        self,
        num_residual_layers: int = 5,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Create a simpler model that can handle our dimensions
        base_model = SimpleCNN(in_channels=11, out_channels=11)

        # Use the residual repeated architecture
        self.model = ResidualRepeatedFixedWeightsUNet(
            num_layers=num_residual_layers, base_module=base_model
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Store learning rate as a class attribute
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def _calculate_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        # Convert one-hot targets to class indices for CrossEntropyLoss
        targets_indices = torch.argmax(targets, dim=3)

        return self.criterion(outputs, targets_indices)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss = self._calculate_loss(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}


class SingleExampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        target_filename: str,
        num_permutations: int = 100,
        batch_size: int = 16,
        val_split: float = 0.2,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.target_filename = target_filename
        self.num_permutations = num_permutations
        self.batch_size = batch_size
        self.val_split = val_split

        # Will be set during setup
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None

    def prepare_data(self) -> None:
        # This method is called only once and on only one GPU
        # Download/prepare data if needed (not applicable in our case)
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Load all data
        filenames, _, inputs, outputs = load_parquet_data(self.data_path)

        # Find the target example
        target_indices: List[int] = [
            i for i, fname in enumerate(filenames) if fname == self.target_filename
        ]

        if not target_indices:
            raise ValueError(
                f"Filename {self.target_filename} not found in the dataset"
            )

        # Get the first matching example
        target_idx: int = target_indices[0]
        target_input = inputs[target_idx : target_idx + 1]  # Add batch dimension
        target_output = outputs[target_idx : target_idx + 1]

        # Generate permutations
        permuted_inputs, permuted_outputs = repeat_and_permute(
            target_input, target_output, self.num_permutations
        )

        # Permute dimensions for model input (B, H, W, C) -> (B, C, H, W)
        inputs_for_model = permuted_inputs.permute(0, 3, 1, 2)
        outputs_for_loss = permuted_outputs

        # Split into training and validation sets
        total_samples = inputs_for_model.size(0)
        val_size = int(total_samples * self.val_split)
        train_size = total_samples - val_size

        train_inputs = inputs_for_model[:train_size]
        train_outputs = outputs_for_loss[:train_size]

        val_inputs = inputs_for_model[train_size:]
        val_outputs = outputs_for_loss[train_size:]

        # Create datasets
        self.train_dataset = TensorDataset(train_inputs, train_outputs)
        self.val_dataset = TensorDataset(val_inputs, val_outputs)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set. Did you call setup()?")
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("val_dataset is not set. Did you call setup()?")
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )


def main() -> None:
    # Initialize data module with correct path (one directory up)
    data_module = SingleExampleDataModule(
        data_path="../processed_data/train.parquet",
        target_filename="025d127b.json",  # Example filename
        num_permutations=100,
        batch_size=16,
    )

    # Initialize model
    model = SingleExampleUNetLightning(
        num_residual_layers=5,
        learning_rate=1e-4,
    )

    # Define callbacks
    callbacks: List[pl.Callback] = [
        ModelCheckpoint(
            monitor="val_loss",
            filename="unet-{epoch:02d}-{val_loss:.6f}",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Visualize a sample prediction
    try:
        from utils.terminal_imshow import imshow

        # Set model to eval mode
        model.eval()

        # Load a single example for visualization
        dataloader = data_module.val_dataloader()
        batch = next(iter(dataloader))
        inputs, targets = batch
        sample_input = inputs[0].unsqueeze(0)  # Add batch dimension

        # Get prediction
        with torch.no_grad():
            output = model(sample_input)

        # Convert to categorical indices
        # Convert input from (C, H, W) to (H, W, C) for argmax
        sample_input_cat = torch.argmax(sample_input[0].permute(1, 2, 0), dim=2)

        # Convert targets from (H, W, C) format to indices
        target_cat = torch.argmax(targets[0], dim=2)

        # Convert output from (C, H, W) to indices
        pred_cat = torch.argmax(output[0], dim=0)

        # Display
        print("\nSample Input:")
        imshow(sample_input_cat)

        print("\nSample Prediction:")
        imshow(pred_cat)

        print("\nSample Target:")
        imshow(target_cat)

    except ImportError:
        print("Could not import terminal_imshow for visualization")


if __name__ == "__main__":
    main()
