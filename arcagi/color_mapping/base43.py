import hashlib
import json
import os
from typing import List, Optional, Tuple, Type

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydanclick import from_pydantic
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore
from loguru import logger

from arcagi.data_loader import create_shot_dataloader


class TrainingConfig(BaseModel):
    """Configuration for training the simple MLP few-shot learning model."""

    # Data parameters
    npz_path: str = "/home/chris/arcagi/processed_data/train_all.npz"
    batch_size: int = 32
    shuffle: bool = True
    filename_filter: Optional[str] = None

    # Model parameters
    hidden_dim: int = 512
    num_hidden_layers: int = 3
    dropout: float = 0.1

    # Training parameters
    max_epochs: int = 1000
    lr: float = 0.0001
    weight_decay: float = 1e-4

    # Checkpoint parameters
    checkpoint_dir: str = "ex43_checkpoints"

    # Logging parameters
    project_name: str = "arcagi-ex43"
    log_every_n_steps: int = 10


class SimpleMLP(nn.Module):
    """Simple MLP that embeds input images and predicts output images."""

    def __init__(
        self, hidden_dim: int = 512, num_hidden_layers: int = 3, dropout: float = 0.1
    ):
        super().__init__()  # type: ignore

        # Input dimension: 30x30x11 = 9900
        input_dim = 30 * 30 * 11
        output_dim = 30 * 30 * 11

        # Build MLP layers
        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        input_images: torch.Tensor,
        filenames: torch.Tensor,
        example_indices: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the MLP."""
        x = input_images
        batch_size = x.shape[0]

        # Flatten the input
        x_flat = x.view(batch_size, -1)  # [B, 9900]

        # Pass through MLP
        output_flat = self.mlp(x_flat)  # [B, 9900]

        # Reshape to original dimensions
        output = output_flat.view(batch_size, 30, 30, 11)

        return output


class MainModule(pl.LightningModule):
    """PyTorch Lightning module for few-shot learning with simple MLP."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Create the model
        self.model = SimpleMLP(
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )

        # Save hyperparameters for logging
        self.save_hyperparameters(config.model_dump())

    def forward(
        self,
        input_images: torch.Tensor,
        filenames: torch.Tensor,
        example_indices: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_images=input_images,
            filenames=filenames,
            example_indices=example_indices,
            context=context,
            current_epoch=self.current_epoch,
        )

    def step(
        self,
        batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        is_train: bool = True,
    ) -> torch.Tensor:
        """Common step logic for both training and validation."""
        # filenames shape [B]
        # example_indices shape [B]
        # output_images shape [B, 30, 30, 11]
        # input_images shape [B, 30, 30, 11]
        # context shape [B, n_examples-1, 2, 30, 30, 11]
        filenames, example_indices, input_images, output_images, context = batch

        # Forward pass
        predicted_outputs = self.model(
            input_images,
            filenames=filenames,
            example_indices=example_indices,
            context=context,
            current_epoch=self.current_epoch,
        )  # [B, 30, 30, 11] logits

        # Convert one-hot encoded targets to class indices
        true_classes = output_images.argmax(dim=-1)  # [B, 30, 30]

        # Reshape for cross-entropy loss: expects [B, C, H, W]
        logits_reshaped = predicted_outputs.permute(0, 3, 1, 2)  # [B, 11, 30, 30]

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits_reshaped, true_classes)

        # Calculate accuracy (argmax accuracy)
        pred_classes = predicted_outputs.argmax(dim=-1)  # [B, 30, 30]
        accuracy = (pred_classes == true_classes).float().mean()

        # Calculate pixel-level accuracy metrics
        incorrect_pixels = (pred_classes != true_classes).sum(
            dim=(1, 2)
        )  # [B] - incorrect pixels per image
        avg_incorrect_pixels = incorrect_pixels.float().mean()  # Average across batch

        # Determine prefix for metrics
        prefix = "train" if is_train else "val"

        # Log basic metrics
        self.log(f"{prefix}_loss", loss, on_step=is_train, prog_bar=True)
        self.log(
            f"{prefix}_accuracy",
            accuracy,
            on_step=is_train,
            prog_bar=True,
        )
        self.log(
            f"{prefix}_bad_pixels",
            avg_incorrect_pixels,
            on_step=is_train,
            prog_bar=True,
        )
        return loss

    def training_step(
        self,
        batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step for one batch."""
        return self.step(batch, batch_idx, is_train=True)

    def validation_step(
        self,
        batch: Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Validation step for one batch."""
        self.step(batch, batch_idx, is_train=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        return optimizer


@click.command()
@from_pydantic(
    TrainingConfig,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfig.model_fields.keys()
        if "_" in field_name
    },
)
def main_click(training_config: TrainingConfig):
    return main(training_config, MainModule)


def main(training_config: TrainingConfig, ModelClass: Type[MainModule]):
    """Main training function."""

    # Log configuration
    logger.info(f"Starting training with config: {training_config.model_dump()}")

    # Create checkpoint directory
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    # Create data loader
    logger.info(f"Loading data from {training_config.npz_path}")
    train_dataloader = create_shot_dataloader(
        npz_path=training_config.npz_path,
        batch_size=training_config.batch_size,
        shuffle=training_config.shuffle,
        filename_filter=training_config.filename_filter,
        context_includes_train=True,
        context_includes_test=False,
        subset="train",  # Only include train examples in training batches
    )

    # Create validation dataloader (using same data but no shuffle for now)
    val_dataloader = create_shot_dataloader(
        npz_path=training_config.npz_path,
        batch_size=training_config.batch_size,
        shuffle=False,
        filename_filter=training_config.filename_filter,
        context_includes_train=True,
        context_includes_test=True,
        subset="test",  # Only include test examples in validation batches
    )

    # Log dataset statistics
    # Note: create_shot_dataloader returns a custom dataset
    logger.info(f"Total training batches: {len(train_dataloader)}")
    logger.info(f"Number of batches: {len(train_dataloader)}")

    # Sample one batch to check shapes
    for batch in train_dataloader:
        filenames, _, input_images, output_images, context = batch
        logger.info(f"Batch shapes:")
        logger.info(f"  - Input images: {input_images.shape}")
        logger.info(f"  - Output images: {output_images.shape}")
        logger.info(f"  - Context: {context.shape}")
        logger.info(f"  - Number of unique files in batch: {len(set(filenames))}")
        break

    # Create Lightning module
    model = ModelClass(training_config)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=training_config.checkpoint_dir,
        filename="ex43-{epoch:04d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    # Setup wandb logger if available
    # Generate hash of training config for unique identification
    config_dict = training_config.model_dump()
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    # Create run name with config hash
    run_name = f"ex43-mlp-{config_hash}"

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=training_config.project_name,
        name=run_name,
        config=config_dict,
        save_dir=training_config.checkpoint_dir,
    )

    logger.info(f"Initialized wandb logging: {run_name}")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=training_config.checkpoint_dir,
        logger=wandb_logger,  # type: ignore[arg-type]
        log_every_n_steps=training_config.log_every_n_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",  # Use bfloat16 mixed precision
    )

    # Train the model
    logger.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    logger.info("Training completed!")

    # Log final validation metrics
    final_metrics = trainer.validate(model, val_dataloader)
    logger.info(f"Final validation metrics: {final_metrics}")


if __name__ == "__main__":
    main_click()
