import argparse
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from utils.terminal_imshow import imshow

import wandb
from arcagi.data_loader import (
    create_dataloader,
    one_hot_to_categorical,
    prepare_dataset,
)
from arcagi.TGA import RepeatedTGA


class ARCModel(pl.LightningModule):
    """PyTorch Lightning module for ARC-AGI task using TGA."""

    def __init__(
        self,
        embed_dim: int = 48,
        num_q_heads: int = 6,
        num_kv_heads: int = 2,
        attn_dropout: float = 0.01,
        ff_dropout: float = 0.01,
        swiglu_factor: int = 8,
        max_timesteps: int = 50,
        learning_rate: float = 1e-3,
        visualize_freq: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters as instance variables
        self.embed_dim = embed_dim
        self.max_timesteps = max_timesteps
        self.learning_rate = learning_rate
        self.visualize_freq = visualize_freq

        # Input/output projections for the 11 color channels
        self.input_proj = nn.Linear(11, embed_dim)
        self.output_proj = nn.Linear(embed_dim, 11)

        # TGA model
        self.tga = RepeatedTGA(
            embed_dim=embed_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            swiglu_factor=swiglu_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, H, W, 11) with one-hot encoded colors

        Returns:
            Output tensor of shape (B, H, W, 11) with predicted color logits
        """
        B, H, W, _ = x.shape

        # Project input to embedding dimension
        embeddings = self.input_proj(x)  # (B, H, W, embed_dim)

        # Process each example in the batch
        outputs: List[torch.Tensor] = []
        for i in range(B):
            # Get single example
            grid = embeddings[i]  # (H, W, embed_dim)

            # Initialize empty history
            history = torch.empty(H, W, 0, self.embed_dim, device=x.device)

            # Run TGA iterations
            grid, _ = self.tga(grid, history, self.max_timesteps, local_mix=False)

            outputs.append(grid)

        # Stack outputs back to batch
        outputs_tensor = torch.stack(outputs, dim=0)  # (B, H, W, embed_dim)

        # Project to output space
        logits = self.output_proj(outputs_tensor)  # (B, H, W, 11)

        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch

        # Forward pass
        logits = self(inputs)

        # Compute loss - cross entropy over the color dimension
        loss = F.cross_entropy(
            logits.reshape(-1, 11),
            targets.argmax(dim=-1).reshape(-1),
            ignore_index=-1,  # In case we have padding
        )

        # Compute fraction of correct pixels
        predictions = logits.argmax(dim=-1)
        targets_cat = targets.argmax(dim=-1)
        fraction_correct = (predictions == targets_cat).float().mean()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_fraction_correct", fraction_correct, prog_bar=True)

        # Make one-hot predictions & targets
        pred_one_hot = F.one_hot(predictions, num_classes=11)
        target_one_hot = F.one_hot(targets_cat, num_classes=11)

        # Visualize first example in batch based on frequency
        if (
            self.visualize_freq > 0
            and batch_idx % self.visualize_freq == 0
            and batch_idx < 50
        ):  # Limit to first 50 batches
            # Convert to numpy for visualization
            input_grid = one_hot_to_categorical(inputs[0]).cpu()
            pred_grid = one_hot_to_categorical(pred_one_hot[0]).cpu()
            target_grid = one_hot_to_categorical(target_one_hot[0]).cpu()

            print(f"\n=== Validation Batch {batch_idx}, Example 0 ===")
            print("Input:")
            imshow(input_grid)
            print("\nPredicted Output:")
            imshow(pred_grid)
            print("\nActual Output:")
            imshow(target_grid)

            # Calculate accuracy for this specific example
            example_accuracy = (pred_grid == target_grid).float().mean().item()
            print(f"Example accuracy: {example_accuracy:.2%}")
            print("=" * 40)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch

        # Forward pass
        logits = self(inputs)

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, 11), targets.argmax(dim=-1).reshape(-1), ignore_index=-1
        )

        # Compute fraction of correct pixels
        predictions = logits.argmax(dim=-1)
        targets_cat = targets.argmax(dim=-1)
        fraction_correct = (predictions == targets_cat).float().mean()

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_fraction_correct", fraction_correct, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    limit_examples: Optional[int] = None,
    augment_factor: int = 1,
    filter_filename: Optional[str] = None,
) -> Tuple[DataLoader[Any], Optional[DataLoader[Any]]]:
    """Create train and validation dataloaders from parquet files."""

    # Prepare training data
    train_inputs, train_outputs = prepare_dataset(
        parquet_path=train_path,
        filter_filename=filter_filename,
        limit_examples=limit_examples,
        augment_factor=augment_factor,
        dataset_name="training data",
    )

    # Create training dataloader
    train_loader = create_dataloader(
        inputs=train_inputs,
        outputs=train_outputs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Create validation dataloader if path provided
    val_loader = None
    if val_path is not None:
        val_inputs, val_outputs = prepare_dataset(
            parquet_path=val_path,
            filter_filename=filter_filename,
            limit_examples=limit_examples,
            augment_factor=1,  # No augmentation for validation
            dataset_name="validation data",
        )

        val_loader = create_dataloader(
            inputs=val_inputs,
            outputs=val_outputs,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=num_workers,
        )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train ARC-AGI model with TGA")

    # Data arguments
    parser.add_argument(
        "--train_path",
        type=str,
        default="processed_data/train.parquet",
        help="Path to training parquet file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to validation parquet file (optional)",
    )
    parser.add_argument(
        "--limit_examples",
        type=int,
        default=None,
        help="Limit number of examples to use (for debugging)",
    )
    parser.add_argument(
        "--augment_factor",
        type=int,
        default=1,
        help="Factor for data augmentation via color permutation",
    )
    parser.add_argument(
        "--filter_filename",
        type=str,
        default=None,
        help="Filter examples to only include those from this filename (e.g., 025d127b.json)",
    )

    # Model arguments
    parser.add_argument("--embed_dim", type=int, default=48, help="Embedding dimension")
    parser.add_argument(
        "--num_q_heads", type=int, default=6, help="Number of query heads"
    )
    parser.add_argument(
        "--num_kv_heads", type=int, default=2, help="Number of key/value heads"
    )
    parser.add_argument(
        "--attn_dropout", type=float, default=0.1, help="Attention dropout"
    )
    parser.add_argument(
        "--ff_dropout", type=float, default=0.1, help="Feed-forward dropout"
    )
    parser.add_argument(
        "--swiglu_factor", type=int, default=8, help="SwiGLU expansion factor"
    )
    parser.add_argument(
        "--max_timesteps", type=int, default=3, help="Number of TGA iterations"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")

    # Wandb arguments
    parser.add_argument(
        "--wandb_project", type=str, default="arc-agi", help="Wandb project name"
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Visualization arguments
    parser.add_argument(
        "--visualize_freq",
        type=int,
        default=10,
        help="Frequency of visualization during validation (0 to disable)",
    )

    args = parser.parse_args()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_examples=args.limit_examples,
        augment_factor=args.augment_factor,
        filter_filename=args.filter_filename,
    )

    # Create model
    model = ARCModel(
        embed_dim=args.embed_dim,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout,
        swiglu_factor=args.swiglu_factor,
        max_timesteps=args.max_timesteps,
        learning_rate=args.learning_rate,
        visualize_freq=args.visualize_freq,
    )

    # Initialize wandb logger if not disabled
    logger = None
    if not args.no_wandb:
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "embed_dim": args.embed_dim,
                "num_q_heads": args.num_q_heads,
                "num_kv_heads": args.num_kv_heads,
                "attn_dropout": args.attn_dropout,
                "ff_dropout": args.ff_dropout,
                "swiglu_factor": args.swiglu_factor,
                "max_timesteps": args.max_timesteps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "augment_factor": args.augment_factor,
                "filter_filename": args.filter_filename,
                "limit_examples": args.limit_examples,
            },
        )
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            log_model=True,  # Log model checkpoints
        )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=16 if args.gpus > 0 else 32,  # Use mixed precision on GPU
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Training complete!")

    # Finish wandb run
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
