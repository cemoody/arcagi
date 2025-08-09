import os

# Import from parent module
import sys
from typing import Any, Dict, List, Literal, Tuple

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydanclick import from_pydantic
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our visualization tools
# Import data loading functions
from data_loader import create_dataloader, prepare_dataset
from models import Batch, BatchData
from utils.metrics import image_metrics
from utils.terminal_imshow import imshow

# Import Order2Features from lib
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TrainingConfig(BaseModel):
    """Configuration for training the order2-based neural cellular automata model."""

    # Data and checkpoint paths
    data_dir: str = "/tmp/arc_data"
    checkpoint_dir: str = "order2_checkpoints"

    # Training parameters
    max_epochs: int = 2000
    lr: float = 1e-2
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    num_message_rounds: int = 24

    # Self-healing noise parameters
    enable_self_healing: bool = True
    death_prob: float = 0.02
    gaussian_std: float = 0.05
    salt_pepper_prob: float = 0.01
    spatial_corruption_prob: float = 0.01

    # Model parameters
    dropout: float = 0.1
    temperature: float = 1.0
    filename: str = "3345333e"

    # Early stopping parameters
    patience: int = 100
    min_epochs: int = 500

    # Training mode
    single_file_mode: bool = True

    # Noise parameters
    noise_prob: float = 0.10
    num_final_steps: int = 6


def batch_to_dataclass(
    batch: Tuple[torch.Tensor, ...],
) -> BatchData:
    inputs_one_hot, outputs_one_hot, input_features, output_features, indices = batch

    # Ensure features are float tensors for MPS compatibility
    input_features = input_features.float()
    output_features = output_features.float()

    # Extract colors and masks from the grid data
    input_colors = inputs_one_hot.argmax(dim=-1).long()
    output_colors = outputs_one_hot.argmax(dim=-1).long()

    # Change any colors >= 10 to -1 (mask)
    input_colors = torch.where(input_colors >= 10, -1, input_colors)
    output_colors = torch.where(output_colors >= 10, -1, output_colors)

    # Create masks (True where valid colors exist)
    input_masks = input_colors >= 0
    output_masks = output_colors >= 0

    # Get example indices for this batch
    input_colors_flat = input_colors.reshape(-1)
    output_colors_flat = output_colors.reshape(-1)
    input_masks_flat = input_masks.reshape(-1)
    output_masks_flat = output_masks.reshape(-1)

    inp = Batch(
        one=inputs_one_hot,
        fea=input_features,
        col=input_colors,
        msk=input_masks,
        colf=input_colors_flat,
        mskf=input_masks_flat.float(),
        idx=indices,
    )

    out = Batch(
        one=outputs_one_hot,
        fea=output_features,
        col=output_colors,
        msk=output_masks,
        colf=output_colors_flat,
        mskf=output_masks_flat.float(),
        idx=indices,
    )

    return BatchData(inp=inp, out=out)


def cross_entropy_shifted(
    logits: torch.Tensor, targets: torch.Tensor, start_index: int = -1, **kwargs: Any
) -> torch.Tensor:
    """
    Cross-entropy loss for class indices starting at `start_index` instead of 0.

    Args:
        logits: [N, C, ...] raw predictions (unnormalized)
        targets: [N, ...] integer class labels in range [start_index, start_index + C - 1]
        start_index: the smallest valid class index (default: -1)
        **kwargs: extra args for F.cross_entropy (e.g., reduction, ignore_index)
    """
    # Shift targets so that smallest index becomes 0
    shifted_targets = targets - start_index
    return F.cross_entropy(logits, shifted_targets, **kwargs)


class MainModel(pl.LightningModule):
    """
    Super simple linear model.
    """

    training_index_metrics: Dict[int, Dict[str, int]] = {}
    validation_index_metrics: Dict[int, Dict[str, int]] = {}
    force_visualize: bool = False

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(32, 11)
        self.embeddings = nn.Embedding(11, 32)

    def forward(
        self,
        colors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass processing colors through order2 features.

        Args:
            colors: [B, 30, 30] integer color values (-1 for mask)

        Returns:
            color_logits: [B, 30, 30, num_classes]
            mask_logits: [B, 30, 30, 1]
        """
        # Step 1: Convert colors to order2 features
        embeddings = self.embeddings(colors.long())
        logits = self.linear(embeddings.float())  # [B, 30, 30, 11]
        # Rearrange for loss: CrossEntropyLoss expects [B, C, H, W]
        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, step_type="val")

    def step(
        self,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        step_type: Literal["train", "val"] = "train",
    ) -> torch.Tensor:
        i = batch_to_dataclass(batch)

        # Compute losses for color predictions
        metrics: Dict[int, Dict[str, float]] = {}
        # Use color grids instead of features
        logits = self(i.inp.col)
        logits_perm = logits.permute(0, 3, 1, 2)  # [B, 30, 30, 11]

        # Compute loss
        loss = cross_entropy_shifted(logits_perm, i.out.col.long(), start_index=-1)

        # col_log = logits[:, :, :, 1:]
        # msk_log = logits[:, :, :, 0]
        image_metrics(i.out, logits, metrics, prefix=step_type)

        # Visualize every 10 epochs
        if self.force_visualize or (self.current_epoch % 10 == 0 and batch_idx == 0):
            self.visualize_predictions(
                i,
                i.out.col,
                logits,
                f"step-{step_type}",
            )
        # Convert float metrics to int for compatibility
        int_metrics: Dict[int, Dict[str, int]] = {}
        for idx, m in metrics.items():
            int_metrics[idx] = {k: int(v) for k, v in m.items()}

        if step_type == "train":
            self.training_index_metrics = int_metrics
        else:
            self.validation_index_metrics = int_metrics

        # Only log metrics during training or validation, not during test evaluation
        self.log_metrics(int_metrics, step_type)
        self.log(f"{step_type}_total_loss", loss)  # type: ignore

        return loss

    def log_metrics(
        self, metrics: Dict[int, Dict[str, int]], prefix: str = "train"
    ) -> None:
        for idx in metrics.keys():
            for key, value in metrics[idx].items():
                self.log(f"{prefix}_{key}", value)  # type: ignore

    def on_train_epoch_end(self) -> None:
        """Print training per-index accuracy at the end of each epoch."""
        self.epoch_end("train")

    def on_train_epoch_start(self) -> None:
        # logger.info(f"Training epoch start: {self.current_epoch}")
        pass

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at the start of each epoch."""
        self.validation_index_metrics = {}

    def on_validation_epoch_end(self) -> None:
        """Compute and log epoch-level validation metrics."""
        self.epoch_end("val")

        # Compute epoch-level metrics from validation step outputs
        val_loss = self.trainer.callback_metrics.get("val_total_loss", 0.0)

        # Calculate color and mask accuracy from validation metrics
        val_input_pixels_incorrect = self.trainer.callback_metrics.get(
            "val_input_pixels_incorrect", 0.0
        )
        val_output_pixels_incorrect = self.trainer.callback_metrics.get(
            "val_output_pixels_incorrect", 0.0
        )
        val_input_pixels_per_index = self.trainer.callback_metrics.get(
            "val_input_pixels_per_index", 1.0
        )
        val_output_pixels_per_index = self.trainer.callback_metrics.get(
            "val_output_pixels_per_index", 1.0
        )

        # Calculate accuracy as (1 - error_rate)
        # Handle both tensor and float cases
        if isinstance(val_input_pixels_per_index, torch.Tensor):
            val_input_pixels_per_index = val_input_pixels_per_index.item()
        if isinstance(val_output_pixels_per_index, torch.Tensor):
            val_output_pixels_per_index = val_output_pixels_per_index.item()
        if isinstance(val_input_pixels_incorrect, torch.Tensor):
            val_input_pixels_incorrect = val_input_pixels_incorrect.item()
        if isinstance(val_output_pixels_incorrect, torch.Tensor):
            val_output_pixels_incorrect = val_output_pixels_incorrect.item()

        val_input_color_acc = 1.0 - (
            val_input_pixels_incorrect / max(val_input_pixels_per_index, 1.0)
        )
        val_output_color_acc = 1.0 - (
            val_output_pixels_incorrect / max(val_output_pixels_per_index, 1.0)
        )

        # For now, use color accuracy for mask accuracy
        val_input_mask_acc = val_input_color_acc
        val_output_mask_acc = val_output_color_acc

        # Log epoch-level metrics that ModelCheckpoint expects
        self.log("val_epoch_loss", val_loss, prog_bar=True)
        self.log(
            "val_epoch_color_acc", val_output_color_acc, prog_bar=True
        )  # Main metric for checkpointing
        self.log("val_epoch_mask_acc", val_output_mask_acc, prog_bar=True)
        self.log("val_input_color_acc", val_input_color_acc)
        self.log("val_output_color_acc", val_output_color_acc)
        self.log("val_input_mask_acc", val_input_mask_acc)
        self.log("val_output_mask_acc", val_output_mask_acc)

    def epoch_end(self, step_type: Literal["train", "val"]) -> None:
        if step_type == "train":
            metrics = self.training_index_metrics
        else:
            metrics = self.validation_index_metrics

        if metrics:
            print(
                f"\n{step_type.upper()} Per-Index Accuracy (Epoch {self.current_epoch}):"
            )
            print(
                f"{'Index':<8} {'Output Pixels':<12} {'Output Mask':<12} {'All Perfect':<12}"
            )
            print("-" * 50)

            for idx in sorted(self.training_index_metrics.keys()):
                metrics = self.training_index_metrics[idx]
                out_pix_incor = metrics["train_n_incorrect_num_color"]
                out_msk_incor = metrics["train_n_incorrect_num_mask"]

                all_perfect = "✓" if out_pix_incor == 0 else "✗"
                print(
                    f"{idx:<8} {out_pix_incor:<12} {out_msk_incor:<12} {all_perfect:<12}"
                )

        # Clear training metrics for next epoch
        self.training_index_metrics = {}

    def configure_optimizers(self) -> Dict[str, Any]:
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.01,
            weight_decay=0.0,
        )

        # Learning rate schedule
        def lr_lambda(epoch: int) -> float:
            if epoch < 2:
                return (epoch + 1) / 2  # Quick warmup
            elif epoch < 30:
                return 1.0  # Full LR
            elif epoch < 50:
                return 0.5  # Half LR
            elif epoch < 500:
                return 0.1  # Half LR
            elif epoch < 1000:
                return 0.01  # Half LR
            else:
                return 0.001  # Low LR for fine-tuning

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def visualize_predictions(
        self,
        i: BatchData,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        prefix: str,
        start_index: int = -1,
    ) -> None:
        """Visualize predictions vs ground truth for training."""
        # Get the first unique index from the batch
        _, inverse_indices = torch.unique(i.inp.idx, return_inverse=True)  # type: ignore

        for idx in inverse_indices:
            pred_colors = predictions[idx].argmax(dim=-1).cpu() + start_index
            true_colors = targets[idx].cpu()

            # Create visualization
            print(f"\n{'='*60}")
            print(f"{prefix.upper()} - Epoch {self.current_epoch}")
            print(f"{'='*60}")

            # Show ground truth
            print("\nGround Truth colors:")
            imshow(true_colors, title=None, show_legend=True)

            # Show predictions
            print("\nPredicted colors:")
            correct = (true_colors == pred_colors) | (true_colors == -1)
            imshow(pred_colors, title=None, show_legend=True, correct=correct)

            # Calculate accuracy for this example
            valid_mask = true_colors != -1
            if valid_mask.any():
                accuracy = (
                    (pred_colors[valid_mask] == true_colors[valid_mask]).float().mean()
                )

                # Also show mask predictions vs ground truth
                print("\nMask predictions (True=active pixel, False=inactive):")
                print(f"\nColor Accuracy: {accuracy:.2%}")


@click.command()
@from_pydantic(
    TrainingConfig,
    rename={
        field_name: f"--{field_name}"
        for field_name in TrainingConfig.model_fields.keys()
        if "_" in field_name
    },
)
def main(training_config: TrainingConfig):

    # Target filename
    filename = training_config.filename

    print("\n=== Order2-based Model (ex30) ===")
    print(f"Training on 'train' subset of {filename}")
    print(f"Evaluating on 'test' subset of {filename}")
    print("Architecture: color -> order2 -> NCA -> order2 -> color")

    # Load train subset using data_loader.py
    (
        train_inputs,
        train_outputs,
        train_input_features,
        train_output_features,
        train_indices,
    ) = prepare_dataset(
        "processed_data/train_all_d4aug.npz",
        filter_filename=f"{filename}.json",
        use_features=True,
        dataset_name="train",
        use_train_subset=True,
    )

    # Load test subset using data_loader.py
    (
        test_inputs,
        test_outputs,
        test_input_features,
        test_output_features,
        test_indices,
    ) = prepare_dataset(
        "processed_data/train_all.npz",
        filter_filename=f"{filename}.json",
        use_features=True,
        dataset_name="test",
        use_train_subset=False,
    )

    # Get available colors from training data
    # Convert one-hot encoded tensors back to color indices
    train_input_colors = train_inputs.argmax(dim=-1)  # [B, 30, 30]
    train_output_colors = train_outputs.argmax(dim=-1)  # [B, 30, 30]

    # Handle mask: if max value is at index 10, it means masked (-1)
    train_input_colors = torch.where(train_input_colors == 10, -1, train_input_colors)
    train_output_colors = torch.where(
        train_output_colors == 10, -1, train_output_colors
    )

    all_colors = torch.cat(
        [train_input_colors.flatten(), train_output_colors.flatten()]
    )

    # Create dataloaders using data_loader.py
    train_loader = create_dataloader(
        train_inputs,
        train_outputs,
        batch_size=len(train_inputs),
        shuffle=True,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        inputs_features=train_input_features,
        outputs_features=train_output_features,
        indices=train_indices,
    )
    val_loader = create_dataloader(
        test_inputs,
        test_outputs,
        batch_size=len(test_inputs),
        shuffle=False,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        inputs_features=test_input_features,
        outputs_features=test_output_features,
        indices=test_indices,
    )
    test_loader = create_dataloader(
        test_inputs,
        test_outputs,
        batch_size=len(test_inputs),
        shuffle=False,
        num_workers=0,  # Eliminate multiprocessing overhead for small datasets
        inputs_features=test_input_features,
        outputs_features=test_output_features,
        indices=test_indices,
    )

    # Create model
    model = MainModel()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel has {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Create trainer with callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch_color_acc",  # Monitor epoch-level color accuracy
        dirpath=os.path.join(training_config.checkpoint_dir, "checkpoints"),
        filename="order2-{epoch:02d}-{val_epoch_loss:.4f}-{val_epoch_color_acc:.4f}-{val_epoch_mask_acc:.4f}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=training_config.checkpoint_dir,
        gradient_clip_val=1.0,  # Add gradient clipping for stability
    )

    trainer.fit(model, train_loader, val_loader)

    # Evaluate on test set if in single file mode
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    # Run test evaluation
    model.eval()
    model.force_visualize = True
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model.validation_step(batch, batch_idx)

    # Save the final trained model with filename-specific name
    final_model_path = os.path.join(
        training_config.checkpoint_dir, f"order2_model_{filename}.pt"
    )

    # Save both model state dict and metadata
    model_save_data = {
        "model_state_dict": model.state_dict(),
        "filename": filename,
        "hyperparameters": (
            dict(model.hparams) if hasattr(model.hparams, "__dict__") else model.hparams
        ),
    }

    torch.save(model_save_data, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
