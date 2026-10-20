from pathlib import Path
import time
import click
import pytorch_lightning as pl

from conformal_regressor.datasets import (
    prepare_utkface_data,
    prepare_trees_data,
)
from conformal_regressor.models import ImageRegressionModel, Method


@click.command()
@click.option(
    "--max-epoch", type=int, default=32, help="Maximum number of training epochs"
)
@click.option(
    "--model-name",
    type=str,
    default="drift",
    help="Name of the experiment: (drift, utkface)",
)
@click.option("--batch-size", type=int, default=64, help="Batch size")
@click.option(
    "--method",
    type=click.Choice([m.value for m in Method], case_sensitive=False),
    default=Method.NICP.value,
    help="Method for training (SICP, SPACR, DOICR, NICP, CQR)",
)
@click.option("--epsilon", type=float, default=0.05, help="Conformal training")
@click.option("--lam", type=float, default=5.0, help="Conformal training lambda (for SPACR)")
@click.option("--seed", type=int, default=42, help="Seed for random state")
@click.option("--learning_rate", type=float, default=0.001, help="Learning rate")
@click.option(
    "--checkpoint-dir",
    type=str,
    default=None,
    help="Custom directory to save checkpoints (overrides default naming)"
)
def main(
        max_epoch,
        model_name,
        batch_size,
        method,
        epsilon,
        lam,
        seed,
        learning_rate,
        checkpoint_dir,
):
    method_enum = Method(method.upper())
    print(f"\n--- Initializing Experiment: {model_name.upper()} ---")

    data_prep_fns = {
        "utkface": prepare_utkface_data,
        "drift": prepare_trees_data,
    }

    if model_name in data_prep_fns:
        selected_fn = data_prep_fns[model_name]
        print(f"Loading data for experiment: {model_name}")
        original_train_loader, calib_loader, test_loader = selected_fn(batch_size=batch_size, seed=seed)
        save_name = f"{model_name}_{method_enum.value.lower()}"

    if checkpoint_dir:
        model_root = Path(checkpoint_dir)
    else:
        model_root = Path(__file__).parents[1] / "models_images" / save_name

    model_root.mkdir(parents=True, exist_ok=True)

    print(f"Training Method: {method_enum.value}")
    print(f"Saving checkpoints to: {model_root}")

    if method_enum == Method.CQR:
        print(f"Training CQR with target quantiles: {epsilon / 2} and {1 - epsilon / 2}")

    model = ImageRegressionModel(
        method=method_enum,
        epsilon=epsilon,
        lam=lam,
        max_epochs=max_epoch,
        learning_rate=learning_rate,
        seed=seed,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator="auto",
        devices="auto",
        default_root_dir=model_root,
        precision="32-true",
        deterministic=True,
        enable_checkpointing=True
    )

    start_time = time.time()

    trainer.fit(model, original_train_loader)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds.")

    print("\n--- Running Final Evaluation on Test Set ---")
    trainer.test(model, dataloaders=test_loader)

    print(f"Model saved to {model_root}")

    time_file = model_root / "training_time.txt"
    with open(time_file, "w") as f:
        f.write(f"Experiment: {model_name}\n")
        f.write(f"Method: {method_enum.value}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
