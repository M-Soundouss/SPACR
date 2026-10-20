from pathlib import Path
import time
import click
import pytorch_lightning as pl
from conformal_regressor.datasets import (
    prepare_superconduct_tabular_data,
    prepare_california_tabular_data,
    prepare_pol_tabular_data,
    prepare_isolet_tabular_data,
    prepare_winequality_tabular_data,
    prepare_brazilian_houses_tabular_data,
    prepare_bikesharing_tabular_data,
    prepare_fifa_tabular_data,
    prepare_cpu_act_tabular_data,
    prepare_diamonds_tabular_data,
    prepare_medical_charges_tabular_data,
    prepare_house_sales_tabular_data,
)
from conformal_regressor.models import RegressionModel, Method


@click.command()
@click.option(
    "--max-epoch", type=int, default=64, help="Maximum number of training epochs"
)
@click.option(
    "--model-name",
    type=str,
    default="superconduct",
    help="Name for saving the model: (superconduct, california, pol,...)",
)
@click.option("--batch-size", type=int, default=256, help="Batch size")
@click.option(
    "--method",
    type=click.Choice([m.value for m in Method], case_sensitive=False),
    default=Method.NICP.value,
    help="Method for training (SICP, SPACR, DOICR, NICP, CQR)",
)
@click.option("--epsilon", type=float, default=0.05, help="Conformal training")
@click.option("--lam", type=float, default=5.0, help="Conformal training lambda (used for SPACR)")
@click.option("--seed", type=int, default=42, help="Seed for random state")
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
        checkpoint_dir,
):
    data_prep_fns = {
        "superconduct": prepare_superconduct_tabular_data,
        "pol": prepare_pol_tabular_data,
        "isolet": prepare_isolet_tabular_data,
        "wine_quality": prepare_winequality_tabular_data,
        "brazilian_houses": prepare_brazilian_houses_tabular_data,
        "bike_sharing": prepare_bikesharing_tabular_data,
        "california": prepare_california_tabular_data,
        "fifa": prepare_fifa_tabular_data,
        "cpu_act": prepare_cpu_act_tabular_data,
        "diamonds": prepare_diamonds_tabular_data,
        "medical_charges": prepare_medical_charges_tabular_data,
        "house_sales": prepare_house_sales_tabular_data,
    }

    if model_name not in data_prep_fns:
        raise ValueError(f"Invalid model name: {model_name}. Available keys: {list(data_prep_fns.keys())}")

    selected_data_fn = data_prep_fns[model_name]

    print(f"Loading data for: {model_name}")

    input_size, original_train_loader, calib_loader, test_loader = selected_data_fn(
        batch_size=batch_size, seed=seed
    )

    method = Method(method.upper())

    if checkpoint_dir:
        model_root = Path(checkpoint_dir)
    else:
        save_name = f"{model_name}_{method.value.lower()}"
        model_root = Path(__file__).parents[1] / "models_tabular" / save_name

    model_root.mkdir(parents=True, exist_ok=True)

    print(f"Training Method: {method.value}")
    print(f"Saving checkpoints to: {model_root}")

    if method == Method.CQR:
        print(f"Training CQR with target quantiles: {epsilon / 2} and {1 - epsilon / 2}")

    model = RegressionModel(
        input_size=input_size,
        method=method,
        epsilon=epsilon,
        lam=lam,
        max_epochs=max_epoch,
        learning_rate=1e-4,
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

    trainer.test(model, dataloaders=test_loader)

    print(f"Model saved to {model_root}")

    time_file = model_root / "training_time.txt"
    with open(time_file, "w") as f:
        f.write(f"Method: {method.value}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")


if __name__ == "__main__":
    main()