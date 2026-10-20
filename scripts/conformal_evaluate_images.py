from pathlib import Path
import click
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import warnings
import os
import time
import torch

from conformal_regressor.datasets import (
    prepare_utkface_data,
    prepare_trees_data,
)

from conformal_regressor.models import Method, ImageRegressionModel

torch.serialization.add_safe_globals([Method, ImageRegressionModel])

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')
os.environ["MKL_THREADING_LAYER"] = "GNU"


def get_predictions(model, data_loader):
    predictions, uncertainties, error_estimations = [], []
    quant_lows, quant_highs, true_labels = [], [], []

    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    start_time = time.time()
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)

            predictions.append(output["y_hat"].cpu().numpy())
            uncertainties.append(output["uncertainty"].cpu().numpy())
            error_estimations.append(output["error_estimation"].cpu().numpy())
            quant_lows.append(output["quant_low"].cpu().numpy().flatten())
            quant_highs.append(output["quant_high"].cpu().numpy().flatten())
            true_labels.append(y.cpu().numpy())

    end_time = time.time()
    total_inference_time = end_time - start_time

    return (
        np.concatenate(predictions).ravel(), np.concatenate(uncertainties).ravel(),
        np.concatenate(error_estimations).ravel(), np.concatenate(quant_lows).ravel(),
        np.concatenate(quant_highs).ravel(), np.concatenate(true_labels).ravel(),
        total_inference_time
    )


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = list(Path(checkpoint_dir).rglob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
    latest_hparams = latest_checkpoint.parents[1] / "hparams.yaml"
    return latest_checkpoint, latest_hparams


def get_conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if q_level > 1.0:
        print(f"Warning: Calibration set size ({n}) is too small for strict coverage at alpha={alpha}. Returning max.")
        return np.max(scores)
    return np.quantile(scores, q_level, method='higher')


@click.command()
@click.option("--batch-size", type=int, default=512, help="Batch size")
@click.option("--model-name", type=click.Choice(["drift", "utkface"]), default="drift",
              help="Image dataset to evaluate")
@click.option("--method", type=click.Choice([m.value for m in Method], case_sensitive=False), default=Method.SICP.value,
              help="Training method")
@click.option("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
@click.option("--epsilon", type=float, default=0.05, help="Conformal alpha")
@click.option("--seed", type=int, default=42, help="Seed for random state")
@click.option("--output-dir", type=str, default="experimental_results", help="Directory for experimental results")
def test(batch_size, model_name, method, checkpoint_dir, epsilon, seed, output_dir):
    method = Method(method.upper())

    data_prep_fns = {
        "utkface": prepare_utkface_data,
        "drift": prepare_trees_data,
    }

    print(f"\n--- Initializing Evaluation: {model_name.upper()} ---")

    selected_fn = data_prep_fns[model_name]
    print(f"Loading data for experiment: {model_name}")
    train_loader, val_loader, test_loader = selected_fn(batch_size=batch_size, seed=seed)
    experiment_name = model_name
    save_name = f"{model_name}_{method.value.lower()}"

    if not checkpoint_dir:
        project_root = Path(__file__).parent.parent
        checkpoint_dir = project_root / f"models_images/{save_name}/"

    print(f"Searching for checkpoint in: {checkpoint_dir}")
    checkpoint, hparams = find_latest_checkpoint(checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint}")

    model = ImageRegressionModel.load_from_checkpoint(checkpoint, hparams_file=hparams)
    trainer = pl.Trainer(
        accelerator="auto", devices="auto", enable_progress_bar=False,
        enable_model_summary=False, logger=False, deterministic=True, precision="32-true"
    )
    trainer.test(model, dataloaders=test_loader)

    print("Generating predictions...")
    val_pred, val_unc, val_err, val_q_low, val_q_high, val_true, _ = get_predictions(model, val_loader)
    test_pred, test_unc, test_err, test_q_low, test_q_high, test_true, test_time = get_predictions(model, test_loader)

    num_samples = len(test_true)
    time_per_sample = test_time / num_samples if num_samples > 0 else 0
    test_true_f = np.ravel(test_true)

    def export_results(intervals, raw_unc, beta_val, target_dir):
        coverage = np.mean([(l <= y <= u) for (l, u), y in zip(intervals, test_true_f)])
        widths = np.array([u - l for (l, u) in intervals])
        eff_median = np.median(widths)

        print(f"\n--- Results ({method.value} | Beta: {beta_val:.6f}) ---")
        print(f"Coverage: {coverage * 100:.2f}%  |  Efficiency (median): {eff_median:.2f}")

        t_resid = np.abs(np.ravel(test_pred) - test_true_f)
        results_df = pd.DataFrame({
            "y_true": test_true_f, "y_pred": np.ravel(test_pred), "residual": t_resid,
            "raw_uncertainty": raw_unc, "quant_low": np.ravel(test_q_low), "quant_high": np.ravel(test_q_high),
            "lower": [l for (l, u) in intervals], "upper": [u for (l, u) in intervals],
            "width": widths, "covered": [(l <= y <= u) for (l, u), y in zip(intervals, test_true_f)]
        })

        save_dir = Path(target_dir) / experiment_name / f"seed_{seed}" / method.value
        save_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(save_dir / f"eps_{epsilon}.csv", index=False)

        with open(save_dir / f"timing_eps_{epsilon}.txt", "w") as f:
            f.write(f"Inference Performance for {method.value} (alpha={epsilon})\n")
            f.write(f"Total Inference Time:  {test_time:.6f} seconds\n")
            f.write(f"Total Samples:         {num_samples}\n")
            f.write(f"Time Per Sample:       {time_per_sample:.8f} seconds\n")
            f.write(f"Beta Parameter:        {beta_val:.6f}\n")

    if method == Method.CQR:
        val_scores = np.maximum(np.ravel(val_q_low) - np.ravel(val_true), np.ravel(val_true) - np.ravel(val_q_high))
        q_hat = get_conformal_quantile(val_scores, epsilon)
        intervals = list(zip(np.ravel(test_q_low) - q_hat, np.ravel(test_q_high) + q_hat))
        raw_unc = np.ravel(test_q_high) - np.ravel(test_q_low)

        export_results(intervals, raw_unc, 0.0, output_dir)

    elif method == Method.SICP:
        val_residuals = np.abs(np.ravel(val_pred) - np.ravel(val_true))
        q_hat = get_conformal_quantile(val_residuals, epsilon)
        intervals = [(y - q_hat, y + q_hat) for y in np.ravel(test_pred)]

        export_results(intervals, np.ravel(test_unc), 0.0, output_dir)

    else:
        val_residuals = np.abs(np.ravel(val_pred) - np.ravel(val_true))
        source_val = np.ravel(val_err) if method == Method.NICP else np.ravel(val_unc)
        source_test = np.ravel(test_err) if method == Method.NICP else np.ravel(test_unc)

        # ADAPTIVE BETA EVALUATION
        beta_adapt = max(np.mean(source_val) * 0.05, 1e-6)
        q_val_adapt = get_conformal_quantile(val_residuals / (source_val + beta_adapt), epsilon)
        intervals_adapt = [(y - q_val_adapt * (s + beta_adapt), y + q_val_adapt * (s + beta_adapt))
                           for y, s in zip(np.ravel(test_pred), source_test)]

        export_results(intervals_adapt, source_test, beta_adapt, output_dir)


if __name__ == "__main__":
    test()
