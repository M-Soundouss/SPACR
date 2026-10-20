# SPACR

### Single-Pass Adaptive Training of Uncertainty-Aware Conformal Regressors

Research code accompanying the paper **“SPACR: Single-Pass Adaptive Training of Uncertainty-Aware Conformal Regressors”** by Soundouss Messoudi, Sylvain Rousseau and Sébastien Destercke, accepted at *Transactions on Machine Learning Research (TMLR)*.

SPACR jointly learns point predictions and input-dependent uncertainty using a differentiable objective that balances prediction error, interval efficiency, and a validity penalty. Training does not require a fixed target coverage level or an internal calibration split within each batch: one trained model can be calibrated at multiple confidence levels.

**Post-hoc calibration is still required.** The training objective learns useful predictions and uncertainty scales; split-conformal calibration on held-out data is what provides marginal coverage guarantees under exchangeability.

[Setup](#setup) · [Quick start](#quick-start) · [Datasets](#datasets) · [Experiments](#experiments) · [Outputs](#outputs) · [Citation](#citation)

## Method

For a prediction $\hat{y}$ and a positive uncertainty scale $\hat{\sigma}$, the SPACR training objective is

$$
\mathcal{L}_{\text{SPACR}} = \underbrace{\frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i|}_{\mathcal{L}_{\text{Accuracy}}} + \underbrace{\frac{1}{n}\sum_{i=1}^n \hat{\sigma}_i}_{\mathcal{L}_{\text{Efficiency}}} + \lambda \underbrace{\left[\frac{1}{n}\sum_{i=1}^n \phi(y_i, \hat{y}_i, \hat{\sigma}_i)\right]}_{{\mathcal{L}_{\text{Validity}}}}.
$$

The three terms encourage accurate point predictions, small uncertainty scales, and a penalty for targets outside the predicted uncertainty band. The parameter `--lam` controls the last term; the experiment scripts use `5.0`.

The implementation includes:

| CLI method | Description | Training depends on `--epsilon`? |
| --- | --- | --- |
| `SICP` | Standard inductive conformal regression | No |
| `NICP` | Normalized inductive conformal regression | No |
| `CQR` | Conformalized quantile regression | Yes |
| `DOICR` | Directly optimized inductive conformal regression | Yes |
| `SPACR` | Full SPACR objective | No |
| `SPACR_EFF` | Point prediction loss plus efficiency term | No |
| `SPACR_VAL` | Point prediction loss plus validity penalty | No |

Tabular experiments use a residual multilayer perceptron. Image experiments fine-tune an ImageNet-pretrained ResNet-18.

## Setup

Use **Python 3.11**. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

For GPU experiments, install a compatible CUDA build of PyTorch and torchvision for your machine. The trainers automatically select available hardware; CPU execution is also supported by the trainer configuration. The experiment sweeps use Bash and GNU `date` (Linux).

Keep the virtual environment active and set `PYTHONPATH` again when opening a new terminal. This repository is used directly from source and does not contain a package installer. The requirements retain the supplied version constraints.

Internet access is needed on the first run to fetch OpenML data, UTKFace, and pretrained image weights. DRIFT requires a separate download, described below.

## Quick start

### Train a tabular SPACR model

Run from the repository root after setup:

```bash
python scripts/train_tabular.py \
  --model-name california \
  --method SPACR \
  --max-epoch 200 \
  --batch-size 256 \
  --lam 5.0 \
  --seed 93 \
  --checkpoint-dir models_tabular/california_spacr_seed93
```

For a short execution check, reduce `--max-epoch` to `1`; this does not reproduce the paper's results.

### Calibrate and evaluate at several confidence levels

Reuse the same checkpoint for 90%, 95%, and 99% nominal coverage:

```bash
for epsilon in 0.1 0.05 0.01; do
  python scripts/conformal_evaluate_tabular.py \
    --model-name california \
    --method SPACR \
    --batch-size 512 \
    --epsilon "$epsilon" \
    --seed 93 \
    --checkpoint-dir models_tabular/california_spacr_seed93 \
    --output-dir results/quickstart
done
```

`--epsilon` is the significance level, so target coverage is `1 - epsilon`. Use the same dataset and seed for training and evaluation to reconstruct the same data split. The evaluator selects the most recently modified `.ckpt` file inside the checkpoint directory; use a separate directory for each independent run.

The training command reports point-prediction test metrics. Run the separate evaluation command to calibrate and measure conformal intervals.

### Train and evaluate an image model

The UTKFace loader downloads its data automatically:

```bash
python scripts/train_images.py \
  --model-name utkface --method SPACR \
  --max-epoch 200 --batch-size 64 \
  --learning_rate 0.001 --lam 5.0 --seed 93 \
  --checkpoint-dir models_images/utkface_spacr_seed93

python scripts/conformal_evaluate_images.py \
  --model-name utkface --method SPACR \
  --batch-size 512 --epsilon 0.05 --seed 93 \
  --checkpoint-dir models_images/utkface_spacr_seed93 \
  --output-dir results/quickstart
```

Use `--model-name drift` for the Denmark tree-counting experiment after preparing DRIFT.

## Datasets

The loaders create **60% training / 20% calibration / 20% test** splits. For tabular data, preprocessing is fitted only on the training split: numeric features are mean-imputed and standardized, and categorical features are imputed and one-hot encoded.

### Tabular data

OpenML datasets are downloaded automatically. The following IDs and target columns reflect the actual loaders in this release:

| Dataset / `--model-name` | OpenML ID | Target column |
| --- | --- | --- |
| Bike Sharing / `bike_sharing` | [42712](https://www.openml.org/d/42712) | `count` |
| Brazilian Houses / `brazilian_houses` | [42688](https://www.openml.org/d/42688) | `total_(BRL)` |
| California / `california` | [44025](https://www.openml.org/d/44025) | `price` |
| CPU Act / `cpu_act` | [197](https://www.openml.org/d/197) | `usr` |
| Diamonds / `diamonds` | [42225](https://www.openml.org/d/42225) | `price` |
| FIFA / `fifa` | [44026](https://www.openml.org/d/44026) | `wage_eur` |
| House Sales / `house_sales` | [42731](https://www.openml.org/d/42731) | `price` |
| Isolet / `isolet` | [44135](https://www.openml.org/d/44135) | `class` |
| Medical Charges / `medical_charges` | [42720](https://www.openml.org/d/42720) | `Average_Total_Payments` |
| Pol / `pol` | [201](https://www.openml.org/d/201) | `foo` |
| Superconduct / `superconduct` | [44148](https://www.openml.org/d/44148) | `criticaltemp` |
| Wine Quality / `wine_quality` | [287](https://www.openml.org/d/287) | `quality` |

### Image data

- **UTKFace:** age regression using [`py97/UTKFace-Cropped`](https://huggingface.co/datasets/py97/UTKFace-Cropped). The loader extracts age from the filename and filters records without usable image/filename metadata.
- **DRIFT:** tree-count regression on the Denmark subset of [DRIFT](https://dgominski.github.io/drift/). Download the data and arrange it exactly as follows:

```text
datasets/
└── f1Hmpeh6O2/
    └── DRIFT/
        └── Denmark/
            ├── labels.csv
            └── images/
                └── <image files referenced by labels.csv>
```

`labels.csv` must contain `filename` and `tree_count` columns. Images are resized to 224 × 224 and normalized with ImageNet statistics. Datasets and pretrained checkpoints are not bundled with this repository.

## Experiments

The four supplied scripts run the multi-seed comparisons and lambda sweeps. **Launch them from `scripts/`**, after exporting the repository root on `PYTHONPATH` as shown in setup:

```bash
cd scripts
bash run_tabular_seed.sh
bash run_image_seed.sh
bash run_tabular_lambda.sh
bash run_image_lambda.sh
```

Each command launches a full sweep; choose the experiment you need. Edit the arrays and settings at the top of its script to run a smaller subset.

| Setting | Supplied sweep configuration |
| --- | --- |
| Seeds | `93`, `753`, `1416`, `2001`, `6789` |
| Significance levels | `0.01`, `0.05`, `0.1` |
| Training epochs | `200` |
| Methods in seed sweeps | All seven methods listed above |
| Lambda in seed sweeps | `5.0` |
| Lambda sweep | `0.01`, `0.05`, `0.1`, `0.5`, `1.0`, `10.0`, `50.0`, `100.0`, `500.0`, `1000.0`, `5000.0`, `10000.0` |

The seed sweeps train CQR and DOICR separately for each significance level. Other methods reuse a trained model across all three rates. The lambda sweeps train SPACR for each lambda value and evaluate it at all three rates.

Run sweeps sequentially: the supplied scripts share some checkpoint and timing locations. Rerunning an experiment can overwrite per-run result files and append to timing summaries. Seed-sweep training output is suppressed by the shell scripts; if training fails, run the corresponding Python training command directly to see the error.

These commands expose the experiment configuration supplied with the code. Exact reproduction of the paper's tables also requires matching its data versions, preprocessing, environment, and hyperparameters.

## Outputs

Evaluators print empirical coverage and median interval width and save:

```text
<output-dir>/<dataset>/seed_<seed>/<METHOD>/
├── eps_<epsilon>.csv
└── timing_eps_<epsilon>.txt
```

Each CSV contains targets, point predictions, absolute residuals, raw uncertainty, quantile predictions, interval endpoints, widths, and coverage indicators. Timing files contain total inference time, time per sample, and the calibration scale offset `beta`.

For normalized methods, the evaluators use `beta = max(0.05 * mean(calibration_scale), 1e-6)` before computing normalized calibration scores. For calibration sets too small for the requested rate, they return the maximum observed score rather than an infinite bound; that fallback does not preserve the usual finite-sample coverage guarantee at those extreme rates.

When launched from `scripts/`, the sweep outputs are:

| Sweep | Result location |
| --- | --- |
| Tabular seeds | `scripts/results/experimental_results/` |
| Image seeds | `scripts/results/experimental_results/` |
| Lambda sweeps | `scripts/experiment_results_lambda/lambda_<value>/output/` |

Default training checkpoints are stored under `models_tabular/` or `models_images/` at the repository root, with Lightning logs and `training_time.txt`. A custom `--checkpoint-dir` overrides this location. Generated data, checkpoints, and experiment outputs are excluded by `.gitignore`.

## Repository structure

```text
conformal_regressor/
├── datasets.py                     # Data loading, preprocessing, and splits
└── models.py                       # Architectures, objectives, and method variants
scripts/
├── train_tabular.py                 # Tabular training
├── train_images.py                  # Image training
├── conformal_evaluate_tabular.py    # Tabular calibration and evaluation
├── conformal_evaluate_images.py     # Image calibration and evaluation
├── run_tabular_seed.sh              # Tabular multi-seed comparisons
├── run_image_seed.sh                # Image multi-seed comparisons
├── run_tabular_lambda.sh            # Tabular lambda sensitivity
└── run_image_lambda.sh              # Image lambda sensitivity
requirements.txt
README.md
```

## Citation

If you use this code, please cite **“SPACR: Single-Pass Adaptive Training of Uncertainty-Aware Conformal Regressors”**, accepted at *Transactions on Machine Learning Research (TMLR)*.
