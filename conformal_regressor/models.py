import pytorch_lightning as pl
import torch
from torch import nn
from enum import Enum
from torchvision import models
from transformers import AutoModel, AutoConfig


class Method(str, Enum):
    SICP = "SICP"
    SPACR = "SPACR"
    DOICR = "DOICR"
    NICP = "NICP"
    CQR = "CQR"
    SPACR_EFF = "SPACR_EFF"
    SPACR_VAL = "SPACR_VAL"


class RegressionModel(pl.LightningModule):
    def __init__(
            self,
            input_size,
            hidden_size=64,
            num_blocks=3,
            dropout=0.2,
            weight_decay=0.01,
            learning_rate=1e-4,
            method: Method = Method.SICP,
            epsilon=0.05,
            lam=5.0,
            max_epochs=32,
            seed=42,
    ):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.save_hyperparameters()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(num_blocks)
            ]
        )

        # Output layers
        self.output_layer = nn.Linear(hidden_size, 1)  # Mean/Median prediction
        self.uncertainty_layer = nn.Linear(hidden_size, 1)  # For SPACR/DOICR
        self.error_estimation_layer = nn.Linear(input_size, 1)  # For NICP
        self.quant_low_layer = nn.Linear(hidden_size, 1)  # For CQR Lower quantile
        self.quant_high_layer = nn.Linear(hidden_size, 1)  # For CQR Upper quantile

        # Loss function
        self.base_loss = nn.L1Loss()

    def doicr_loss(self, output, y):
        batch_size = output["y_hat"].size(0)
        size_r = batch_size // 2
        y_hat_d1, y_hat_d2 = output["y_hat"][:size_r], output["y_hat"][size_r:]
        uncertainty_d1, uncertainty_d2 = (
            output["uncertainty"][:size_r],
            output["uncertainty"][size_r:],
        )
        y_d1, y_d2 = y[:size_r], y[size_r:]

        efficiency = torch.mean(uncertainty_d1)
        non_conformity_d2 = torch.abs(y_hat_d2 - y_d2) / uncertainty_d2

        loss = (
                2 * torch.quantile(non_conformity_d2, 1 - self.hparams.epsilon) * efficiency
        )

        return loss

    def spacr_loss(self, output, y):
        y_hat = output["y_hat"]
        uncertainty = output["uncertainty"]
        efficiency = torch.mean(uncertainty)
        validity = torch.mean(torch.relu(y - (y_hat + uncertainty))) + torch.mean(
            torch.relu((y_hat - uncertainty) - y)
        )
        loss = efficiency + self.hparams.lam * validity

        return loss

    def spacr_eff_only_loss(self, output, y):
        uncertainty = output["uncertainty"]
        efficiency = torch.mean(uncertainty)
        return efficiency

    def spacr_val_only_loss(self, output, y):
        y_hat = output["y_hat"]
        uncertainty = output["uncertainty"]
        validity = torch.mean(torch.relu(y - (y_hat + uncertainty))) + torch.mean(
            torch.relu((y_hat - uncertainty) - y)
        )
        loss = self.hparams.lam * validity
        return loss

    def error_estimation_loss(self, output, y):
        y_hat = output["y_hat"]
        error_estimation = output["error_estimation"]
        target = torch.abs(y_hat - y).detach()  # detach so we don't impact base network
        return self.base_loss(error_estimation, target)

    # --- NEW: CQR Loss (Pinball Loss) ---
    def quantile_loss(self, output, y):
        q_low = output["quant_low"]
        q_high = output["quant_high"]

        tau_low = self.hparams.epsilon / 2
        tau_high = 1 - (self.hparams.epsilon / 2)

        # Pinball Loss Function: max(tau * (y - y_hat), (tau - 1) * (y - y_hat))
        err_low = y - q_low
        loss_low = torch.mean(torch.max(tau_low * err_low, (tau_low - 1) * err_low))

        err_high = y - q_high
        loss_high = torch.mean(torch.max(tau_high * err_high, (tau_high - 1) * err_high))

        return loss_low + loss_high

    def loss(self, output, y, fold):
        base_loss = self.base_loss(output["y_hat"], y)
        self.log(f"{fold}_base_loss", base_loss)

        total_loss = base_loss

        if self.hparams.method == Method.SICP:
            pass

        elif self.hparams.method == Method.SPACR:
            spacr = self.spacr_loss(output, y)
            self.log(f"{fold}_spacr_loss", spacr)
            total_loss += spacr

        elif self.hparams.method == Method.DOICR:
            doicr = self.doicr_loss(output, y)
            self.log(f"{fold}_doicr_loss", doicr)
            total_loss = doicr  # Override with DOICR loss

        elif self.hparams.method == Method.NICP:
            error_estimation = self.error_estimation_loss(output, y)
            self.log(f"{fold}_error_estimation_loss", error_estimation)
            total_loss += error_estimation

        elif self.hparams.method == Method.SPACR_EFF:
            spacr_eff_only = self.spacr_eff_only_loss(output, y)
            self.log(f"{fold}_spacr_eff_only_loss", spacr_eff_only)
            total_loss += spacr_eff_only

        elif self.hparams.method == Method.SPACR_VAL:
            spacr_val_only = self.spacr_val_only_loss(output, y)
            self.log(f"{fold}_spacr_val_only_loss", spacr_val_only)
            total_loss += spacr_val_only

        elif self.hparams.method == Method.CQR:
            cqr_loss = self.quantile_loss(output, y)
            self.log(f"{fold}_cqr_loss", cqr_loss)
            total_loss += cqr_loss

        else:
            raise ValueError(f"Unknown method: {self.hparams.method}")

        return total_loss

    def forward(self, x):
        out = self.input_layer(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out += residual
            out = torch.relu(out)

        output = {
            "y_hat": self.output_layer(out),
            "uncertainty": torch.exp(self.uncertainty_layer(out)),
            "error_estimation": torch.exp(self.error_estimation_layer(x)),
            "quant_low": self.quant_low_layer(out),
            "quant_high": self.quant_high_layer(out),
        }

        return output

    def _step(self, batch, batch_idx, fold):
        x, y = batch
        output = self(x)
        loss = self.loss(output, y, fold=fold)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="val")
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="test")
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        torch.manual_seed(self.hparams.seed)
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


class ImageRegressionModel(pl.LightningModule):
    def __init__(
            self,
            dropout=0.2,
            weight_decay=0.01,
            learning_rate=1e-4,
            method: Method = Method.SICP,
            epsilon=0.05,
            lam=5.0,
            max_epochs=32,
            seed=42,
            pretrained=True,
    ):

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        self.save_hyperparameters()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.feature_dropout = nn.Dropout(self.hparams.dropout)

        self.output_layer = nn.Linear(num_ftrs, 1)
        self.uncertainty_layer = nn.Linear(num_ftrs, 1)
        self.error_estimation_layer = nn.Linear(num_ftrs, 1)

        # --- NEW HEADS FOR CQR ---
        self.quant_low_layer = nn.Linear(num_ftrs, 1)
        self.quant_high_layer = nn.Linear(num_ftrs, 1)

        self.base_loss = nn.L1Loss()  # Or nn.MSELoss()

    def doicr_loss(self, output, y):
        batch_size = output["y_hat"].size(0)
        size_r = batch_size // 2
        y_hat_d1, y_hat_d2 = output["y_hat"][:size_r], output["y_hat"][size_r:]
        uncertainty_d1, uncertainty_d2 = (
            output["uncertainty"][:size_r],
            output["uncertainty"][size_r:],
        )
        y_d1, y_d2 = y[:size_r], y[size_r:]

        efficiency = torch.mean(uncertainty_d1)
        non_conformity_d2 = torch.abs(y_hat_d2 - y_d2) / uncertainty_d2

        loss = (
                2 * torch.quantile(non_conformity_d2, 1 - self.hparams.epsilon) * efficiency
        )

        return loss

    def spacr_loss(self, output, y):
        y_hat = output["y_hat"]
        uncertainty = output["uncertainty"]
        efficiency = torch.mean(uncertainty)
        validity = torch.mean(torch.relu(y - (y_hat + uncertainty))) + torch.mean(
            torch.relu((y_hat - uncertainty) - y)
        )
        loss = efficiency + self.hparams.lam * validity

        return loss

    def spacr_eff_only_loss(self, output, y):
        uncertainty = output["uncertainty"]
        efficiency = torch.mean(uncertainty)
        return efficiency

    def spacr_val_only_loss(self, output, y):
        y_hat = output["y_hat"]
        uncertainty = output["uncertainty"]
        validity = torch.mean(torch.relu(y - (y_hat + uncertainty))) + torch.mean(
            torch.relu((y_hat - uncertainty) - y)
        )
        return self.hparams.lam * validity

    def error_estimation_loss(self, output, y):
        y_hat = output["y_hat"]
        error_estimation = output["error_estimation"]
        target = torch.abs(y_hat - y).detach()  # detach so we don't impact base network

        return self.base_loss(error_estimation, target)

    def quantile_loss(self, output, y):
        q_low = output["quant_low"]
        q_high = output["quant_high"]

        tau_low = self.hparams.epsilon / 2
        tau_high = 1 - (self.hparams.epsilon / 2)

        err_low = y - q_low
        loss_low = torch.mean(torch.max(tau_low * err_low, (tau_low - 1) * err_low))

        err_high = y - q_high
        loss_high = torch.mean(torch.max(tau_high * err_high, (tau_high - 1) * err_high))

        return loss_low + loss_high

    def loss(self, output, y, fold):
        base_loss = self.base_loss(output["y_hat"], y)
        self.log(f"{fold}_base_loss", base_loss)

        total_loss = base_loss

        if self.hparams.method == Method.SICP:
            pass

        elif self.hparams.method == Method.SPACR:
            spacr = self.spacr_loss(output, y)
            self.log(f"{fold}_spacr_loss", spacr)
            total_loss += spacr

        elif self.hparams.method == Method.DOICR:
            doicr = self.doicr_loss(output, y)
            self.log(f"{fold}_doicr_loss", doicr)
            total_loss = doicr  # Override with DOICR loss

        elif self.hparams.method == Method.NICP:
            error_estimation = self.error_estimation_loss(output, y)
            self.log(f"{fold}_error_estimation_loss", error_estimation)
            total_loss += error_estimation

        elif self.hparams.method == Method.SPACR_EFF:
            spacr_eff_only = self.spacr_eff_only_loss(output, y)
            self.log(f"{fold}_spacr_eff_only_loss", spacr_eff_only)
            total_loss += spacr_eff_only

        elif self.hparams.method == Method.SPACR_VAL:
            spacr_val_only = self.spacr_val_only_loss(output, y)
            self.log(f"{fold}_spacr_val_only_loss", spacr_val_only)
            total_loss += spacr_val_only

        elif self.hparams.method == Method.CQR:
            cqr_loss = self.quantile_loss(output, y)
            self.log(f"{fold}_cqr_loss", cqr_loss)
            total_loss += cqr_loss

        else:
            raise ValueError(f"Unknown method: {self.hparams.method}")

        return total_loss


    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_dropout(features)
        y_hat = self.output_layer(features)
        uncertainty_ = self.uncertainty_layer(features)
        error_est_ = self.error_estimation_layer(features)

        q_low = self.quant_low_layer(features)
        q_high = self.quant_high_layer(features)

        uncertainty_activated = torch.exp(uncertainty_)
        error_est_activated = torch.exp(error_est_)

        output = {
            "y_hat": y_hat,
            "uncertainty": uncertainty_activated,
            "error_estimation": error_est_activated,
            "quant_low": q_low,
            "quant_high": q_high,
        }

        return output

    def _step(self, batch, batch_idx, fold):
        x, y = batch
        output = self(x)
        loss = self.loss(output, y, fold=fold)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="val")
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="test")
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        torch.manual_seed(self.hparams.seed)
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


class TextRegressionModel(pl.LightningModule):
    def __init__(
            self,
            model_name="distilbert-base-uncased",
            dropout=0.2,
            weight_decay=0.01,
            learning_rate=2e-5,
            method: Method = Method.SICP,
            epsilon=0.05,
            lam=5.0,
            max_epochs=32,
            seed=42,
            pretrained=True,
    ):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_config(config)

        num_ftrs = config.hidden_size

        self.feature_dropout = nn.Dropout(self.hparams.dropout)

        self.output_layer = nn.Linear(num_ftrs, 1)
        self.uncertainty_layer = nn.Linear(num_ftrs, 1)
        self.error_estimation_layer = nn.Linear(num_ftrs, 1)

        self.quant_low_layer = nn.Linear(num_ftrs, 1)
        self.quant_high_layer = nn.Linear(num_ftrs, 1)

        self.base_loss = nn.MSELoss()

    def doicr_loss(self, output, y):
        batch_size = output["y_hat"].size(0)
        size_r = batch_size // 2

        if batch_size % 2 != 0:
            size_r = batch_size // 2

        y_hat_d1, y_hat_d2 = output["y_hat"][:size_r], output["y_hat"][size_r: 2 * size_r]
        uncertainty_d1, uncertainty_d2 = (
            output["uncertainty"][:size_r],
            output["uncertainty"][size_r: 2 * size_r],
        )
        y_d1, y_d2 = y[:size_r], y[size_r: 2 * size_r]

        efficiency = torch.mean(uncertainty_d1)
        non_conformity_d2 = torch.abs(y_hat_d2 - y_d2) / uncertainty_d2
        loss = (
                2 * torch.quantile(non_conformity_d2, 1 - self.hparams.epsilon) * efficiency
        )
        return loss

    def spacr_loss(self, output, y):
        y_hat = output["y_hat"]
        uncertainty = output["uncertainty"]
        efficiency = torch.mean(uncertainty)
        validity = torch.mean(torch.relu(y - (y_hat + uncertainty))) + torch.mean(
            torch.relu((y_hat - uncertainty) - y)
        )
        loss = efficiency + self.hparams.lam * validity
        return loss

    def spacr_eff_only_loss(self, output, y):
        uncertainty = output["uncertainty"]
        efficiency = torch.mean(uncertainty)
        return efficiency

    def spacr_val_only_loss(self, output, y):
        y_hat = output["y_hat"]
        uncertainty = output["uncertainty"]
        validity = torch.mean(torch.relu(y - (y_hat + uncertainty))) + torch.mean(
            torch.relu((y_hat - uncertainty) - y)
        )
        return self.hparams.lam * validity

    def error_estimation_loss(self, output, y):
        y_hat = output["y_hat"]
        error_estimation = output["error_estimation"]
        target = torch.abs(y_hat - y).detach()
        return self.base_loss(error_estimation, target)

    def quantile_loss(self, output, y):
        q_low = output["quant_low"]
        q_high = output["quant_high"]

        tau_low = self.hparams.epsilon / 2
        tau_high = 1 - (self.hparams.epsilon / 2)

        err_low = y - q_low
        loss_low = torch.mean(torch.max(tau_low * err_low, (tau_low - 1) * err_low))

        err_high = y - q_high
        loss_high = torch.mean(torch.max(tau_high * err_high, (tau_high - 1) * err_high))

        return loss_low + loss_high

    def loss(self, output, y, fold):
        base_loss = self.base_loss(output["y_hat"], y)
        self.log(f"{fold}_base_loss", base_loss)
        total_loss = base_loss

        if self.hparams.method == Method.SICP:
            pass
        elif self.hparams.method == Method.SPACR:
            spacr = self.spacr_loss(output, y)
            self.log(f"{fold}_spacr_loss", spacr)
            total_loss += spacr
        elif self.hparams.method == Method.DOICR:
            doicr = self.doicr_loss(output, y)
            self.log(f"{fold}_doicr_loss", doicr)
            total_loss = doicr
        elif self.hparams.method == Method.NICP:
            error_estimation = self.error_estimation_loss(output, y)
            self.log(f"{fold}_error_estimation_loss", error_estimation)
            total_loss += error_estimation
        elif self.hparams.method == Method.SPACR_EFF:
            spacr_eff_only = self.spacr_eff_only_loss(output, y)
            self.log(f"{fold}_spacr_eff_only_loss", spacr_eff_only)
            total_loss += spacr_eff_only
        elif self.hparams.method == Method.SPACR_VAL:
            spacr_val_only = self.spacr_val_only_loss(output, y)
            self.log(f"{fold}_spacr_val_only_loss", spacr_val_only)
            total_loss += spacr_val_only
        elif self.hparams.method == Method.CQR:
            cqr_loss = self.quantile_loss(output, y)
            self.log(f"{fold}_cqr_loss", cqr_loss)
            total_loss += cqr_loss

        return total_loss

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        features = self.feature_dropout(cls_embedding)

        y_hat = self.output_layer(features).squeeze(-1)
        uncertainty_ = self.uncertainty_layer(features).squeeze(-1)
        error_est_ = self.error_estimation_layer(features).squeeze(-1)

        q_low = self.quant_low_layer(features).squeeze(-1)
        q_high = self.quant_high_layer(features).squeeze(-1)

        uncertainty_activated = torch.exp(uncertainty_)
        error_est_activated = torch.exp(error_est_)

        output = {
            "y_hat": y_hat,
            "uncertainty": uncertainty_activated,
            "error_estimation": error_est_activated,
            "quant_low": q_low,
            "quant_high": q_high,
        }
        return output

    def _step(self, batch, batch_idx, fold):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        y = batch['labels']

        output = self(input_ids, attention_mask)
        loss = self.loss(output, y, fold=fold)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="val")
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, fold="test")
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )