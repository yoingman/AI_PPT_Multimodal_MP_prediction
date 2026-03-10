"""학습/검증 루프, 스케줄러, 체크포인트 저장을 수행하는 학습 모듈."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import DataConfig, ModelConfig, TrainConfig
from dataset import MoleculeDataset
from model import MultimodalMPModel
from utils import compute_regression_metrics, get_device, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """분자 물성 예측 모델 학습기.

    Parameters
    ----------
    data_cfg : DataConfig
    model_cfg : ModelConfig
    train_cfg : TrainConfig
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = get_device()
        logger.info("Device: %s", self.device)

    def _prepare_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """CSV를 읽고 train/val/test DataLoader를 생성한다."""
        tcfg = self.train_cfg
        dcfg = self.data_cfg

        df = pd.read_csv(tcfg.data_path)
        logger.info("Loaded %d rows from %s", len(df), tcfg.data_path)

        if not dcfg.descriptor_cols:
            raise ValueError("descriptor_cols must not be empty.")
        if not dcfg.target_cols:
            raise ValueError("target_cols must not be empty.")

        self.model_cfg.descriptor_input_dim = len(dcfg.descriptor_cols)
        self.model_cfg.num_targets = len(dcfg.target_cols)

        train_df, temp_df = train_test_split(
            df, test_size=tcfg.val_ratio + tcfg.test_ratio, random_state=tcfg.seed
        )
        relative_test = tcfg.test_ratio / (tcfg.val_ratio + tcfg.test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test, random_state=tcfg.seed
        )
        logger.info(
            "Split -> train: %d, val: %d, test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.smiles_model_name)

        fill_values = MoleculeDataset.compute_descriptor_fill_values(
            train_df, dcfg.descriptor_cols
        )

        common_kwargs = dict(
            smiles_col=dcfg.smiles_col,
            descriptor_cols=dcfg.descriptor_cols,
            target_cols=dcfg.target_cols,
            tokenizer=tokenizer,
            max_length=self.model_cfg.smiles_max_length,
            descriptor_fill_values=fill_values,
        )

        train_ds = MoleculeDataset(df=train_df, **common_kwargs)
        val_ds = MoleculeDataset(df=val_df, **common_kwargs)
        test_ds = MoleculeDataset(df=test_df, **common_kwargs)

        loader_kwargs: dict = dict(num_workers=0, pin_memory=True)
        train_loader = DataLoader(
            train_ds, batch_size=tcfg.batch_size, shuffle=True, **loader_kwargs
        )
        val_loader = DataLoader(
            val_ds, batch_size=tcfg.batch_size, shuffle=False, **loader_kwargs
        )
        test_loader = DataLoader(
            test_ds, batch_size=tcfg.batch_size, shuffle=False, **loader_kwargs
        )
        return train_loader, val_loader, test_loader

    def _train_one_epoch(
        self,
        model: MultimodalMPModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        criterion: nn.Module,
    ) -> float:
        """1 epoch 학습 후 평균 loss를 반환한다."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            descriptors = batch["descriptors"].to(self.device)
            targets = batch["targets"].to(self.device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, descriptors)
            loss = criterion(preds, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), self.train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        model: MultimodalMPModel,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict[str, float]]:
        """검증/테스트 평가를 수행하여 loss와 metric을 반환한다."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            descriptors = batch["descriptors"].to(self.device)
            targets = batch["targets"].to(self.device)

            preds = model(input_ids, attention_mask, descriptors)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        metrics_list = [
            compute_regression_metrics(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
        avg_metrics = {
            k: float(np.mean([m[k] for m in metrics_list]))
            for k in metrics_list[0]
        }
        return avg_loss, avg_metrics

    def run(self) -> None:
        """전체 학습 파이프라인을 실행한다."""
        tcfg = self.train_cfg
        set_seed(tcfg.seed)

        train_loader, val_loader, test_loader = self._prepare_data()

        model = MultimodalMPModel(self.model_cfg).to(self.device)
        logger.info("Model config: %s", asdict(self.model_cfg))

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay
        )

        total_steps = len(train_loader) * tcfg.num_epochs
        warmup_steps = int(total_steps * tcfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        ckpt_dir = pathlib.Path(tcfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, tcfg.num_epochs + 1):
            train_loss = self._train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )
            val_loss, val_metrics = self._evaluate(model, val_loader, criterion)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "val_rmse=%.4f  val_mae=%.4f  val_r2=%.4f",
                epoch,
                tcfg.num_epochs,
                train_loss,
                val_loss,
                val_metrics["rmse"],
                val_metrics["mae"],
                val_metrics["r2"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "model_cfg": asdict(self.model_cfg),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    },
                    ckpt_dir / "best_model.pt",
                )
                logger.info("  -> Best model saved (val_loss=%.4f)", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= tcfg.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        best_ckpt = torch.load(
            ckpt_dir / "best_model.pt", map_location=self.device, weights_only=False
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_metrics = self._evaluate(model, test_loader, criterion)
        logger.info(
            "Test  loss=%.4f  rmse=%.4f  mae=%.4f  r2=%.4f",
            test_loss,
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
        )
