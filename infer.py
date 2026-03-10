"""저장된 체크포인트를 로드하여 새 데이터에 대해 추론을 수행하는 모듈."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import DataConfig, InferConfig, ModelConfig
from dataset import MoleculeDataset
from model import MultimodalMPModel
from utils import get_device

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[MultimodalMPModel, dict]:
    """체크포인트에서 모델을 복원하여 반환한다."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = MultimodalMPModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded from %s (epoch %d)", checkpoint_path, ckpt["epoch"])
    return model, ckpt


@torch.no_grad()
def predict(
    model: MultimodalMPModel,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """DataLoader의 전체 데이터에 대해 예측값을 반환한다."""
    all_preds: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        descriptors = batch["descriptors"].to(device)

        preds = model(input_ids, attention_mask, descriptors)
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def run_inference(data_cfg: DataConfig, infer_cfg: InferConfig) -> pd.DataFrame:
    """추론 파이프라인을 실행하고 예측 결과가 포함된 DataFrame을 반환한다."""
    device = get_device()
    model, ckpt = load_model(infer_cfg.checkpoint_path, device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])

    df = pd.read_csv(infer_cfg.data_path)
    logger.info("Loaded %d rows from %s", len(df), infer_cfg.data_path)

    if not data_cfg.descriptor_cols:
        raise ValueError("descriptor_cols must not be empty.")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.smiles_model_name)

    fill_values = MoleculeDataset.compute_descriptor_fill_values(
        df, data_cfg.descriptor_cols
    )

    has_targets = bool(data_cfg.target_cols) and all(
        c in df.columns for c in data_cfg.target_cols
    )
    if not has_targets:
        dummy_target = "__dummy_target__"
        df[dummy_target] = 0.0
        target_cols_for_ds = [dummy_target]
    else:
        target_cols_for_ds = data_cfg.target_cols

    ds = MoleculeDataset(
        df=df,
        smiles_col=data_cfg.smiles_col,
        descriptor_cols=data_cfg.descriptor_cols,
        target_cols=target_cols_for_ds,
        tokenizer=tokenizer,
        max_length=model_cfg.smiles_max_length,
        descriptor_fill_values=fill_values,
    )
    loader = DataLoader(ds, batch_size=infer_cfg.batch_size, shuffle=False, num_workers=0)

    predictions = predict(model, loader, device)  # (N, num_targets)

    output_dir = Path(infer_cfg.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_cols = (
        data_cfg.target_cols
        if data_cfg.target_cols
        else [f"pred_{i}" for i in range(predictions.shape[1])]
    )
    for i, col_name in enumerate(pred_cols):
        df[f"pred_{col_name}"] = predictions[:, i]

    df.to_csv(infer_cfg.output_path, index=False)
    logger.info("Predictions saved to %s", infer_cfg.output_path)
    return df
