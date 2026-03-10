"""SMILES 문자열로부터 descriptor를 계산하고 학습된 모델로 물성을 예측하는 서비스 모듈.

흐름: raw SMILES → canonical SMILES + descriptor 계산(RDKit) → 모델 추론 → 결과 반환
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import DataConfig, ModelConfig, ServiceConfig
from dataset import MoleculeDataset
from infer import load_model, predict
from utils import get_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit descriptor 계산
# ---------------------------------------------------------------------------

_DESCRIPTOR_FUNCTIONS: dict[str, callable] = {
    "MW": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
}


def canonicalize_smiles(smiles: str) -> str | None:
    """SMILES를 canonical 형태로 변환한다. 유효하지 않으면 None을 반환한다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """SMILES 목록으로부터 canonical SMILES와 분자 descriptor를 계산한다.

    Parameters
    ----------
    smiles_list : list[str]
        원본 SMILES 문자열 목록.

    Returns
    -------
    pd.DataFrame
        컬럼: input_smiles, cano_smiles, MW, TPSA, HBD, HBA
        유효하지 않은 SMILES 행의 descriptor 값은 NaN이 된다.
    """
    records: list[dict] = []
    for smi in smiles_list:
        row: dict = {"input_smiles": smi}
        cano = canonicalize_smiles(smi)
        row["cano_smiles"] = cano

        mol = Chem.MolFromSmiles(smi) if cano is not None else None
        for desc_name, func in _DESCRIPTOR_FUNCTIONS.items():
            row[desc_name] = float(func(mol)) if mol is not None else np.nan

        if mol is None:
            logger.warning("Invalid SMILES skipped for descriptor computation: %s", smi)

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Service pipeline
# ---------------------------------------------------------------------------

def run_service(
    smiles_list: list[str],
    data_cfg: DataConfig,
    service_cfg: ServiceConfig,
) -> pd.DataFrame:
    """SMILES → descriptor 계산 → 모델 추론 → 결과 DataFrame을 반환한다.

    Parameters
    ----------
    smiles_list : list[str]
        예측할 SMILES 문자열 목록.
    data_cfg : DataConfig
        컬럼 설정 (smiles_col, descriptor_cols, target_cols).
    service_cfg : ServiceConfig
        체크포인트 경로, 출력 경로 등.

    Returns
    -------
    pd.DataFrame
        입력 정보 + descriptor + 예측값이 포함된 DataFrame.
    """
    if not smiles_list:
        raise ValueError("smiles_list must not be empty.")

    # 1) Descriptor 계산
    df = compute_descriptors(smiles_list)
    valid_mask = df["cano_smiles"].notna()
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        logger.warning(
            "%d invalid SMILES detected; predictions for these rows will be NaN.", n_invalid
        )
    logger.info("Computed descriptors for %d molecules (%d valid)", len(df), int(valid_mask.sum()))

    # 유효한 행만 추론에 사용
    df_valid = df.loc[valid_mask].copy().reset_index(drop=True)

    if df_valid.empty:
        raise ValueError("No valid SMILES found. Cannot run inference.")

    # 2) 모델 로드
    device = get_device()
    model, ckpt = load_model(service_cfg.checkpoint_path, device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.smiles_model_name)

    # descriptor NaN 대치 (유효 분자에서는 보통 없지만 안전장치)
    fill_values = MoleculeDataset.compute_descriptor_fill_values(
        df_valid, data_cfg.descriptor_cols
    )

    dummy_target = "__dummy_target__"
    df_valid[dummy_target] = 0.0

    ds = MoleculeDataset(
        df=df_valid,
        smiles_col=data_cfg.smiles_col,
        descriptor_cols=data_cfg.descriptor_cols,
        target_cols=[dummy_target],
        tokenizer=tokenizer,
        max_length=model_cfg.smiles_max_length,
        descriptor_fill_values=fill_values,
    )
    loader = DataLoader(ds, batch_size=service_cfg.batch_size, shuffle=False, num_workers=0)

    # 3) 추론
    predictions = predict(model, loader, device)  # (N_valid, num_targets)

    # 예측 컬럼 이름
    pred_col_names = [f"pred_{t}" for t in data_cfg.target_cols] if data_cfg.target_cols else [
        f"pred_{i}" for i in range(predictions.shape[1])
    ]

    # 4) 결과 조립 — 원본 df에 예측값을 매핑 (invalid 행은 NaN)
    for i, col_name in enumerate(pred_col_names):
        df[col_name] = np.nan
        df.loc[valid_mask, col_name] = predictions[:, i]

    df.drop(columns=[dummy_target], errors="ignore", inplace=True)

    # 5) 저장 (output_path가 지정된 경우)
    if service_cfg.output_path:
        output_dir = Path(service_cfg.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(service_cfg.output_path, index=False)
        logger.info("Service results saved to %s", service_cfg.output_path)

    return df
