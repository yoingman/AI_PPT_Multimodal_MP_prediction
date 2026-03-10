"""분자 물성 예측용 Dataset 클래스.

SMILES 토큰화, descriptor tensor 생성, NaN 평균 대치를 처리한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MoleculeDataset(Dataset):
    """SMILES + descriptor → target 을 제공하는 PyTorch Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임.
    smiles_col : str
        SMILES 문자열이 담긴 컬럼 이름.
    descriptor_cols : list[str]
        수치형 descriptor 컬럼 이름 목록.
    target_cols : list[str]
        예측 target 컬럼 이름 목록.
    tokenizer : AutoTokenizer
        Hugging Face tokenizer 인스턴스.
    max_length : int
        토큰 시퀀스 최대 길이.
    descriptor_fill_values : dict[str, float] | None
        descriptor NaN을 채울 컬럼별 평균값. None이면 내부에서 계산한다.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        descriptor_cols: list[str],
        target_cols: list[str],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        descriptor_fill_values: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in DataFrame.")
        missing_desc = [c for c in descriptor_cols if c not in df.columns]
        if missing_desc:
            raise ValueError(f"Descriptor columns not found: {missing_desc}")
        missing_tgt = [c for c in target_cols if c not in df.columns]
        if missing_tgt:
            raise ValueError(f"Target columns not found: {missing_tgt}")

        self.smiles: list[str] = df[smiles_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        desc_df = df[descriptor_cols].copy().astype(np.float32)
        if descriptor_fill_values is None:
            descriptor_fill_values = self.compute_descriptor_fill_values(df, descriptor_cols)
        self.descriptor_fill_values = descriptor_fill_values
        desc_df = desc_df.fillna(descriptor_fill_values)
        self.descriptors: np.ndarray = desc_df.values  # (N, D)

        self.targets: np.ndarray = df[target_cols].values.astype(np.float32)  # (N, T)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "descriptors": torch.tensor(self.descriptors[idx], dtype=torch.float32),  # (D,)
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),  # (T,)
        }

    @staticmethod
    def compute_descriptor_fill_values(
        df: pd.DataFrame,
        descriptor_cols: list[str],
    ) -> dict[str, float]:
        """Train set 기준 descriptor 컬럼별 평균값을 계산하여 반환한다."""
        return {col: float(df[col].mean(skipna=True)) for col in descriptor_cols}
