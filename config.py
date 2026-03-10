"""프로젝트 전체 설정을 dataclass로 관리하고, YAML에서 로드한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """공통 컬럼 설정. 학습과 추론 모두에서 공유한다."""

    smiles_col: str = "SMILES"
    descriptor_cols: list[str] = field(default_factory=list)
    target_cols: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """모델 아키텍처 설정."""

    smiles_model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    smiles_max_length: int = 128
    smiles_hidden_dim: int = 768

    descriptor_input_dim: int = 0
    descriptor_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    descriptor_dropout: float = 0.1
    descriptor_use_layernorm: bool = True

    fusion_method: str = "concat"
    fusion_hidden_dim: int = 256

    prediction_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    prediction_dropout: float = 0.1
    num_targets: int = 1

    freeze_transformer: bool = False


@dataclass
class TrainConfig:
    """학습 하이퍼파라미터 및 경로 설정."""

    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    val_ratio: float = 0.1
    test_ratio: float = 0.1

    checkpoint_dir: str = "checkpoints"
    patience: int = 10

    data_path: str = "data/train.csv"


@dataclass
class InferConfig:
    """추론 경로 설정."""

    checkpoint_path: str = "checkpoints/best_model.pt"
    batch_size: int = 64
    data_path: str = "data/test.csv"
    output_path: str = "outputs/predictions.csv"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _safe_update(dc: Any, overrides: dict) -> None:
    """dataclass 인스턴스의 필드만 선택적으로 덮어쓴다."""
    for k, v in overrides.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(
    yaml_path: str | Path,
) -> tuple[DataConfig, ModelConfig, TrainConfig, InferConfig]:
    """YAML 파일을 읽어 4개의 config dataclass를 생성한다.

    Parameters
    ----------
    yaml_path : str | Path
        설정 YAML 파일 경로.

    Returns
    -------
    (DataConfig, ModelConfig, TrainConfig, InferConfig)
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    infer_cfg = InferConfig()

    if "data" in raw:
        _safe_update(data_cfg, raw["data"])
    if "model" in raw:
        _safe_update(model_cfg, raw["model"])
    if "train" in raw:
        _safe_update(train_cfg, raw["train"])
    if "infer" in raw:
        _safe_update(infer_cfg, raw["infer"])

    return data_cfg, model_cfg, train_cfg, infer_cfg
