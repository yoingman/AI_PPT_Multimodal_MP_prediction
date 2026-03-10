"""프로젝트 전체 설정을 dataclass로 관리하고, YAML에서 로드한다.

단일 YAML 파일 또는 configs/ 디렉토리를 지정할 수 있다.
디렉토리를 지정하면 내부의 모든 *.yaml 파일을 읽어 병합한다.
"""

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


@dataclass
class ServiceConfig:
    """서비스 모드 설정. SMILES 입력 → descriptor 계산 → 추론."""

    checkpoint_path: str = "checkpoints/best_model.pt"
    batch_size: int = 64
    output_path: str = "outputs/service_predictions.csv"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _safe_update(dc: Any, overrides: dict) -> None:
    """dataclass 인스턴스의 필드만 선택적으로 덮어쓴다."""
    for k, v in overrides.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def _load_yaml_raw(config_path: Path) -> dict:
    """단일 YAML 파일 또는 디렉토리의 모든 *.yaml 파일을 읽어 병합된 dict를 반환한다.

    디렉토리인 경우 파일 이름순으로 로드하므로,
    base.yaml → infer.yaml → service.yaml → train.yaml 순서가 된다.
    동일 top-level 키가 여러 파일에 있으면 나중 파일이 덮어쓴다.
    """
    if config_path.is_dir():
        merged: dict = {}
        yaml_files = sorted(config_path.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No *.yaml files found in {config_path}")
        for yf in yaml_files:
            with open(yf, encoding="utf-8") as f:
                part = yaml.safe_load(f) or {}
            merged.update(part)
        return merged

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: str | Path,
) -> tuple[DataConfig, ModelConfig, TrainConfig, InferConfig, ServiceConfig]:
    """YAML 파일(또는 디렉토리)을 읽어 5개의 config dataclass를 생성한다.

    Parameters
    ----------
    config_path : str | Path
        단일 YAML 파일 경로 또는 *.yaml 파일들이 담긴 디렉토리 경로.

    Returns
    -------
    (DataConfig, ModelConfig, TrainConfig, InferConfig, ServiceConfig)
    """
    raw = _load_yaml_raw(Path(config_path))

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    infer_cfg = InferConfig()
    service_cfg = ServiceConfig()

    if "data" in raw:
        _safe_update(data_cfg, raw["data"])
    if "model" in raw:
        _safe_update(model_cfg, raw["model"])
    if "train" in raw:
        _safe_update(train_cfg, raw["train"])
    if "infer" in raw:
        _safe_update(infer_cfg, raw["infer"])
    if "service" in raw:
        _safe_update(service_cfg, raw["service"])

    return data_cfg, model_cfg, train_cfg, infer_cfg, service_cfg
