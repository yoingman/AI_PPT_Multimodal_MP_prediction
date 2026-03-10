"""Multimodal Molecular Property Prediction 모델.

SMILES Transformer + Descriptor MLP → Fusion → Prediction Head
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from config import ModelConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DescriptorMLP(nn.Module):
    """수치형 descriptor를 인코딩하는 MLP.

    GELU activation, Dropout, 선택적 LayerNorm을 사용한다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"descriptor input_dim must be > 0, got {input_dim}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, descriptor_input_dim) → (batch, output_dim)"""
        return self.mlp(x)


class PredictionHead(nn.Module):
    """Fused embedding → target 값을 예측하는 MLP head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_targets: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_targets))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, num_targets)"""
        return self.head(x)


# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """단순 concat fusion. 두 벡터를 이어붙인다."""

    def __init__(self, smiles_dim: int, desc_dim: int) -> None:
        super().__init__()
        self.output_dim = smiles_dim + desc_dim

    def forward(
        self, smiles_emb: torch.Tensor, desc_emb: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([smiles_emb, desc_emb], dim=-1)


class GatedFusion(nn.Module):
    """Gated fusion: 학습 가능한 gate로 두 modality 비중을 조절한다."""

    def __init__(self, smiles_dim: int, desc_dim: int, hidden_dim: int) -> None:
        super().__init__()
        total_dim = smiles_dim + desc_dim
        self.gate = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, smiles_dim),
            nn.Sigmoid(),
        )
        self.project_desc = nn.Linear(desc_dim, smiles_dim)
        self.output_dim = smiles_dim

    def forward(
        self, smiles_emb: torch.Tensor, desc_emb: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([smiles_emb, desc_emb], dim=-1)
        g = self.gate(combined)  # (batch, smiles_dim), values in [0,1]
        desc_proj = self.project_desc(desc_emb)  # (batch, smiles_dim)
        return g * smiles_emb + (1 - g) * desc_proj


class AttentionFusion(nn.Module):
    """Cross-attention style fusion: descriptor가 SMILES embedding에 attend한다."""

    def __init__(self, smiles_dim: int, desc_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(desc_dim, hidden_dim)
        self.key = nn.Linear(smiles_dim, hidden_dim)
        self.value = nn.Linear(smiles_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.output_proj = nn.Linear(hidden_dim + desc_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(
        self, smiles_emb: torch.Tensor, desc_emb: torch.Tensor
    ) -> torch.Tensor:
        # smiles_emb: (batch, smiles_dim), desc_emb: (batch, desc_dim)
        q = self.query(desc_emb).unsqueeze(1)   # (batch, 1, hidden)
        k = self.key(smiles_emb).unsqueeze(1)    # (batch, 1, hidden)
        v = self.value(smiles_emb).unsqueeze(1)  # (batch, 1, hidden)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (batch, 1, 1)
        attn = attn.softmax(dim=-1)
        attended = (attn @ v).squeeze(1)  # (batch, hidden)

        fused = torch.cat([attended, desc_emb], dim=-1)  # (batch, hidden + desc_dim)
        return self.output_proj(fused)  # (batch, hidden)


def build_fusion(
    method: str,
    smiles_dim: int,
    desc_dim: int,
    hidden_dim: int,
) -> ConcatFusion | GatedFusion | AttentionFusion:
    """설정에 따라 적절한 Fusion 모듈을 생성한다."""
    if method == "concat":
        return ConcatFusion(smiles_dim, desc_dim)
    if method == "gated":
        return GatedFusion(smiles_dim, desc_dim, hidden_dim)
    if method == "attention":
        return AttentionFusion(smiles_dim, desc_dim, hidden_dim)
    raise ValueError(f"Unknown fusion method: '{method}'. Choose from: concat, gated, attention")


# ---------------------------------------------------------------------------
# Masked mean pooling
# ---------------------------------------------------------------------------

def masked_mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Attention mask를 사용한 mean pooling.

    Parameters
    ----------
    last_hidden_state : (batch, seq_len, hidden)
    attention_mask : (batch, seq_len)

    Returns
    -------
    (batch, hidden)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
    sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)  # (batch, hidden)
    mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
    return sum_hidden / mask_sum


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MultimodalMPModel(nn.Module):
    """SMILES Transformer + Descriptor MLP → Fusion → Prediction Head.

    Parameters
    ----------
    cfg : ModelConfig
        모델 아키텍처 전체 설정.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1) SMILES Transformer encoder
        self.transformer = AutoModel.from_pretrained(cfg.smiles_model_name)
        if cfg.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # 2) Descriptor MLP encoder
        if cfg.descriptor_input_dim <= 0:
            raise ValueError(
                "descriptor_input_dim must be > 0. "
                "Set it to the number of descriptor columns."
            )
        self.descriptor_encoder = DescriptorMLP(
            input_dim=cfg.descriptor_input_dim,
            hidden_dims=cfg.descriptor_hidden_dims,
            dropout=cfg.descriptor_dropout,
            use_layernorm=cfg.descriptor_use_layernorm,
        )

        # 3) Fusion
        smiles_dim = cfg.smiles_hidden_dim
        desc_dim = self.descriptor_encoder.output_dim
        self.fusion = build_fusion(
            method=cfg.fusion_method,
            smiles_dim=smiles_dim,
            desc_dim=desc_dim,
            hidden_dim=cfg.fusion_hidden_dim,
        )

        # 4) Prediction head
        self.prediction_head = PredictionHead(
            input_dim=self.fusion.output_dim,
            hidden_dims=cfg.prediction_hidden_dims,
            num_targets=cfg.num_targets,
            dropout=cfg.prediction_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len)
        attention_mask : (batch, seq_len)
        descriptors : (batch, descriptor_input_dim)

        Returns
        -------
        predictions : (batch, num_targets)
        """
        # SMILES encoding
        transformer_out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        smiles_emb = masked_mean_pooling(
            transformer_out.last_hidden_state, attention_mask
        )  # (batch, smiles_hidden_dim)

        # Descriptor encoding
        desc_emb = self.descriptor_encoder(descriptors)  # (batch, desc_output_dim)

        # Fusion
        fused = self.fusion(smiles_emb, desc_emb)  # (batch, fusion_output_dim)

        # Prediction
        return self.prediction_head(fused)  # (batch, num_targets)
