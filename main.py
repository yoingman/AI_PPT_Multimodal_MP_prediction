"""단일 진입점: configs/ 디렉토리(또는 단일 YAML)로 train / infer / service 모드를 실행한다.

사용법::

    python main.py train   --config configs/
    python main.py infer   --config configs/
    python main.py service --config configs/ --smiles "CCO" "c1ccccc1"
    python main.py service --config configs/ --smiles-file smiles.txt
"""

from __future__ import annotations

import argparse
import logging

from config import load_config
from infer import run_inference
from service import run_service
from train import Trainer


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal Molecular Property Prediction",
    )
    parser.add_argument(
        "mode",
        choices=["train", "infer", "service"],
        help="실행 모드: train(학습) / infer(추론) / service(SMILES→descriptor→추론)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/",
        help="설정 YAML 파일 또는 디렉토리 경로 (기본: configs/)",
    )
    parser.add_argument(
        "--smiles",
        nargs="+",
        type=str,
        default=None,
        help="[service] 예측할 SMILES 문자열 목록",
    )
    parser.add_argument(
        "--smiles-file",
        type=str,
        default=None,
        help="[service] SMILES가 한 줄씩 담긴 텍스트 파일 경로",
    )
    args = parser.parse_args()

    _setup_logging()
    logger = logging.getLogger(__name__)

    data_cfg, model_cfg, train_cfg, infer_cfg, service_cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    if args.mode == "train":
        trainer = Trainer(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
        )
        trainer.run()

    elif args.mode == "infer":
        run_inference(data_cfg=data_cfg, infer_cfg=infer_cfg)

    elif args.mode == "service":
        smiles_list = _collect_smiles(args)
        df = run_service(
            smiles_list=smiles_list,
            data_cfg=data_cfg,
            service_cfg=service_cfg,
        )
        print("\n===== Service Prediction Results =====")
        print(df.to_string(index=False))


def _collect_smiles(args: argparse.Namespace) -> list[str]:
    """CLI 인자에서 SMILES 목록을 수집한다."""
    smiles_list: list[str] = []

    if args.smiles:
        smiles_list.extend(args.smiles)

    if args.smiles_file:
        with open(args.smiles_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    smiles_list.append(line)

    if not smiles_list:
        raise ValueError(
            "service 모드에서는 --smiles 또는 --smiles-file 중 하나 이상을 지정해야 합니다."
        )

    return smiles_list


if __name__ == "__main__":
    main()
