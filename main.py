"""단일 진입점: config.yaml 하나로 학습(train)과 추론(infer)을 실행한다.

사용법::

    python main.py train --config config.yaml
    python main.py infer  --config config.yaml
"""

from __future__ import annotations

import argparse
import logging

from config import load_config
from infer import run_inference
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
        choices=["train", "infer"],
        help="실행 모드: train(학습) 또는 infer(추론)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="설정 YAML 파일 경로 (기본: config.yaml)",
    )
    args = parser.parse_args()

    _setup_logging()
    logger = logging.getLogger(__name__)

    data_cfg, model_cfg, train_cfg, infer_cfg = load_config(args.config)
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


if __name__ == "__main__":
    main()
