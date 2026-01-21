#!/usr/bin/env python
"""QAT (Quantization-Aware Training) training script.

Args:
    --config: Path to config file (yaml)
    --checkpoint: Pretrained checkpoint path (required)
    --work_dir: Working directory for saving checkpoints and logs
    --seed: Random seed
    --backend: Quantization backend (fbgemm/qnnpack)
    --fusion_strategy: Module fusion strategy
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from FunFlow.registry import MODELS
from FunFlow.datasets.dataset import build_dataset
from FunFlow.trainer import Trainer
from FunFlow.logger import setup_logger, reset_logger
from FunFlow.compression.QAT import QATQuantizer
from FunFlow.utils import load_config, save_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="FunFlow QAT Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file (yaml)"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./work_dirs",
        help="Working directory for saving checkpoints and logs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Pretrained checkpoint path for QAT training (required)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["fbgemm", "qnnpack", "x86", "onednn"],
        help="Quantization backend (overrides config)",
    )
    parser.add_argument(
        "--fusion_strategy",
        type=str,
        default=None,
        choices=[
            "resnet",
            "resnet18",
            "resnet34",
            "resnet50",
            "audiocnn",
            "conv1d",
            "default",
        ],
        help="Module fusion strategy (overrides config, null=auto)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Max epochs (overrides config)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loading workers (overrides config)",
    )

    return parser.parse_args()


def update_config(config: dict, args) -> dict:
    """Update config with command line arguments."""
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    if args.epochs is not None:
        config["trainer"]["max_epochs"] = args.epochs
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    if args.backend is not None:
        if "quantization" not in config:
            config["quantization"] = {}
        config["quantization"]["backend"] = args.backend
    if args.fusion_strategy is not None:
        if "quantization" not in config:
            config["quantization"] = {}
        config["quantization"]["fusion_strategy"] = args.fusion_strategy

    return config


def apply_qat_config(config: dict) -> dict:
    """Apply QAT training config overrides.

    Args:
        config: Configuration dict

    Returns:
        Updated configuration dict
    """
    qat_cfg = config.get("quantization", {})

    if "qat_max_epochs" in qat_cfg:
        config["trainer"]["max_epochs"] = qat_cfg["qat_max_epochs"]

    if "qat_lr" in qat_cfg:
        config["optimizer"]["lr"] = qat_cfg["qat_lr"]

    return config


def main():
    args = parse_args()

    from FunFlow.utils.distributed import init_distributed, is_main_process

    is_distributed = init_distributed()

    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    config = load_config(args.config)
    config = update_config(config, args)
    config = apply_qat_config(config)

    trainer_cfg = config.get("trainer", {})
    work_dir = Path(args.work_dir)

    reset_logger("FunFlow")

    logger = setup_logger(
        name="FunFlow",
        work_dir=work_dir,
        file_mode="w",
    )

    if is_main_process():
        logger.section("FunFlow QAT Training")
        logger.console(f"Config: {args.config}")
        logger.console(f"Pretrained checkpoint: {args.checkpoint}")
        logger.console(f"Work dir: {work_dir}")
        logger.console(f"Distributed: {is_distributed}")
        logger.console(f"Seed: {args.seed}")
        logger.separator()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.console(f"ERROR: Checkpoint not found: {args.checkpoint}")
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    qat_cfg = config.get("quantization", {})
    if not qat_cfg.get("enabled", True):
        logger.console(
            "WARNING: quantization.enabled is False, but running QAT training script"
        )

    backend = qat_cfg.get("backend", "fbgemm")
    fuse_modules = qat_cfg.get("fuse_modules", True)
    fusion_strategy = qat_cfg.get("fusion_strategy", None)

    logger.console(f"QAT Backend: {backend}")
    logger.console(f"Fuse modules: {fuse_modules}")
    if fusion_strategy:
        logger.console(f"Fusion strategy: {fusion_strategy}")
    logger.separator()

    config_save_path = work_dir / "config.yaml"
    save_config(config, config_save_path)
    logger.console(f"Config saved to: {config_save_path}")

    logger.console("\n[1/3] Building data loaders...")

    data_cfg = config["data"]
    num_workers = data_cfg.get("num_workers", 4)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)

    train_data_file = data_cfg["train"]["data_file"]
    train_conf = data_cfg["train"].get("conf", data_cfg["train"])
    train_dataset = build_dataset(
        data_file=train_data_file,
        conf=train_conf,
        partition=True,
        is_distributed=is_distributed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
    )

    val_loader = None
    if "val" in data_cfg:
        val_data_file = data_cfg["val"]["data_file"]
        val_conf = data_cfg["val"].get("conf", data_cfg["val"])
        val_dataset = build_dataset(
            data_file=val_data_file,
            conf=val_conf,
            partition=False,
            is_distributed=is_distributed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
        )

    logger.console(f"Train data file: {train_data_file}")
    if val_loader is not None:
        logger.console(f"Val data file: {val_data_file}")

    logger.console("\n[2/3] Building model for QAT...")

    model_config = config["model"].copy()
    model_config["quantization"] = True
    model = MODELS.build(model_config)
    logger.console(f"Model: {config['model']['type']}")
    logger.console(f"Parameters: {model.get_num_params():,}")

    logger.console(f"Loading pretrained weights from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.console(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.info(f"Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        logger.info(f"Unexpected keys in state_dict: {unexpected_keys}")
    logger.console("  Pretrained weights loaded successfully")

    logger.console("\nPreparing model for QAT...")
    model = QATQuantizer.prepare_qat(
        model,
        backend=backend,
        fuse_modules=fuse_modules,
        fusion_strategy=fusion_strategy,
    )

    logger.console(
        f"Parameters after QAT prep: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info(f"Model architecture: \n{model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from FunFlow.utils.distributed import get_rank

        model = DDP(
            model,
            device_ids=[get_rank()],
            output_device=get_rank(),
            find_unused_parameters=False,
        )
        if is_main_process():
            logger.console(f"Model wrapped with DDP (rank {get_rank()})")

    if is_main_process():
        logger.console("\n[3/3] Setting up trainer...")

    scheduler_cfg = config.get("scheduler", {})
    paramwise_cfg = config.get("paramwise_cfg", None)
    freeze_cfg = config.get("freeze_cfg", None)

    hook_cfgs = config.get("hooks", [])
    hook_cfgs = [h.copy() for h in hook_cfgs]

    has_qat_hook = any(h.get("type") == "QATHook" for h in hook_cfgs)
    if not has_qat_hook:
        freeze_observer_epoch = qat_cfg.get("freeze_observer_epoch", 2)
        freeze_bn_epoch = qat_cfg.get("freeze_bn_epoch", 3)
        hook_cfgs.append(
            {
                "type": "QATHook",
                "freeze_observer_epoch": freeze_observer_epoch,
                "freeze_bn_epoch": freeze_bn_epoch,
                "verbose": True,
            }
        )
        logger.console(
            f"QATHook auto-enabled: freeze_observer_epoch={freeze_observer_epoch}, "
            f"freeze_bn_epoch={freeze_bn_epoch}"
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_cfg=config.get("optimizer", {"type": "AdamW", "lr": 1e-5}),
        paramwise_cfg=paramwise_cfg,
        freeze_cfg=freeze_cfg,
        scheduler_cfg=scheduler_cfg,
        max_epochs=trainer_cfg.get("max_epochs", 20),
        device=device,
        fp16=False,
        gradient_accumulation_steps=trainer_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=trainer_cfg.get("max_grad_norm", 1.0),
        work_dir=str(work_dir),
        seed=args.seed,
        resume_from=None,
        is_distributed=is_distributed,
        logger_cfg=config.get(
            "logger_cfg", {"log_interval": trainer_cfg.get("log_interval", 10)}
        ),
        checkpoint_cfg=config.get(
            "checkpoint_cfg",
            {
                "max_keep_ckpts": 5,
                "save_best": True,
                "save_last": True,
                "metric_name": "accuracy",
                "metric_mode": "max",
            },
        ),
        eval_cfg=config.get(
            "eval_cfg", {"eval_interval": 1, "metric_name": "accuracy"}
        ),
        hook_cfgs=hook_cfgs,
    )

    if is_main_process():
        logger.console(f"Max epochs: {trainer_cfg.get('max_epochs', 20)}")
        logger.console(f"Learning rate: {config.get('optimizer', {}).get('lr', 1e-5)}")
        logger.console(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        logger.console("FP16: Disabled (QAT requires FP32)")

        logger.separator()
        logger.console("Starting QAT training...")
        logger.separator()

    trainer.train()

    if is_main_process():
        logger.separator()
        logger.console("QAT training completed!")
        logger.console(
            f"Best metric: {trainer.best_metric:.4f} at epoch {trainer.best_epoch}"
        )
        logger.console(f"Checkpoints saved in: {trainer.work_dir}")
        logger.separator()

        logger.console("\nNext steps:")
        logger.console("1. Convert QAT model to quantized model:")
        logger.console(
            f"   python -m FunFlow.export.export_cli --exporter_type onnx \\"
        )
        logger.console(
            f"       --config {args.config} --checkpoint {work_dir}/best.pth"
        )
        logger.console("2. Or use QATQuantizer.convert_qat() to convert in Python")


if __name__ == "__main__":
    main()
