#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from FunFlow.registry import MODELS
from FunFlow.datasets.dataset import build_dataset
from FunFlow.trainer import Trainer
from FunFlow.logger import setup_logger, reset_logger
from FunFlow.utils import load_config, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="FunFlow Training Script")

    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to config file"
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
        default=None,
        help="Checkpoint path for resume training",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        help="Use mixed precision training"
    )

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

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

    return config


def main():
    args = parse_args()

    config = load_config(args.config)
    config = update_config(config, args)

    trainer_cfg = config.get("trainer", {})
    work_dir = Path(args.work_dir)

    reset_logger("FunFlow")

    if args.checkpoint is not None:
        logger_file_mode = "a"
    else:
        logger_file_mode = "w"

    logger = setup_logger(
        name="FunFlow",
        work_dir=work_dir,
        log_file="train.log",
        file_mode=logger_file_mode,
    )

    logger.section("FunFlow Training")
    logger.console(f"Config: {args.config}")
    logger.console(f"Seed: {args.seed}")
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
        data_file=train_data_file, conf=train_conf, partition=True
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
            data_file=val_data_file, conf=val_conf, partition=False
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

    logger.console("\n[2/3] Building model...")

    model = MODELS.build(config["model"])
    logger.console(f"Model: {config['model']['type']}")
    logger.console(f"Parameters: {model.get_num_params():,}")
    logger.info(f"Model architecture: \n{model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.console("\n[3/3] Setting up trainer...")

    scheduler_cfg = config.get("scheduler", {})
    paramwise_cfg = config.get("paramwise_cfg", None)
    freeze_cfg = config.get("freeze_cfg", None)

    hook_cfgs = config.get("hooks", [])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_cfg=config.get("optimizer", {"type": "AdamW", "lr": 1e-4}),
        paramwise_cfg=paramwise_cfg,
        freeze_cfg=freeze_cfg,
        scheduler_cfg=scheduler_cfg,
        max_epochs=trainer_cfg.get("max_epochs", None),
        max_steps=trainer_cfg.get("max_steps", None),
        device=device,
        fp16=args.fp16,
        gradient_accumulation_steps=trainer_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=trainer_cfg.get("max_grad_norm", 1.0),
        work_dir=str(work_dir),
        seed=args.seed,
        resume_from=args.checkpoint,
        logger_cfg=config.get(
            "logger_cfg", {"log_interval": trainer_cfg.get("log_interval", 10)}
        ),
        checkpoint_cfg=config.get(
            "checkpoint_cfg", {"save_interval": trainer_cfg.get("save_interval", 1)}
        ),
        eval_cfg=config.get(
            "eval_cfg", {"eval_interval": trainer_cfg.get("eval_interval", 1)}
        ),
        hook_cfgs=hook_cfgs,
    )

    logger.separator()
    logger.console("Starting training...")
    logger.separator()

    trainer.train()

    logger.separator()
    logger.console("Training completed!")
    logger.console(
        f"Best metric: {trainer.best_metric:.4f} at epoch {trainer.best_epoch}"
    )
    logger.console(f"Checkpoints saved in: {trainer.work_dir}")
    logger.separator()


if __name__ == "__main__":
    main()
