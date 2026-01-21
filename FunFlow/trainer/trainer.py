"""Training module providing Trainer with hooks system

Hook mechanism:
1. Default hooks (auto-registered):
   - LoggerHook: Logging
   - CheckpointHook: Checkpoint saving
   - EvalHook: Model evaluation (only when val_loader is provided)

2. Optional hooks (configured via hook_cfgs):
   - TensorBoardHook: TensorBoard logging
   - EarlyStoppingHook: Early stopping
   - LRSchedulerHook: Learning rate warmup
   - QATHook: Quantization-aware training

3. Hook configuration format:
   hooks:
     - type: TensorBoardHook
       log_dir: ./tensorboard
     - type: EarlyStoppingHook
       patience: 10
       metric_name: accuracy
       mode: max
"""

from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from FunFlow.trainer.hooks import Hook, LoggerHook, CheckpointHook, EvalHook, build_hook
from FunFlow.trainer.optimizer import build_optimizer
from FunFlow.trainer.scheduler import build_scheduler, infer_scheduler_stepping
from FunFlow.trainer.evaluator import BaseEvaluator, ClassificationEvaluator
from FunFlow.utils.misc import freeze_layers
from FunFlow.logger import setup_logger


class Trainer:
    """Training orchestrator with hooks support

    Features:
    - Mixed precision training support
    - Gradient accumulation
    - Hook mechanism (default hooks auto-registered + optional hooks via config)
    - Automatic checkpoint saving
    - Early stopping mechanism

    Hook configuration:
    - Default hooks: LoggerHook, CheckpointHook, EvalHook are auto-registered
    - Optional hooks: Configured via hook_cfgs list, each using {type: HookName, ...params} format

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer_cfg={'type': 'AdamW', 'lr': 1e-4},
        ...     max_epochs=100,
        ...     # Default hook configs
        ...     logger_cfg={'log_interval': 10},
        ...     checkpoint_cfg={'save_interval': 5, 'max_keep_ckpts': 5},
        ...     eval_cfg={'eval_interval': 1, 'metric_name': 'accuracy'},
        ...     # Optional hooks
        ...     hook_cfgs=[
        ...         {'type': 'TensorBoardHook'},
        ...         {'type': 'EarlyStoppingHook', 'patience': 10},
        ...     ],
        ... )
        >>> trainer.train()

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer_cfg: Optimizer configuration
        paramwise_cfg: Parameter group config for differentiated learning rates
            - backbone_lr_mult: Backbone LR multiplier (e.g., 0.1 for 1/10 LR)
            - head_lr_mult: Head LR multiplier
            - neck_lr_mult: Neck LR multiplier
            - bias_lr_mult: Bias LR multiplier
            - bias_decay_mult: Bias weight decay multiplier
            - custom_patterns: Custom patterns {'layer4': {'lr_mult': 1.0}}
        freeze_cfg: Layer freezing configuration
            - freeze_layers: Layer names to freeze at initialization, supports substring and regex matching
                Substring match: ['backbone.layer1', 'backbone.layer2']
                Regex match: ['regex:backbone\\.layer[0-2]']
        scheduler_cfg: Learning rate scheduler configuration
        max_epochs: Maximum training epochs, default None. If both max_epochs and max_steps are unset, defaults to 100
        max_steps: Maximum training steps, default None. Commonly used for pretraining scenarios
            - If both max_epochs and max_steps are set, training stops when either condition is met first
            - If only max_steps is set, suitable for large-scale pretraining
            - If only max_epochs is set, suitable for regular supervised learning
        device: Training device
        fp16: Enable mixed precision training
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Gradient clipping threshold
        work_dir: Working directory
        seed: Random seed
        resume_from: Checkpoint path to resume training from
        scheduler_step_per_epoch: Scheduler stepping behavior
            - None (default): Auto-infer from scheduler type
            - True: Step once per epoch (for CosineAnnealingLR, StepLR, etc.)
            - False: Step once per optimizer step (for OneCycleLR, CyclicLR, etc.)

        # Default hook configs (auto-registered)
        logger_cfg: LoggerHook configuration
            - log_interval: Logging interval in steps, default 10
        checkpoint_cfg: CheckpointHook configuration
            - max_keep_ckpts: Max checkpoints to keep, default 5 (keeps top N by metric)
            - save_best: Create best symlink to best model, default True
            - save_last: Save last model, default True
            - metric_name: Metric name for ranking, default 'accuracy'
            - metric_mode: 'max' (higher is better) or 'min' (lower is better), default 'max'
        eval_cfg: EvalHook configuration (only active when val_loader is provided)
            - eval_interval: Evaluation interval in epochs, default 1
            - eval_interval_by_step: Evaluation interval in steps, optional
                * If explicitly specified, forces step-level evaluation
                * If unspecified, auto-selects based on max_epochs/max_steps:
                    - Only max_steps set: Use step-level eval (auto-sets to max_steps/100)
                    - max_epochs set (with or without max_steps): Use epoch-level eval
            - metric_name: Best model metric name, default 'accuracy'
            - evaluator_cfg: Evaluator config dict, e.g.:
                {'type': 'ClassificationEvaluator', 'num_classes': 2, 'compute_auc': True}
            - custom_metrics_fn: Custom metrics function with signature:
                (aggregated_outputs: Dict) -> Dict[str, float]
                aggregated_outputs contains: preds, probs, logits, features, labels, avg_loss

        # Optional hooks config list
        hook_cfgs: Optional hooks config list, each element is a hook config dict
            Example: [
                {'type': 'TensorBoardHook', 'log_dir': './tb_logs'},
                {'type': 'EarlyStoppingHook', 'patience': 10, 'metric_name': 'loss', 'mode': 'min'},
                {'type': 'LRSchedulerHook', 'warmup_epochs': 3},
            ]
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer_cfg: Dict[str, Any] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        device: Union[str, torch.device] = "cuda",
        resume_from: Optional[str] = None,
        work_dir: str = "./exp",
        fp16: bool = False,
        max_grad_norm: Optional[float] = 1.0,
        paramwise_cfg: Optional[Dict[str, Any]] = None,
        freeze_cfg: Optional[Dict[str, Any]] = None,
        gradient_accumulation_steps: int = 1,
        scheduler_step_per_epoch: Optional[bool] = None,
        seed: int = 42,
        logger_cfg: Optional[Dict[str, Any]] = None,
        checkpoint_cfg: Optional[Dict[str, Any]] = None,
        eval_cfg: Optional[Dict[str, Any]] = None,
        hook_cfgs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training parameters validation and setup
        # If both are unset, default to max_epochs=100
        if max_epochs is None and max_steps is None:
            max_epochs = 100

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        # scheduler_step_per_epoch will be set after building scheduler
        self._scheduler_step_per_epoch_override = scheduler_step_per_epoch

        # Save configs for later use (e.g., rebuilding optimizer)
        self.optimizer_cfg = optimizer_cfg or {"type": "AdamW", "lr": 1e-4}
        self.paramwise_cfg = paramwise_cfg
        self.freeze_cfg = freeze_cfg

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        self._set_seed(seed)

        self._setup_logger()

        if freeze_cfg:
            freeze_layer_names = freeze_cfg.get("freeze_layers", [])
            if freeze_layer_names:
                frozen_count = freeze_layers(model, freeze_layer_names, verbose=True)
                self.logger.console(
                    f"Frozen {frozen_count} parameters matching: {freeze_layer_names}"
                )
                self._log_trainable_params()

        self.optimizer = build_optimizer(self.model, self.optimizer_cfg, paramwise_cfg)

        if paramwise_cfg:
            self._log_param_groups()

        if scheduler_cfg:
            self.scheduler = build_scheduler(self.optimizer, scheduler_cfg)
            scheduler_type = scheduler_cfg.get("type", "")
            if self._scheduler_step_per_epoch_override is None:
                self.scheduler_step_per_epoch = infer_scheduler_stepping(
                    scheduler=self.scheduler, scheduler_type=scheduler_type
                )
                step_mode = (
                    "epoch-level" if self.scheduler_step_per_epoch else "step-level"
                )
                self.logger.console(
                    f"Scheduler '{scheduler_type}' auto-detected as {step_mode}"
                )
            else:
                self.scheduler_step_per_epoch = self._scheduler_step_per_epoch_override
        else:
            self.scheduler = None
            self.scheduler_step_per_epoch = True

        self.scaler = GradScaler() if fp16 else None

        self.epoch = 0
        self.global_step = 0
        checkpoint_cfg_temp = checkpoint_cfg or {}
        if val_loader is not None:
            self._metric_mode = checkpoint_cfg_temp.get("metric_mode", "max")
        else:
            self._metric_mode = "min"
        self.best_metric = float("-inf") if self._metric_mode == "max" else float("inf")
        self.best_epoch = 0
        self.should_stop = False

        self.current_metrics = {"train": {}, "val": {}}

        self.hooks = []
        self._register_default_hooks(
            logger_cfg=logger_cfg,
            checkpoint_cfg=checkpoint_cfg,
            eval_cfg=eval_cfg,
        )

        if hook_cfgs:
            self._register_optional_hooks(hook_cfgs)

        # Resume training from checkpoint if specified
        if resume_from:
            self.resume(resume_from)

    def _set_seed(self, seed: int):
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_logger(self):
        self._fun_logger = setup_logger(
            name="FunFlow",
            work_dir=self.work_dir,
            log_file="train.log",
            file_mode="a",
        )
        self.logger = self._fun_logger

    def _log_trainable_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        self.logger.console(
            f"Parameters: total={total_params:,}, "
            f"trainable={trainable_params:,} ({100*trainable_params/total_params:.1f}%), "
            f"frozen={frozen_params:,} ({100*frozen_params/total_params:.1f}%)"
        )

    def _log_param_groups(self):
        lr_info = {}
        for group in self.optimizer.param_groups:
            lr = group["lr"]
            name = group.get("group_name", "default")
            if name not in lr_info:
                lr_info[name] = {"lr": lr, "count": 0, "params": 0}
            lr_info[name]["count"] += 1
            lr_info[name]["params"] += sum(p.numel() for p in group["params"])

        self.logger.debug("Parameter groups learning rates:")
        for name, info in lr_info.items():
            self.logger.debug(
                f"  {name}: lr={info['lr']:.2e}, params={info['params']:,}"
            )

    def _register_default_hooks(
        self,
        logger_cfg: Optional[Dict[str, Any]] = None,
        checkpoint_cfg: Optional[Dict[str, Any]] = None,
        eval_cfg: Optional[Dict[str, Any]] = None,
    ):
        """Register default hooks: Logger, Eval, and Checkpoint"""
        # LoggerHook: Training progress logging
        logger_cfg = logger_cfg or {}
        log_interval = logger_cfg.get("log_interval", 10)
        self.register_hook(LoggerHook(log_interval=log_interval))

        # EvalHook: Model evaluation (must be registered before CheckpointHook)
        if self.val_loader is not None:
            eval_cfg = eval_cfg or {}
            self.register_hook(
                EvalHook(
                    eval_interval=eval_cfg.get("eval_interval", 1),
                    eval_interval_by_step=eval_cfg.get("eval_interval_by_step", None),
                    evaluator_cfg=eval_cfg.get("evaluator_cfg", None),
                    custom_metrics_fn=eval_cfg.get("custom_metrics_fn", None),
                )
            )

        # CheckpointHook: Automatic checkpoint saving
        checkpoint_cfg = checkpoint_cfg or {}

        self.register_hook(
            CheckpointHook(
                save_dir=str(self.work_dir),
                save_best=checkpoint_cfg.get("save_best", True),
                save_last=checkpoint_cfg.get("save_last", True),
                max_keep_ckpts=checkpoint_cfg.get("max_keep_ckpts", 5),
                metric_name=checkpoint_cfg.get("metric_name", "accuracy"),
                metric_mode=checkpoint_cfg.get("metric_mode", "max"),
            )
        )

    def _register_optional_hooks(self, hook_cfgs: List[Dict[str, Any]]):
        for cfg in hook_cfgs:
            if not cfg.get("enabled", True):
                continue

            hook_type = cfg.get("type")
            if not hook_type:
                self.logger.warning(f"Hook config missing 'type' field: {cfg}")
                continue

            try:
                hook_cfg = {k: v for k, v in cfg.items() if k != "enabled"}
                hook = build_hook(hook_cfg)
                self.register_hook(hook)
                self.logger.info(f"Registered optional hook: {hook_type}")
            except Exception as e:
                self.logger.error(f"Failed to build hook {hook_type}: {e}")
                raise

    def register_hook(self, hook: Hook):
        hook.trainer = self
        self.hooks.append(hook)

    def call_hook(self, fn_name: str, **kwargs):
        for hook in self.hooks:
            getattr(hook, fn_name, lambda **kw: None)(**kwargs)

    def train(self):
        """Main training loop"""
        self.call_hook("before_train")

        # Display training plan based on actual settings
        if self.max_steps is not None and self.max_epochs is not None:
            self.logger.console(
                f"Start training for {self.max_epochs} epochs or {self.max_steps} steps (whichever comes first)"
            )
        elif self.max_steps is not None:
            self.logger.console(f"Start training for {self.max_steps} steps")
        else:
            self.logger.console(f"Start training for {self.max_epochs} epochs")

        while True:
            self.call_hook("before_epoch")

            self.train_epoch()

            if self.scheduler and self.scheduler_step_per_epoch:
                self.scheduler.step()

            self.call_hook("after_epoch")

            self.epoch += 1

            if self.should_stop:
                break

            if self.max_epochs is not None and self.epoch >= self.max_epochs:
                self.logger.console(f"Reached max epochs {self.max_epochs}")
                break

        self.call_hook("after_train")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        total_loss_dict = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.call_hook("before_step")

            batch = self._move_to_device(batch)

            loss_dict = self._forward_step(batch)

            loss = loss_dict["loss"] / self.gradient_accumulation_steps

            self._backward_step(loss)

            # Perform optimizer step after accumulation completes
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step()
                self.global_step += 1

            num_batches += 1

            # Accumulate loss_dict (contains all loss terms including total loss)
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0
                if isinstance(value, torch.Tensor):
                    total_loss_dict[key] += value.item()
                else:
                    total_loss_dict[key] += value

            self.call_hook("after_step", loss_dict=loss_dict)

            # Check if max_steps is reached
            if self.max_steps and self.global_step >= self.max_steps:
                self.logger.console(f"Reached max steps {self.max_steps}")
                self.should_stop = True
                break

        # Handle remaining accumulated gradients
        if not self.should_stop and num_batches % self.gradient_accumulation_steps != 0:
            self._optimizer_step()
            self.global_step += 1

        # Compute average metrics for the epoch
        metrics = {key: value / num_batches for key, value in total_loss_dict.items()}

        self.current_metrics["train"] = metrics.copy()

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.fp16:
            with autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = self.model.forward_train(batch)
                assert (
                    "loss" in outputs
                ), "Model's forward_train must return a dict with 'loss'"
                loss = outputs.get("loss")
                if isinstance(loss, dict):
                    loss_dict = loss
                    assert (
                        "loss" in loss_dict
                    ), "loss_dict returned by model's forward_train must contain 'loss' key"
                else:
                    loss_dict = {"loss": loss}
        else:
            outputs = self.model.forward_train(batch)
            assert (
                "loss" in outputs
            ), "Model's forward_train must return a dict with 'loss'"
            loss = outputs.get("loss")
            if isinstance(loss, dict):
                loss_dict = loss
                assert (
                    "loss" in loss_dict
                ), "loss_dict returned by model's forward_train must contain 'loss' key"
            else:
                loss_dict = {"loss": loss}

        return loss_dict

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        evaluator: Optional[BaseEvaluator] = None,
    ) -> Dict[str, float]:
        """Evaluate model

        Args:
            dataloader: Data loader (default: val_loader)
            evaluator: Evaluator instance

        Returns:
            Metrics dictionary
        """
        dataloader = dataloader or self.val_loader
        if dataloader is None:
            return {}

        if evaluator is None:
            num_classes = getattr(self.model, "num_classes", None)
            if num_classes is None and hasattr(self.model, "head"):
                num_classes = getattr(self.model.head, "num_classes", None)
            if num_classes is None:
                num_classes = 2
            evaluator = ClassificationEvaluator(num_classes=num_classes)

        evaluator.reset()

        self.model.eval()

        for batch in dataloader:
            batch = self._move_to_device(batch)

            if self.fp16:
                with autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model.forward_train(batch)
            else:
                outputs = self.model.forward_train(batch)

            evaluator.process_batch(outputs, batch)

        metrics = evaluator.evaluate()

        self.model.train()

        return metrics

    def _backward_step(self, loss: torch.Tensor):
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self):
        if self.fp16:
            self.scaler.unscale_(self.optimizer)

        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.scheduler and not self.scheduler_step_per_epoch:
            self.scheduler.step()

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def save_checkpoint(self, filename: str = "checkpoint.pth"):
        """Save training checkpoint

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metric_mode": self._metric_mode,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        save_path = self.work_dir / filename
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        self.best_epoch = checkpoint["best_epoch"]

        if "metric_mode" in checkpoint:
            self._metric_mode = checkpoint["metric_mode"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.logger.console(
            f"Resumed from {checkpoint_path}, will start from epoch {self.epoch}"
        )
