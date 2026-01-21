"""Training hooks for callbacks during training process"""

from abc import ABC
from typing import Dict, Any, Optional, Type, Callable
from pathlib import Path
import torch
import yaml

from FunFlow.registry import HOOKS


def build_hook(cfg: Dict[str, Any]) -> "Hook":
    """Build hook from config

    Args:
        cfg: Hook config with 'type' field

    Returns:
        Hook instance
    """
    return HOOKS.build(cfg)


class Hook(ABC):
    """Base hook class"""

    def __init__(self):
        self.trainer = None

    def before_train(self, **kwargs):
        """Before training starts"""
        pass

    def after_train(self, **kwargs):
        """After training ends"""
        pass

    def before_epoch(self, **kwargs):
        """Before each epoch"""
        pass

    def after_epoch(self, **kwargs):
        """After each epoch"""
        pass

    def before_step(self, **kwargs):
        """Before each step"""
        pass

    def after_step(self, **kwargs):
        """After each step"""
        pass


@HOOKS.register_module()
class LoggerHook(Hook):
    """Logger hook for recording training metrics"""

    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self.loss_dict_buffer = {}

    def after_step(self, loss_dict: Dict[str, Any] = None, **kwargs):
        """Record loss"""
        if loss_dict:
            for key, value in loss_dict.items():
                if key not in self.loss_dict_buffer:
                    self.loss_dict_buffer[key] = []
                if isinstance(value, torch.Tensor):
                    self.loss_dict_buffer[key].append(value.item())
                else:
                    self.loss_dict_buffer[key].append(value)

        if self.trainer.global_step % self.log_interval == 0:
            loss_details = []
            for key, values in self.loss_dict_buffer.items():
                recent_values = values[-self.log_interval :]
                avg_value = (
                    sum(recent_values) / len(recent_values) if recent_values else 0
                )
                loss_details.append(f"{key}: {avg_value:.4f}")
            loss_str = " ".join(loss_details) if loss_details else "loss: N/A"

            self.trainer.logger.info(
                f"Epoch [{self.trainer.epoch}] Step [{self.trainer.global_step}] "
                f"{loss_str} LR: {self.trainer.optimizer.param_groups[0]['lr']:.2e}"
            )

    def after_epoch(self, **kwargs):
        """Record epoch metrics"""
        train_metrics = self.trainer.current_metrics.get("train", {})
        if train_metrics:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            self.trainer.logger.info(
                f"Epoch [{self.trainer.epoch}] Train - {metrics_str}"
            )

        self.loss_dict_buffer = {}


@HOOKS.register_module()
class EvalHook(Hook):
    """Evaluation hook supporting epoch and step modes"""

    def __init__(
        self,
        eval_interval: int = 1,
        eval_interval_by_step: Optional[int] = None,
        evaluator_cfg: Optional[Dict[str, Any]] = None,
        custom_metrics_fn: Optional[
            Callable[[Dict[str, Any]], Dict[str, float]]
        ] = None,
    ):
        """Args:
        eval_interval: Evaluation interval (epoch level)
        eval_interval_by_step: Evaluation interval (step level)
        evaluator_cfg: Evaluator config
        custom_metrics_fn: Custom metrics function
        """
        super().__init__()
        self.eval_interval = eval_interval
        self.eval_interval_by_step = eval_interval_by_step
        self.evaluator_cfg = evaluator_cfg
        self.custom_metrics_fn = custom_metrics_fn
        self._evaluator = None
        self._eval_mode = None

    def before_train(self, **kwargs):
        """Determine evaluation mode before training"""
        if self.eval_interval_by_step is not None:
            self._eval_mode = "step"
        elif self.trainer.max_epochs is None and self.trainer.max_steps is not None:
            self._eval_mode = "step"
            if self.eval_interval_by_step is None:
                self.eval_interval_by_step = max(1, self.trainer.max_steps // 100)
        else:
            self._eval_mode = "epoch"

        if self._eval_mode == "step":
            mode_str = f"step-level (every {self.eval_interval_by_step} steps)"
        else:
            mode_str = f"epoch-level (every {self.eval_interval} epochs)"
        self.trainer.logger.console(f"Evaluation mode: {mode_str}")

    def _get_evaluator(self):
        """Get or create evaluator"""
        if self._evaluator is None:
            from FunFlow.trainer.evaluator import (
                ClassificationEvaluator,
                build_evaluator,
            )

            if self.evaluator_cfg is not None:
                cfg = self.evaluator_cfg.copy()
                if self.custom_metrics_fn is not None:
                    cfg["custom_metrics_fn"] = self.custom_metrics_fn
                self._evaluator = build_evaluator(cfg)
            else:
                num_classes = getattr(self.trainer.model, "num_classes", None)
                if num_classes is None and hasattr(self.trainer.model, "head"):
                    num_classes = getattr(self.trainer.model.head, "num_classes", None)
                if num_classes is None:
                    num_classes = 2
                self._evaluator = ClassificationEvaluator(
                    num_classes=num_classes,
                    custom_metrics_fn=self.custom_metrics_fn,
                )

        return self._evaluator

    def _do_evaluate(self):
        """Execute evaluation"""
        evaluator = self._get_evaluator()
        metrics = self.trainer.evaluate(evaluator=evaluator)

        if metrics:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.trainer.logger.info(
                f"Epoch [{self.trainer.epoch}] Step [{self.trainer.global_step}] Val - {metrics_str}"
            )

            self.trainer.current_metrics["val"] = metrics.copy()

    def after_epoch(self, **kwargs):
        """Evaluate after epoch (epoch mode)"""
        if (
            self._eval_mode == "epoch"
            and (self.trainer.epoch + 1) % self.eval_interval == 0
        ):
            self._do_evaluate()

    def after_step(self, **kwargs):
        """Evaluate after step (step mode)"""
        if (
            self._eval_mode == "step"
            and self.trainer.global_step % self.eval_interval_by_step == 0
        ):
            self._do_evaluate()


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Checkpoint hook for saving models and metrics"""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        save_best: bool = True,
        save_last: bool = True,
        max_keep_ckpts: int = 5,
        metric_name: str = "accuracy",
        metric_mode: str = "max",
    ):
        """Args:
        save_dir: Save directory
        save_best: Create best symlink
        save_last: Save last model
        max_keep_ckpts: Max checkpoints to keep
        metric_name: Metric name for ranking
        metric_mode: 'max' or 'min'
        """
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_best = save_best
        self.save_last = save_last
        self.max_keep_ckpts = max_keep_ckpts
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.saved_ckpts = []

    def _save_metrics_yaml(self, yaml_path: Path):
        """Save metrics to YAML file"""
        metrics_dict = {
            "epoch": self.trainer.epoch,
            "global_step": self.trainer.global_step,
            "lr": self.trainer.optimizer.param_groups[0]["lr"],
        }

        train_metrics = self.trainer.current_metrics.get("train", {})
        if train_metrics:
            for k, v in train_metrics.items():
                metrics_dict[f"train_{k}"] = float(v)

        val_metrics = self.trainer.current_metrics.get("val", {})
        if val_metrics:
            for k, v in val_metrics.items():
                metrics_dict[f"cv_{k}"] = float(v)
            if "loss" in val_metrics:
                metrics_dict["cv_loss"] = float(val_metrics["loss"])

        with open(yaml_path, "w") as f:
            yaml.dump(metrics_dict, f, default_flow_style=False)

    def _is_better(
        self, new_metric: float, old_metric: float, mode: Optional[str] = None
    ) -> bool:
        """Check if new metric is better"""
        mode = mode or self.metric_mode
        if mode == "max":
            return new_metric > old_metric
        else:
            return new_metric < old_metric

    def _get_worst_ckpt(self, mode: Optional[str] = None):
        """Get worst checkpoint from saved list"""
        if not self.saved_ckpts:
            return None, None

        mode = mode or self.metric_mode

        if mode == "max":
            worst_idx = min(
                range(len(self.saved_ckpts)), key=lambda i: self.saved_ckpts[i][1]
            )
        else:
            worst_idx = max(
                range(len(self.saved_ckpts)), key=lambda i: self.saved_ckpts[i][1]
            )

        return worst_idx, self.saved_ckpts[worst_idx]

    def _get_best_ckpt(self, mode: Optional[str] = None):
        """Get best checkpoint from saved list"""
        if not self.saved_ckpts:
            return None

        mode = mode or self.metric_mode

        if mode == "max":
            best_idx = max(
                range(len(self.saved_ckpts)), key=lambda i: self.saved_ckpts[i][1]
            )
        else:
            best_idx = min(
                range(len(self.saved_ckpts)), key=lambda i: self.saved_ckpts[i][1]
            )

        return self.saved_ckpts[best_idx][0]

    def _update_best_link(self):
        """Update best symlink to point to best model"""
        if not self.save_best:
            return

        best_ckpt = self._get_best_ckpt()
        if best_ckpt is None:
            return

        best_link = self.save_dir / "best.pth"
        best_yaml_link = self.save_dir / "best.yaml"

        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        if best_yaml_link.exists() or best_yaml_link.is_symlink():
            best_yaml_link.unlink()

        best_link.symlink_to(best_ckpt.name)
        best_yaml_link.symlink_to(best_ckpt.with_suffix(".yaml").name)

    def _get_current_metric(self) -> tuple:
        """Get current metric for comparison

        Returns:
            (metric_value, metric_name, metric_mode, source)
        """
        val_metrics = self.trainer.current_metrics.get("val", {})
        if val_metrics and self.metric_name in val_metrics:
            return (
                val_metrics[self.metric_name],
                self.metric_name,
                self.metric_mode,
                "val",
            )

        train_metrics = self.trainer.current_metrics.get("train", {})
        if train_metrics and "loss" in train_metrics:
            return (
                train_metrics["loss"],
                "loss",
                "min",
                "train",
            )

        return (None, None, None, None)

    def after_epoch(self, **kwargs):
        """Save checkpoint after epoch based on metrics"""
        if self.save_dir is None:
            self.save_dir = self.trainer.work_dir

        current_metric, metric_name, metric_mode, source = self._get_current_metric()

        if current_metric is None:
            if self.save_last:
                self.trainer.save_checkpoint("last.pth")
                self._save_metrics_yaml(self.save_dir / "last.yaml")
            return

        is_global_best = self._is_better(
            current_metric, self.trainer.best_metric, mode=metric_mode
        )
        if is_global_best:
            self.trainer.best_metric = current_metric
            self.trainer.best_epoch = self.trainer.epoch
            self.trainer.logger.console(
                f"New best model! {metric_name}: {current_metric:.4f} (from {source})"
            )

        should_save = False

        if len(self.saved_ckpts) < self.max_keep_ckpts:
            should_save = True
        else:
            worst_idx, (worst_ckpt, worst_metric) = self._get_worst_ckpt(
                mode=metric_mode
            )
            if self._is_better(current_metric, worst_metric, mode=metric_mode):
                should_save = True
                if worst_ckpt.exists():
                    worst_ckpt.unlink()
                worst_yaml = worst_ckpt.with_suffix(".yaml")
                if worst_yaml.exists():
                    worst_yaml.unlink()
                self.saved_ckpts.pop(worst_idx)

        if should_save:
            filename = f"epoch_{self.trainer.epoch}.pth"
            self.trainer.save_checkpoint(filename)
            ckpt_path = self.save_dir / filename

            yaml_filename = f"epoch_{self.trainer.epoch}.yaml"
            self._save_metrics_yaml(self.save_dir / yaml_filename)

            self.saved_ckpts.append((ckpt_path, current_metric))

            self._update_best_link()

        if self.save_last:
            self.trainer.save_checkpoint("last.pth")
            self._save_metrics_yaml(self.save_dir / "last.yaml")


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early stopping hook"""

    def __init__(
        self,
        patience: int = 10,
        metric_name: str = "accuracy",
        metric_mode: str = "max",
        min_delta: float = 0.0001,
    ):
        """Args:
        patience: Number of epochs with no improvement
        metric_name: Metric name to monitor
        metric_mode: 'min' or 'max'
        min_delta: Minimum change to qualify as improvement
        """
        super().__init__()
        self.patience = patience
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.min_delta = min_delta

        self.counter = 0
        self.best_value = None
        self._initialized = False

    def _get_current_metric(self) -> tuple:
        """Get current metric for comparison

        Returns:
            (metric_value, metric_name, metric_mode, source)
        """
        val_metrics = self.trainer.current_metrics.get("val", {})
        if val_metrics and self.metric_name in val_metrics:
            return (
                val_metrics[self.metric_name],
                self.metric_name,
                self.metric_mode,
                "val",
            )

        train_metrics = self.trainer.current_metrics.get("train", {})
        if train_metrics and "loss" in train_metrics:
            return (
                train_metrics["loss"],
                "loss",
                "min",
                "train",
            )

        return (None, None, None, None)

    def after_epoch(self, **kwargs):
        """Check for early stopping"""
        current, metric_name, metric_mode, source = self._get_current_metric()

        if current is None:
            return

        if not self._initialized:
            self.best_value = float("-inf") if metric_mode == "max" else float("inf")
            self._initialized = True

        if metric_mode == "max":
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.trainer.logger.console(
                f"Early stopping! No improvement in {metric_name} (from {source}) "
                f"for {self.patience} epochs."
            )
            self.trainer.should_stop = True


@HOOKS.register_module()
class LRSchedulerHook(Hook):
    """Learning rate scheduler hook with warmup"""

    def __init__(
        self,
        warmup_epochs: int = 0,
        warmup_steps: int = 0,
        warmup_lr: float = 1e-6,
    ):
        """Args:
        warmup_epochs: Warmup epochs (used when warmup_steps=0)
        warmup_steps: Warmup steps (higher priority)
        warmup_lr: Initial warmup learning rate
        """
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.base_lr = None

    def before_train(self, **kwargs):
        """Record base learning rate"""
        self.base_lr = self.trainer.optimizer.param_groups[0]["lr"]

        if self.warmup_steps > 0:
            self.trainer.logger.console(
                f"LR Warmup: {self.warmup_steps} steps, "
                f"from {self.warmup_lr:.2e} to {self.base_lr:.2e}"
            )
        elif self.warmup_epochs > 0:
            self.trainer.logger.console(
                f"LR Warmup: {self.warmup_epochs} epochs, "
                f"from {self.warmup_lr:.2e} to {self.base_lr:.2e}"
            )

    def before_step(self, **kwargs):
        """Step-level warmup"""
        if self.warmup_steps > 0 and self.trainer.global_step < self.warmup_steps:
            progress = self.trainer.global_step / self.warmup_steps
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * progress

            for param_group in self.trainer.optimizer.param_groups:
                param_group["lr"] = lr

    def before_epoch(self, **kwargs):
        """Epoch-level warmup"""
        if self.warmup_steps == 0 and self.trainer.epoch < self.warmup_epochs:
            progress = self.trainer.epoch / self.warmup_epochs
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * progress

            for param_group in self.trainer.optimizer.param_groups:
                param_group["lr"] = lr


@HOOKS.register_module()
class TensorBoardHook(Hook):
    """TensorBoard logging hook"""

    def __init__(self, log_dir: Optional[str] = None):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        self.use_step_granularity = False

    def before_train(self, **kwargs):
        """Initialize TensorBoard"""
        from torch.utils.tensorboard import SummaryWriter

        log_dir = self.log_dir or (self.trainer.work_dir / "tensorboard")
        self.writer = SummaryWriter(log_dir)

        if self.trainer.max_steps is not None:
            self.use_step_granularity = True
            self.trainer.logger.info(
                "TensorBoard: Using step granularity (x-axis: global_step)"
            )
        else:
            self.use_step_granularity = False
            self.trainer.logger.info(
                "TensorBoard: Using epoch granularity (x-axis: epoch)"
            )

    def after_step(self, loss_dict: Dict[str, Any] = None, **kwargs):
        """Record training step metrics"""
        if not self.use_step_granularity:
            return

        if loss_dict:
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"train/{key}", value, self.trainer.global_step)

        self.writer.add_scalar(
            "train/lr",
            self.trainer.optimizer.param_groups[0]["lr"],
            self.trainer.global_step,
        )

    def after_epoch(self, **kwargs):
        """Record epoch-level metrics"""
        if self.use_step_granularity:
            val_metrics = self.trainer.current_metrics.get("val", {})
            if val_metrics:
                for name, value in val_metrics.items():
                    self.writer.add_scalar(
                        f"val/{name}", value, self.trainer.global_step
                    )
        else:
            x_value = self.trainer.epoch

            train_metrics = self.trainer.current_metrics.get("train", {})
            if train_metrics:
                for name, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{name}", value, x_value)

            val_metrics = self.trainer.current_metrics.get("val", {})
            if val_metrics:
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{name}", value, x_value)

            self.writer.add_scalar(
                "train/lr",
                self.trainer.optimizer.param_groups[0]["lr"],
                x_value,
            )

    def after_train(self, **kwargs):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


@HOOKS.register_module()
class QATHook(Hook):
    """Quantization-aware training hook"""

    def __init__(
        self,
        freeze_observer_epoch: int = 2,
        freeze_bn_epoch: int = 3,
        verbose: bool = True,
    ):
        """Args:
        freeze_observer_epoch: Epoch to freeze observers
        freeze_bn_epoch: Epoch to freeze BatchNorm stats
        verbose: Whether to print logs
        """
        super().__init__()
        self.freeze_observer_epoch = freeze_observer_epoch
        self.freeze_bn_epoch = freeze_bn_epoch
        self.verbose = verbose
        self._observer_frozen = False
        self._bn_frozen = False

    def before_epoch(self, **kwargs):
        """Check freezing before each epoch"""
        model = self.trainer.model

        if (
            self.trainer.epoch >= self.freeze_observer_epoch
            and not self._observer_frozen
        ):
            self._freeze_observers(model)
            self._observer_frozen = True
            if self.verbose:
                self.trainer.logger.console(
                    f"[QAT] Epoch {self.trainer.epoch}: Froze quantization observers"
                )

        if self.trainer.epoch >= self.freeze_bn_epoch and not self._bn_frozen:
            self._freeze_bn_stats(model)
            self._bn_frozen = True
            if self.verbose:
                self.trainer.logger.console(
                    f"[QAT] Epoch {self.trainer.epoch}: Froze BatchNorm statistics"
                )

    def _freeze_observers(self, model: torch.nn.Module):
        """Freeze quantization observers"""
        try:
            model.apply(torch.ao.quantization.disable_observer)
        except AttributeError:
            model.apply(torch.quantization.disable_observer)

    def _freeze_bn_stats(self, model: torch.nn.Module):
        """Freeze BatchNorm running statistics"""
        try:
            qat_module = torch.ao.nn.intrinsic.qat
        except AttributeError:
            qat_module = torch.nn.intrinsic.qat

        for module in model.modules():
            if hasattr(module, "freeze_bn_stats"):
                module.freeze_bn_stats()

    def after_train(self, **kwargs):
        """Log after training"""
        if self.verbose:
            self.trainer.logger.console(
                f"[QAT] Training completed. Observer frozen: {self._observer_frozen}, "
                f"BN frozen: {self._bn_frozen}"
            )
