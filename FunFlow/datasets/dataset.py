"""General dataset module for multimodal data (audio, image, time series)."""

import random
import yaml
import importlib
from typing import Dict, Any, List, Optional, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from FunFlow.utils.file_utils import read_jsonl
from FunFlow.logger import get_logger

logger = get_logger("FunFlow")


def _import_processor_module(module_path: str):
    """Dynamically import processor module.

    Args:
        module_path: Full module path, e.g., 'FunFlow.datasets.audio_processor'

    Returns:
        Imported module object

    Raises:
        ImportError: If module import fails
    """
    try:
        module = importlib.import_module(module_path)
        logger.info(f"Successfully imported processor module: {module_path}")
        return module
    except ImportError as e:
        logger.error(f"Failed to import processor module '{module_path}': {e}")
        raise ImportError(
            f"Cannot import processor module '{module_path}'. "
            f"Please check the module path in your config. Error: {e}"
        )


class Processor(IterableDataset):
    """Wrapper class for processing functions as iterable dataset."""

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def __iter__(self):
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    """Distributed sampler for data partitioning."""

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        data = data.copy()
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank :: self.world_size]
        data = data[self.worker_id :: self.num_workers]
        return data


class DataList(IterableDataset):
    """Data list class with distributed sampling support."""

    def __init__(self, data_list: List, shuffle=True, partition=True):
        self.data_list = data_list
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        data_list = self.sampler.sample(self.data_list)
        for item in data_list:
            data = dict(raw=item)
            data.update(sampler_info)
            yield data


def shuffle(data: Iterator, shuffle_size: int = 1000) -> Iterator:
    """Shuffle data locally with buffer.

    Args:
        data: Input data iterator
        shuffle_size: Buffer size for shuffling

    Yields:
        Shuffled samples
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    random.shuffle(buf)
    for x in buf:
        yield x


def batch(data: Iterator, batch_size: int = 32) -> Iterator:
    """Batch data into groups.

    Args:
        data: Input data iterator
        batch_size: Number of samples per batch

    Yields:
        Batched samples
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def build_dataset(
    data_file: str, conf: Dict[str, Any], partition: bool = True
) -> IterableDataset:
    """Build dataset from config.

    Args:
        data_file: Path to jsonl data file
        conf: Config dict with processor_module, shuffle, processors, batch_conf, etc.
        partition: Whether to partition data in distributed mode

    Returns:
        IterableDataset
    """
    data_list = read_jsonl(data_file)
    logger.info(f"Loaded {len(data_list)} samples from {data_file}")

    shuffle_flag = conf.get("shuffle", True)

    dataset = DataList(data_list, shuffle=shuffle_flag, partition=partition)

    processor_module_path = conf.get("processor_module")
    if not processor_module_path:
        raise ValueError(
            "Missing 'processor_module' in config. "
            "Please specify the processor module path, e.g., 'local.dataset.processor'"
        )
    proc_module = _import_processor_module(processor_module_path)

    parse_raw_conf = conf.get("parse_raw_conf", {})
    dataset = Processor(dataset, proc_module.parse_raw, **parse_raw_conf)
    logger.info(f"Apply parse_raw")

    processors_conf = conf.get("processors", [])
    for proc_conf in processors_conf:
        if isinstance(proc_conf, str):
            proc_name = proc_conf
            proc_args = {}
            enabled = True
        elif isinstance(proc_conf, dict):
            proc_name = proc_conf.get("name")
            enabled = proc_conf.get("enabled", True)
            proc_args = {
                k: v for k, v in proc_conf.items() if k not in ["name", "enabled"]
            }
        else:
            continue

        if not enabled:
            logger.info(f"Skip disabled processor: {proc_name}")
            continue

        if hasattr(proc_module, proc_name):
            proc_fn = getattr(proc_module, proc_name)
        else:
            logger.warning(f"Processor '{proc_name}' not found, skipping")
            continue

        dataset = Processor(dataset, proc_fn, **proc_args)
        logger.info(f"Apply {proc_name} with config: {proc_args}")

    if shuffle_flag:
        shuffle_conf = conf.get("shuffle_conf", {"shuffle_size": 1000})
        dataset = Processor(dataset, shuffle, **shuffle_conf)
        logger.info(f"Apply shuffle with config: {shuffle_conf}")

    batch_conf = conf.get("batch_conf", {"batch_size": 32})
    dataset = Processor(dataset, batch, **batch_conf)
    logger.info(f"Apply batch with config: {batch_conf}")

    if hasattr(proc_module, "collate_fn"):
        dataset = Processor(dataset, proc_module.collate_fn)
        logger.info(f"Apply collate_fn")

    return dataset


if __name__ == "__main__":
    config_path = "conf/config_tdcsfog.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_file = config["data"]["train"]["data_file"]
    dataset_conf = config["data"]["train"].get("conf", config["data"]["train"])
    dataset = build_dataset(data_file=data_file, conf=dataset_conf, partition=False)
    num_workers = config["data"].get("num_workers", 0)
    prefetch_factor = (
        None if num_workers == 0 else config["data"].get("prefetch_factor", None)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    num = 0
    num_0 = 0
    num_1 = 0
    num_x = 0
    normal_frame = 0
    fog_frame = 0
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape {value.shape}")
            else:
                print(f"  {key}: {value[:5]}")
            if key == "segment_labels":
                for idx, label_seq in enumerate(value):
                    if label_seq.item() == 0:
                        num_0 += 1
                    elif label_seq.item() == 1:
                        num_1 += 1
                    elif label_seq.item() == -1:
                        num_x += 1
                    else:
                        raise ValueError(f"Unexpected label value: {label_seq.item()}")
            if key == "labels":
                for idx, label_seq in enumerate(value):
                    normal_frame += (label_seq == 0).sum().item()
                    fog_frame += (label_seq == 1).sum().item()
        if "feats" in batch:
            num += batch["feats"].shape[0]
    print(f"Segment label counts - 0: {num_0}, 1: {num_1}, -1: {num_x}")
    print(f"Frame label counts - Normal (0): {normal_frame}, FOG (1): {fog_frame}")
    print(f"{fog_frame / (normal_frame + fog_frame):.4f} frames are fog")
    print(f"{normal_frame / (normal_frame + fog_frame):.4f} frames are normal")
    print(f"Total samples processed: {num}")
