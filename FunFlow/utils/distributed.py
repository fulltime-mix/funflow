"""Distributed training utilities"""

import os
import pickle
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist


def init_distributed(
    backend: str = None,
    init_method: str = "env://",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
) -> bool:
    """Initialize distributed training environment

    Args:
        backend: Backend (auto: 'nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method
        rank: Global process rank
        world_size: Total processes
        local_rank: Local process rank

    Returns:
        True if initialized successfully
    """
    if "RANK" not in os.environ and rank is None:
        return False

    rank = rank if rank is not None else int(os.environ["RANK"])
    world_size = world_size if world_size is not None else int(os.environ["WORLD_SIZE"])
    local_rank = (
        local_rank if local_rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    )

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_id = local_rank % device_count if device_count > 0 else 0
        torch.cuda.set_device(device_id)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    synchronize()

    return True


def get_world_size() -> int:
    """Get total number of processes"""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank"""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get local process rank"""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if current process is main"""
    return get_rank() == 0


def is_distributed() -> bool:
    """Check if in distributed mode"""
    return dist.is_available() and dist.is_initialized()


def synchronize() -> None:
    """Synchronize all processes"""
    if not is_distributed():
        return

    if get_world_size() == 1:
        return

    dist.barrier()


def all_gather(data: Any) -> List[Any]:
    """Gather data from all processes

    Args:
        data: Current process data

    Returns:
        List of data from all processes
    """
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))

    if local_size != max_size:
        padding = torch.empty(
            (max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat([tensor, padding], dim=0)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True,
) -> Dict[str, torch.Tensor]:
    """All-reduce tensors in dict

    Args:
        input_dict: Input dictionary
        average: Whether to average

    Returns:
        Reduced dictionary
    """
    world_size = get_world_size()

    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


def broadcast(data: Any, src: int = 0) -> Any:
    """Broadcast data from src to all processes

    Args:
        data: Data to broadcast
        src: Source process rank

    Returns:
        Broadcasted data
    """
    if get_world_size() == 1:
        return data

    rank = get_rank()

    if rank == src:
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
        size = torch.tensor([tensor.numel()], device="cuda")
    else:
        size = torch.tensor([0], device="cuda")

    dist.broadcast(size, src)

    if rank != src:
        tensor = torch.empty((size.item(),), dtype=torch.uint8, device="cuda")

    dist.broadcast(tensor, src)

    if rank != src:
        buffer = tensor.cpu().numpy().tobytes()
        data = pickle.loads(buffer)

    return data


class DistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler ensuring each process handles different data subset"""

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = len(self.dataset) // self.num_replicas
        else:
            self.num_samples = (
                len(self.dataset) + self.num_replicas - 1
            ) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for shuffling"""
        self.epoch = epoch
