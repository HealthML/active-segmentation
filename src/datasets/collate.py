""" Module to collate batches """

import math
from typing import List, Union

import torch
import torch.nn.functional as F


def batch_padding_collate_fn(
    batch: List[Union[tuple, str, torch.Tensor]]
) -> torch.Tensor:
    """
    Collates a batch and padds tensors to the same size before stacking them.

    Args:
        batch (List[Union[tuple, str, torch.Tensor]]): The batch in List form.

    Returns:
        The batch collated.
    """
    elem = batch[0]
    if isinstance(elem, tuple):
        return tuple((batch_padding_collate_fn(samples) for samples in zip(*batch)))
    if isinstance(elem, str):
        return batch

    max_size = [max([item.size(i) for item in batch]) for i in range(batch[0].dim())]
    total_paddings = [
        [max_size[i] - item.size(i) for i in range(len(max_size))] for item in batch
    ]
    split_paddings = [
        [
            p
            for dim in reversed(padding)
            for p in [math.floor(dim / 2), math.ceil(dim / 2)]
        ]
        for padding in total_paddings
    ]
    batch = [
        F.pad(item, tuple(pad), "constant", 0)
        for item, pad in zip(batch, split_paddings)
    ]

    return torch.stack(batch)