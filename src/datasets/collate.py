""" Module to collate batches """

import math
from typing import List, Union

import torch
import torch.nn.functional as F


def batch_padding_collate_fn(
    batch: List[Union[tuple, str, torch.Tensor]], pad_value: int = 0
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
        return tuple(
            (
                # Always pad labels with -1.
                batch_padding_collate_fn(samples, -1 if idx == 1 else pad_value)
                for idx, samples in enumerate(zip(*batch))
            )
        )
    if isinstance(elem, (bool, str)):
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
        F.pad(item, tuple(pad), "constant", pad_value)
        for item, pad in zip(batch, split_paddings)
    ]

    return torch.stack(batch)
