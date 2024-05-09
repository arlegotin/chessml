import torch.nn as nn
from typing import Callable, Tuple, List, Dict


def make_conv_layers(
    input_shape: tuple,
    kernel_size: int,
    calc_next_channels: Callable[[int, int], float],
) -> Tuple[List[nn.Module], List[Dict]]:
    """
    Creates consequent convolutional layers and returns them with corresponding meta
    """

    assert (
        input_shape[1] == input_shape[2]
    ), f"input expected to be square, got {input_shape[1]}×{input_shape[2]} instead"

    layers = []
    meta = []
    current_shape = input_shape
    current_index = 0

    while current_shape[1] > 1:
        if current_index == 0:
            current_kernel_size = 1
        else:
            current_kernel_size = min(kernel_size, current_shape[1])

        next_channels = calc_next_channels(current_shape[0], current_index)

        next_shape = (
            int(next_channels),
            current_shape[1] + 1 - current_kernel_size,
            current_shape[2] + 1 - current_kernel_size,
        )

        layers.append(nn.Conv2d(current_shape[0], next_shape[0], current_kernel_size))

        meta.append(
            {
                "name": f"conv2d #{current_index}: {current_shape} -> {next_shape}",
                "in_shape": current_shape,
                "out_shape": next_shape,
                "kernel_size": current_kernel_size,
                "is_first": current_index == 0,
                "is_last": next_shape[1] == 1,
                "index": current_index,
            }
        )

        current_shape = next_shape
        current_index += 1

    return layers, meta


def mirror_conv_layers(
    reference_meta: List[Dict],
) -> Tuple[List[nn.Module], List[Dict]]:
    """
    Takes meta of convolutional layers
    nad creates a mirrored one
    """

    layers = []
    meta = []

    for meta_item in reversed(reference_meta):
        layers.append(
            nn.ConvTranspose2d(
                meta_item["out_shape"][0],
                meta_item["in_shape"][0],
                meta_item["kernel_size"],
            )
        )

        index = len(reference_meta) - 1 - meta_item["index"]

        meta.append(
            {
                "name": f"transposedConv2d #{index}: {meta_item['out_shape']} -> {meta_item['in_shape']}",
                "in_shape": meta_item["out_shape"],
                "out_shape": meta_item["in_shape"],
                "kernel_size": meta_item["kernel_size"],
                "is_first": meta_item["is_last"],
                "is_last": meta_item["is_first"],
                "index": index,
            }
        )

    return layers, meta
