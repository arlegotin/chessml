import torch.nn as nn
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CustomSequential(nn.Module):
    """
    A custom sequential model that has a extract_features method.

    Attributes:
    -----------
    layers: nn.ModuleList
        A list that holds the layers of the sequential model.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([])

    def add_module(self, name: str, layer: nn.Module):
        """
        Adds a module to the custom sequential model and logs the module's name.

        Parameters:
        -----------
        name : str
            The name of the module being added.
        layer : nn.Module
            The module that is added to the list of layers.
        """
        logger.info(name)
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def extract_features(
        self, x, condition: Optional[Callable[[nn.Module], bool]] = None
    ):
        """
        Computes the forward pass and conditionally extracts intermediate features.

        Parameters:
        -----------
        x : Tensor
            The input tensor.
        condition : Optional[Callable[[nn.Module], bool]]
            An optional function that takes a module as input and returns a boolean.
            If the function returns True for a module, the output of that module is
            included in the returned features. If the condition is None, all
            intermediate outputs are returned.

        Returns:
        --------
        extracted : list
            A list of tensors containing the extracted features.
        """
        extracted = []

        for layer in self.layers:
            x = layer(x)

            if not condition or condition(layer):
                extracted.append(x)

        return extracted
