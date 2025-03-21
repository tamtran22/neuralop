import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models import FNO1d

# from functools import partialmethod
from typing import Tuple, List, Union, Literal

Number = Union[float, int]

# from neuralop.model.layers.embeddings import GridEmbeddingND, GridEmbedding2D
# from neuralop.model.layers.spectral_convolution import SpectralConv
# from neuralop.model.layers.padding import DomainPadding
# from neuralop.model.layers.fno_block import FNOBlocks
# from neuralop.model.layers.channel_mlp import ChannelMLP
# from neuralop.model.layers.complex import ComplexValued


class FNO1d_activation(nn.Module):

    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        max_n_modes=None,
        n_layers=4,
        resolution_scaling_factor=None,
        non_linearity=F.gelu,
        stabilizer=None,
        complex_data=False,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="symmetric",
        post_activation=F.tanh,
        **kwargs
    ):
        super().__init__()
        self.net = FNO1d(
            n_modes_height=n_modes_height,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            max_n_modes=max_n_modes,
            n_layers=n_layers,
            resolution_scaling_factor=resolution_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            complex_data=complex_data,
            fno_block_precision=fno_block_precision,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            **kwargs
        )
        self.post_activation = post_activation

    def forward(self, x, output_shape=None, **kwargs):
        x = self.net(x)
        x = self.post_activation(x)
        return x