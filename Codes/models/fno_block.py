from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.complex import CGELU, apply_complex, ctanh, ComplexValued
from neuralop.layers.normalization_layers import AdaIN, InstanceNorm
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.utils import validate_scaling_factor

from .unet import UNet


Number = Union[int, float]


class IU_FNOBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False, #undoc
        implementation="factorized", #undoc
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        # apply real nonlin if data is real, otherwise CGELU
        if self.complex_data:
            self.non_linearity = CGELU
        else:
            self.non_linearity = non_linearity
        
        self.convs = conv_module(
                self.in_channels,
                self.out_channels,
                self.n_modes,
                resolution_scaling_factor=None if resolution_scaling_factor is None else self.resolution_scaling_factor[i],
                max_n_modes=max_n_modes,
                rank=rank,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                fno_block_precision=fno_block_precision,
                decomposition_kwargs=decomposition_kwargs
            )

        self.fno_skips = skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )

        self.channel_mlp = ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(self.out_channels * channel_mlp_expansion),
                    dropout=channel_mlp_dropout,
                    n_dim=self.n_dim,
                )
                
        self.channel_mlp_skips = skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=channel_mlp_skip,
                    n_dim=self.n_dim,
                )

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                    [
                        InstanceNorm()
                        for _ in range(self.n_norms)
                    ]
                )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(self.n_norms)
                ]
            )
        
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, ada_in]"
            )

        #
        self.unet = UNet(n_channels=self.out_channels, n_classes=self.out_channels, 
                         hidden_channel=out_channels, bilinear=False)

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):

        x_skip = x

        x_skip_fno = self.fno_skips(x)
        x_skip_fno = self.convs.transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips(x)
        x_skip_channel_mlp = self.convs.transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs(x, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms](x_fno)

        x = x_fno + x_skip_fno

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        # x = self.channel_mlp[index](x) + x_skip_channel_mlp # after Fourier and W
        x_channel_mlp = self.channel_mlp(x)
        x_unet = self.unet(x_channel_mlp-x)
        x = x_channel_mlp + x_unet + x_skip_channel_mlp

        if self.norm is not None:
            x = self.norm[self.n_norms + 1](x)

        # activation sigma
        # if index < (self.n_layers - 1):
        x = self.non_linearity(x)

        # add skip connection
        x = x + x_skip
        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        raise NotImplementedError
    #     # Apply non-linear activation (and norm)
    #     # before this block's convolution/forward pass:
    #     x = self.non_linearity(x)

    #     if self.norm is not None:
    #         x = self.norm[self.n_norms * index](x)

    #     x_skip_fno = self.fno_skips[index](x)
    #     x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

    #     x_skip_channel_mlp = self.channel_mlp_skips[index](x)
    #     x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

    #     if self.stabilizer == "tanh":
    #         if self.complex_data:
    #             x = ctanh(x)
    #         else:
    #             x = torch.tanh(x)

    #     x_fno = self.convs[index](x, output_shape=output_shape)

    #     x = x_fno + x_skip_fno

    #     if index < (self.n_layers - 1):
    #         x = self.non_linearity(x)

    #     if self.norm is not None:
    #         x = self.norm[self.n_norms * index + 1](x)

    #     x = self.channel_mlp[index](x) + x_skip_channel_mlp

    #     return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)