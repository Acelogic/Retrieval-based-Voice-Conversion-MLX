"""
Multi-Period Discriminator for RVC MLX Training

MLX implementation of the HiFi-GAN Multi-Period Discriminator (v2).
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' output size."""
    return int((kernel_size * dilation - dilation) / 2)


class GroupedConv1d(nn.Module):
    """
    Grouped Conv1d implementation for MLX.

    MLX doesn't have native grouped convolution support, so we implement it
    by splitting channels into groups and applying separate convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"

        self.groups = groups
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        # Create separate convolutions for each group
        for i in range(groups):
            conv = nn.Conv1d(
                self.in_channels_per_group,
                self.out_channels_per_group,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            setattr(self, f"conv_{i}", conv)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with grouped convolution.

        Args:
            x: Input (B, T, C) in MLX format

        Returns:
            Output (B, T', C_out)
        """
        # For groups=1, we still have conv_0 (same as regular conv)
        if self.groups == 1:
            conv = getattr(self, "conv_0")
            return conv(x)

        # Split input along channel dimension
        # x is (B, T, C)
        b, t, c = x.shape

        # Reshape to (B, T, groups, channels_per_group)
        x_grouped = x.reshape(b, t, self.groups, self.in_channels_per_group)

        # Apply convolutions to each group
        outputs = []
        for i in range(self.groups):
            conv = getattr(self, f"conv_{i}")
            x_i = x_grouped[:, :, i, :]  # (B, T, in_channels_per_group)
            out_i = conv(x_i)  # (B, T', out_channels_per_group)
            outputs.append(out_i)

        # Stack along last axis and reshape
        # outputs: list of (B, T', out_channels_per_group)
        out = mx.stack(outputs, axis=-2)  # (B, T', groups, out_channels_per_group)
        b_out, t_out, _, _ = out.shape
        out = out.reshape(b_out, t_out, -1)  # (B, T', out_channels)

        return out


class DiscriminatorS(nn.Module):
    """
    Scale Discriminator using 1D convolutions.

    Processes audio at multiple scales to capture different frequency patterns.
    Uses grouped convolutions to match PyTorch HiFi-GAN implementation.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        # Note: MLX doesn't have spectral_norm/weight_norm built-in
        # We implement without normalization (weights are normalized during training via optimizer)

        # Conv1d layers with downsampling (matching PyTorch groups)
        # PyTorch: groups=4, 16, 64, 256 for conv1-4
        # MLX Conv1d: input (B, T, C), kernel (out_channels, kernel_size, in_channels)
        self.conv_0 = nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)
        self.conv_1 = GroupedConv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)
        self.conv_2 = GroupedConv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)
        self.conv_3 = GroupedConv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)
        self.conv_4 = GroupedConv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)
        self.conv_5 = nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)
        self.conv_post = nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)

        self.n_convs = 6

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """
        Forward pass.

        Args:
            x: Input audio (B, T, 1) in MLX format

        Returns:
            output: Discriminator score (B, T')
            fmap: List of intermediate feature maps
        """
        fmap = []

        # Process through conv layers
        x = nn.leaky_relu(self.conv_0(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_1(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_2(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_3(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_4(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_5(x), LRELU_SLOPE)
        fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        # Flatten for output
        x = x.reshape(x.shape[0], -1)

        return x, fmap


class DiscriminatorP(nn.Module):
    """
    Period Discriminator using 2D convolutions.

    Reshapes audio into 2D based on period and applies 2D convolutions.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period

        # Conv2d layers
        # MLX Conv2d: input (B, H, W, C), kernel (out_channels, kH, kW, in_channels)
        self.conv_0 = nn.Conv2d(1, 32, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))
        self.conv_1 = nn.Conv2d(32, 128, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))
        self.conv_2 = nn.Conv2d(128, 512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))
        self.conv_3 = nn.Conv2d(512, 1024, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))
        self.conv_4 = nn.Conv2d(1024, 1024, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(get_padding(kernel_size, 1), 0))
        self.conv_post = nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """
        Forward pass.

        Args:
            x: Input audio (B, T, 1) in MLX format

        Returns:
            output: Discriminator score (B, T')
            fmap: List of intermediate feature maps
        """
        fmap = []
        b, t, c = x.shape

        # Pad to make length divisible by period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = mx.pad(x, [(0, 0), (0, n_pad), (0, 0)])
            t = t + n_pad

        # Reshape: (B, T, C) -> (B, T//period, period, C)
        x = x.reshape(b, t // self.period, self.period, c)

        # Apply conv layers
        x = nn.leaky_relu(self.conv_0(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_1(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_2(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_3(x), LRELU_SLOPE)
        fmap.append(x)

        x = nn.leaky_relu(self.conv_4(x), LRELU_SLOPE)
        fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        # Flatten
        x = x.reshape(b, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) v2.

    Combines one scale discriminator with multiple period discriminators
    (periods: 2, 3, 5, 7, 11, 17, 23, 37).
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        version: str = "v2",
    ):
        super().__init__()

        # Define periods based on version
        if version == "v1":
            periods = [2, 3, 5, 7, 11, 17]
        elif version == "v2":
            periods = [2, 3, 5, 7, 11, 17, 23, 37]
        else:
            periods = [2, 3, 5, 7, 11, 17, 23, 37]

        self.version = version
        self.periods = periods

        # Scale discriminator
        self.discriminator_s = DiscriminatorS(use_spectral_norm)

        # Period discriminators - using named attributes for MLX weight loading
        for i, p in enumerate(periods):
            setattr(self, f"discriminator_p_{i}", DiscriminatorP(p, use_spectral_norm=use_spectral_norm))

        self.n_period_discriminators = len(periods)

    def __call__(
        self,
        y: mx.array,
        y_hat: mx.array,
    ) -> Tuple[List[mx.array], List[mx.array], List[List[mx.array]], List[List[mx.array]]]:
        """
        Forward pass for both real and generated audio.

        Args:
            y: Real audio (B, T, 1)
            y_hat: Generated audio (B, T, 1)

        Returns:
            y_d_rs: List of discriminator outputs for real audio
            y_d_gs: List of discriminator outputs for generated audio
            fmap_rs: List of feature maps for real audio
            fmap_gs: List of feature maps for generated audio
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # Scale discriminator
        y_d_r, fmap_r = self.discriminator_s(y)
        y_d_g, fmap_g = self.discriminator_s(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)

        # Period discriminators
        for i in range(self.n_period_discriminators):
            disc = getattr(self, f"discriminator_p_{i}")
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def forward_single(self, x: mx.array) -> Tuple[List[mx.array], List[List[mx.array]]]:
        """
        Forward pass for single input (real or generated).

        Args:
            x: Audio input (B, T, 1)

        Returns:
            outputs: List of discriminator outputs
            fmaps: List of feature maps
        """
        outputs = []
        fmaps = []

        # Scale discriminator
        out, fmap = self.discriminator_s(x)
        outputs.append(out)
        fmaps.append(fmap)

        # Period discriminators
        for i in range(self.n_period_discriminators):
            disc = getattr(self, f"discriminator_p_{i}")
            out, fmap = disc(x)
            outputs.append(out)
            fmaps.append(fmap)

        return outputs, fmaps
