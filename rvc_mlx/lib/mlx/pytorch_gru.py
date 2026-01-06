"""
Custom GRU implementation that exactly matches PyTorch's GRU formula.

PyTorch GRU formula:
    r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
    z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
    n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}

Where weights are concatenated:
    weight_ih = [W_ir; W_iz; W_in]  # shape (3H, I)
    weight_hh = [W_hr; W_hz; W_hn]  # shape (3H, H)
    bias_ih = [b_ir; b_iz; b_in]    # shape (3H,)
    bias_hh = [b_hr; b_hz; b_hn]    # shape (3H,)
"""

import mlx.core as mx
import mlx.nn as nn


class ModuleList(nn.Module):
    """Custom ModuleList that properly registers submodules."""
    def __init__(self, modules=None):
        super().__init__()
        self._length = 0
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module):
        setattr(self, str(self._length), module)
        self._length += 1

    def __getitem__(self, idx):
        if idx < 0: idx += self._length
        if idx < 0 or idx >= self._length: raise IndexError()
        return getattr(self, str(idx))

    def __len__(self):
        return self._length

    def __iter__(self):
        for i in range(self._length):
            yield self[i]


class PyTorchGRU(nn.Module):
    """GRU cell that exactly matches PyTorch's implementation."""

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Initialize weights (will be loaded from converted PyTorch weights)
        # weight_ih: (3*hidden_size, input_size)
        # weight_hh: (3*hidden_size, hidden_size)
        scale = (1.0 / hidden_size) ** 0.5
        self.weight_ih = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, input_size)
        )
        self.weight_hh = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_size, hidden_size)
        )

        if bias:
            self.bias_ih = mx.zeros((3 * hidden_size,))
            self.bias_hh = mx.zeros((3 * hidden_size,))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        hidden_size = self.hidden_size

        # Initialize hidden state
        h = mx.zeros((batch_size, hidden_size))

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # Compute input transformations
            gi = x_t @ self.weight_ih.T  # (batch_size, 3*hidden_size)
            if self.bias_ih is not None:
                gi = gi + self.bias_ih

            # Compute hidden transformations
            gh = h @ self.weight_hh.T  # (batch_size, 3*hidden_size)
            if self.bias_hh is not None:
                gh = gh + self.bias_hh

            # Split into gates
            i_r, i_z, i_n = mx.split(gi, 3, axis=1)
            h_r, h_z, h_n = mx.split(gh, 3, axis=1)

            # Reset gate
            r_t = mx.sigmoid(i_r + h_r)

            # Update gate
            z_t = mx.sigmoid(i_z + h_z)

            # New gate
            n_t = mx.tanh(i_n + r_t * h_n)

            # New hidden state
            h = (1 - z_t) * n_t + z_t * h

            outputs.append(h)

        # Stack outputs
        output = mx.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_size)

        return output


class BiGRU(nn.Module):
    """Bidirectional GRU using PyTorchGRU cells."""

    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.forward_grus = ModuleList()
        self.backward_grus = ModuleList()

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size * 2
            self.forward_grus.append(PyTorchGRU(in_size, hidden_size, bias=bias))
            self.backward_grus.append(PyTorchGRU(in_size, hidden_size, bias=bias))

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, 2*hidden_size)
        """
        for i in range(self.num_layers):
            fwd_gru = self.forward_grus[i]
            bwd_gru = self.backward_grus[i]

            # Forward direction
            out_fwd = fwd_gru(x)

            # Backward direction
            x_rev = x[:, ::-1, :]
            out_bwd_rev = bwd_gru(x_rev)
            out_bwd = out_bwd_rev[:, ::-1, :]

            # Concatenate
            x = mx.concatenate([out_fwd, out_bwd], axis=-1)

        return x
