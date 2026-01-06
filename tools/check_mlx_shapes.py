
import mlx.core as mx
import mlx.nn as nn

print("Checking MLX Layer Shapes...")

c1 = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=5)
# PyTorch: (10, 3, 5) -> (Out, In, K)
print(f"Conv1d(3, 10, 5) weight shape: {c1.weight.shape}")

ct1 = nn.ConvTranspose1d(in_channels=3, out_channels=10, kernel_size=5)
# PyTorch: (3, 10, 5) -> (In, Out, K) usually
print(f"ConvTranspose1d(3, 10, 5) weight shape: {ct1.weight.shape}")

l = nn.Linear(3, 10)
# PyTorch: (10, 3) -> (Out, In)
print(f"Linear(3, 10) weight shape: {l.weight.shape}")
