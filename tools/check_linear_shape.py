
import mlx.nn as nn
import mlx.core as mx

l = nn.Linear(10, 20) # in=10, out=20
print(f"nn.Linear(10, 20) weight shape: {l.weight.shape}")
