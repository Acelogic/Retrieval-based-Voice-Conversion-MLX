
import mlx.core as mx
import mlx.nn as nn

try:
    print(f"Checking for conv_transpose2d in mx ({mx.__version__})...")
    if hasattr(mx, "conv_transpose2d"):
        print("mx.conv_transpose2d EXISTS")
    else:
        print("mx.conv_transpose2d MISSING")
        
    print(f"Checking for ConvTranspose2d in nn...")
    if hasattr(nn, "ConvTranspose2d"):
        print("nn.ConvTranspose2d EXISTS")
    else:
        print("nn.ConvTranspose2d MISSING")

except Exception as e:
    print(f"Error: {e}")
