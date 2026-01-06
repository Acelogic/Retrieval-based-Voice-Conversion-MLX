
import mlx.core as mx
import mlx.nn as nn

def check_conv_transpose():
    # Instantiate
    ct = nn.ConvTranspose1d(in_channels=1, out_channels=2, kernel_size=3)
    print(f"Shapes: in=1, out=2, k=3")
    print(f"Weight shape: {ct.weight.shape}")
    
    # MLX source usually says (Out, K, In) for Conv1d.
    # For ConvTranspose1d?
    # It wraps conv1d_transpose?
    
    # Let's perform a manual operation to verify what dims do what.
    # Set weight to deterministic.
    # We want to see how (Out, In) interactions happen.
    
    # Weight shape should be (Out, K, In) or (In, K, Out)?
    # If shape is (2, 3, 1), it suggests (Out, K, In).
    
    # Let's try to map PyTorch (1, 2, 3) [(In, Out, K)] to this.
    
    pass

if __name__ == "__main__":
    check_conv_transpose()
