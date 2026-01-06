
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def test_transpose_padding():
    # Test case equivalent to one upsample layer
    # u=2, k=4.
    # PyTorch padding: (k - u) // 2 = (4-2)//2 = 1.
    # Input L=10. Output should be L*2 = 20 (if padding works as expected in PyTorch).
    # L_out = (L_in - 1)*stride - 2*padding + kernel_size
    #       = (10 - 1)*2 - 2*1 + 4
    #       = 18 - 2 + 4 = 20. Correct.
    
    L_in = 10
    u = 2
    k = 4
    p = 1
    
    ct = nn.ConvTranspose1d(1, 1, k, stride=u, padding=p)
    # Init weights to 1 to easy track
    ct.weight = mx.ones_like(ct.weight)
    if ct.bias is not None: ct.bias = mx.zeros_like(ct.bias)
    
    x = mx.ones((1, L_in, 1))
    y = ct(x)
    
    print(f"L_in={L_in}, u={u}, k={k}, p={p}")
    print(f"MLX Output shape: {y.shape}")
    
    expected_L = (L_in - 1)*u - 2*p + k
    print(f"Expected L_out (PyTorch formula): {expected_L}")
    
    if y.shape[1] == expected_L:
        print("✅ Shapes match PyTorch formula.")
    else:
        print("❌ Shape MISMATCH!")

    # Test case 2: u=10, k=20 (Like first layer in RVC)
    # p = (20 - 10)//2 = 5
    # L_in = 100
    # Expected: (99)*10 - 10 + 20 = 990 - 10 + 20 = 1000.
    
    u=10; k=20; p=5; L_in=100
    ct2 = nn.ConvTranspose1d(1, 1, k, stride=u, padding=p)
    x2 = mx.ones((1, L_in, 1))
    y2 = ct2(x2)
    ev2 = (L_in-1)*u - 2*p + k
    
    print(f"\nTest 2: L_in={L_in}, u={u}, k={k}, p={p}")
    print(f"MLX Output shape: {y2.shape}")
    print(f"Expected: {ev2}")
    
    if y2.shape[1] == ev2:
         print("✅ Shapes match.")
    else:
         print("❌ Shape MISMATCH!")

if __name__ == "__main__":
    test_transpose_padding()
