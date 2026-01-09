#!/usr/bin/env python3
"""Compare all model weights between weights/ and Assets/."""

import safetensors.numpy
import os

WEIGHTS_DIR = "/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/weights"
ASSETS_DIR = "/Users/mcruz/Developer/Retrieval-based-Voice-Conversion-MLX/Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets"

models = [
    "Drake.safetensors",
    "Slim_Shady_New.safetensors",
    "Bob Marley (RVC v2) (500 Epochs) RMVPE.safetensors",
    "Eminem Modern.safetensors",
    "Juice WRLD (RVC v2) 310 Epochs.safetensors",
]

def check_model(name):
    weights_path = os.path.join(WEIGHTS_DIR, name)
    assets_path = os.path.join(ASSETS_DIR, name)

    if not os.path.exists(weights_path) or not os.path.exists(assets_path):
        print(f"  SKIP: Missing file(s)")
        return

    with safetensors.numpy.safe_open(weights_path, framework="numpy") as f:
        w1 = f.get_tensor("dec.resblock_0.c1_0.weight")
        range1 = (w1.min(), w1.max())
        val1 = w1[0, 0, 0]

    with safetensors.numpy.safe_open(assets_path, framework="numpy") as f:
        w2 = f.get_tensor("dec.resblock_0.c1_0.weight")
        range2 = (w2.min(), w2.max())
        val2 = w2[0, 0, 0]

    ratio = abs(val2 / val1) if val1 != 0 else 0
    match = "✓ MATCH" if abs(ratio - 1.0) < 0.01 else f"✗ WRONG ({ratio:.2f}x)"

    print(f"  weights/: range=[{range1[0]:.4f}, {range1[1]:.4f}], val[0,0,0]={val1:.6f}")
    print(f"  Assets/:  range=[{range2[0]:.4f}, {range2[1]:.4f}], val[0,0,0]={val2:.6f}")
    print(f"  Status: {match}")


print("=" * 70)
print("Comparing all bundled models")
print("=" * 70)

for model in models:
    print(f"\n{model}:")
    check_model(model)
