#!/usr/bin/env python3
"""Validate iOS weights match the reference Python MLX weights.

This script compares the bundled iOS Assets weights against the reference
weights in the weights/ directory to catch any conversion regressions.

The key check is the ResBlock Conv1d weights, which were affected by a bug
in fuse_weight_norm() that caused ~2x scaling.

Usage:
    python3 tools/validate_ios_weights.py
"""

import sys
from pathlib import Path

try:
    import safetensors.numpy
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)

WEIGHTS_DIR = Path("weights")
ASSETS_DIR = Path("Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets")

# Models that should be validated
MODELS = [
    "Drake.safetensors",
    "Slim_Shady_New.safetensors",
    "Bob Marley (RVC v2) (500 Epochs) RMVPE.safetensors",
    "Eminem Modern.safetensors",
    "Juice WRLD (RVC v2) 310 Epochs.safetensors",
    "Coder999V2.safetensors",
]

# Key to check for weight scaling issues (ResBlock conv weights)
CHECK_KEY = "dec.resblock_0.c1_0.weight"


def validate_model(name: str) -> tuple[str, bool]:
    """Validate a single model's weights match the reference.

    Returns:
        Tuple of (status message, passed boolean)
    """
    weights_path = WEIGHTS_DIR / name
    assets_path = ASSETS_DIR / name

    if not weights_path.exists():
        return "SKIP (no reference in weights/)", True

    if not assets_path.exists():
        return "SKIP (not in Assets/)", True

    try:
        with safetensors.numpy.safe_open(str(weights_path), framework="numpy") as f:
            ref_tensor = f.get_tensor(CHECK_KEY)
            ref_val = ref_tensor[0, 0, 0]
            ref_range = (ref_tensor.min(), ref_tensor.max())

        with safetensors.numpy.safe_open(str(assets_path), framework="numpy") as f:
            ios_tensor = f.get_tensor(CHECK_KEY)
            ios_val = ios_tensor[0, 0, 0]
            ios_range = (ios_tensor.min(), ios_tensor.max())

        # Check ratio
        ratio = abs(ios_val / ref_val) if ref_val != 0 else 0

        if abs(ratio - 1.0) < 0.01:
            return f"MATCH (ratio={ratio:.4f})", True
        else:
            return (
                f"MISMATCH (ratio={ratio:.2f}x)\n"
                f"    Reference: val[0,0,0]={ref_val:.6f}, range=[{ref_range[0]:.4f}, {ref_range[1]:.4f}]\n"
                f"    iOS:       val[0,0,0]={ios_val:.6f}, range=[{ios_range[0]:.4f}, {ios_range[1]:.4f}]",
                False,
            )

    except Exception as e:
        return f"ERROR: {e}", False


def main():
    print("=" * 70)
    print("iOS Weights Validation")
    print("=" * 70)
    print(f"Reference dir: {WEIGHTS_DIR}")
    print(f"Assets dir:    {ASSETS_DIR}")
    print(f"Check key:     {CHECK_KEY}")
    print()

    all_pass = True
    results = []

    for model in MODELS:
        result, passed = validate_model(model)
        results.append((model, result, passed))
        if not passed:
            all_pass = False

    # Print results
    for model, result, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {model}")
        for line in result.split("\n"):
            print(f"    {line}")

    print()
    if all_pass:
        print("=" * 70)
        print("✓ All weights validated successfully!")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("✗ Some weights have mismatches - check above for details")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
