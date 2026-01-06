import mlx.core as mx
import numpy as np
from rvc.lib.mlx.hubert import HubertModel, HubertConfig
import os


def inspect():
    conf = HubertConfig()
    model = HubertModel(conf)

    print("Model Parameters:")
    params = dict(model.parameters())
    model_keys = sorted(params.keys())
    for k in model_keys[:20]:
        print(f"  {k}")
    print(f"Total model parameters: {len(model_keys)}")

    weights_path = "rvc/models/embedders/contentvec/hubert_mlx.npz"
    if os.path.exists(weights_path):
        weights = mx.load(weights_path)
        print("\nNPZ Weights:")
        weight_keys = sorted(weights.keys())
        for k in weight_keys[:20]:
            print(f"  {k}")
        print(f"Total weight parameters: {len(weight_keys)}")

        # Check for intersection
        m_set = set(model_keys)
        w_set = set(weight_keys)
        common = m_set.intersection(w_set)
        only_m = m_set - w_set
        only_w = w_set - m_set

        print(f"\nCommon: {len(common)}")
        print(f"Only in Model: {len(only_m)} (subset: {sorted(list(only_m))[:5]})")
        print(f"Only in NPZ: {len(only_w)} (subset: {sorted(list(only_w))[:5]})")


if __name__ == "__main__":
    inspect()
