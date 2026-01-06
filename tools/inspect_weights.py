import torch


def inspect(model_path):
    print(f"Inspecting {model_path}...")
    try:
        ckpt = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    if isinstance(ckpt, dict):
        if "weight" in ckpt:
            sd = ckpt["weight"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    print(f"Keys found: {list(sd.keys())[:5]}")

    for k, v in sd.items():
        if "dec.ups" in k and "weight_g" in k:
            prefix = k.replace(".weight_g", "")
            if prefix + ".weight_v" in sd:
                w_v = sd[prefix + ".weight_v"]
                print(f"Key: {prefix}")
                print(f"  weight_v: {w_v.shape}")
                print(f"  weight_g: {v.shape}")

                # Input Ch = Dim 0 for weight_v: (In, Out, K)
                # Output Ch = Dim 1

                if v.shape[0] == w_v.shape[0]:
                    print("  Matches Dim 0 (In)")
                elif v.shape[0] == w_v.shape[1]:
                    print("  Matches Dim 1 (Out)")
                else:
                    print("  Matches NEITHER!")


if __name__ == "__main__":
    inspect("rvc/models/pretraineds/hifi-gan/f0G40k.pth")
