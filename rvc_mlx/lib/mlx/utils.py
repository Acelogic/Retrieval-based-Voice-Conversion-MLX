import os
import mlx.core as mx
import mlx.nn as nn
from rvc_mlx.lib.mlx.hubert import HubertModel, HubertConfig


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        # The base HubertModel already has `final_proj` initialized if `classifier_proj_size` is set.
        # We just need to ensure the config passed has the right projection size.


def load_embedding(
    embedder_model: str = None, embedder_model_custom: str = None
) -> nn.Module:
    """
    Load the Hubert/ContentVec model for MLX.
    """
    model_path = embedder_model_custom

    # If custom path not provided or not valid, fallbacks could be added here
    if not model_path or not os.path.exists(model_path):
        # Fallback to default name if provided
        if embedder_model:
            # Assume it's a file in current dir or assets?
            # For now, simplistic logic.
            if os.path.exists(embedder_model):
                model_path = embedder_model

    if not model_path or not os.path.exists(model_path):
        # Try finding standard hubert
        candidates = [
            "hubert_base.pt",
            "models/hubert_base.pt",
            "rvc_mlx/models/hubert_base.pt",
            "hubert_base.safetensors",
            "models/hubert_base.safetensors",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if not model_path:
        # Check standard location for mlx weights?
        # Maybe "models/hubert_mlx.safetensors"
        pass

    print(f"Loading generic HuBERT/ContentVec from {model_path}...")

    # Config for standard ContentVec/HuBERT Base
    config = HubertConfig(
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,  # No dropout for inference
        attention_probs_dropout_prob=0.0,
        classifier_proj_size=256,  # ContentVec: 256. Hubert Base: 768 usually?
        # RVC usually uses ContentVec which projects to 768->256?
        # Actually RVC v1/v2 diff.
        # v1 uses 256. v2 uses 768.
    )

    # If loading from .pt (PyTorch) via some conversion on the fly? No, we want pure MLX runtime.
    # We expect .safetensors or .npz converted weights.

    # If user provides .pt, we should warn/error.
    # If user provides .pt, we should warn/error.
    if model_path and model_path.endswith(".pt"):
        print(
            f"Warning: {model_path} is a PyTorch checkpoint. Please convert it to MLX format using tools/convert_hubert.py first."
        )
        # Attempt to load .safetensors version
        safe_path = model_path.replace(".pt", ".safetensors")
        if os.path.exists(safe_path):
            print(f"Found {safe_path}, executing using MLX weights.")
            model_path = safe_path
        else:
            raise RuntimeError(
                "Cannot load .pt file in pure MLX mode. Please convert weights."
            )

    model = HubertModelWithFinalProj(config)

    try:
        logging_level_backup = None  # mx.load doesn't span logs
        weights = mx.load(model_path)
        model.load_weights(list(weights.items()), strict=False)
    except Exception as e:
        print(f"Failed to load MLX weights from {model_path}: {e}")
        raise

    return model
