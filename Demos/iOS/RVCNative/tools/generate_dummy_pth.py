import torch
import os

# Create a simple state dict with known values
state_dict = {
    "emb.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    "layer.weight_g": torch.tensor([[[1.0, 1.0]]], dtype=torch.float32), # Shape [1, 1, 2]
    "layer.weight_v": torch.tensor([[[0.5, 0.5]]], dtype=torch.float32), # Shape [1, 1, 2]
    "bias": torch.tensor([0.1, 0.2], dtype=torch.float32)
}

# The weight norm logic in converter expects weight_v to be normalized and combined with weight_g.
# We will verify if the Swift converter handles this structure.

os.makedirs("tests", exist_ok=True)
torch.save(state_dict, "tests/dummy.pth")
print("Saved tests/dummy.pth")
