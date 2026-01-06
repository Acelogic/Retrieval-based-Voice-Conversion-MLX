import json
import os
import mlx.core as mx

# Minimal pure MLX Config

class Config:
    def __init__(self):
        # MLX automatically uses GPU on Apple Silicon if available
        self.device = "gpu" # Abstract concept, MLX handles it
        self.gpu_name = "Apple Silicon" 
        self.x_pad = 1
        self.x_query = 6
        self.x_center = 38
        self.x_max = 41
        self.is_half = False # MLX handles precision dynamically usually, or float32/float16

    # Minimal method to satisfy legacy calls if any
    def device_config(self):
        return self.x_pad, self.x_query, self.x_center, self.x_max
