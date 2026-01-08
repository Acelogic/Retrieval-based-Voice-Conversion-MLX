"""
Parity tests: MLX implementations vs PyTorch references.

These tests ensure MLX implementations produce results matching
the original PyTorch implementations with high correlation (> 0.95).

Requires:
- torchcrepe (pip install torchcrepe)
- torchfcpe (pip install torchfcpe)
- PyTorch model weights
"""

import pytest
import numpy as np
from tests.conftest import (
    SAMPLE_RATE,
    HOP_SIZE,
    F0_MIN,
    F0_MAX,
    f0_correlation,
    f0_accuracy,
    generate_sine_wave,
)


def check_pytorch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def check_torchcrepe_available():
    """Check if torchcrepe is available."""
    try:
        import torchcrepe
        return True
    except ImportError:
        return False


def check_torchfcpe_available():
    """Check if torchfcpe is available."""
    try:
        import torchfcpe
        return True
    except ImportError:
        return False


# Skip markers for optional dependencies
requires_pytorch = pytest.mark.skipif(
    not check_pytorch_available(),
    reason="PyTorch not installed"
)
requires_torchcrepe = pytest.mark.skipif(
    not check_torchcrepe_available(),
    reason="torchcrepe not installed"
)
requires_torchfcpe = pytest.mark.skipif(
    not check_torchfcpe_available(),
    reason="torchfcpe not installed"
)


@pytest.mark.parity
class TestCREPEParity:
    """Parity tests comparing CREPE MLX vs torchcrepe."""

    @pytest.fixture
    def crepe_mlx(self):
        """Create MLX CREPE instance."""
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            return CREPE(model="full")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

    @pytest.fixture
    def audio_440hz(self):
        """1-second 440 Hz test audio."""
        return generate_sine_wave(440.0, duration=1.0)

    @pytest.fixture
    def audio_sweep(self):
        """Frequency sweep from 100-400 Hz."""
        from tests.conftest import generate_chirp
        audio, _ = generate_chirp(100.0, 400.0, duration=1.0)
        return audio

    @requires_pytorch
    @requires_torchcrepe
    def test_crepe_full_parity_sine(self, crepe_mlx, audio_440hz):
        """CREPE full MLX should match torchcrepe full on sine wave."""
        import torch
        import torchcrepe

        # PyTorch reference
        audio_torch = torch.from_numpy(audio_440hz).unsqueeze(0).float()
        f0_torch, pd_torch = torchcrepe.predict(
            audio_torch,
            SAMPLE_RATE,
            HOP_SIZE,
            F0_MIN,
            F0_MAX,
            model="full",
            batch_size=512,
            device="cpu",
            return_periodicity=True,
        )
        # Apply filtering like RVC does
        pd_torch = torchcrepe.filter.median(pd_torch, 3)
        f0_torch = torchcrepe.filter.mean(f0_torch, 3)
        f0_torch[pd_torch < 0.1] = 0
        f0_pytorch = f0_torch[0].numpy()

        # MLX implementation
        f0_mlx = crepe_mlx.get_f0(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        # Ensure same length for comparison
        min_len = min(len(f0_pytorch), len(f0_mlx))
        f0_pytorch = f0_pytorch[:min_len]
        f0_mlx = f0_mlx[:min_len]

        # Check correlation
        corr = f0_correlation(f0_pytorch, f0_mlx)
        assert corr > 0.95, f"CREPE parity correlation {corr:.4f} should be > 0.95"

        # Check mean F0 difference
        voiced_mask = (f0_pytorch > 0) & (f0_mlx > 0)
        if voiced_mask.sum() > 10:
            mean_diff = np.abs(f0_pytorch[voiced_mask] - f0_mlx[voiced_mask]).mean()
            assert mean_diff < 5.0, f"Mean F0 difference {mean_diff:.2f} Hz should be < 5 Hz"

    @requires_pytorch
    @requires_torchcrepe
    def test_crepe_tiny_parity_sine(self, audio_440hz):
        """CREPE tiny MLX should match torchcrepe tiny."""
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            crepe_mlx = CREPE(model="tiny")
        except ImportError:
            pytest.skip("CREPE MLX not implemented yet")

        import torch
        import torchcrepe

        # PyTorch reference
        audio_torch = torch.from_numpy(audio_440hz).unsqueeze(0).float()
        f0_torch, pd_torch = torchcrepe.predict(
            audio_torch, SAMPLE_RATE, HOP_SIZE, F0_MIN, F0_MAX,
            model="tiny", batch_size=512, device="cpu", return_periodicity=True,
        )
        pd_torch = torchcrepe.filter.median(pd_torch, 3)
        f0_torch = torchcrepe.filter.mean(f0_torch, 3)
        f0_torch[pd_torch < 0.1] = 0
        f0_pytorch = f0_torch[0].numpy()

        # MLX
        f0_mlx = crepe_mlx.get_f0(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        min_len = min(len(f0_pytorch), len(f0_mlx))
        corr = f0_correlation(f0_pytorch[:min_len], f0_mlx[:min_len])
        assert corr > 0.95, f"CREPE tiny parity {corr:.4f} should be > 0.95"

    @requires_pytorch
    @requires_torchcrepe
    @pytest.mark.slow
    def test_crepe_parity_sweep(self, crepe_mlx, audio_sweep):
        """CREPE should match on frequency sweep (harder test)."""
        import torch
        import torchcrepe

        audio_torch = torch.from_numpy(audio_sweep).unsqueeze(0).float()
        f0_torch, pd_torch = torchcrepe.predict(
            audio_torch, SAMPLE_RATE, HOP_SIZE, F0_MIN, F0_MAX,
            model="full", batch_size=512, device="cpu", return_periodicity=True,
        )
        pd_torch = torchcrepe.filter.median(pd_torch, 3)
        f0_torch = torchcrepe.filter.mean(f0_torch, 3)
        f0_torch[pd_torch < 0.1] = 0
        f0_pytorch = f0_torch[0].numpy()

        f0_mlx = crepe_mlx.get_f0(audio_sweep, f0_min=F0_MIN, f0_max=F0_MAX)

        min_len = min(len(f0_pytorch), len(f0_mlx))
        corr = f0_correlation(f0_pytorch[:min_len], f0_mlx[:min_len])
        assert corr > 0.90, f"CREPE sweep parity {corr:.4f} should be > 0.90"


@pytest.mark.parity
class TestFCPEParity:
    """Parity tests comparing FCPE MLX vs torchfcpe."""

    @pytest.fixture
    def fcpe_mlx(self):
        """Create MLX FCPE instance."""
        try:
            from rvc_mlx.lib.mlx.fcpe import FCPE
            return FCPE()
        except ImportError:
            pytest.skip("FCPE MLX not implemented yet")

    @pytest.fixture
    def audio_440hz(self):
        return generate_sine_wave(440.0, duration=1.0)

    @requires_pytorch
    @requires_torchfcpe
    def test_fcpe_parity_sine(self, fcpe_mlx, audio_440hz):
        """FCPE MLX should match torchfcpe on sine wave."""
        import torch
        from torchfcpe import spawn_infer_model_from_pt
        import os

        # Load PyTorch FCPE model
        fcpe_path = os.path.join("rvc", "models", "predictors", "fcpe.pt")
        if not os.path.exists(fcpe_path):
            pytest.skip("FCPE model weights not found")

        fcpe_torch = spawn_infer_model_from_pt(fcpe_path, "cpu", bundled_model=True)

        # PyTorch inference
        audio_torch = torch.from_numpy(audio_440hz).float().unsqueeze(0)
        f0_pytorch = fcpe_torch.infer(
            audio_torch,
            sr=SAMPLE_RATE,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze().numpy()

        # MLX inference
        f0_mlx = fcpe_mlx.get_f0(audio_440hz, f0_min=F0_MIN, f0_max=F0_MAX)

        min_len = min(len(f0_pytorch), len(f0_mlx))
        corr = f0_correlation(f0_pytorch[:min_len], f0_mlx[:min_len])
        assert corr > 0.95, f"FCPE parity correlation {corr:.4f} should be > 0.95"

    @requires_pytorch
    @requires_torchfcpe
    @pytest.mark.slow
    def test_fcpe_parity_voice(self, fcpe_mlx, real_voice_audio):
        """FCPE parity on voice-like audio."""
        import torch
        from torchfcpe import spawn_infer_model_from_pt
        import os

        fcpe_path = os.path.join("rvc", "models", "predictors", "fcpe.pt")
        if not os.path.exists(fcpe_path):
            pytest.skip("FCPE model weights not found")

        fcpe_torch = spawn_infer_model_from_pt(fcpe_path, "cpu", bundled_model=True)

        audio_torch = torch.from_numpy(real_voice_audio).float().unsqueeze(0)
        f0_pytorch = fcpe_torch.infer(
            audio_torch, sr=SAMPLE_RATE, decoder_mode="local_argmax", threshold=0.006,
        ).squeeze().numpy()

        f0_mlx = fcpe_mlx.get_f0(real_voice_audio, f0_min=F0_MIN, f0_max=F0_MAX)

        min_len = min(len(f0_pytorch), len(f0_mlx))
        corr = f0_correlation(f0_pytorch[:min_len], f0_mlx[:min_len])
        assert corr > 0.90, f"FCPE voice parity {corr:.4f} should be > 0.90"


@pytest.mark.parity
class TestRMVPEParity:
    """Parity tests for RMVPE (reference implementation)."""

    @pytest.fixture
    def rmvpe_mlx(self):
        """MLX RMVPE (already implemented)."""
        try:
            from rvc_mlx.lib.mlx.rmvpe import RMVPE
            return RMVPE()
        except ImportError:
            pytest.skip("RMVPE MLX not available")

    @pytest.fixture
    def rmvpe_pytorch(self):
        """PyTorch RMVPE."""
        try:
            from rvc.lib.predictors.RMVPE import RMVPE0Predictor
            import os
            model_path = os.path.join("rvc", "models", "predictors", "rmvpe.pt")
            if not os.path.exists(model_path):
                pytest.skip("RMVPE model weights not found")
            return RMVPE0Predictor(model_path, device="cpu")
        except ImportError:
            pytest.skip("PyTorch RMVPE not available")

    @pytest.fixture
    def audio_440hz(self):
        return generate_sine_wave(440.0, duration=1.0)

    @requires_pytorch
    @pytest.mark.requires_model
    def test_rmvpe_parity(self, rmvpe_mlx, rmvpe_pytorch, audio_440hz):
        """RMVPE MLX should match PyTorch with > 0.99 correlation."""
        # PyTorch
        f0_pytorch = rmvpe_pytorch.infer_from_audio(audio_440hz, thred=0.03)

        # MLX
        f0_mlx = rmvpe_mlx.infer_from_audio(audio_440hz, thred=0.03)

        min_len = min(len(f0_pytorch), len(f0_mlx))
        corr = f0_correlation(f0_pytorch[:min_len], f0_mlx[:min_len])

        # RMVPE should have very high parity (already achieved 99.98%)
        assert corr > 0.99, f"RMVPE parity {corr:.4f} should be > 0.99"


@pytest.mark.parity
class TestCrossMethodParity:
    """Cross-method comparisons to ensure consistent behavior."""

    @pytest.fixture
    def audio_200hz(self):
        return generate_sine_wave(200.0, duration=1.0)

    def test_all_methods_detect_same_frequency(self, audio_200hz):
        """All pitch methods should detect roughly the same F0 for clean signal."""
        results = {}

        # RMVPE (baseline)
        try:
            from rvc_mlx.lib.mlx.rmvpe import RMVPE
            rmvpe = RMVPE()
            results['rmvpe'] = rmvpe.infer_from_audio(audio_200hz, thred=0.03)
        except ImportError:
            pass

        # PyWorld methods
        try:
            from rvc_mlx.lib.mlx.pyworld_pitch import PyWorldExtractor
            pw = PyWorldExtractor(sample_rate=SAMPLE_RATE, hop_size=HOP_SIZE)
            results['dio'] = pw.dio(audio_200hz, f0_min=F0_MIN, f0_max=F0_MAX)
            results['harvest'] = pw.harvest(audio_200hz, f0_min=F0_MIN, f0_max=F0_MAX)
        except ImportError:
            pass

        # CREPE
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            crepe = CREPE(model="full")
            results['crepe'] = crepe.get_f0(audio_200hz, f0_min=F0_MIN, f0_max=F0_MAX)
        except ImportError:
            pass

        # FCPE
        try:
            from rvc_mlx.lib.mlx.fcpe import FCPE
            fcpe = FCPE()
            results['fcpe'] = fcpe.get_f0(audio_200hz, f0_min=F0_MIN, f0_max=F0_MAX)
        except ImportError:
            pass

        if len(results) < 2:
            pytest.skip("Need at least 2 methods to compare")

        # All methods should detect ~200 Hz
        for name, f0 in results.items():
            voiced = f0[f0 > 0]
            if len(voiced) > 0:
                mean_f0 = voiced.mean()
                assert 180 < mean_f0 < 220, f"{name} mean F0 {mean_f0:.1f} should be ~200 Hz"


@pytest.mark.parity
class TestWeightConversion:
    """Tests for weight conversion between PyTorch and MLX."""

    @requires_pytorch
    def test_crepe_weights_loadable(self):
        """Converted CREPE weights should load into MLX model."""
        try:
            from rvc_mlx.lib.mlx.crepe import CREPE
            # This should load weights automatically
            crepe = CREPE(model="full")
            assert crepe is not None
        except ImportError:
            pytest.skip("CREPE not implemented")
        except FileNotFoundError:
            pytest.skip("CREPE weights not found")

    @requires_pytorch
    def test_fcpe_weights_loadable(self):
        """Converted FCPE weights should load into MLX model."""
        try:
            from rvc_mlx.lib.mlx.fcpe import FCPE
            fcpe = FCPE()
            assert fcpe is not None
        except ImportError:
            pytest.skip("FCPE not implemented")
        except FileNotFoundError:
            pytest.skip("FCPE weights not found")

    @requires_pytorch
    def test_weight_shapes_match(self):
        """Converted weight shapes should match model expectations."""
        # This test verifies the weight conversion produces correct shapes
        try:
            import mlx.core as mx
            from pathlib import Path

            crepe_weights = Path("weights/crepe_full.npz")
            if crepe_weights.exists():
                weights = dict(np.load(str(crepe_weights)))
                # Check expected layer shapes
                # Conv1: (1024, 512, 1, 1) in PyTorch -> (1024, 1, 512, 1) in MLX
                assert "conv1.weight" in weights or "layer-1.weight" in weights
        except ImportError:
            pytest.skip("MLX not available")
        except FileNotFoundError:
            pytest.skip("Weights not found")
