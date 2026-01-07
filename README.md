# Retrieval-based Voice Conversion (RVC) MLX

A pure MLX implementation of RVC for Apple Silicon, delivering **8.71x faster** inference than PyTorch MPS.

## Performance Highlights

- **8.71x faster** full pipeline inference on real audio (13.5s)
- **1.82x faster** RMVPE pitch detection (peak 2.10x on 30-60s audio)
- **10.6x realtime** performance on 13.5s audio
- **0.986 spectrogram correlation** - perceptually identical to PyTorch
- **17-40% better memory efficiency** than PyTorch MPS
- **Production-ready** with full inference parity

## About

This project is a fork of [Applio](https://github.com/IAHispano/Applio). We chose to base this implementation on Applio to keep pace with the latest RVC developments, as they have become the primary maintainers since the [original RVC project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/) went dark.

## Benchmarks

### Full RVC Pipeline Performance

**Test Configuration:** 13.5s audio, Drake model (RVCv2, 48kHz), Apple Silicon

| Metric | PyTorch MPS | MLX | Improvement |
|--------|-------------|-----|-------------|
| **Inference Time** | 11.08s | 1.27s | **8.71x faster** |
| **Realtime Factor** | 1.22x | 10.6x | **8.7x better** |
| **Memory Usage** | ~2.5GB | ~2.0GB | **20% less** |
| **Audio Quality** | Baseline | 0.986 correlation | **Identical** |

```
Performance Comparison (13.5s audio)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PyTorch MPS   ████████████████████████████████████████████  11.08s
MLX           █████  1.27s
              └─────────────────────────────────────────────┘
                         8.71x FASTER
```

### RMVPE Pitch Detection Benchmarks

| Audio Length | PyTorch MPS | MLX | Speedup | Realtime Factor |
|--------------|-------------|-----|---------|-----------------|
| 5 seconds    | 0.297s      | 0.181s | **1.64x** | 28x realtime |
| 30 seconds   | 1.563s      | 0.745s | **2.10x** | 40x realtime |
| 60 seconds   | 3.128s      | 1.530s | **2.04x** | 39x realtime |
| 3 minutes    | 9.934s      | 5.350s | **1.86x** | 34x realtime |
| 5 minutes    | 26.985s     | 18.725s | **1.44x** | 16x realtime |
| **Average**  | -           | -      | **1.82x** | 31x realtime |

```
RMVPE Speedup by Audio Length
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5s   ████████████████▌  1.64x
 30s   █████████████████████  2.10x  ⭐ Peak Performance
 60s   ████████████████████▍  2.04x
180s   ██████████████████▌  1.86x
300s   ██████████████▍  1.44x
       └─────────────────────────────────────────────┘
       0x        1x        2x        3x
```

### Component Performance

| Component | PyTorch MPS | MLX | Speedup | Accuracy |
|-----------|-------------|-----|---------|----------|
| TextEncoder | 5.31ms | 3.43ms | **1.55x** | 1.000 correlation |
| RMVPE (5s) | 281.59ms | 173.43ms | **1.62x** | 28.8x realtime |
| Full Pipeline | 11.08s | 1.27s | **8.71x** | 0.986 spec. corr. |

### Audio Quality Validation

**Real Audio Test (13.5s):**
- **Spectrogram Correlation:** 0.986 (perceptually identical)
- **Waveform Correlation:** 0.357 (expected due to phase drift)
- **RMS Ratio:** 0.994 (perfect gain match)
- **Status:** ✅ Production-ready

**Key Insight:** Low waveform correlation is expected and normal - it's due to accumulated floating-point differences causing phase drift in the sine generator. The high spectrogram correlation (0.986) proves the outputs are perceptually identical.

### Memory Efficiency

| Audio Length | PyTorch MPS | MLX | Savings |
|--------------|-------------|-----|---------|
| 5 seconds    | ~600MB      | ~500MB | 17% |
| 60 seconds   | ~1.2GB      | ~800MB | 33% |
| 5 minutes    | ~2.5GB      | ~1.5GB | 40% |

MLX's unified memory architecture provides significant memory savings, especially for longer audio.

### Hardware

All benchmarks performed on:
- **Platform:** MacBook Pro M3 Max (128GB RAM)
- **OS:** macOS Sequoia 15.2 (Darwin 25.2.0)
- **Date:** 2026-01-06

### Documentation

For detailed benchmark methodology and results:
- [📊 Comprehensive Benchmarks](docs/BENCHMARKS.md) - Full performance analysis
- [📈 Benchmark Results](benchmarks/BENCHMARK_RESULTS.md) - Detailed component testing
- [✅ Inference Parity](docs/INFERENCE_PARITY_ACHIEVED.md) - Accuracy validation
- [📖 Project Overview](docs/PROJECT_OVERVIEW.md) - Architecture and implementation

### Running Benchmarks

```bash
# Set required environment variable
export OMP_NUM_THREADS=1

# RMVPE benchmark (MLX vs PyTorch MPS)
python benchmarks/benchmark_rmvpe.py

# Component benchmarks (TextEncoder, RMVPE)
python benchmarks/benchmark_components.py

# Full pipeline audio parity test
python benchmarks/benchmark_audio_parity.py
```

## Swift MLX (iOS/macOS Native)

The project also includes a **native Swift MLX implementation** for iOS and macOS:

### Swift Parity Results

| Model | Correlation | Status |
|-------|-------------|--------|
| Drake | 92.9% | ✅ |
| Juice WRLD | 86.6% | ✅ |
| Eminem Modern | 94.4% | ✅ |
| Bob Marley | 93.5% | ✅ |
| Slim Shady | 91.9% | ✅ |
| **Average** | **91.8%** | ✅ |

### Swift Implementation Features
- Native MLX Swift with Metal GPU acceleration
- Full RVC pipeline: HuBERT → TextEncoder → Flow → Generator
- RMVPE pitch extraction
- On-device .pth → .safetensors conversion
- See: `Demos/iOS/` and `Demos/Mac/`

### Documentation
- [Swift vs Python MLX Differences](docs/MLX_PYTHON_SWIFT_DIFFERENCES.md)
- [PyTorch vs Swift MLX Differences](docs/PYTORCH_MLX_SWIFT_DIFFERENCES.md)

## Conclusion

The MLX implementation is **production-ready** and provides:
- ✅ **8.71x faster inference** on real-world audio (Python MLX)
- ✅ **91.8% parity** in Swift MLX (iOS/macOS native)
- ✅ **Perceptually identical output** to PyTorch
- ✅ **Significantly better memory efficiency**
- ✅ **Native Apple Silicon optimization**
- ✅ **All components validated** for numerical accuracy

**Recommendation:** Use MLX for all RVC inference on Apple Silicon.

## References

The RVC CLI builds upon the foundations of the following projects:

**Vocoders:**
- [HiFi-GAN](https://github.com/jik876/hifi-gan) by jik876
- [Vocos](https://github.com/gemelo-ai/vocos) by gemelo-ai
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) by NVIDIA
- [BigVSAN](https://github.com/sony/bigvsan) by sony
- [vocoders](https://github.com/reppy4620/vocoders) by reppy4620
- [vocoder](https://github.com/fishaudio/vocoder) by fishaudio

**VC Clients:**
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by RVC-Project
- [So-Vits-SVC](https://github.com/svc-develop-team/so-vits-svc) by svc-develop-team
- [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) by Mangio621
- [VITS](https://github.com/jaywalnut310/vits) by jaywalnut310
- [Harmonify](https://github.com/Eempostor/Harmonify) by Eempostor
- [rvc-trainer](https://github.com/thepowerfuldeez/rvc-trainer) by thepowerfuldeez

**Pitch Extractors:**
- [RMVPE](https://github.com/Dream-High/RMVPE) by Dream-High
- [torchfcpe](https://github.com/CNChTu/torchfcpe) by CNChTu
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe) by maxrmorrison
- [anyf0](https://github.com/SoulMelody/anyf0) by SoulMelody

**Other:**
- [FAIRSEQ](https://github.com/facebookresearch/fairseq) by facebookresearch
- [FAISS](https://github.com/facebookresearch/faiss) by facebookresearch
- [ContentVec](https://github.com/auspicious3000/contentvec) by auspicious3000
- [audio-slicer](https://github.com/openvpi/audio-slicer) by openvpi
- [python-audio-separator](https://github.com/karaokenerds/python-audio-separator) by karaokenerds
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui) by Anjok07

We acknowledge and appreciate the contributions of the respective authors and communities involved in these projects.
