# RVC MLX Documentation

**Last Updated:** 2026-01-07

## 📚 Documentation Structure

This directory contains comprehensive documentation for the RVC MLX project, organized into focused documents for easy navigation.

## 🗂️ Document Index

### 📖 Overview & Getting Started

**[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Start here!
- Project status and achievements
- Performance benchmarks summary
- Architecture overview
- Usage examples
- Future roadmap

### 🎯 Technical Deep Dives

**[INFERENCE_PARITY_ACHIEVED.md](INFERENCE_PARITY_ACHIEVED.md)**
- Complete RVC inference parity details
- All critical fixes applied
- Numerical accuracy results
- Verification procedures
- Model configuration details

**[RMVPE_OPTIMIZATION.md](RMVPE_OPTIMIZATION.md)**
- RMVPE F0 optimization journey
- All bug fixes and solutions
- Component verification
- Performance vs accuracy analysis
- Debugging tools used

**[PYTORCH_MLX_DIFFERENCES.md](PYTORCH_MLX_DIFFERENCES.md)**
- PyTorch vs Python MLX conventions
- Weight conversion guide
- Dimension ordering differences
- Common pitfalls and solutions
- Best practices for porting

**[PYTORCH_MLX_SWIFT_DIFFERENCES.md](PYTORCH_MLX_SWIFT_DIFFERENCES.md)**
- PyTorch vs MLX Swift conventions
- Swift-specific weight conversion
- Module definition patterns
- API differences
- Conversion checklist

**[MLX_PYTHON_SWIFT_DIFFERENCES.md](MLX_PYTHON_SWIFT_DIFFERENCES.md)**
- Python MLX vs Swift MLX conventions
- Module parameter registration
- Array slicing syntax
- **Flow reverse pass order** (critical!)
- Parity results (91.8% average)

**[BENCHMARKS.md](BENCHMARKS.md)**
- Detailed performance benchmarks
- MLX vs PyTorch MPS comparison
- Memory usage analysis
- Optimization impact breakdown
- Benchmark reproduction guide

### 📱 Platform-Specific

**[IOS_DEVELOPMENT.md](IOS_DEVELOPMENT.md)**
- iOS/Swift port progress
- Model conversion for mobile
- iOS-specific issues and solutions
- Swift implementation status
- Testing procedures

## 🎯 Quick Navigation

### I want to...

**...understand the project**
→ Start with [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

**...convert a PyTorch model**
→ See "Usage" section in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
→ Check conversion details in [PYTORCH_MLX_DIFFERENCES.md](PYTORCH_MLX_DIFFERENCES.md)

**...debug numerical differences**
→ Read [INFERENCE_PARITY_ACHIEVED.md](INFERENCE_PARITY_ACHIEVED.md)
→ Check debugging tools in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

**...optimize performance**
→ See [BENCHMARKS.md](BENCHMARKS.md)
→ Check "Future Work" in [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

**...port to iOS/Swift**
→ Read [IOS_DEVELOPMENT.md](IOS_DEVELOPMENT.md)
→ Check [PYTORCH_MLX_DIFFERENCES.md](PYTORCH_MLX_DIFFERENCES.md) for conversion guide

**...understand RMVPE pitch detection**
→ Read [RMVPE_OPTIMIZATION.md](RMVPE_OPTIMIZATION.md)
→ Check RMVPE benchmarks in [BENCHMARKS.md](BENCHMARKS.md)

## 📊 Project Status

| Component | Status | Documentation |
|-----------|--------|---------------|
| Python MLX Implementation | ✅ Complete | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
| RVC Inference Parity | ✅ Complete | [INFERENCE_PARITY_ACHIEVED.md](INFERENCE_PARITY_ACHIEVED.md) |
| RMVPE Optimization | ✅ Complete | [RMVPE_OPTIMIZATION.md](RMVPE_OPTIMIZATION.md) |
| Performance Benchmarks | ✅ Complete | [BENCHMARKS.md](BENCHMARKS.md) |
| **Swift MLX Port** | ✅ **91.8% Parity** | [MLX_PYTHON_SWIFT_DIFFERENCES.md](MLX_PYTHON_SWIFT_DIFFERENCES.md) |
| iOS App | ✅ Complete | [IOS_DEVELOPMENT.md](IOS_DEVELOPMENT.md) |

## 🔍 Key Achievements

### RVC Inference Parity (Python MLX) ✅
- **Correlation:** 0.999847 (nearly perfect!)
- **Max Difference:** 0.015762 (audio samples)
- **Status:** Production ready

### Swift MLX Parity ✅ (NEW!)
- **Average Correlation:** 91.8%
- **Best Model:** 94.4% (Eminem Modern)
- **Key Fixes:** Flow reverse pass order, WaveNet architecture, weight key mapping

### RMVPE Optimization ✅
- **Voiced Detection Error:** 0.8%
- **F0 Accuracy:** 18.2% error (acceptable)
- **Speed:** 1.78-2.05x faster than PyTorch MPS

### Performance ✅
- **RMVPE:** 0.18s for 5s audio
- **Full Pipeline:** 2.9s for 5s audio
- **Memory:** ~2GB for typical inference

## 🛠️ Development Tools

### Debugging Scripts (`tools/`)

**RVC Debugging:**
- `compare_rvc_full.py` - Full inference comparison
- `debug_attention.py` - Attention layer analysis
- `debug_encoder.py` - TextEncoder debugging
- `check_layernorm.py` - LayerNorm verification
- `check_weights.py` - Weight conversion check

**RMVPE Debugging:**
- `debug_rmvpe.py` - Layer-by-layer analysis
- `debug_first_layer.py` - First layer debugging
- `compare_bigru_real_data.py` - BiGRU verification

**Conversion Tools:**
- `convert_rvc_model.py` - PyTorch → MLX RVC
- `convert_hubert.py` - PyTorch → MLX HuBERT
- `convert_rmvpe.py` - PyTorch → MLX RMVPE

## 📈 Performance Highlights

### MLX vs PyTorch MPS

| Component | MLX | PyTorch MPS | Speedup |
|-----------|-----|-------------|---------|
| RMVPE (5s) | 0.18s | 0.29s | **1.78x** |
| Full Pipeline (5s) | 2.9s | 2.8s | 0.97x |
| Memory | ~2GB | ~2.5GB | **20% less** |

See [BENCHMARKS.md](BENCHMARKS.md) for detailed analysis.

## 🎓 Learning Path

### New to the Project?

1. **Start:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Get the big picture
2. **Understand:** [PYTORCH_MLX_DIFFERENCES.md](PYTORCH_MLX_DIFFERENCES.md) - Learn MLX conventions
3. **Deep Dive:** [INFERENCE_PARITY_ACHIEVED.md](INFERENCE_PARITY_ACHIEVED.md) - See how parity was achieved
4. **Optimize:** [BENCHMARKS.md](BENCHMARKS.md) - Understand performance characteristics

### Porting from PyTorch?

1. **Read:** [PYTORCH_MLX_DIFFERENCES.md](PYTORCH_MLX_DIFFERENCES.md) - Essential conversion guide
2. **Study:** [INFERENCE_PARITY_ACHIEVED.md](INFERENCE_PARITY_ACHIEVED.md) - Common pitfalls and fixes
3. **Reference:** [RMVPE_OPTIMIZATION.md](RMVPE_OPTIMIZATION.md) - Complex debugging example
4. **Test:** Use debugging tools from [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

### Working on iOS?

1. **Start:** [IOS_DEVELOPMENT.md](IOS_DEVELOPMENT.md) - iOS-specific guide
2. **Refer:** [PYTORCH_MLX_DIFFERENCES.md](PYTORCH_MLX_DIFFERENCES.md) - Conversion patterns
3. **Compare:** [INFERENCE_PARITY_ACHIEVED.md](INFERENCE_PARITY_ACHIEVED.md) - Expected accuracy
4. **Benchmark:** [BENCHMARKS.md](BENCHMARKS.md) - Target performance

## 📝 Contributing to Documentation

When updating docs:

1. **Keep focused:** Each doc should cover one main topic
2. **Link extensively:** Reference related docs for details
3. **Update index:** Add new docs to this README
4. **Date stamp:** Update "Last Updated" in each doc
5. **Examples:** Include code examples and commands

## 🔗 External Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

## 📋 Document History

### 2026-01-06
- ✅ Created new focused documentation structure
- ✅ Split monolithic context.md into:
  - PROJECT_OVERVIEW.md (main entry point)
  - INFERENCE_PARITY_ACHIEVED.md (RVC parity details)
  - RMVPE_OPTIMIZATION.md (from context2.md)
  - PYTORCH_MLX_DIFFERENCES.md (conversion guide)
  - IOS_DEVELOPMENT.md (iOS-specific)
  - BENCHMARKS.md (performance details)
- ✅ Added this README for navigation
- 📦 Backed up original context.md to context.md.backup

### Previous History
- context.md: Original monolithic documentation (2026-01-05)
- context2.md: RMVPE optimization deep dive (2026-01-06)

---

**Questions?** Start with [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

**Need Help?** Check the specific focused document for your topic

**Found an Issue?** Update the relevant document and this README
