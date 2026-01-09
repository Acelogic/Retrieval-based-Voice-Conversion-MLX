# MLX Training Architecture for RVC Voice Conversion

## Overview

This document describes the architecture for MLX-based fine-tuning of RVC voice conversion models on Apple Silicon, achieving faster-than-PyTorch training performance.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MLX RVC Training Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   Raw Audio      │    │   Audio Slicer   │    │  Sliced Segments │          │
│  │   (.wav/.mp3)    │───▶│   (VAD + HPF)    │───▶│  (3-10s chunks)  │          │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘          │
│                                                          │                      │
│                                                          ▼                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │  Feature Files   │◀───│ Feature Extractor│◀───│  16kHz Resampled │          │
│  │  (HuBERT + F0)   │    │ (HuBERT + RMVPE) │    │     Audio        │          │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘          │
│          │                                                                      │
│          ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                     Training Pipeline                             │          │
│  ├──────────────────────────────────────────────────────────────────┤          │
│  │                                                                   │          │
│  │  ┌────────────┐    ┌────────────────┐    ┌────────────────┐      │          │
│  │  │  PyTorch   │    │    Weight      │    │   MLX Models   │      │          │
│  │  │ Pretrained │───▶│   Converter    │───▶│   (G + D)      │      │          │
│  │  │  (.pth)    │    │  (On-the-fly)  │    │   (.npz)       │      │          │
│  │  └────────────┘    └────────────────┘    └────────────────┘      │          │
│  │                                                  │               │          │
│  │                                                  ▼               │          │
│  │  ┌────────────┐    ┌────────────────┐    ┌────────────────┐      │          │
│  │  │   MLX      │    │   Training     │◀───│  DataLoader    │      │          │
│  │  │ DataLoader │───▶│     Loop       │    │   (Batched)    │      │          │
│  │  └────────────┘    └────────────────┘    └────────────────┘      │          │
│  │                           │                                      │          │
│  │                           ▼                                      │          │
│  │         ┌─────────────────────────────────────────┐              │          │
│  │         │           Loss Functions                │              │          │
│  │         ├─────────────────────────────────────────┤              │          │
│  │         │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │              │          │
│  │         │  │   KL    │  │   Mel   │  │ Feature │ │              │          │
│  │         │  │Divergence│  │  L1     │  │Matching │ │              │          │
│  │         │  └─────────┘  └─────────┘  └─────────┘ │              │          │
│  │         │  ┌─────────┐  ┌─────────┐              │              │          │
│  │         │  │Generator│  │  Disc   │              │              │          │
│  │         │  │   Adv   │  │   Adv   │              │              │          │
│  │         │  └─────────┘  └─────────┘              │              │          │
│  │         └─────────────────────────────────────────┘              │          │
│  │                           │                                      │          │
│  │                           ▼                                      │          │
│  │  ┌────────────┐    ┌────────────────┐    ┌────────────────┐      │          │
│  │  │   AdamW    │◀───│   Gradients    │◀───│ value_and_grad │      │          │
│  │  │ Optimizer  │    │   (G + D)      │    │    (MLX)       │      │          │
│  │  └────────────┘    └────────────────┘    └────────────────┘      │          │
│  │         │                                                        │          │
│  └─────────│────────────────────────────────────────────────────────┘          │
│            │                                                                    │
│            ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                     Monitoring & Output                           │          │
│  ├──────────────────────────────────────────────────────────────────┤          │
│  │                                                                   │          │
│  │  ┌────────────┐    ┌────────────────┐    ┌────────────────┐      │          │
│  │  │    Aim     │    │  Checkpoints   │    │   Benchmark    │      │          │
│  │  │  Tracker   │    │   (.npz)       │    │    Suite       │      │          │
│  │  └────────────┘    └────────────────┘    └────────────────┘      │          │
│  │       │                    │                     │               │          │
│  │       ▼                    ▼                     ▼               │          │
│  │  ┌────────────┐    ┌────────────────┐    ┌────────────────┐      │          │
│  │  │   Voice    │    │  Final Model   │    │   Performance  │      │          │
│  │  │  Metrics   │    │   (Inference)  │    │    Report      │      │          │
│  │  └────────────┘    └────────────────┘    └────────────────┘      │          │
│  │                                                                   │          │
│  └───────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### Generator (Synthesizer)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Synthesizer (Generator)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Inputs:                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Phone     │  │   Pitch     │  │   PitchF    │  │ Speaker ID │ │
│  │ (HuBERT)    │  │  (Coarse)   │  │   (Fine)    │  │            │ │
│  │ (B,T,768)   │  │  (B,T)      │  │   (B,T)     │  │   (B,)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│         │                │                │                │        │
│         ▼                ▼                │                ▼        │
│  ┌─────────────────────────────┐          │    ┌────────────────┐  │
│  │       TextEncoder (enc_p)   │          │    │   emb_g        │  │
│  │  ┌─────────┐  ┌──────────┐  │          │    │ (Speaker Emb)  │  │
│  │  │emb_phone│  │emb_pitch │  │          │    │ (B,1,256)      │  │
│  │  │Linear768│  │Embed(256)│  │          │    └───────┬────────┘  │
│  │  └────┬────┘  └────┬─────┘  │          │            │           │
│  │       └─────┬──────┘        │          │            │           │
│  │             ▼               │          │            │           │
│  │  ┌──────────────────────┐   │          │            │           │
│  │  │ Encoder (6 layers)   │   │          │            │           │
│  │  │ MultiHead Attention  │   │          │            │           │
│  │  │ + FFN + LayerNorm    │   │          │            │           │
│  │  └──────────┬───────────┘   │          │            │           │
│  │             ▼               │          │            │           │
│  │  ┌──────────────────────┐   │          │            │           │
│  │  │   proj (Conv1d)      │   │          │            │           │
│  │  │  → m_p, logs_p       │   │          │            │           │
│  │  └──────────┬───────────┘   │          │            │           │
│  └─────────────│───────────────┘          │            │           │
│                │                          │            │           │
│                ▼                          │            │           │
│  ┌─────────────────────────────────────────────────────│───────┐   │
│  │              Flow (ResidualCouplingBlock)           │       │   │
│  │  ┌─────────────────────────────────────────────────┐│       │   │
│  │  │  4 x ResidualCouplingLayer                      ││       │   │
│  │  │  (with channel flipping between each)           ││◀──────┘   │
│  │  │                                                 ││           │
│  │  │  Forward:  z_q → z_p  (training)               ││           │
│  │  │  Reverse:  z_p → z    (inference)              ││           │
│  │  └─────────────────────────────────────────────────┘│           │
│  └─────────────────────────────────────────────────────┘           │
│                │                          │                        │
│                ▼                          ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                  Decoder (HiFiGAN-NSF)                       │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  SineGenerator (pitch-conditioned source)              │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  4 x Upsample (ConvTranspose1d)                        │ │  │
│  │  │  Rates: [10, 8, 2, 2] → 320x total (hop_length)        │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  3 x ResBlock (dilated convolutions)                   │ │  │
│  │  │  Kernels: [3, 7, 11], Dilations: [[1,3,5]...]         │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                │                                                   │
│                ▼                                                   │
│  Output: Audio Waveform (B, T_audio)                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Training-Only Component:
┌────────────────────────────────────────────────────────────────────┐
│              PosteriorEncoder (enc_q) - Training Only              │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Input: Ground-truth Mel Spectrogram (B, T_mel, 80)        │   │
│  │                                                             │   │
│  │  pre (Conv1d) → WaveNet (5 layers) → proj (Conv1d)         │   │
│  │                                                             │   │
│  │  Output: z_q, m_q, logs_q (latent distribution)            │   │
│  └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### Discriminator (MultiPeriodDiscriminator v2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  MultiPeriodDiscriminator v2                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: Audio Waveform (B, T, 1)                                    │
│                │                                                     │
│                ├────────────────────────────────────────────┐       │
│                ▼                                            │       │
│  ┌─────────────────────────┐                               │       │
│  │    DiscriminatorS       │  (Scale-based, Conv1d)        │       │
│  │                         │                               │       │
│  │  Conv1d(1→16, k=15)     │                               │       │
│  │  Conv1d(16→64, k=41)    │  Downsampling with            │       │
│  │  Conv1d(64→256, k=41)   │  grouped convolutions         │       │
│  │  Conv1d(256→1024, k=41) │                               │       │
│  │  Conv1d(1024→1024, k=41)│                               │       │
│  │  Conv1d(1024→1024, k=5) │                               │       │
│  │  Conv1d(1024→1, k=3)    │  → Real/Fake score            │       │
│  └─────────────────────────┘                               │       │
│                                                             │       │
│                ┌────────────────────────────────────────────┘       │
│                │                                                     │
│                ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              8 x DiscriminatorP (Period-based, Conv2d)       │   │
│  │                                                               │   │
│  │  Periods: [2, 3, 5, 7, 11, 17, 23, 37]                       │   │
│  │                                                               │   │
│  │  For each period p:                                          │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  Reshape: (B, T, 1) → (B, T//p, p, 1)                   │ │   │
│  │  │                                                          │ │   │
│  │  │  Conv2d(1→32, k=(5,1))                                   │ │   │
│  │  │  Conv2d(32→128, k=(5,1), s=(3,1))                       │ │   │
│  │  │  Conv2d(128→512, k=(5,1), s=(3,1))                      │ │   │
│  │  │  Conv2d(512→1024, k=(5,1), s=(3,1))                     │ │   │
│  │  │  Conv2d(1024→1024, k=(5,1))                              │ │   │
│  │  │  Conv2d(1024→1, k=(3,1))  → Real/Fake score             │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  Output:                                                             │
│  - 9 Real/Fake scores (1 scale + 8 period)                          │
│  - 9 Feature map lists (for feature matching loss)                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Step                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  1. FORWARD PASS (Generator)                                  │   │
│  │                                                               │   │
│  │  y_hat, ids, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)│   │
│  │      = net_g.forward(phone, phone_len, pitch, pitchf,        │   │
│  │                      spec, spec_len, speaker_id)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  2. DISCRIMINATOR UPDATE                                      │   │
│  │                                                               │   │
│  │  y_hat_detached = mx.stop_gradient(y_hat)                    │   │
│  │  y_d_real, y_d_fake, _, _ = net_d(wave_slice, y_hat_detached)│   │
│  │                                                               │   │
│  │  loss_d = discriminator_loss(y_d_real, y_d_fake)             │   │
│  │                                                               │   │
│  │  grads_d = value_and_grad(loss_d)                            │   │
│  │  optim_d.update(net_d, grads_d)                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  3. GENERATOR UPDATE                                          │   │
│  │                                                               │   │
│  │  _, y_d_fake, fmap_r, fmap_g = net_d(wave_slice, y_hat)      │   │
│  │                                                               │   │
│  │  loss_mel = L1(mel(wave), mel(y_hat)) * 45                   │   │
│  │  loss_kl  = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)        │   │
│  │  loss_fm  = feature_loss(fmap_r, fmap_g)                     │   │
│  │  loss_gen = generator_loss(y_d_fake)                         │   │
│  │                                                               │   │
│  │  loss_g_all = loss_gen + loss_fm + loss_mel + loss_kl        │   │
│  │                                                               │   │
│  │  grads_g = value_and_grad(loss_g_all)                        │   │
│  │  optim_g.update(net_g, grads_g)                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  4. EVALUATE & LOG                                            │   │
│  │                                                               │   │
│  │  mx.eval(net_g.parameters(), net_d.parameters(),             │   │
│  │          optim_g.state, optim_d.state)                       │   │
│  │                                                               │   │
│  │  aim_tracker.log_losses(losses, step)                        │   │
│  │  aim_tracker.log_voice_metrics(metrics, step)                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
rvc_mlx/
├── preprocess/                      # Pure MLX preprocessing
│   ├── __init__.py
│   ├── audio_slicer.py             # VAD, high-pass filter, normalization
│   ├── feature_extractor.py        # HuBERT + F0 (RMVPE)
│   └── dataset_builder.py          # Filelist generation
│
├── train/                           # Training module
│   ├── __init__.py
│   ├── trainer.py                  # Main training loop (~400 lines)
│   ├── losses.py                   # Loss functions (~150 lines)
│   ├── discriminators.py           # MPD v2 (~350 lines)
│   ├── data_loader.py              # Dataset + batching (~200 lines)
│   ├── mel_processing.py           # MLX STFT + mel (~150 lines)
│   ├── schedulers.py               # LR scheduling (~50 lines)
│   └── utils.py                    # Checkpointing (~100 lines)
│
├── monitoring/                      # Aim integration
│   ├── __init__.py
│   ├── aim_tracker.py              # Experiment tracking (~150 lines)
│   └── voice_metrics.py            # F0 accuracy, MCD (~200 lines)
│
└── lib/mlx/                         # Existing (modified)
    ├── synthesizers.py             # + forward() method
    ├── commons.py                  # + slice_segments
    └── residuals.py                # + forward mode for flow

tools/
├── convert_rvc_model.py            # + discriminator conversion
└── train_mlx.py                    # Training CLI entry point

benchmarks/training/
├── benchmark_training.py           # Performance benchmarks
└── compare_pytorch_mlx.py          # PyTorch vs MLX comparison
```

---

## Loss Functions

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **KL Divergence** | `logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p)² * exp(-2*logs_p))` | 1.0 | Regularize latent space |
| **Mel L1** | `mean(\|mel_real - mel_gen\|)` | 45.0 | Spectral similarity |
| **Feature Matching** | `2 * sum(mean(\|fmap_r - fmap_g\|))` | 1.0 | Stabilize GAN training |
| **Generator Adv** | `sum((1 - D(fake))²)` | 1.0 | Fool discriminator |
| **Discriminator** | `sum((1 - D(real))² + D(fake)²)` | 1.0 | Distinguish real/fake |

---

## Dimension Reference

| Component | PyTorch Format | MLX Format | Notes |
|-----------|---------------|------------|-------|
| Audio data | `(B, 1, T)` | `(B, T, 1)` | Transpose at boundary |
| Mel spectrogram | `(B, 80, T_mel)` | `(B, T_mel, 80)` | Transpose at boundary |
| Conv1d weight | `(Out, In, K)` | `(Out, K, In)` | Rearrange in converter |
| Conv2d weight | `(Out, In, H, W)` | `(Out, H, W, In)` | Rearrange in converter |
| Flow latent | `(B, C, T)` | `(B, T, C)` | Internal transpose |

---

## Performance Targets

| Metric | PyTorch (M3 Max) | MLX Target | Improvement |
|--------|-----------------|------------|-------------|
| Samples/sec | ~100 | ~400+ | 4x |
| Epoch time (1hr data) | ~30 min | ~10 min | 3x |
| Peak memory | ~16 GB | ~12 GB | 25% less |
| GPU utilization | ~70% | ~90% | Better efficiency |

---

## Voice-Specific Metrics (Aim)

| Metric | Description | Good Value |
|--------|-------------|------------|
| `voice/f0_accuracy` | % of frames within 50 cents | > 90% |
| `voice/mcd` | Mel Cepstral Distortion | < 5.0 dB |
| `voice/spec_correlation` | Spectrogram Pearson r | > 0.95 |
| `voice/speaker_similarity` | Embedding cosine sim | > 0.85 |

---

## CLI Reference

### Prerequisites Download

Download pretrained models and dependencies:

```bash
# Download base RVC v2 pretrains + RMVPE/ContentVec models
rvc-mlx-cli prerequisites --pretraineds_hifigan true --models true

# Include TITAN community pretrain (recommended by AI Hub)
rvc-mlx-cli prerequisites --pretraineds_hifigan true --models true --titan true
```

**Pretrained Models:**

| Pretrain | Description | Best For | Path |
|----------|-------------|----------|------|
| Base RVC v2 | Original RVC pretrains | General use | `rvc/models/pretraineds/hifi-gan/` |
| TITAN | 11.15 hours, fine-tuned RVC V2 | General purpose | `rvc/models/pretraineds/titan/` |
| RefineGAN | Highest fidelity vocoder | High quality output | `rvc/models/pretraineds/refinegan/` |

### Preprocessing

Slice and normalize audio files:

```bash
rvc-mlx-cli preprocess \
    --model_name "my_voice" \
    --input_folder /path/to/audio \
    --sample_rate 40000
```

### Feature Extraction

Extract HuBERT embeddings and F0 pitch:

```bash
rvc-mlx-cli extract \
    --model_name "my_voice" \
    --f0_method rmvpe
```

### Training

Full training with all options:

```bash
rvc-mlx-cli train \
    --model_name "my_voice" \
    --sample_rate 40000 \
    --pretrain base \
    --batch_size 8 \
    --total_epoch 200 \
    --save_every_epoch 10 \
    --overtraining_detector true \
    --overtraining_patience 10
```

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | (required) | Name for your voice model |
| `--sample_rate` | 40000 | Sample rate (32000, 40000, 48000) |
| `--pretrain` | base | Pretrain to use: `base`, `titan` |
| `--vocoder` | hifigan | Vocoder type: `hifigan`, `refinegan` |
| `--batch_size` | 8 | Training batch size |
| `--total_epoch` | 200 | Total epochs to train |
| `--save_every_epoch` | 10 | Save checkpoint every N epochs |
| `--auto_batch_size` | false | Auto-detect batch size based on dataset duration |
| `--overtraining_detector` | true | Enable g/total loss monitoring |
| `--overtraining_patience` | 10 | Epochs without improvement before stopping |

#### Smart Batch Size (AI Hub Recommendation)

Based on AI Hub guidelines, batch size should be adjusted based on dataset duration:

- **Dataset ≥ 30 minutes** → Batch size 8 (smoother gradients)
- **Dataset < 30 minutes** → Batch size 4 (prevents overtraining)

```bash
# Auto-detect optimal batch size
rvc-mlx-cli train \
    --model_name "my_voice" \
    --auto_batch_size true
```

#### Overtraining Detection

The overtraining detector monitors `g/total` loss and stops training when:
- Loss plateaus for N epochs (patience)
- Loss rises for 5+ consecutive epochs

```bash
# Enable with custom patience
rvc-mlx-cli train \
    --model_name "my_voice" \
    --overtraining_detector true \
    --overtraining_patience 15
```

#### Using Community Pretrains

```bash
# Train with TITAN pretrain (recommended for general purpose)
rvc-mlx-cli train \
    --model_name "my_voice" \
    --pretrain titan \
    --sample_rate 40000

# Train with base RVC v2 pretrain
rvc-mlx-cli train \
    --model_name "my_voice" \
    --pretrain base \
    --sample_rate 40000
```

### Benchmarking

Run training performance benchmarks:

```bash
# Basic benchmark with random weights
python benchmarks/training/benchmark_training.py \
    --batch-sizes 1,2,4,8 \
    --num-steps 50

# Benchmark with real pretrained weights
python benchmarks/training/benchmark_training.py \
    --use-real-weights \
    --pretrain-g rvc/models/pretraineds/hifi-gan/f0G40k.pth \
    --pretrain-d rvc/models/pretraineds/hifi-gan/f0D40k.pth \
    --sample-rate 40000 \
    --output benchmark_results.json

# Forward-only benchmark (no gradients)
python benchmarks/training/benchmark_training.py \
    --forward-only \
    --batch-sizes 1,2,4,8,16
```

**Benchmark Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-sizes` | 1,2,4,8 | Comma-separated batch sizes to test |
| `--num-steps` | 50 | Steps per benchmark |
| `--warmup-steps` | 5 | Warmup steps before timing |
| `--forward-only` | false | Only benchmark forward pass |
| `--use-real-weights` | false | Load actual pretrained weights |
| `--pretrain-g` | (auto) | Path to generator pretrain (.pth/.npz) |
| `--pretrain-d` | (auto) | Path to discriminator pretrain (.pth/.npz) |
| `--sample-rate` | 40000 | Sample rate for pretrain selection |
| `--output` | benchmark_results.json | Output file for results |

### Complete Training Workflow

```bash
# Step 1: Download prerequisites (including TITAN)
rvc-mlx-cli prerequisites --pretraineds_hifigan true --models true --titan true

# Step 2: Preprocess audio
rvc-mlx-cli preprocess \
    --model_name "my_voice" \
    --input_folder ~/my_audio_samples \
    --sample_rate 40000

# Step 3: Extract features
rvc-mlx-cli extract \
    --model_name "my_voice" \
    --f0_method rmvpe

# Step 4: Train with TITAN + overtraining detection + auto batch size
rvc-mlx-cli train \
    --model_name "my_voice" \
    --pretrain titan \
    --auto_batch_size true \
    --overtraining_detector true \
    --total_epoch 200
```

---

## Implementation Status

### Core Training
- [x] Preprocessing module (audio slicer, feature extractor)
- [x] Discriminator weight conversion
- [x] Synthesizer forward() method
- [x] MultiPeriodDiscriminator v2 (MLX)
- [x] Loss functions (MLX)
- [x] Data loader with batching
- [x] Mel spectrogram processing (MLX FFT)
- [x] Trainer class with value_and_grad
- [x] Aim integration with voice metrics
- [x] Training CLI
- [x] Checkpoint saving/loading
- [x] Benchmark suite

### AI Hub Improvements (January 2026)
- [x] Smart batch size auto-detection (30+ min → batch 8, <30 min → batch 4)
- [x] Enhanced overtraining detector with g/total loss monitoring
- [x] TITAN community pretrain download support
- [x] RefineGAN vocoder CLI option
- [x] Benchmark suite with real pretrained weights support
