import Foundation
import MLX
import MLXNN
import MLXRandom

// Constants
let LRELU_SLOPE: Float = 0.1

// MARK: - Modules

class Conv1d: Module {
    let conv: MLXNN.Conv1d
    
    init(_ inChannels: Int, _ outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, dilation: Int = 1, bias: Bool = true) {
        self.conv = MLXNN.Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize, stride: stride, padding: padding, dilation: dilation, bias: bias)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

class ResBlock: Module {
    // CRITICAL: Named properties to match Python weight keys (c1_0, c1_1, c1_2, c2_0, c2_1, c2_2)
    let c1_0: Conv1d
    let c1_1: Conv1d
    let c1_2: Conv1d
    let c2_0: Conv1d
    let c2_1: Conv1d
    let c2_2: Conv1d

    init(channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5]) {
        // Create convs with correct dilation for each layer
        // convs1 use dilation, convs2 use dilation=1
        self.c1_0 = Conv1d(channels, channels, kernelSize: kernelSize, stride: 1,
                           padding: (kernelSize - 1) * dilation[0] / 2, dilation: dilation[0], bias: true)
        self.c1_1 = Conv1d(channels, channels, kernelSize: kernelSize, stride: 1,
                           padding: (kernelSize - 1) * dilation[1] / 2, dilation: dilation[1], bias: true)
        self.c1_2 = Conv1d(channels, channels, kernelSize: kernelSize, stride: 1,
                           padding: (kernelSize - 1) * dilation[2] / 2, dilation: dilation[2], bias: true)

        self.c2_0 = Conv1d(channels, channels, kernelSize: kernelSize, stride: 1,
                           padding: (kernelSize - 1) / 2, bias: true)
        self.c2_1 = Conv1d(channels, channels, kernelSize: kernelSize, stride: 1,
                           padding: (kernelSize - 1) / 2, bias: true)
        self.c2_2 = Conv1d(channels, channels, kernelSize: kernelSize, stride: 1,
                           padding: (kernelSize - 1) / 2, bias: true)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, debug: Bool = false) -> MLXArray {
        let convs1 = [c1_0, c1_1, c1_2]
        let convs2 = [c2_0, c2_1, c2_2]

        // DEBUG: Check conv weights in detail
        if debug {
            MLX.eval(x)
            print("DEBUG ResBlock: INPUT shape=\(x.shape), range=[\(x.min().item(Float.self))...\(x.max().item(Float.self))]")

            // Print weight statistics for ALL convolutions with full detail
            for (i, c1) in convs1.enumerated() {
                let w = c1.conv.weight
                MLX.eval(w)
                print("DEBUG ResBlock: c1_\(i).conv.weight shape=\(w.shape), range=[\(w.min().item(Float.self))...\(w.max().item(Float.self))], mean=\(w.mean().item(Float.self))")

                // Print a few actual weight values to compare with Python
                let wSlice = w[0, 0, 0..<5]  // First 5 values of first filter
                MLX.eval(wSlice)
                print("DEBUG ResBlock: c1_\(i) weight[0,0,:5] = \(wSlice.asArray(Float.self))")

                if let b = c1.conv.bias {
                    MLX.eval(b)
                    print("DEBUG ResBlock: c1_\(i).conv.bias range=[\(b.min().item(Float.self))...\(b.max().item(Float.self))]")
                }
            }
            for (i, c2) in convs2.enumerated() {
                let w = c2.conv.weight
                MLX.eval(w)
                print("DEBUG ResBlock: c2_\(i).conv.weight shape=\(w.shape), range=[\(w.min().item(Float.self))...\(w.max().item(Float.self))], mean=\(w.mean().item(Float.self))")
            }
        }

        var out = x
        for (idx, (c1, c2)) in zip(convs1, convs2).enumerated() {
            var xt = out
            MLX.eval(xt)

            xt = leakyRelu(xt, negativeSlope: LRELU_SLOPE)
            MLX.eval(xt)
            if debug {
                print("DEBUG ResBlock[\(idx)]: after lrelu1 range=[\(xt.min().item(Float.self))...\(xt.max().item(Float.self))]")
                // Print a few input values to c1
                let xtSlice = xt[0, 0, 0..<5]
                MLX.eval(xtSlice)
                print("DEBUG ResBlock[\(idx)]: c1 input[0,0,:5] = \(xtSlice.asArray(Float.self))")
            }

            xt = c1(xt)
            MLX.eval(xt)
            if debug {
                print("DEBUG ResBlock[\(idx)]: after c1 range=[\(xt.min().item(Float.self))...\(xt.max().item(Float.self))]")
                // Print a few output values from c1
                let xtSlice = xt[0, 0, 0..<5]
                MLX.eval(xtSlice)
                print("DEBUG ResBlock[\(idx)]: c1 output[0,0,:5] = \(xtSlice.asArray(Float.self))")
            }

            xt = leakyRelu(xt, negativeSlope: LRELU_SLOPE)
            MLX.eval(xt)
            if debug { print("DEBUG ResBlock[\(idx)]: after lrelu2 range=[\(xt.min().item(Float.self))...\(xt.max().item(Float.self))]") }

            xt = c2(xt)
            MLX.eval(xt)
            if debug { print("DEBUG ResBlock[\(idx)]: after c2 range=[\(xt.min().item(Float.self))...\(xt.max().item(Float.self))]") }

            out = out + xt
            MLX.eval(out)
            if debug { print("DEBUG ResBlock[\(idx)]: after residual range=[\(out.min().item(Float.self))...\(out.max().item(Float.self))]") }
        }
        return out
    }
}

// MARK: - NSF Modules

class SineGenerator: Module {
    let sample_rate: Int
    let harmonic_num: Int
    let sine_amp: Float
    let noise_std: Float
    let voiced_threshold: Float
    let waveform_dim: Int
    
    init(sample_rate: Int, harmonic_num: Int = 0, sine_amp: Float = 0.1, add_noise_std: Float = 0.003, voiced_threshold: Float = 0) {
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = harmonic_num + 1
        super.init()
    }
    
    func _generate_sine_wave(_ f0: MLXArray, upsampling_factor: Int) -> MLXArray {
        // f0: [B, L, 1]
        let B = f0.shape[0]
        let L = f0.shape[1]
        
        // Upsampling grid: [1, 1, U]
        let upsampling_grid = MLXArray(1...upsampling_factor).asType(f0.dtype)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
            
        // phase_increments: (f0 / sr) * grid -> [B, L, U]
        var phase_increments = (f0 / Float(sample_rate)) * upsampling_grid
        
        // Slicing: [B, L-1, 1] (taking last upsample point of previous frames)
        // Slice logic: MLXArray doesn't support generic slice objects easily in Swift yet?
        // We use Range.
        // [0..., 0..<(L-1), -1...]
        // Note: -1... in MLX Swift subscript might return suffix?
        // We need explicit index -1.
        // If Swift MLX subscript( ranges ) returns slice, we good.
        // If we use integer index, it drops dim. We want [B, L-1, 1].
        // If we do [..., -1], result is [B, L-1]. We need expand.
        
        // Safe approach:
        let prev_last_phase_raw = phase_increments[0..., 0..<(L-1), -1] // [B, L-1]
        let prev_last_phase = prev_last_phase_raw.expandedDimensions(axis: 2) // [B, L-1, 1]
        
        let p = prev_last_phase + 0.5
        // Manually compute remainder 1.0
        // (p.asType(Int).asType(Float)?) No floor.
        let phase_remainder = (p - floor(p)) - 0.5
        
        // cumsum over axis 1
        let cum = cumsum(phase_remainder, axis: 1)
        var cumulative_phase = cum - floor(cum) // % 1.0
        
        // Pad: [[0,0], [1,0], [0,0]] -> add 1 zero at start of axis 1
        cumulative_phase = padded(cumulative_phase, widths: [[0,0], [1,0], [0,0]])
        
        // Broadcast addition
        phase_increments = phase_increments + cumulative_phase
        
        // Reshape to [B, L*U, 1]
        phase_increments = phase_increments.reshaped([B, -1, 1])
        
        // Harmonics
        if waveform_dim > 1 {
            let harmonic_scale = MLXArray(1...waveform_dim).asType(f0.dtype)
                .reshaped([1, 1, -1]) // [1, 1, H]
            phase_increments = phase_increments * harmonic_scale
        }
        
        return sin(2 * Float.pi * phase_increments)
    }
    
    func callAsFunction(_ f0: MLXArray, upsampling_factor: Int) -> (MLXArray, MLXArray, MLXArray) {
        // f0: [B, L, 1]
        let f0 = (f0.ndim == 2) ? f0.expandedDimensions(axis: 2) : f0
        let B = f0.shape[0]
        let L = f0.shape[1]
        
        let sine_waves = _generate_sine_wave(f0, upsampling_factor: upsampling_factor) * sine_amp
        
        // Voiced Mask
        let uv = (f0 .> voiced_threshold).asType(Float32.self)
        
        // Upsample mask: [B, L, 1] -> [B, L*U, 1]
        // CRITICAL: Must repeat on axis 1 (time), NOT axis 2 (channel)
        // Python: voiced_mask = mx.repeat(voiced_mask, upsampling_factor, axis=1)
        let uv_final = MLX.repeat(uv, count: upsampling_factor, axis: 1)
        
        // Noise
        // Python: noise_amp = voiced_mask * noise_std + (1 - voiced_mask) * (sine_amp / 3)
        let noise_amp = uv_final * noise_std + (1 - uv_final) * (sine_amp / 3)
        
        let noise = MLXRandom.normal(sine_waves.shape) * noise_amp
        
        let sine_waveforms = sine_waves * uv_final + noise
        
        return (sine_waveforms, uv_final, noise)
    }
}

class SourceModuleHnNSF: Module {
    let l_sin_gen: SineGenerator
    let l_linear: Linear
    let l_tanh: Tanh
    
    init(sample_rate: Int, harmonic_num: Int = 0) {
        self.l_sin_gen = SineGenerator(sample_rate: sample_rate, harmonic_num: harmonic_num)
        self.l_linear = Linear(harmonic_num + 1, 1)
        self.l_tanh = Tanh()
        super.init()
    }
    
    func callAsFunction(_ f0: MLXArray, upsampling_factor: Int) -> MLXArray {
        let (sine_wavs, _, _) = l_sin_gen(f0, upsampling_factor: upsampling_factor)
        return l_tanh(l_linear(sine_wavs))
    }
}

class Generator: Module {
    // Porting HiFiGANNSFGenerator structure
    // Properties named to match Python weights (dec.conv_pre, dec.ups, dec.resblocks, dec.noise_convs, dec.m_source)
    let conv_pre: Conv1d
    let conv_post: Conv1d
    let m_source: SourceModuleHnNSF
    let cond: Conv1d?  // Speaker conditioning

    // CRITICAL: These must be registered as individual attributes to match Python weight keys
    // Python: setattr(self, f"up_{i}", l) creates dec.up_0, dec.up_1, etc.
    let up_0: MLXNN.ConvTransposed1d
    let up_1: MLXNN.ConvTransposed1d
    let up_2: MLXNN.ConvTransposed1d
    let up_3: MLXNN.ConvTransposed1d

    // Resblocks: 3 per upsample layer * 4 layers = 12 total
    let resblock_0: ResBlock
    let resblock_1: ResBlock
    let resblock_2: ResBlock
    let resblock_3: ResBlock
    let resblock_4: ResBlock
    let resblock_5: ResBlock
    let resblock_6: ResBlock
    let resblock_7: ResBlock
    let resblock_8: ResBlock
    let resblock_9: ResBlock
    let resblock_10: ResBlock
    let resblock_11: ResBlock

    // Noise convs: 1 per upsample layer = 4 total
    let noise_conv_0: Conv1d
    let noise_conv_1: Conv1d
    let noise_conv_2: Conv1d
    let noise_conv_3: Conv1d

    let totalUpsample: Int
    let outputPaddings: [Int]

    init(inputChannels: Int = 768, ginChannels: Int = 0, upsampleRates: [Int] = [10, 10, 2, 2], upsampleKernelSizes: [Int] = [16, 16, 4, 4], sampleRate: Int = 40000) {
        self.totalUpsample = upsampleRates.reduce(1, *)
        self.outputPaddings = upsampleRates.map { $0 % 2 }

        let upsampleInitialChannel = 512
        self.conv_pre = Conv1d(inputChannels, upsampleInitialChannel, kernelSize: 7, stride: 1, padding: 3)
        self.cond = ginChannels > 0 ? Conv1d(ginChannels, upsampleInitialChannel, kernelSize: 1, stride: 1, padding: 0) : nil
        self.m_source = SourceModuleHnNSF(sample_rate: sampleRate, harmonic_num: 0)

        // Calculate channel sizes (512 -> 256 -> 128 -> 64 -> 32)
        let channels = (0..<4).map { upsampleInitialChannel / (1 << ($0 + 1)) }

        // Helper: calculate padding for ConvTransposed1d (matches Python generators.py:180-183)
        func calcPadding(_ stride: Int, _ kernel: Int) -> Int {
            return (stride % 2 == 0) ? (kernel - stride) / 2 : stride / 2 + stride % 2
        }

        // Helper: calculate noise conv parameters (matches Python generators.py:196-200)
        func calcNoiseParams(_ layerIdx: Int) -> (kernel: Int, stride: Int, padding: Int) {
            let stride = (layerIdx + 1 < 4) ? upsampleRates[(layerIdx + 1)...].reduce(1, *) : 1
            let kernel = stride == 1 ? 1 : stride * 2 - stride % 2
            let padding = stride == 1 ? 0 : (kernel - stride) / 2
            return (kernel, stride, padding)
        }

        // Create upsample layers with config-specific parameters
        let pad0 = calcPadding(upsampleRates[0], upsampleKernelSizes[0])
        self.up_0 = MLXNN.ConvTransposed1d(inputChannels: 512, outputChannels: channels[0], kernelSize: upsampleKernelSizes[0], stride: upsampleRates[0], padding: pad0)

        let pad1 = calcPadding(upsampleRates[1], upsampleKernelSizes[1])
        self.up_1 = MLXNN.ConvTransposed1d(inputChannels: channels[0], outputChannels: channels[1], kernelSize: upsampleKernelSizes[1], stride: upsampleRates[1], padding: pad1)

        let pad2 = calcPadding(upsampleRates[2], upsampleKernelSizes[2])
        self.up_2 = MLXNN.ConvTransposed1d(inputChannels: channels[1], outputChannels: channels[2], kernelSize: upsampleKernelSizes[2], stride: upsampleRates[2], padding: pad2)

        let pad3 = calcPadding(upsampleRates[3], upsampleKernelSizes[3])
        self.up_3 = MLXNN.ConvTransposed1d(inputChannels: channels[2], outputChannels: channels[3], kernelSize: upsampleKernelSizes[3], stride: upsampleRates[3], padding: pad3)

        // Create resblocks (3 per layer: kernels 3, 7, 11)
        self.resblock_0 = ResBlock(channels: channels[0], kernelSize: 3, dilation: [1, 3, 5])
        self.resblock_1 = ResBlock(channels: channels[0], kernelSize: 7, dilation: [1, 3, 5])
        self.resblock_2 = ResBlock(channels: channels[0], kernelSize: 11, dilation: [1, 3, 5])
        self.resblock_3 = ResBlock(channels: channels[1], kernelSize: 3, dilation: [1, 3, 5])
        self.resblock_4 = ResBlock(channels: channels[1], kernelSize: 7, dilation: [1, 3, 5])
        self.resblock_5 = ResBlock(channels: channels[1], kernelSize: 11, dilation: [1, 3, 5])
        self.resblock_6 = ResBlock(channels: channels[2], kernelSize: 3, dilation: [1, 3, 5])
        self.resblock_7 = ResBlock(channels: channels[2], kernelSize: 7, dilation: [1, 3, 5])
        self.resblock_8 = ResBlock(channels: channels[2], kernelSize: 11, dilation: [1, 3, 5])
        self.resblock_9 = ResBlock(channels: channels[3], kernelSize: 3, dilation: [1, 3, 5])
        self.resblock_10 = ResBlock(channels: channels[3], kernelSize: 7, dilation: [1, 3, 5])
        self.resblock_11 = ResBlock(channels: channels[3], kernelSize: 11, dilation: [1, 3, 5])

        // Create noise convs
        let n0 = calcNoiseParams(0)
        self.noise_conv_0 = Conv1d(1, channels[0], kernelSize: n0.kernel, stride: n0.stride, padding: n0.padding)
        let n1 = calcNoiseParams(1)
        self.noise_conv_1 = Conv1d(1, channels[1], kernelSize: n1.kernel, stride: n1.stride, padding: n1.padding)
        let n2 = calcNoiseParams(2)
        self.noise_conv_2 = Conv1d(1, channels[2], kernelSize: n2.kernel, stride: n2.stride, padding: n2.padding)
        let n3 = calcNoiseParams(3)
        self.noise_conv_3 = Conv1d(1, channels[3], kernelSize: n3.kernel, stride: n3.stride, padding: n3.padding)

        self.conv_post = Conv1d(channels[3], 1, kernelSize: 7, stride: 1, padding: 3)

        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, f0: MLXArray, g: MLXArray? = nil) -> MLXArray {
        // CRITICAL: MLX uses channels-last format (B, T, C)
        // Input x is in PyTorch format (B, C, T), transpose to MLX format
        // Python line 222-223: x = x.transpose(0, 2, 1)
        var out = x.transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
        print("DEBUG: Generator input transposed: \(out.shape)")

        out = conv_pre(out)
        print("DEBUG: Generator.conv_pre out: \(out.shape), [\(out.min().item(Float.self))...\(out.max().item(Float.self))]")

        // Add speaker conditioning if available
        if let g = g, let condLayer = cond {
            out = out + condLayer(g)
            print("DEBUG: Generator.cond added: [\(out.min().item(Float.self))...\(out.max().item(Float.self))]")
        }

        // NSF Source Signal: [B, L*U, 1] (High Resolution)
        let har_source = m_source(f0, upsampling_factor: totalUpsample)
        print("DEBUG: Generator.har_source: [\(har_source.min().item(Float.self))...\(har_source.max().item(Float.self))]")
        // har_source is now [B, AudioLen, 1]

        // Use arrays for iteration but reference the named properties
        let ups: [MLXNN.ConvTransposed1d] = [up_0, up_1, up_2, up_3]
        let noise_convs: [Conv1d] = [noise_conv_0, noise_conv_1, noise_conv_2, noise_conv_3]
        let resblocks: [ResBlock] = [
            resblock_0, resblock_1, resblock_2,
            resblock_3, resblock_4, resblock_5,
            resblock_6, resblock_7, resblock_8,
            resblock_9, resblock_10, resblock_11
        ]

        var resIdx = 0
        // Iterate through upsampling layers
        for (i, up) in ups.enumerated() {
            out = leakyRelu(out, negativeSlope: LRELU_SLOPE)
            out = up(out)

            // Apply output_padding if needed (matches Python generators.py:240-244)
            if outputPaddings[i] > 0 {
                out = padded(out, widths: [[0,0], [0, outputPaddings[i]], [0,0]])
            }

            // MEMORY FIX: Force evaluation after each upsample to prevent graph accumulation
            MLX.eval(out)
            if i == 0 { print("DEBUG: After up_0: shape=\(out.shape), range=[\(out.min().item(Float.self))...\(out.max().item(Float.self))]") }

            // Add NSF Noise
            // noise_conv reduces har_source resolution to match current 'out'
            let noise_conv = noise_convs[i]
            let n = noise_conv(har_source)
            if i == 0 { print("DEBUG: noise_conv_0 output: shape=\(n.shape), range=[\(n.min().item(Float.self))...\(n.max().item(Float.self))]") }

            // Crop if necessary
            if out.shape[1] != n.shape[1] {
                let minLen = min(out.shape[1], n.shape[1])
                out = out[0..., 0..<minLen, 0...]
                out = out + n[0..., 0..<minLen, 0...]
            } else {
                out = out + n
            }
            if i == 0 { print("DEBUG: After noise add: range=[\(out.min().item(Float.self))...\(out.max().item(Float.self))]") }

            // Multi-ResBlocks
            var xs: MLXArray? = nil
            for j in 0..<3 { // 3 kernels (3, 7, 11)
                let res = resblocks[resIdx]
                // Debug all 3 resblocks in first upsample stage to understand explosion
                let r = res(out, debug: i == 0 && j == 0)  // Full debug on first resblock only
                if i == 0 { print("DEBUG: resblock_\(resIdx) (kernel \([3, 7, 11][j])) output: range=[\(r.min().item(Float.self))...\(r.max().item(Float.self))]") }
                if xs == nil { xs = r }
                else { xs = xs! + r }
                resIdx += 1
            }
            // Average
            out = xs! / 3.0
            
            // MEMORY FIX: Evaluate and clear cache after each stage
            MLX.eval(out)
            MLX.Memory.clearCache()
            print("DEBUG: Generator.ups[\(i)] (after resblocks) out: [\(out.min().item(Float.self))...\(out.max().item(Float.self))]")
        }

        out = leakyRelu(out)
        out = conv_post(out)
        print("DEBUG: Generator.conv_post out: \(out.shape), [\(out.min().item(Float.self))...\(out.max().item(Float.self))]")

        out = tanh(out)
        print("DEBUG: Generator.final output: \(out.shape), [\(out.min().item(Float.self))...\(out.max().item(Float.self))]")

        // Output is already (B, T, 1) since we're operating in channels-last format
        // Python line 269-271: conv_post reduces to 1 channel, returns (B, T, 1)
        // No transpose needed - MLX Swift Conv1d outputs (B, T, C) format
        return out
    }
}
