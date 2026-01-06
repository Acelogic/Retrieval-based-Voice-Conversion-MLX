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

class ConvTranspose1d: Module {
    let conv: MLXNN.Conv1d
    let stride: Int
    let padding: Int
    let kernelSize: Int
    
    init(_ inChannels: Int, _ outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, bias: Bool = true) {
        // Transposed conv via upsampling + conv
        self.conv = MLXNN.Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize, stride: 1, padding: 0, bias: bias)
        self.stride = stride
        self.padding = padding
        self.kernelSize = kernelSize
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [N, L, C]
        let shape = x.shape
        let N = shape[0]
        let L = shape[1]
        let C = shape[2]
        
        var xUp: MLXArray
        if stride > 1 {
            // Upsample by inserting zeros: [N, L, C] -> [N, L * stride, C]
            // expand to [N, L, 1, C]
            let xExp = x.reshaped([N, L, 1, C])
            // Create zeros [N, L, stride-1, C]
            let xZeros = MLX.zeros([N, L, stride - 1, C], dtype: x.dtype)
            // Concat -> [N, L, stride, C]
            let xCat = MLX.concatenated([xExp, xZeros], axis: 2)
            // Reshape -> [N, L * stride, C]
            xUp = xCat.reshaped([N, L * stride, C])
        } else {
            xUp = x
        }
        
        // Padding
        let pad = kernelSize - 1 - padding
        if pad > 0 {
             // Use padded functionality
             xUp = padded(xUp, widths: [[0,0], [pad, pad], [0,0]])
        }
        
        return conv(xUp)
    }
}

class ResBlock: Module {
    let convs1: [Conv1d]
    let convs2: [Conv1d]
    
    init(channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5]) {
        var c1: [Conv1d] = []
        var c2: [Conv1d] = []
        
        for d in dilation {
            c1.append(Conv1d(
                channels, channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) * d / 2,
                dilation: d,
                bias: true
            ))
            
            c2.append(Conv1d(
                channels, channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: (kernelSize - 1) / 2
            ))
        }
        self.convs1 = c1
        self.convs2 = c2
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for (c1, c2) in zip(convs1, convs2) {
            var xt = out
            xt = leakyRelu(xt, negativeSlope: LRELU_SLOPE)
            xt = c1(xt)
            xt = leakyRelu(xt, negativeSlope: LRELU_SLOPE)
            xt = c2(xt)
            out = out + xt
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
    let ups: [ConvTranspose1d]
    let resblocks: [ResBlock]
    let noise_convs: [Conv1d]
    let m_source: SourceModuleHnNSF
    let cond: Conv1d?  // Speaker conditioning
    
    init(inputChannels: Int = 768, ginChannels: Int = 0) {
        // Standard RVC V2 40k configuration from configs/40000.json
        let upsampleRates = [10, 10, 2, 2]
        let upsampleKernels = [16, 16, 4, 4]
        let resblockKernels = [3, 7, 11]
        let resblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        let upsampleInitialChannel = 512
        
        self.conv_pre = Conv1d(inputChannels, upsampleInitialChannel, kernelSize: 7, stride: 1, padding: 3)
        
        // Speaker conditioning
        self.cond = ginChannels > 0 ? Conv1d(ginChannels, upsampleInitialChannel, kernelSize: 1, stride: 1, padding: 0) : nil
        
        // Initialize NSF Source Module
        self.m_source = SourceModuleHnNSF(sample_rate: 40000, harmonic_num: 0)
        
        var _ups: [ConvTranspose1d] = []
        var _resblocks: [ResBlock] = []
        var _noise_convs: [Conv1d] = []
        
        var ch = upsampleInitialChannel
        
        // Calculate stride products for noise_convs (matching Python "stride_f0s")
        // stride_f0s[i] = prod(upsample_rates[i+1:])
        var stride_f0s: [Int] = []
        for i in 0..<upsampleRates.count {
            if i + 1 < upsampleRates.count {
                let s = upsampleRates[(i+1)...].reduce(1, *)
                stride_f0s.append(s)
            } else {
                stride_f0s.append(1)
            }
        }
        
        for (i, (u, k)) in zip(upsampleRates, upsampleKernels).enumerated() {
            let outCh = ch / 2
            // Padding logic matches Python implementation
            let p = (u % 2 == 0) ? (k - u) / 2 : (u / 2 + u % 2)
            
            _ups.append(ConvTranspose1d(ch, outCh, kernelSize: k, stride: u, padding: p))
            
            // 3 ResBlocks per upsample layer
            for (rk, rd) in zip(resblockKernels, resblockDilations) {
                _resblocks.append(ResBlock(channels: outCh, kernelSize: rk, dilation: rd))
            }
            
            // Noise Conv (NSF)
            let stride = stride_f0s[i]
            let kernel = (stride == 1) ? 1 : (stride * 2 - stride % 2)
            let padding = (stride == 1) ? 0 : (kernel - stride) / 2
            _noise_convs.append(Conv1d(1, outCh, kernelSize: kernel, stride: stride, padding: padding))
            
            ch = outCh
        }
        
        self.ups = _ups
        self.resblocks = _resblocks
        self.noise_convs = _noise_convs
        self.conv_post = Conv1d(ch, 1, kernelSize: 7, stride: 1, padding: 3)
        
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
        // upp = prod(upsampleRates) = 400
        let upp = 400
        let har_source = m_source(f0, upsampling_factor: upp)
        print("DEBUG: Generator.har_source: [\(har_source.min().item(Float.self))...\(har_source.max().item(Float.self))]")
        // har_source is now [B, AudioLen, 1]
        
        var resIdx = 0
        // Iterate through upsampling layers
        for (i, up) in ups.enumerated() {
            out = leakyRelu(out, negativeSlope: LRELU_SLOPE)
            out = up(out)
            
            // Add NSF Noise
            // noise_conv reduces har_source resolution to match current 'out'
            let noise_conv = noise_convs[i]
            let n = noise_conv(har_source)
            
            // Crop if necessary
            if out.shape[1] != n.shape[1] {
                let minLen = min(out.shape[1], n.shape[1])
                out = out[0..., 0..<minLen, 0...]
                out = out + n[0..., 0..<minLen, 0...]
            } else {
                out = out + n
            }
            
            // Multi-ResBlocks
            var xs: MLXArray? = nil
            for _ in 0..<3 { // 3 kernels
                let res = resblocks[resIdx]
                let r = res(out)
                if xs == nil {  xs = r } 
                else { xs = xs! + r }
                resIdx += 1
            }
            // Average
            out = xs! / 3.0
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
