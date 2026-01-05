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

class Generator: Module {
    // Porting HiFiGANNSFGenerator structure
    let convPre: Conv1d
    let convPost: Conv1d
    let upsamples: [ConvTranspose1d]
    let resblocks: [ResBlock]
    
    override init() {
        // Hardcoded generic config for demo scaffolding
        // In real app, these would come from config
        self.convPre = Conv1d(768, 512, kernelSize: 7, stride: 1, padding: 3)
        self.convPost = Conv1d(128, 1, kernelSize: 7, stride: 1, padding: 3)
        self.upsamples = [
            ConvTranspose1d(512, 256, kernelSize: 16, stride: 8, padding: 4), // 8x
            ConvTranspose1d(256, 128, kernelSize: 16, stride: 8, padding: 4), // 8x
             // ... more layers
        ]
        self.resblocks = [
            ResBlock(channels: 256),
            ResBlock(channels: 128)
        ]
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = convPre(x)
        
        for (up, res) in zip(upsamples, resblocks) {
            out = leakyRelu(out, negativeSlope: LRELU_SLOPE)
            out = up(out)
            out = res(out) // Simplified block connection
        }
        
        out = leakyRelu(out)
        out = convPost(out)
        out = tanh(out)
        return out
    }
}
