import MLX
import MLXNN

let up0 = ConvTransposed1d(inputChannels: 512, outputChannels: 256, kernelSize: 20, stride: 10, padding: 5)
let x = MLXArray.zeros([1, 1370, 512])
let out = up0(x)
print("Input shape: \(x.shape)")
print("Output shape: \(out.shape)")
