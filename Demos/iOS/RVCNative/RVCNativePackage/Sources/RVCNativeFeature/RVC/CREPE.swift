import Foundation
import MLX
import MLXNN
import Accelerate

// MARK: - Constants

private let PITCH_BINS = 360
private let SAMPLE_RATE = 16000
private let HOP_SIZE = 160  // ~10ms
private let WINDOW_SIZE = 1024  // 64ms

private let CENTS_PER_BIN: Float = 20
private let FMIN: Float = 10.0  // Reference frequency

/// Precomputed cent values for each bin
private let CENTS: [Float] = (0..<PITCH_BINS).map { Float($0) * CENTS_PER_BIN + 1997.3794 }

// MARK: - Utility Functions

private func centsToFrequency(_ cents: Float) -> Float {
    return FMIN * pow(2.0, cents / 1200.0)
}

private func frequencyToCents(_ freq: Float) -> Float {
    return 1200.0 * log2(freq / FMIN)
}

// MARK: - CREPEModel

/// CREPE pitch estimation model in MLX Swift.
///
/// Architecture:
/// - 6 Conv2d layers with BatchNorm and MaxPool
/// - Final Linear classifier to 360 pitch bins
/// - Sigmoid activation for probabilities
///
/// Note: MLX uses (B, H, W, C) format
public class CREPEModel: Module {

    let modelType: String
    let inFeatures: Int

    // Layer 1
    let conv1: Conv2d
    let conv1_BN: CustomBatchNorm

    // Layer 2
    let conv2: Conv2d
    let conv2_BN: CustomBatchNorm

    // Layer 3
    let conv3: Conv2d
    let conv3_BN: CustomBatchNorm

    // Layer 4
    let conv4: Conv2d
    let conv4_BN: CustomBatchNorm

    // Layer 5
    let conv5: Conv2d
    let conv5_BN: CustomBatchNorm

    // Layer 6
    let conv6: Conv2d
    let conv6_BN: CustomBatchNorm

    // Classifier
    let classifier: Linear

    public init(model: String = "full") {
        self.modelType = model

        // Model-specific parameters
        let inChannels: [Int]
        let outChannels: [Int]

        if model == "full" {
            inChannels = [1, 1024, 128, 128, 128, 256]
            outChannels = [1024, 128, 128, 128, 256, 512]
            self.inFeatures = 2048
        } else {
            // tiny
            inChannels = [1, 128, 16, 16, 16, 32]
            outChannels = [128, 16, 16, 16, 32, 64]
            self.inFeatures = 256
        }

        // First layer has kernel (512, 1), stride (4, 1)
        // Rest have kernel (64, 1), stride (1, 1)

        // Layer 1
        self.conv1 = Conv2d(
            inputChannels: inChannels[0],
            outputChannels: outChannels[0],
            kernelSize: IntOrPair((512, 1)),
            stride: IntOrPair((4, 1))
        )
        self.conv1_BN = CustomBatchNorm(featureCount: outChannels[0], eps: 1e-3, momentum: 0.0)

        // Layer 2
        self.conv2 = Conv2d(
            inputChannels: inChannels[1],
            outputChannels: outChannels[1],
            kernelSize: IntOrPair((64, 1)),
            stride: IntOrPair((1, 1))
        )
        self.conv2_BN = CustomBatchNorm(featureCount: outChannels[1], eps: 1e-3, momentum: 0.0)

        // Layer 3
        self.conv3 = Conv2d(
            inputChannels: inChannels[2],
            outputChannels: outChannels[2],
            kernelSize: IntOrPair((64, 1)),
            stride: IntOrPair((1, 1))
        )
        self.conv3_BN = CustomBatchNorm(featureCount: outChannels[2], eps: 1e-3, momentum: 0.0)

        // Layer 4
        self.conv4 = Conv2d(
            inputChannels: inChannels[3],
            outputChannels: outChannels[3],
            kernelSize: IntOrPair((64, 1)),
            stride: IntOrPair((1, 1))
        )
        self.conv4_BN = CustomBatchNorm(featureCount: outChannels[3], eps: 1e-3, momentum: 0.0)

        // Layer 5
        self.conv5 = Conv2d(
            inputChannels: inChannels[4],
            outputChannels: outChannels[4],
            kernelSize: IntOrPair((64, 1)),
            stride: IntOrPair((1, 1))
        )
        self.conv5_BN = CustomBatchNorm(featureCount: outChannels[4], eps: 1e-3, momentum: 0.0)

        // Layer 6
        self.conv6 = Conv2d(
            inputChannels: inChannels[5],
            outputChannels: outChannels[5],
            kernelSize: IntOrPair((64, 1)),
            stride: IntOrPair((1, 1))
        )
        self.conv6_BN = CustomBatchNorm(featureCount: outChannels[5], eps: 1e-3, momentum: 0.0)

        // Classifier
        self.classifier = Linear(inFeatures, PITCH_BINS)

        super.init()
    }

    /// Forward pass through one layer: Conv -> ReLU -> BatchNorm -> MaxPool
    private func layer(
        _ x: MLXArray,
        conv: Conv2d,
        batchNorm: CustomBatchNorm,
        paddingTop: Int,
        paddingBottom: Int
    ) -> MLXArray {
        var x = x

        // Pad height dimension (axis 1) - MLX format is (B, H, W, C)
        if paddingTop > 0 || paddingBottom > 0 {
            x = MLX.padded(x, widths: [[0, 0], [paddingTop, paddingBottom], [0, 0], [0, 0]])
        }

        // Conv2d
        x = conv(x)

        // ReLU
        x = relu(x)

        // BatchNorm - reshape for proper normalization
        let B = x.shape[0]
        let H = x.shape[1]
        let W = x.shape[2]
        let C = x.shape[3]

        let xFlat = x.reshaped([B * H * W, C])
        let xNorm = batchNorm(xFlat)
        x = xNorm.reshaped([B, H, W, C])

        // MaxPool2d with kernel (2, 1) and stride (2, 1) - pool over height only
        x = x.reshaped([B, H / 2, 2, W, C])
        x = x.max(axis: 2)

        return x
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: (B, 1024) audio frames
        let B = x.shape[0]

        // Reshape to (B, H=1024, W=1, C=1) for Conv2d
        var h = x.reshaped([B, WINDOW_SIZE, 1, 1])

        // Forward through 6 layers
        h = layer(h, conv: conv1, batchNorm: conv1_BN, paddingTop: 254, paddingBottom: 254)
        h = layer(h, conv: conv2, batchNorm: conv2_BN, paddingTop: 31, paddingBottom: 32)
        h = layer(h, conv: conv3, batchNorm: conv3_BN, paddingTop: 31, paddingBottom: 32)
        h = layer(h, conv: conv4, batchNorm: conv4_BN, paddingTop: 31, paddingBottom: 32)
        h = layer(h, conv: conv5, batchNorm: conv5_BN, paddingTop: 31, paddingBottom: 32)
        h = layer(h, conv: conv6, batchNorm: conv6_BN, paddingTop: 31, paddingBottom: 32)

        // Flatten: (B, H, W, C) -> (B, H * W * C) = (B, inFeatures)
        h = h.reshaped([B, inFeatures])

        // Classifier with sigmoid
        h = classifier(h)
        h = sigmoid(h)

        return h
    }

    public func setTrainingMode(_ mode: Bool) {
        conv1_BN.setTrainingMode(mode)
        conv2_BN.setTrainingMode(mode)
        conv3_BN.setTrainingMode(mode)
        conv4_BN.setTrainingMode(mode)
        conv5_BN.setTrainingMode(mode)
        conv6_BN.setTrainingMode(mode)
    }
}

// MARK: - CREPE Interface

/// CREPE pitch extraction interface.
///
/// Example:
/// ```swift
/// let crepe = try CREPE(weightsURL: url)
/// let f0 = crepe.getF0(audio: audioArray, f0Min: 50, f0Max: 1100)
/// ```
public class CREPE {

    private let model: CREPEModel
    private let modelType: String

    /// Initialize CREPE with weights.
    ///
    /// - Parameters:
    ///   - weightsURL: URL to the safetensors weights file
    ///   - modelType: Model variant ("full" or "tiny")
    public init(weightsURL: URL, modelType: String = "full") throws {
        self.modelType = modelType
        self.model = CREPEModel(model: modelType)

        // Load weights
        let weights = try MLX.loadArrays(url: weightsURL)
        let remapped = remapWeights(weights)
        try model.update(parameters: ModuleParameters.unflattened(remapped))

        // Set to eval mode
        model.setTrainingMode(false)

        MLX.eval(model)
    }

    /// Remap weight keys from Python format to Swift format.
    private func remapWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var remapped: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Remap BatchNorm keys
            if key.contains("running_mean") {
                newKey = key.replacingOccurrences(of: "running_mean", with: "runningMean")
            } else if key.contains("running_var") {
                newKey = key.replacingOccurrences(of: "running_var", with: "runningVar")
            }

            remapped[newKey] = value
        }

        return remapped
    }

    /// Extract F0 from audio.
    ///
    /// - Parameters:
    ///   - audio: Audio signal (1D MLXArray, 16kHz)
    ///   - f0Min: Minimum F0 to detect (Hz)
    ///   - f0Max: Maximum F0 to detect (Hz)
    ///   - threshold: Periodicity threshold below which F0 is set to 0
    /// - Returns: F0 contour with shape (1, T, 1) in Hz
    public func getF0(
        audio: MLXArray,
        f0Min: Float = 50.0,
        f0Max: Float = 1100.0,
        threshold: Float = 0.1
    ) -> MLXArray {
        // Frame the audio
        let frames = frameAudio(audio)

        // Run inference
        let probabilities = infer(frames: frames)

        // Decode to F0 and periodicity
        var (f0, periodicity) = decode(probabilities: probabilities, f0Min: f0Min, f0Max: f0Max)

        // Apply filtering
        periodicity = medianFilter(periodicity, window: 3)
        f0 = meanFilter(f0, window: 3)

        // Apply threshold - set F0 to 0 where periodicity is low
        for i in 0..<f0.count {
            if periodicity[i] < threshold {
                f0[i] = 0
            }
        }

        // Convert to MLXArray with shape (1, T, 1)
        let f0Array = MLXArray(f0)
        return f0Array.reshaped([1, f0.count, 1])
    }

    /// Frame audio into overlapping windows.
    private func frameAudio(_ audio: MLXArray) -> MLXArray {
        // Convert to numpy-style array for easier manipulation
        MLX.eval(audio)
        let audioData = audio.asArray(Float.self)

        // Pad to ensure we have complete frames
        let padLength = WINDOW_SIZE / 2
        var audioPadded = [Float](repeating: 0, count: audioData.count + 2 * padLength)

        // Reflect padding
        for i in 0..<padLength {
            audioPadded[padLength - 1 - i] = audioData[min(i + 1, audioData.count - 1)]
        }
        for i in 0..<audioData.count {
            audioPadded[padLength + i] = audioData[i]
        }
        for i in 0..<padLength {
            let srcIdx = max(0, audioData.count - 2 - i)
            audioPadded[padLength + audioData.count + i] = audioData[srcIdx]
        }

        // Number of frames
        let nFrames = 1 + (audioPadded.count - WINDOW_SIZE) / HOP_SIZE

        // Extract and normalize frames
        var frames = [Float](repeating: 0, count: nFrames * WINDOW_SIZE)

        for i in 0..<nFrames {
            let start = i * HOP_SIZE
            let end = start + WINDOW_SIZE

            // Extract frame
            var frame = Array(audioPadded[start..<end])

            // Normalize: subtract mean, divide by std
            let mean = frame.reduce(0, +) / Float(frame.count)
            frame = frame.map { $0 - mean }

            var sumSq: Float = 0
            vDSP_svesq(frame, 1, &sumSq, vDSP_Length(frame.count))
            let std = sqrt(sumSq / Float(frame.count))

            if std > 1e-10 {
                frame = frame.map { $0 / std }
            }

            // Copy to output
            for j in 0..<WINDOW_SIZE {
                frames[i * WINDOW_SIZE + j] = frame[j]
            }
        }

        return MLXArray(frames).reshaped([nFrames, WINDOW_SIZE])
    }

    /// Run model inference on frames.
    private func infer(frames: MLXArray) -> [[Float]] {
        let nFrames = frames.shape[0]
        let batchSize = 512
        var allProbabilities: [[Float]] = []

        for i in stride(from: 0, to: nFrames, by: batchSize) {
            let end = min(i + batchSize, nFrames)
            let batch = frames[i..<end]

            let probs = model(batch)
            MLX.eval(probs)

            // Convert to Swift array
            let probsFlat = probs.asArray(Float.self)
            let batchCount = end - i

            for b in 0..<batchCount {
                var row = [Float](repeating: 0, count: PITCH_BINS)
                for p in 0..<PITCH_BINS {
                    row[p] = probsFlat[b * PITCH_BINS + p]
                }
                allProbabilities.append(row)
            }
        }

        return allProbabilities
    }

    /// Decode probabilities to F0 and periodicity.
    private func decode(
        probabilities: [[Float]],
        f0Min: Float,
        f0Max: Float
    ) -> (f0: [Float], periodicity: [Float]) {
        let f0MinCents = frequencyToCents(f0Min)
        let f0MaxCents = frequencyToCents(f0Max)

        var f0 = [Float](repeating: 0, count: probabilities.count)
        var periodicity = [Float](repeating: 0, count: probabilities.count)

        for (i, probs) in probabilities.enumerated() {
            // Mask bins outside frequency range
            var maskedProbs = probs
            for bin in 0..<PITCH_BINS {
                if CENTS[bin] < f0MinCents || CENTS[bin] > f0MaxCents {
                    maskedProbs[bin] = 0
                }
            }

            // Find peak bin
            var peakBin = 0
            var maxProb: Float = 0
            for bin in 0..<PITCH_BINS {
                if maskedProbs[bin] > maxProb {
                    maxProb = maskedProbs[bin]
                    peakBin = bin
                }
            }

            periodicity[i] = maxProb

            // Weighted average around peak (±4 bins)
            let windowSize = 4
            let start = max(0, peakBin - windowSize)
            let end = min(PITCH_BINS, peakBin + windowSize + 1)

            var weightedSum: Float = 0
            var totalWeight: Float = 0

            for bin in start..<end {
                weightedSum += maskedProbs[bin] * CENTS[bin]
                totalWeight += maskedProbs[bin]
            }

            if totalWeight > 0 {
                let f0Cents = weightedSum / totalWeight
                f0[i] = centsToFrequency(f0Cents)
            }
        }

        return (f0, periodicity)
    }

    /// Apply median filter.
    private func medianFilter(_ x: [Float], window: Int) -> [Float] {
        guard x.count > 0 else { return x }

        var result = x
        let halfWindow = window / 2

        for i in 0..<x.count {
            let start = max(0, i - halfWindow)
            let end = min(x.count, i + halfWindow + 1)
            var windowValues = Array(x[start..<end])
            windowValues.sort()
            result[i] = windowValues[windowValues.count / 2]
        }

        return result
    }

    /// Apply mean filter.
    private func meanFilter(_ x: [Float], window: Int) -> [Float] {
        guard x.count > 0 else { return x }

        var result = x
        let halfWindow = window / 2

        for i in 0..<x.count {
            let start = max(0, i - halfWindow)
            let end = min(x.count, i + halfWindow + 1)
            let sum = x[start..<end].reduce(0, +)
            result[i] = sum / Float(end - start)
        }

        return result
    }
}
