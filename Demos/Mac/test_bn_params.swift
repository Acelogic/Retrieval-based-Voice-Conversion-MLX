import MLX
import MLXNN

// Create a BN and check if running stats are in parameters()
let bn = BatchNorm(featureCount: 10)

// Set it to eval mode
bn.train(false)

// Check parameters
let params = bn.parameters()
let flat = ModuleParameters.flattened(params)

print("BatchNorm parameter keys:")
for key in flat.keys.sorted() {
    print("  \(key)")
}

print("\nChecking for running stats:")
print("  Has runningMean: \(flat.keys.contains("runningMean"))")
print("  Has runningVar: \(flat.keys.contains("runningVar"))")
print("  Has running_mean: \(flat.keys.contains("running_mean"))")
print("  Has running_var: \(flat.keys.contains("running_var"))")

// Try to update with fake running stats
var testParams: [String: MLXArray] = [:]
testParams["weight"] = MLXArray.ones([10])
testParams["bias"] = MLXArray.zeros([10])
testParams["runningMean"] = MLXArray.ones([10]) * 5.0
testParams["runningVar"] = MLXArray.ones([10]) * 2.0

bn.update(parameters: ModuleParameters.unflattened(testParams))

// Check again
let paramsAfter = bn.parameters()
let flatAfter = ModuleParameters.flattened(paramsAfter)

print("\nAfter update, parameter keys:")
for key in flatAfter.keys.sorted() {
    print("  \(key)")
}

print("\nDoes update() load running stats?")
if let rm = flatAfter["runningMean"] {
    print("  runningMean loaded: \(rm.asArray(Float.self))")
} else {
    print("  runningMean NOT FOUND")
}
