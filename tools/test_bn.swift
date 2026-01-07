
import MLX
import MLXNN

let bn = BatchNorm(featureCount: 10)
print("BatchNorm keys: \(bn.children().keys)")
print("BatchNorm parameters: \(bn.parameters().keys)")
// Recursively inspect
func inspect(_ m: Module, prefix: String = "") {
    for (k, v) in m.parameters() {
        print("Param: \(prefix)\(k)")
    }
    for (k, c) in m.children() {
        inspect(c, prefix: "\(prefix)\(k).")
    }
}
// Inspect a dummy module with BN
class TestModule: Module {
    let bn = BatchNorm(featureCount: 10)
}
let m = TestModule()
// print("Module parameters keys: \(ModuleParameters.flattened(m.parameters()))") 
// Update: MLX Swift uses a different way to list params? 
// let sorted = m.parameters().flattened() // This returns [String: MLXArray]
// print(sorted.keys)

// Check non-trainable state?
// BatchNorm stores running stats as ... ?
