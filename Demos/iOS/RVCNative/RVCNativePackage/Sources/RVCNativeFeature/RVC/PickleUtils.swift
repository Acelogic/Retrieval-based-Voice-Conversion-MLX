import Foundation

/// A minimal Pickle Virtual Machine for parsing PyTorch state_dicts.
/// Supports Pickle Protocol 2, 3, 4 (partial).
final class PickleUnpickler {
    enum PickleError: Error {
        case unknownOpcode(UInt8)
        case stackUnderflow
        case invalidStructure
        case unsupported(String)
    }
    
    private var data: Data
    private var position: Int = 0
    private var stack: [Any] = []
    private var memo: [Int: Any] = [:]
    
    // Opcodes (PyTorch commonly uses these)
    private let PROTO: UInt8 = 0x80
    private let STOP: UInt8 = 0x2E
    private let GLOBAL: UInt8 = 0x63
    private let BININT1: UInt8 = 0x4B
    private let BININT2: UInt8 = 0x4D
    private let BININT: UInt8 = 0x4A
    private let BINGET: UInt8 = 0x68
    private let LONG_BINGET: UInt8 = 0x6a
    private let BINPUT: UInt8 = 0x71
    private let LONG_BINPUT: UInt8 = 0x72
    private let BINPERSID: UInt8 = 0x51
    private let EMPTY_DICT: UInt8 = 0x7D
    private let EMPTY_LIST: UInt8 = 0x5d
    private let EMPTY_TUPLE: UInt8 = 0x29
    private let SETITEM: UInt8 = 0x73
    private let SETITEMS: UInt8 = 0x75
    private let APPEND: UInt8 = 0x61
    private let APPENDS: UInt8 = 0x65
    private let BUILD: UInt8 = 0x62
    private let MARK: UInt8 = 0x28
    private let TUPLE: UInt8 = 0x74
    private let TUPLE1: UInt8 = 0x85
    private let TUPLE2: UInt8 = 0x86
    private let TUPLE3: UInt8 = 0x87
    private let REDUCE: UInt8 = 0x52
    private let NEWTRUE: UInt8 = 0x88
    private let NEWFALSE: UInt8 = 0x89
    private let NONE: UInt8 = 0x4E
    private let BINUNICODE: UInt8 = 0x58
    private let SHORT_BINUNICODE: UInt8 = 0x8c
    private let BINFLOAT: UInt8 = 0x47
    
    init(data: Data) {
        self.data = data
    }
    
    func load() throws -> Any {
        while position < data.count {
            let opcode = data[position]
            position += 1
            
            do {
                switch opcode {
                case PROTO:
                    // Ignore version
                    _ = readByte()
                    
                case STOP:
                    guard let value = stack.popLast() else { throw PickleError.stackUnderflow }
                    return value
                    
                case GLOBAL:
                    let module = try readLine()
                    let name = try readLine()
                    // Create a placeholder for the global class/function
                    stack.append(GlobalReference(module: module, name: name))
                    
                case EMPTY_DICT:
                    stack.append(NSMutableDictionary())
                    
                case EMPTY_LIST:
                    stack.append(NSMutableArray())
                    
                case BINUNICODE:
                    let len = Int(try readUInt32())
                    let str = try readString(count: len)
                    stack.append(str)
                    
                case SHORT_BINUNICODE:
                    let len = Int(readByte())
                    let str = try readString(count: len)
                    stack.append(str)
                    
                case BININT1:
                    stack.append(Int(readByte()))
                    
                case BININT2:
                    stack.append(Int(try readUInt16()))
                    
                case BININT:
                    stack.append(Int(try readInt32()))
                    
                case BINPUT:
                    let index = Int(readByte())
                    if let top = stack.last {
                        memo[index] = top
                    }
                    
                case LONG_BINPUT:
                    let index = Int(try readUInt32())
                    if let top = stack.last {
                        memo[index] = top
                    }
                    
                case BINGET:
                    let index = Int(readByte())
                    guard let value = memo[index] else { throw PickleError.invalidStructure }
                    stack.append(value)
                    
                case LONG_BINGET:
                    let index = Int(try readUInt32())
                    guard let value = memo[index] else { throw PickleError.invalidStructure }
                    stack.append(value)
                    
                case BINPERSID:
                    // Persistent ID (often used for storage references in PyTorch)
                    // Pushes the persistent ID onto the stack
                    if let pid = stack.popLast() {
                        stack.append(PersistentID(value: pid))
                    }
                    
                case MARK:
                    stack.append(Mark())
                    
                case EMPTY_TUPLE:
                    stack.append([Any]())
                    
                case TUPLE, TUPLE1, TUPLE2, TUPLE3:
                    try buildTuple(opcode: opcode)
                    
                case SETITEM:
                    guard let value = stack.popLast(),
                          let key = stack.popLast() else {
                        throw PickleError.invalidStructure
                    }
                    guard let dict = stack.last as? NSMutableDictionary else {
                         throw PickleError.invalidStructure
                    }
                    dict[key] = value
                    
                case APPENDS:
                    // print("OP: APPENDS")
                    var items: [Any] = []
                    while true {
                        let top = stack.popLast()
                        if top is Mark { break }
                        guard let value = top else { throw PickleError.stackUnderflow }
                        items.append(value)
                    }
                    // List is below mark
                    guard let list = stack.last as? NSMutableArray else { 
                        print("APPENDS: Object below mark is not NSMutableArray, is \(type(of: stack.last))")
                        throw PickleError.invalidStructure 
                    }
                    for item in items.reversed() {
                        list.add(item)
                    }
                    
                case SETITEMS:
                    // print("OP: SETITEMS")
                    // Pop until mark
                    var items: [(Any, Any)] = []
                    while true {
                        let top = stack.popLast()
                        if top is Mark { break }
                         guard let value = top, let key = stack.popLast() else { 
                             print("SETITEMS: Stack underflow while popping key/value")
                             throw PickleError.stackUnderflow 
                         }
                        items.append((key, value))
                    }
                    // Target dict is below mark
                    guard let dict = stack.last as? NSMutableDictionary else { 
                        if let actual = stack.last {
                            print("SETITEMS: Object below mark is not NSMutableDictionary, is \(type(of: actual))")
                        } else {
                            print("SETITEMS: Stack empty below mark")
                        }
                        throw PickleError.invalidStructure 
                    }
                    for (key, value) in items.reversed() {
                        dict[key] = value
                    }
                    
                case APPEND:
                    guard let value = stack.popLast() else { throw PickleError.stackUnderflow }
                    guard let list = stack.last as? NSMutableArray else {
                        throw PickleError.invalidStructure
                    }
                    list.add(value)
                    
                case APPENDS:
                    var items: [Any] = []
                    while true {
                        let top = stack.popLast()
                        if top is Mark { break }
                        guard let value = top else { throw PickleError.stackUnderflow }
                        items.append(value)
                    }
                    // List is below mark
                    guard var list = stack.popLast() as? [Any] else { throw PickleError.invalidStructure }
                    list.append(contentsOf: items.reversed())
                    stack.append(list)
                    
                case BUILD:
                    // Build object state
                    guard let state = stack.popLast(),
                          let _ = stack.last else { // Instance
                        throw PickleError.stackUnderflow
                    }
                    // For our purposes (parsing dict), we might just keep the state
                    // or merge it if the instance handles it.
                    // Ideally, we'd apply state to the instance.
                    // For simple PyTorch state_dict, the dict IS the object usually.
                    // If instance is a GlobalRef (rebuild_tensor), we attach state.
                    if var global = stack.last as? GlobalReference {
                        stack.removeLast()
                        global.state = state
                        stack.append(global)
                    }
                    
                case REDUCE:
                    // Calls a callable with a tuple of arguments
                    guard let args = stack.popLast() as? [Any],
                          let callable = stack.popLast() as? GlobalReference else {
                        throw PickleError.invalidStructure
                    }
                    
                    // Handle PyTorch specific rebuilds
                    if callable.module == "torch._utils" && (callable.name == "_rebuild_tensor_v2" || callable.name == "_rebuild_tensor") {
                         // Create a TensorReference
                         let tensor = TensorReference(args: args)
                         stack.append(tensor)
                    } else if callable.module == "collections" && callable.name == "OrderedDict" {
                        // Instantiate OrderedDict as NSMutableDictionary
                        let dict = NSMutableDictionary()
                        // OrderedDict constructor might take a list of items
                        if let first = args.first as? [(Any, Any)] {
                            for (k, v) in first {
                                dict[k] = v
                            }
                        } else if let first = args.first as? [Any] {
                            // Sometimes args might be wrapped differently?
                            // Typically OrderedDict([]) -> args = [[]]
                        }
                        stack.append(dict)
                    } else {
                        // Generic reduce, just keep as Reference
                        var newRef = callable
                        newRef.args = args
                        stack.append(newRef)
                    }

                    
                case NEWTRUE:
                    stack.append(true)
                    
                case NEWFALSE:
                    stack.append(false)
                    
                case NONE:
                    // We need a way to represent None. Since [Any] can't hold nil
                    // unless we wrap it or use Optional<Any> which is tricky in mixed stack.
                    // For typical state_dict parsing, None is rarely a key or tensor value.
                    // We can use a unique sentinel or NSNull.
                    stack.append(NSNull())
                    
                default:
                    // Check logic for unknown opcodes
                    print("Unsupported Opcode: \(opcode)")
                    throw PickleError.unknownOpcode(opcode)
                }
            } catch {
                print("Pickle Error at pos \(position) (last opcode: \(opcode)): \(error)")
                throw error
            }
        }
        throw PickleError.stackUnderflow
    }
    
    // MARK: - Helpers
    
    private func readByte() -> UInt8 {
        let b = data[position]
        position += 1
        return b
    }
    
    private func readUInt16() throws -> UInt16 {
        let val = data.subdata(in: position..<position+2).withUnsafeBytes { $0.load(as: UInt16.self) }
        position += 2
        return val
    }
    
    private func readInt32() throws -> Int32 {
        let val = data.subdata(in: position..<position+4).withUnsafeBytes { $0.load(as: Int32.self) }
        position += 4
        return val
    }
    
    private func readUInt32() throws -> UInt32 {
        let val = data.subdata(in: position..<position+4).withUnsafeBytes { $0.load(as: UInt32.self) }
        position += 4
        return val
    }
    
    private func readLine() throws -> String {
        var bytes: [UInt8] = []
        while position < data.count {
            let b = data[position]
            position += 1
            if b == 0x0A { break } // \n
            bytes.append(b)
        }
        guard let str = String(bytes: bytes, encoding: .utf8) else { throw PickleError.invalidStructure }
        return str
    }
    
    private func readString(count: Int) throws -> String {
        let sub = data.subdata(in: position..<position+count)
        position += count
        guard let str = String(data: sub, encoding: .utf8) else { throw PickleError.invalidStructure }
        return str
    }
    
    private func buildTuple(opcode: UInt8) throws {
        var items: [Any] = []
        
        if opcode == MARK {
             // Already handled MARK case in main loop,
             // But TUPLE opcode means pop until mark
             fatalError("Use specific logic for TUPLE vs specific sizes")
        }
        
        switch opcode {
        case TUPLE:
             while true {
                 let top = stack.popLast()
                 if top is Mark { break }
                 guard let val = top else { throw PickleError.stackUnderflow }
                 items.append(val)
             }
             stack.append(Array(items.reversed()))
        case TUPLE1:
             guard let a = stack.popLast() else { throw PickleError.stackUnderflow }
             stack.append([a])
        case TUPLE2:
             guard let b = stack.popLast(), let a = stack.popLast() else { throw PickleError.stackUnderflow }
             stack.append([a, b])
        case TUPLE3:
             guard let c = stack.popLast(), let b = stack.popLast(), let a = stack.popLast() else { throw PickleError.stackUnderflow }
             stack.append([a, b, c])
        default: break
        }
    }
}

// MARK: - Structures

struct GlobalReference {
    let module: String
    let name: String
    var args: [Any]? = nil
    var state: Any? = nil
}

struct Mark {}

struct PersistentID {
    let value: Any
}

struct TensorReference {
    let storageOffset: Int
    let storage: TensorStorage
    let size: [Int]
    let stride: [Int]
    let requiresGrad: Bool
    
    init(args: [Any]) {
        // _rebuild_tensor_v2 args: (storage, storage_offset, size, stride, requires_grad, backward_hooks)
        // Storage is (storage_type, file_key, device, numel, etc) - usually implemented as a PersistentID
        
        // This is a simplification; actual args parsing needs care
        // Assuming args[0] is the storage tuple/obj
        
        self.storage = TensorStorage(from: args[0])
        self.storageOffset = args[1] as? Int ?? 0
        self.size = args[2] as? [Int] ?? []
        self.stride = args[3] as? [Int] ?? []
        self.requiresGrad = args[4] as? Bool ?? false
    }
}

struct TensorStorage {
    let filename: String
    let numel: Int
    let dtype: String
    
    init(from obj: Any) {
         // Storage object from PyTorch pickle involves a PersistentID
         // The PersistentID value is a tuple: ('storage', storage_type, key, device, numel)
         // e.g. ('storage', torch.FloatStorage, '0', 'cpu', 1024)
         
         if let pid = obj as? PersistentID, let tuple = pid.value as? [Any] {
             // Parse tuple
             // Index 2 is usually the key (filename inside zip, e.g. "0", "1")
             self.filename = tuple[2] as? String ?? ""
             self.numel = tuple[4] as? Int ?? 0
             self.dtype = String(describing: tuple[1]) // approximation
         } else {
             self.filename = ""
             self.numel = 0
             self.dtype = "unknown"
         }
    }
}
