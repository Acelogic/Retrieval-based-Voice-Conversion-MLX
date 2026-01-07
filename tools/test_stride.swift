import MLX

func testStride() {
    let arr = MLXArray([1, 2, 3, 4, 5])
    let rev = arr[.stride(by: -1)]
    print("Original: \(arr.asArray(Int.self))")
    print("Reversed: \(rev.asArray(Int.self))")
}

testStride()
