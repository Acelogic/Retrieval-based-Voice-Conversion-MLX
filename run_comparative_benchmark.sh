#!/bin/bash
set -e

# Configuration
TEST_RESULTS_DIR="test_results"
TEST_AUDIO_DIR="test-audio"
REPLAY_MODELS_DIR="/Users/mcruz/Library/Application Support/Replay/com.replay.Replay/models"
IOS_ASSETS_DIR="Demos/iOS/RVCNative/RVCNativePackage/Sources/RVCNativeFeature/Assets"

# Core Models (Sources of Truth)
HUBERT_SOURCE="rvc/models/embedders/contentvec/pytorch_model.bin"
RMVPE_SOURCE="rvc/models/predictors/rmvpe.pt"

# Priority Models (will test all available)
# 48kHz models preferred for better quality
PRIORITY_MODELS=("Drake" "Juice WRLD (RVC v2) 310 Epochs" "Eminem Modern" "Bob Marley (RVC v2) (500 Epochs) RMVPE" "Slim_Shady_New")

# Input Audio (The sole source of truth)
INPUT_AUDIO="${TEST_AUDIO_DIR}/coder_audio_stock.wav"

# Asset paths
HUBERT_PATH="${IOS_ASSETS_DIR}/hubert_base.safetensors"
RMVPE_PATH="${IOS_ASSETS_DIR}/rmvpe.safetensors"

echo "=================================================="
echo "      RVC Comparative Benchmark Suite (ZERO-ARTIFACT)"
echo "=================================================="

# Helper Functions
check_file() {
    local file="$1"
    local desc="$2"
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $desc at $file"
        return 1
    fi
    echo "✅ Found: $desc"
    return 0
}

# 1. Validate Core Models
echo ""
echo "--------------------------------------------------"
echo "Validating Core Models..."
echo "--------------------------------------------------"

CORE_MODELS_OK=true
check_file "${HUBERT_SOURCE}" "HuBERT Model" || CORE_MODELS_OK=false
check_file "${RMVPE_SOURCE}" "RMVPE Model" || CORE_MODELS_OK=false

if [ "$CORE_MODELS_OK" = false ]; then
    echo ""
    echo "❌ Core models missing. Please ensure:"
    echo "   - ${HUBERT_SOURCE}"
    echo "   - ${RMVPE_SOURCE}"
    echo "are available in the rvc/ directory."
    exit 1
fi

# 2. Find Available Priority Models
echo ""
echo "--------------------------------------------------"
echo "Discovering Priority Models..."
echo "--------------------------------------------------"

AVAILABLE_MODELS=()
for model in "${PRIORITY_MODELS[@]}"; do
    model_dir="${REPLAY_MODELS_DIR}/${model}"
    pth_file="${model_dir}/model.pth"
    json_file="${model_dir}/model.json"

    if [ -f "$pth_file" ]; then
        echo "✅ Found: $model at $model_dir"
        AVAILABLE_MODELS+=("$model")
    else
        echo "⚠️  Skipping: $model (not found in Replay models)"
    fi
done

if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
    echo ""
    echo "❌ No priority models found in $REPLAY_MODELS_DIR"
    echo "   Looking for: ${PRIORITY_MODELS[*]}"
    exit 1
fi

echo ""
echo "Will benchmark ${#AVAILABLE_MODELS[@]} model(s): ${AVAILABLE_MODELS[*]}"

# 3. Standardized Cleanup
echo ""
echo "--------------------------------------------------"
echo "Cleaning up old artifacts..."
echo "--------------------------------------------------"
rm -rf "${TEST_RESULTS_DIR}"
mkdir -p "${TEST_RESULTS_DIR}"
rm -f weights/*.safetensors weights/*.npz weights/*.json
rm -f "${IOS_ASSETS_DIR}"/*.safetensors
echo "✅ Cleanup complete"

# 4. Total Weight Reconstruction (From Scratch)
echo ""
echo "--------------------------------------------------"
echo "Building weights from scratch..."
echo "--------------------------------------------------"

# First, convert core models for iOS
echo "Converting Core Models (HuBERT & RMVPE) for iOS..."
# We'll use the first available model's path for the tool, but specify core models
FIRST_MODEL="${AVAILABLE_MODELS[0]}"
FIRST_MODEL_DIR="${REPLAY_MODELS_DIR}/${FIRST_MODEL}"

python3 tools/convert_models_for_ios.py \
    --model-path "${FIRST_MODEL_DIR}" \
    --model-name "${FIRST_MODEL}" \
    --hubert-path "${HUBERT_SOURCE}" \
    --rmvpe-path "${RMVPE_SOURCE}" \
    --output-dir "${IOS_ASSETS_DIR}"

echo "✅ Core models converted for iOS"

# Now convert each priority model
for MODEL_NAME in "${AVAILABLE_MODELS[@]}"; do
    echo ""
    echo "Converting ${MODEL_NAME}..."

    MODEL_DIR="${REPLAY_MODELS_DIR}/${MODEL_NAME}"
    MODEL_PTH="${MODEL_DIR}/model.pth"
    PYTHON_WEIGHTS="weights/${MODEL_NAME}.npz"
    SWIFT_WEIGHTS="weights/${MODEL_NAME}.safetensors"

    # Convert for Python MLX
    python3 tools/convert_rvc_model.py "${MODEL_PTH}" "${PYTHON_WEIGHTS}"

    # Convert for Swift MLX
    python3 tools/convert_npz_to_safetensors.py "${PYTHON_WEIGHTS}" "${SWIFT_WEIGHTS}"

    # Also convert for iOS if not already done
    if [ "$MODEL_NAME" != "$FIRST_MODEL" ]; then
        python3 tools/convert_models_for_ios.py \
            --model-path "${MODEL_DIR}" \
            --model-name "${MODEL_NAME}" \
            --hubert-path "${HUBERT_SOURCE}" \
            --rmvpe-path "${RMVPE_SOURCE}" \
            --output-dir "${IOS_ASSETS_DIR}"
    fi

    echo "✅ ${MODEL_NAME} converted"
done

echo ""
echo "✅ All weights rebuilt and deployed"

# 5. Build Swift CLI Once
echo ""
echo "--------------------------------------------------"
echo "Building Swift CLI..."
echo "--------------------------------------------------"
swift build -c debug --package-path Demos/Mac --product RVCNativeMac

# Deploy Native metallib
BUILD_METALLIB="Demos/Mac/.build/arm64-apple-macosx/debug/default.metallib"
if [ -f "${BUILD_METALLIB}" ]; then
    cp -f "${BUILD_METALLIB}" .
    cp -f "${BUILD_METALLIB}" "${IOS_ASSETS_DIR}/"
    echo "✅ Native macOS metallib deployed"
fi

# 6. Benchmark Each Model
echo ""
echo "=================================================="
echo "      Running Benchmarks for All Models"
echo "=================================================="

for MODEL_NAME in "${AVAILABLE_MODELS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Benchmarking: ${MODEL_NAME}"
    echo "--------------------------------------------------"

    PYTHON_WEIGHTS="weights/${MODEL_NAME}.npz"
    SWIFT_WEIGHTS="weights/${MODEL_NAME}.safetensors"
    CTX_PYTHON_OUT="${TEST_RESULTS_DIR}/${MODEL_NAME}_python_mlx.wav"
    CTX_SWIFT_OUT="${TEST_RESULTS_DIR}/${MODEL_NAME}_swift_mlx.wav"

    # Generate Python MLX Reference
    echo "Generating Python MLX output..."
    python3 rvc-mlx-cli.py infer \
        --model_path "${PYTHON_WEIGHTS}" \
        --input_path "${INPUT_AUDIO}" \
        --output_path "${CTX_PYTHON_OUT}"
    echo "✅ Python MLX output: ${CTX_PYTHON_OUT}"

    # Run Swift MLX Benchmark
    echo ""
    echo "Running Swift MLX benchmark..."
    ./Demos/Mac/.build/arm64-apple-macosx/debug/RVCNativeMac \
        --model "${SWIFT_WEIGHTS}" \
        --audio "${INPUT_AUDIO}" \
        --output "${CTX_SWIFT_OUT}" \
        --ref "${CTX_PYTHON_OUT}" \
        --benchmark \
        --hubert "${HUBERT_PATH}" \
        --rmvpe "${RMVPE_PATH}"

    echo "✅ Swift MLX output: ${CTX_SWIFT_OUT}"
    echo ""
done

echo ""
echo "=================================================="
echo "      Benchmark Suite Complete"
echo "=================================================="
echo ""
echo "Summary of Results:"
echo "--------------------------------------------------"
ls -lh "${TEST_RESULTS_DIR}"

echo ""
echo "✅ All benchmarks completed successfully!"
echo "   Results directory: ${TEST_RESULTS_DIR}"
echo "   Tested models: ${AVAILABLE_MODELS[*]}"
