#!/usr/bin/env bash
# Configure and build the vendored llama.cpp (with the layer-range patch) on Linux/macOS.
#
#   scripts/build-llama.sh                 # CPU (+ Metal on macOS)
#   GGML_VULKAN=ON scripts/build-llama.sh  # add Vulkan
#   GGML_CUDA=ON scripts/build-llama.sh    # add CUDA
#   GGML_NATIVE=OFF scripts/build-llama.sh # portable CPU code (no -march=native): for
#                                          # Docker images and CI caches shared across machines
#   TARGETS=llama scripts/build-llama.sh   # only libllama and the ggml libraries it needs,
#                                          # no tools or examples (what dfin links against)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/vendor/llama.cpp"
build="$src/build"
if [ ! -f "$src/CMakeLists.txt" ]; then
    echo "vendor/llama.cpp is missing; see vendor/README.md" >&2
    exit 1
fi
cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE="${GGML_NATIVE:-ON}" \
    -DGGML_VULKAN="${GGML_VULKAN:-OFF}" -DGGML_CUDA="${GGML_CUDA:-OFF}" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF
# shellcheck disable=SC2086  # TARGETS is a space-separated list
cmake --build "$build" --config Release -j "${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}" \
    ${TARGETS:+--target $TARGETS}
echo "llama.cpp built: $build/bin"
