# syntax=docker/dockerfile:1
#
# dfin: CPU-only Linux image.
#
#   git submodule update --init          # vendor/llama.cpp must be checked out
#   docker build -t dfin .
#   docker run --rm dfin --help
#   docker run --rm -v $PWD/models:/models -p 8080:8080 dfin \
#       pod --model /models/tiny-llama-f32.gguf --split 2,5 --listen 0.0.0.0:8080
#
# Stage 1 builds the vendored, layer-range patched llama.cpp (Release, CPU backend) and
# the Rust workspace; stage 2 is a Debian slim image with just the `dfin` binary, the
# llama.cpp shared libraries and their two runtime dependencies (libstdc++, libgomp).
#
# GPU variants (not implemented here yet):
#   Vulkan: build on an image with the Vulkan SDK (for example ubuntu + lunarg packages,
#           or `vulkan-sdk` from the LunarG apt repository), run the build script with
#           `GGML_VULKAN=ON`, and add `libvulkan1` plus the vendor ICD (mesa-vulkan-drivers
#           for AMD/Intel, the NVIDIA ICD comes from the container toolkit) to the runtime
#           image. The runtime needs `--device /dev/dri` (AMD/Intel) or `--gpus all`.
#   CUDA:   build FROM nvidia/cuda:<ver>-devel-ubuntu24.04 with rustup installed, run the
#           build script with `GGML_CUDA=ON` (optionally CMAKE_CUDA_ARCHITECTURES for the
#           target GPUs), and use nvidia/cuda:<ver>-runtime-ubuntu24.04 as the runtime base;
#           run with `--gpus all`. libggml-cuda.so is picked up by the same COPY below.
# Either variant can be a second Dockerfile or a build arg selecting the base images; the
# two scripts already accept GGML_VULKAN / GGML_CUDA, so the build steps do not change.

FROM rust:1-trixie AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake g++ git make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Layer-range patch (idempotent; the script also copes with a CRLF checkout from Windows,
# since the build context carries no .git for git apply to convert through), then
# llama.cpp. GGML_NATIVE=OFF keeps the CPU backend portable (AVX2/FMA/F16C baseline on
# x86-64, NEON on arm64) instead of -march=native of whatever machine ran `docker build`.
RUN scripts/apply-llama-patch.sh \
    && GGML_NATIVE=OFF scripts/build-llama.sh

# The workspace. build.rs links against vendor/llama.cpp/build and copies the .so files
# next to the binary; the runtime image collects them from the llama.cpp build tree.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release --locked --workspace \
    && strip target/release/dfin \
    && mkdir -p /out/lib \
    && cp target/release/dfin /out/dfin \
    && cp -a vendor/llama.cpp/build/bin/*.so* /out/lib/ \
    && find /out/lib -type f -name '*.so*' -exec strip --strip-unneeded {} +

FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/lib/ /usr/local/lib/
COPY --from=builder /out/dfin /usr/local/bin/dfin
RUN ldconfig

ENTRYPOINT ["dfin"]
CMD ["--help"]
