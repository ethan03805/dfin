# Configure and build the vendored llama.cpp (with the layer-range patch) on Windows.
#
#   .\scripts\build-llama.ps1            # Vulkan + CPU, Release
#   .\scripts\build-llama.ps1 -NoVulkan  # CPU only
#   .\scripts\build-llama.ps1 -Targets llama,llama-simple
#
# Requires Visual Studio 2022 (C++ workload), CMake 3.21+, and the Vulkan SDK for -Vulkan.
# Important: keep the checkout on a short path (e.g. a `subst` drive). MSVC builds break
# beyond 260-character paths.
param(
    [switch]$NoVulkan,
    [string]$Config = "Release",
    [string[]]$Targets = @(),
    [int]$Jobs = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$src = Join-Path $root "vendor\llama.cpp"
$build = Join-Path $src "build"
if (-not (Test-Path (Join-Path $src "CMakeLists.txt"))) {
    throw "vendor/llama.cpp is missing. Clone it: git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp (see vendor/README.md for the pinned commit and patch)"
}
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }
$vulkan = if ($NoVulkan) { "OFF" } else { "ON" }

cmake -S $src -B $build -G "Visual Studio 17 2022" -A x64 `
    -DGGML_VULKAN=$vulkan `
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF `
    -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

$args = @("--build", $build, "--config", $Config, "-j", "$Jobs")
if ($Targets.Count -gt 0) { $args += "--target"; $args += $Targets }
cmake @args
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
Write-Host "llama.cpp built: $build\bin\$Config"
