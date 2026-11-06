# Apply the layer-range patch to the vendored llama.cpp checkout (idempotent).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$src = Join-Path $root "vendor\llama.cpp"
if (-not (Test-Path (Join-Path $src "include\llama.h"))) { throw "vendor/llama.cpp is empty; run: git submodule update --init --depth 1" }
if (Select-String -Path (Join-Path $src "include\llama.h") -Pattern "llama_model_layer_end" -Quiet) { Write-Host "layer-range patch already applied"; exit 0 }
git -C $src apply --check ..\patches\0001-layer-range.patch
if ($LASTEXITCODE -ne 0) { throw "patch does not apply cleanly" }
git -C $src apply ..\patches\0001-layer-range.patch
Write-Host "layer-range patch applied"
