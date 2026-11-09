# Apply the layer-range patch to the vendored llama.cpp checkout (idempotent).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$src = Join-Path $root "vendor\llama.cpp"
if (-not (Test-Path (Join-Path $src "include\llama.h"))) { throw "vendor/llama.cpp is empty; run: git submodule update --init --depth 1" }
# Applied means this exact patch reverses cleanly; a tree that merely contains the patch's
# symbols may carry an older version of it.
git -C $src apply -R --check --quiet ..\patches\0001-layer-range.patch
if ($LASTEXITCODE -eq 0) { Write-Host "layer-range patch already applied (current version)"; exit 0 }
git -C $src apply --check --quiet ..\patches\0001-layer-range.patch
if ($LASTEXITCODE -ne 0) {
    throw "vendor/llama.cpp is neither clean nor carrying the current layer-range patch (an older version of it, or other local changes). Reset the submodule tree and run this script again: git -C vendor/llama.cpp checkout -- . ; Remove-Item -Recurse -Force vendor/llama.cpp/examples/layer-range"
}
git -C $src apply ..\patches\0001-layer-range.patch
Write-Host "layer-range patch applied"
