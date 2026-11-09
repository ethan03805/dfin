#!/usr/bin/env bash
# Apply the layer-range patch to the vendored llama.cpp checkout. Idempotent: a tree
# that already carries the current patch is left alone; a tree that carries anything
# else the patch does not apply to is refused, with the command that resets it.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/vendor/llama.cpp"
if [ ! -f include/llama.h ]; then
    echo "vendor/llama.cpp is empty; run: git submodule update --init --depth 1" >&2
    exit 1
fi
patch=../patches/0001-layer-range.patch
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # A plain tree without .git (a Docker build context) that was checked out on Windows
    # with core.autocrlf=true has CRLF line endings, which the LF patch cannot match;
    # inside a repository git apply converts them itself. Normalise the touched files.
    grep '^+++ b/' "$patch" | sed 's#^+++ b/##' | while read -r f; do
        if [ -f "$f" ]; then
            tr -d '\r' < "$f" > "$f.lf" && mv "$f.lf" "$f"
        fi
    done
fi
# Applied means this exact patch reverses cleanly. A tree that merely contains the
# patch's symbols may carry an older version of it, which builds a libllama the Rust
# side no longer matches.
if git apply -R --check "$patch" 2>/dev/null; then
    echo "layer-range patch already applied (current version)"
    exit 0
fi
if ! git apply --check "$patch" 2>/dev/null; then
    cat >&2 <<'EOF2'
vendor/llama.cpp is neither clean nor carrying the current layer-range patch: it has
an older version of the patch or other local changes, and the current patch does not
apply on top of them. Reset the submodule tree, then run this script again and rebuild:

    git -C vendor/llama.cpp checkout -- . && rm -rf vendor/llama.cpp/examples/layer-range
    scripts/apply-llama-patch.sh
    scripts/build-llama.sh
EOF2
    exit 1
fi
git apply "$patch"
echo "layer-range patch applied to $(git rev-parse --short HEAD 2>/dev/null || echo 'a plain tree')"
