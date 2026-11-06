//! Links against a prebuilt llama.cpp tree and copies its shared libraries next to
//! the produced executables.
//!
//! The llama.cpp checkout lives in `vendor/llama.cpp` and is built separately with
//! CMake (see `scripts/build-llama.ps1` / `scripts/build-llama.sh`). Override the
//! build directory with `DFIN_LLAMA_BUILD_DIR` and the CMake configuration name
//! (multi-config generators only) with `DFIN_LLAMA_BUILD_CONFIG` (default `Release`).
//!
//! Bindings are pre-generated into `src/bindings.rs` by `scripts/gen-bindings.ps1`
//! and must be regenerated whenever `vendor/llama.cpp/include/llama.h` changes.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .ancestors()
        .nth(2)
        .expect("crate lives in <root>/crates/<name>")
        .to_path_buf();
    let llama_src = workspace_root.join("vendor").join("llama.cpp");
    let build_dir = env::var_os("DFIN_LLAMA_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| llama_src.join("build"));
    let config = env::var("DFIN_LLAMA_BUILD_CONFIG").unwrap_or_else(|_| "Release".to_string());

    println!("cargo:rerun-if-env-changed=DFIN_LLAMA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=DFIN_LLAMA_BUILD_CONFIG");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bindings.rs");
    println!("cargo:rustc-check-cfg=cfg(has_layer_range)");

    // Detect whether the vendored llama.h carries the layer-range patch, so the safe
    // wrapper can compile against both a patched and an unpatched tree.
    let header = llama_src.join("include").join("llama.h");
    println!("cargo:rerun-if-changed={}", header.display());
    if let Ok(text) = fs::read_to_string(&header) {
        if text.contains("layer_begin") && text.contains("llama_model_layer_end") {
            println!("cargo:rustc-cfg=has_layer_range");
        } else {
            println!("cargo:warning=vendored llama.h lacks the layer-range patch; partial models are unavailable");
        }
    } else {
        println!(
            "cargo:warning=could not read {}; is the llama.cpp submodule checked out?",
            header.display()
        );
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        // MSVC multi-config layout: import libraries under <subdir>/<Config>/, DLLs in bin/<Config>/.
        let lib_dirs = [
            build_dir.join("src").join(&config),
            build_dir.join("ggml").join("src").join(&config),
        ];
        for d in &lib_dirs {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
        for lib in ["llama", "ggml", "ggml-base"] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
        let dll_dir = build_dir.join("bin").join(&config);
        copy_shared_libs(&dll_dir, "dll");
    } else {
        // Single-config layout: shared objects in bin/ (or lib/ on some setups).
        for d in [build_dir.join("bin"), build_dir.join("lib"), build_dir.join("src"), build_dir.join("ggml").join("src")] {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
        for lib in ["llama", "ggml", "ggml-base"] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
        let ext = if target_os == "macos" { "dylib" } else { "so" };
        copy_shared_libs(&build_dir.join("bin"), ext);
        if target_os == "macos" {
            // llama.cpp's dylibs carry @rpath install names; look next to the executable
            // first, then in the llama.cpp build tree.
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.join("bin").display());
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.join("bin").display());
        }
    }
}

/// Copy every shared library from `dir` into the cargo profile directory
/// (`target/<profile>/`), where executables look for them at run time.
fn copy_shared_libs(dir: &Path, ext: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // OUT_DIR = target/<profile>/build/<crate>-<hash>/out
    let profile_dir = match out_dir.ancestors().nth(3) {
        Some(p) => p.to_path_buf(),
        None => return,
    };
    let Ok(entries) = fs::read_dir(dir) else {
        println!(
            "cargo:warning=llama.cpp build directory {} not found; build it first (see README)",
            dir.display()
        );
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_lib = path.extension().and_then(|e| e.to_str()) == Some(ext)
            || (ext == "so" && path.to_string_lossy().contains(".so"));
        if !is_lib {
            continue;
        }
        let name = path.file_name().unwrap();
        // Re-run (and re-copy) whenever llama.cpp is rebuilt.
        println!("cargo:rerun-if-changed={}", path.display());
        // Executables live in target/<profile>/, test binaries in target/<profile>/deps/.
        let deps_dir = profile_dir.join("deps");
        let _ = fs::create_dir_all(&deps_dir);
        for dest in [profile_dir.join(name), deps_dir.join(name)] {
            let stale = match (fs::metadata(&path), fs::metadata(&dest)) {
                (Ok(a), Ok(b)) => a.len() != b.len() || a.modified().ok() > b.modified().ok(),
                _ => true,
            };
            if stale {
                if let Err(e) = fs::copy(&path, &dest) {
                    println!("cargo:warning=failed to copy {} to {}: {e}", path.display(), dest.display());
                }
            }
        }
    }
}
