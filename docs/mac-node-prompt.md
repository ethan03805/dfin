# Prompt for the Claude session on the MacBook Air

Copy everything below the line into a new Claude Code session on the Mac.

---

Set this MacBook Air up as a **stage node** of my pipeline-parallel inference project and
leave the stage running so a client on my Windows PC (LAN address 10.0.0.105) can connect
to it. Work autonomously; only stop for things that need my click (Xcode tools install,
firewall prompt) or my decision. Report at the end with the exact lines listed in step 8.

**Background.** The project is `dfin`: a model is cut into contiguous layer ranges, each
machine runs one range, hidden states flow over TCP. The repo is private at
https://github.com/ethan03805/dfin (I am logged in as ethan03805; use `gh auth login` if
cloning asks for credentials). `README.md` explains the layout, `docs/design-v0.1.md` the
design, `vendor/README.md` the llama.cpp patch. The Mac only needs to serve layers
`1:N` of GLM-4.7-Flash; it never holds the whole model in memory, but the GGUF file
(18.5 GB) must be byte-identical to the one on the PC because the handshake checks a
fingerprint.

1. **Inventory.** Report chip (`sysctl -n machdep.cpu.brand_string`), RAM in GB
   (`sysctl -n hw.memsize`), free disk (`df -h ~`), macOS version, and whether
   `xcode-select -p`, `cmake`, `cargo`, `git`, `python3` exist. Install what is missing:
   Xcode command line tools (`xcode-select --install`, I will click the dialog), Homebrew
   if absent, `brew install cmake`, Rust via `curl https://sh.rustup.rs -sSf | sh -s -- -y`
   then `source ~/.cargo/env`. At least 25 GB of free disk is needed.

2. **Source.**
   ```bash
   git clone --recurse-submodules --shallow-submodules https://github.com/ethan03805/dfin.git ~/dfin
   cd ~/dfin && scripts/apply-llama-patch.sh
   ```
   If the shallow submodule fetch fails, run `git submodule update --init` (full clone of
   llama.cpp) and then the patch script again. The script must print that the patch was
   applied (or already applied).

3. **Build.** `scripts/build-llama.sh` (Metal is on by default on Apple Silicon), then
   `cargo build --release`. The Rust build links against `vendor/llama.cpp/build` and
   copies the dylibs next to the binary. This code was developed on Windows and has not
   been built on macOS before, so expect small platform issues: dylib names or rpaths in
   `crates/dfin-llama-sys/build.rs`, the FFI layout tests (`cargo test -p dfin-llama-sys`;
   if they fail, `cargo install bindgen-cli` and run `scripts/gen-bindings.sh`), compiler
   warnings treated differently. Fix what is needed, keep fixes minimal, and commit them
   on a local branch `mac-node` (do not push unless I say so).

4. **Verify with the tiny test model** before touching the big one:
   ```bash
   python3 -m venv .venv && .venv/bin/pip install numpy -e vendor/llama.cpp/gguf-py
   .venv/bin/python tools/make_tiny_llama.py models/tiny-llama-f32.gguf
   cargo test --release
   ```
   All tests should pass; `tests/e2e.rs` and `tests/stage_proto.rs` spawn real processes.
   Then run `target/release/dfin bench --model models/tiny-llama-f32.gguf --split 3,6 --n-gpu-layers -1`
   and paste the table (this exercises the Metal path with partial models).

5. **Model.** Download to the exact path below and check the size is 18474983296 bytes:
   ```bash
   mkdir -p models/glm-4.7-flash && cd models/glm-4.7-flash
   curl -L -C - --retry 8 --retry-delay 15 -o zai-org_GLM-4.7-Flash-Q4_K_M.gguf \
     https://huggingface.co/bartowski/zai-org_GLM-4.7-Flash-GGUF/resolve/main/zai-org_GLM-4.7-Flash-Q4_K_M.gguf
   ls -l zai-org_GLM-4.7-Flash-Q4_K_M.gguf && cd ~/dfin
   ```
   Start this early (it takes 10 to 30 minutes) and do steps 3 and 4 while it runs.

6. **Choose the layer range from RAM.** The model has 47 layers of about 0.39 GB each at
   this quantization; layer 0 is dense and stays on the PC. Use `1:9` for 8 GB, `1:21`
   for 16 GB, `1:31` for 24 GB, `1:41` for 32 GB or more. Call the upper bound N.

7. **Run the stage** and leave it running (a separate terminal tab, or `nohup ... &` with
   output to `stage.log`):
   ```bash
   target/release/dfin stage --model models/glm-4.7-flash/zai-org_GLM-4.7-Flash-Q4_K_M.gguf \
     --layers 1:N --listen 0.0.0.0:7001 --n-gpu-layers -1 --n-ctx 8192 --n-seq-max 2 \
     --act-dtype f16 --log info
   ```
   Replace `1:N` with the range from step 6. Wait for the line `stage ready: layers 1:N`.
   If macOS asks whether `dfin` may accept incoming network connections, tell me so I can
   click Allow (or allow it under System Settings, Network, Firewall, Options). Check the
   port is open with `nc -z 127.0.0.1 7001 && echo open`.

8. **Report**, each on its own line: the Mac's LAN IP (`ipconfig getifaddr en0`, or `en1`
   if en0 is empty), the port (7001), the layer range `1:N`, RAM and chip, free disk after
   the download, the `dfin` git commit (`git rev-parse --short HEAD`) and any files you
   changed, the tiny-model bench table from step 4, and the stage's startup log lines
   about model size and Metal buffers. Then keep the stage running.


---

## Follow-up prompt: update the Mac stage to phase 1 (6 November 2026)

Copy everything below the line into the same Claude session on the Mac.

---

The `dfin` project moved from raw TCP to an authenticated QUIC transport with node
identities (branch `main`, commit 897bed2 or later). Update the stage on this Mac and
leave it running. Work autonomously; only stop for a firewall click or a decision.

1. Stop the running `dfin stage` (find it with `pgrep -fl "dfin stage"`).
2. In `~/dfin`: `git fetch origin && git checkout main && git pull`. If you had local
   commits on `mac-node` that are not on `main`, rebase them on top of `main` and tell me
   which files they touch. Then `cargo build --release` (llama.cpp does not need
   rebuilding; the patch did not change). Run `cargo test --release` with the tiny model.
3. Start the stage with the same model, layer range and flags as before, plus an identity
   file and locked weights:
   ```bash
   target/release/dfin stage --model models/glm-4.7-flash/zai-org_GLM-4.7-Flash-Q4_K_M.gguf \
     --layers 1:N --listen 0.0.0.0:7001 --identity ~/dfin/mac-stage.key --mlock \
     --n-gpu-layers -1 --n-ctx 8192 --n-seq-max 2 --act-dtype f16 --log info
   ```
   The stage is now QUIC over UDP: if macOS asks about incoming connections for `dfin`,
   tell me so I can click Allow. `nc -z` will not see a UDP listener; use the log line
   `stage ready` instead.
4. Report, each on its own line: the line `node id <64 hex characters>` from the stage
   log, the Mac's LAN IP, the port, the layer range, the git commit, and the `cargo test`
   result. Then keep the stage running.
