# Sourced by the benchmark and check scripts here, from the repository root. Resolves the
# dfin binary, Python, curl and the process tools on Linux, macOS and Windows Git Bash.
#
# Overrides: DFIN_BIN (the dfin binary), PYTHON (the interpreter).

case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) DFIN_OS=windows ;;
    Darwin) DFIN_OS=macos ;;
    *) DFIN_OS=linux ;;
esac

EXE=""
[ "$DFIN_OS" = windows ] && EXE=.exe

# The release binary cargo built.
B="${DFIN_BIN:-target/release/dfin$EXE}"

# Python with numpy and gguf: the project venv if there is one, else whatever is on PATH.
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

# These scripts were written against the Windows curl.exe; keep using it there.
CURL=curl
if [ "$DFIN_OS" = windows ] && command -v curl.exe >/dev/null 2>&1; then
    CURL=curl.exe
fi

# Where the llama.cpp build put its tools: bin/Release with a multi-config generator
# (Visual Studio), bin otherwise.
LLAMA_BIN=vendor/llama.cpp/build/bin
[ -d "$LLAMA_BIN/Release" ] && LLAMA_BIN="$LLAMA_BIN/Release"

need_dfin() {
    if [ ! -x "$B" ]; then
        echo "no dfin binary at $B; run: cargo build --release --locked" >&2
        exit 1
    fi
}

# The name a client serves a model file under when it is given no --model-name: the
# file name without its last extension. Requests must name this or no model at all.
model_name() {
    local base
    base="$(basename "$1")"
    echo "${base%.*}"
}

# Remove terminal colour codes from a log. The escape byte is written by bash, since
# BSD sed (macOS) does not read \x1b.
strip_ansi() {
    sed $'s/\x1b\\[[0-9;]*m//g'
}

ping_host() {
    if [ "$DFIN_OS" = windows ]; then
        ping -n 10 "$1"
    else
        ping -c 10 "$1"
    fi
}

# Stop processes this script started. Ctrl-C first, so a pod stops its own children;
# then a plain kill for whatever did not react.
stop_pids() {
    kill -INT "$@" 2>/dev/null
    local pid waited=0 alive=1
    while [ "$alive" = 1 ] && [ "$waited" -lt 5 ]; do
        alive=0
        for pid in "$@"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" = 1 ] && sleep 1
        waited=$((waited + 1))
    done
    kill "$@" 2>/dev/null
    return 0
}

# Stop every dfin process started from this binary, including a pod's children that
# outlived it.
stop_all_dfin() {
    if [ "$DFIN_OS" = windows ]; then
        taskkill //F //IM dfin.exe >/dev/null 2>&1
    else
        pkill -f "$B " >/dev/null 2>&1
    fi
    return 0
}

# One line per running dfin process: pid, resident memory in MB, role and layers.
dfin_memory() {
    if [ "$DFIN_OS" = windows ]; then
        powershell.exe -NoProfile -Command "Get-Process dfin | Sort-Object Id | ForEach-Object { '{0,8}  WS {1,7:N0} MB  private {2,7:N0} MB  cmd: {3}' -f \$_.Id, (\$_.WorkingSet64/1MB), (\$_.PrivateMemorySize64/1MB), ((Get-CimInstance Win32_Process -Filter ('ProcessId=' + \$_.Id)).CommandLine -replace '.*dfin(\.exe)? ','' -replace ' --model \S+','' -replace ' --listen \S+','' -replace ' --n-ctx.*','') }"
    else
        ps -eo pid=,rss=,args= | awk -v bin="$B" '
            index($0, bin " ") {
                pid = $1; rss = $2
                cmd = substr($0, index($0, bin " ") + length(bin) + 1)
                sub(/ --model [^ ]+/, "", cmd)
                sub(/ --listen [^ ]+/, "", cmd)
                sub(/ --n-ctx.*/, "", cmd)
                printf "%8d  RSS %7.0f MB  cmd: %s\n", pid, rss / 1024, cmd
            }'
    fi
}

system_memory() {
    case "$DFIN_OS" in
        windows)
            powershell.exe -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | ForEach-Object { 'system RAM: total {0:N0} MB, free {1:N0} MB' -f (\$_.TotalVisibleMemorySize/1KB), (\$_.FreePhysicalMemory/1KB) }"
            ;;
        macos)
            vm_stat | awk -v total="$(sysctl -n hw.memsize)" '
                /page size of/ { page = $8 }
                /^Pages free/ { gsub(/\./, "", $3); free = $3 }
                END { printf "system RAM: total %.0f MB, free %.0f MB\n", total / 1048576, free * page / 1048576 }'
            ;;
        *)
            free -m | awk '/^Mem:/ { printf "system RAM: total %d MB, free %d MB\n", $2, $4 }'
            ;;
    esac
}
