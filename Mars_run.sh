#!/usr/bin/env bash
# ==============================================================================
# GALE CRATER 5-HAZARD EVA SIMULATOR — CROSS-PLATFORM LAUNCHER
# Works on: Linux, macOS, Windows (Git Bash / WSL / MSYS2)
# Usage:
#   ./run_mars.sh           -> Launches Python Backend + 3D Desktop App (LWJGL3)
#   ./run_mars.sh --web     -> Launches Python Backend + 3D WebAssembly Server (port 8181)
#   ./run_mars.sh --js      -> Launches Python Backend + TeaVM JS Dev Server
#   ./run_mars.sh --backup  -> Uses mars_nav.py instead of mars_backend_server.py
# ==============================================================================

set -e

# Always run relative to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="desktop"
USE_BACKUP=0

for arg in "$@"; do
    case "$arg" in
        --web|--wasm) MODE="wasm" ;;
        --js)         MODE="js" ;;
        --backup)     USE_BACKUP=1 ;;
        --desktop)    MODE="desktop" ;;
    esac
done

echo "============================================================"
echo " MARS EVA CO-PILOT — UNIVERSAL LAUNCHER (OS: $(uname -s))"
echo "============================================================"

# ------------------------------------------------------------------------------
# 1. DETECT PYTHON INTERPRETER (Linux / macOS / Windows)
# ------------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in python3 python py; do
    if command -v "$candidate" >/dev/null 2>&1; then
        # Verify it is Python 3 and not Windows Store stub
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info.major == 3 else 1)" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "        Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

echo "[1/4] Found Python: $($PYTHON_BIN --version)"

# ------------------------------------------------------------------------------
# 2. CREATE VIRTUAL ENVIRONMENT & PULL PYTHON DEPENDENCIES
# ------------------------------------------------------------------------------
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating Python virtual environment in .venv ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate venv across Linux/macOS (.venv/bin) or Windows (.venv/Scripts)
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    VENV_PY="$VENV_DIR/bin/python"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
    VENV_PY="$VENV_DIR/Scripts/python.exe"
else
    VENV_PY="$PYTHON_BIN"
fi

# Generate requirements.txt if missing so dependencies are tracked cleanly
REQ_FILE="$SCRIPT_DIR/requirements.txt"
cat << 'EOF' > "$REQ_FILE"
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
requests>=2.31.0
matplotlib>=3.7.0
flask>=3.0.0
flask-cors>=4.0.0
paho-mqtt>=1.6.1
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
EOF

echo "[2/4] Pulling & verifying Python dependencies (CPU-safe)..."
"$VENV_PY" -m pip install --upgrade pip --quiet

# On Linux/Windows, install CPU-only PyTorch wheel first if torch isn't installed yet
# (prevents downloading a 2.5 GB CUDA bundle when we run on CPU anyway)
if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    echo "      Installing lightweight CPU PyTorch..."
    "$VENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
    "$VENV_PY" -m pip install torch torchvision
fi

"$VENV_PY" -m pip install -r "$REQ_FILE"

# ------------------------------------------------------------------------------
# 3. START PYTHON BACKEND (mars_backend_server.py with fallback to mars_nav.py)
# ------------------------------------------------------------------------------
BACKEND_PID=""
cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping Mars EVA Backend..."
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [ "$USE_BACKUP" -eq 1 ]; then
    echo "[3/4] Starting Backup Backend (mars_nav.py)..."
    "$VENV_PY" "$SCRIPT_DIR/mars_nav.py" 3 &
    BACKEND_PID=$!
else
    echo "[3/4] Starting Main Backend (mars_backend_server.py) on http://localhost:5000 ..."
    "$VENV_PY" "$SCRIPT_DIR/mars_backend_server.py" &
    BACKEND_PID=$!
fi

# Wait up to 25 seconds for http://localhost:5000/api/state to come online
echo "      Waiting for backend server on port 5000..."
for _ in $(seq 1 25); do
    if "$VENV_PY" -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/state', timeout=1)" >/dev/null 2>&1; then
        echo "      Backend is ONLINE at http://localhost:5000 !"
        break
    fi
    # If main backend crashed unexpectedly, automatically fail over to mars_nav.py
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "[WARN] Main backend exited early. Failing over to backup mars_nav.py..."
        "$VENV_PY" "$SCRIPT_DIR/mars_nav.py" 3 &
        BACKEND_PID=$!
        sleep 2
        break
    fi
    sleep 1
done

# ------------------------------------------------------------------------------
# 4. CHECK JAVA 21+ & LAUNCH GRADLE FRONTEND (gale-frontend)
# ------------------------------------------------------------------------------
cd "$SCRIPT_DIR/gale-frontend"

# Choose correct Gradle wrapper for OS
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        GRADLE_CMD="./gradlew.bat"
        ;;
    *)
        chmod +x ./gradlew
        GRADLE_CMD="./gradlew"
        ;;
esac

echo "[4/4] Pulling Java/LibGDX dependencies & launching 3D Gale Crater Frontend ($MODE)..."
echo "      Controls in 3D Window:"
echo "        - Right-Click (RMB):       Set Point A (Start)"
echo "        - Shift+RMB / Middle-Click: Set Point B (Target)"
echo "        - Keys [1] [2] [3]:        Switch NASA Preset Targets"
echo "        - Keys [R][S][D][T][A][C]: Toggle Wokwi Hazards Live"
echo "============================================================"

if [ "$MODE" = "wasm" ]; then
    echo ">>> Opening TeaVM WebAssembly release on http://localhost:8181 ..."
    "$GRADLE_CMD" :teavm:gdx_teavm_web_wasm_release_run
elif [ "$MODE" = "js" ]; then
    echo ">>> Opening TeaVM JS Dev Server on http://localhost:8080 ..."
    "$GRADLE_CMD" :teavm:gdx_teavm_web_js_run
else
    # Default: Run native 3D desktop viewer (LWJGL3)
    "$GRADLE_CMD" :lwjgl3:run
fi
