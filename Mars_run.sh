#!/usr/bin/env bash
# ==============================================================================
# GALE CRATER 5-HAZARD EVA SIMULATOR — CROSS-PLATFORM LAUNCHER
# Handles "gale-frontend (new).zip", "gale-frontend.zip", "Wokwi.zip",
# installs dependencies, injects the updated 3D Java + Python code,
# starts the Python Backend, and launches the 3D Gale Crater Frontend.
# ==============================================================================

set -e

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
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info.major == 3 else 1)" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

echo "[1/5] Found Python: $($PYTHON_BIN --version)"

# ------------------------------------------------------------------------------
# 2. CREATE VIRTUAL ENVIRONMENT & PULL PYTHON DEPENDENCIES
# ------------------------------------------------------------------------------
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating Python virtual environment in .venv ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

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

echo "[2/5] Verifying Python dependencies..."
"$VENV_PY" -m pip install --upgrade pip --quiet

if ! "$VENV_PY" -c "import torch" >/dev/null 2>&1; then
    echo "      Installing lightweight CPU PyTorch..."
    "$VENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
    "$VENV_PY" -m pip install torch torchvision
fi

"$VENV_PY" -m pip install -r "$REQ_FILE" --quiet

# ------------------------------------------------------------------------------
# 3. LOCATE OR UNZIP "gale-frontend (new).zip" / "gale-frontend.zip" / "Wokwi.zip"
# ------------------------------------------------------------------------------
echo "[3/5] Checking archives and project directories..."

"$VENV_PY" - << 'PYEOF'
import zipfile
from pathlib import Path

root = Path(".").resolve()

# 1. Unpack Wokwi.zip if present and not unpacked
for wokwi_zip in root.glob("*[Ww]okwi*.zip"):
    if not (root / "Wokwi").exists() and not (root / "project(1)").exists():
        print(f"      Extracting {wokwi_zip.name} ...")
        with zipfile.ZipFile(wokwi_zip, "r") as zf:
            zf.extractall(root / "Wokwi")

# 2. Unpack gale-frontend (new).zip or gale-frontend.zip if gradlew not yet extracted
has_gradlew = any(root.rglob("gradlew"))
if not has_gradlew:
    frontend_zips = sorted(root.glob("*gale-frontend*.zip"))
    if frontend_zips:
        target_zip = frontend_zips[0]
        print(f"      Extracting '{target_zip.name}' ...")
        with zipfile.ZipFile(target_zip, "r") as zf:
            zf.extractall(root)
        # If zip extracted core/ and build.gradle directly without gale-frontend/ wrapper, organize or detect it
PYEOF

FRONTEND_DIR=""
if [ -d "$SCRIPT_DIR/gale-frontend" ] && [ -f "$SCRIPT_DIR/gale-frontend/build.gradle" ]; then
    FRONTEND_DIR="$SCRIPT_DIR/gale-frontend"
elif [ -f "$SCRIPT_DIR/build.gradle" ] && [ -d "$SCRIPT_DIR/core" ]; then
    FRONTEND_DIR="$SCRIPT_DIR"
else
    FOUND_GRADLEW="$(find "$SCRIPT_DIR" -maxdepth 3 -name "gradlew" | head -n 1)"
    if [ -n "$FOUND_GRADLEW" ]; then
        FRONTEND_DIR="$(dirname "$FOUND_GRADLEW")"
    fi
fi

if [ -z "$FRONTEND_DIR" ] || [ ! -d "$FRONTEND_DIR" ]; then
    echo "[ERROR] Could not find gale-frontend directory or *gale-frontend*.zip in $SCRIPT_DIR"
    ls -la "$SCRIPT_DIR"
    exit 1
fi

echo "      Frontend located at: $FRONTEND_DIR"

# If updated GalePanoramaApp.java is placed in root, copy it into the extracted frontend automatically
JAVA_TARGET="$FRONTEND_DIR/core/src/main/java/com/gale/viewer/GalePanoramaApp.java"
if [ -f "$SCRIPT_DIR/GalePanoramaApp.java" ] && [ -d "$(dirname "$JAVA_TARGET")" ]; then
    cp "$SCRIPT_DIR/GalePanoramaApp.java" "$JAVA_TARGET"
    echo "      Synced updated GalePanoramaApp.java into frontend."
fi

# ------------------------------------------------------------------------------
# 4. START PYTHON BACKEND (mars_backend_server.py with fallback to mars_nav.py)
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

if [ "$USE_BACKUP" -eq 1 ] && [ -f "$SCRIPT_DIR/mars_nav.py" ]; then
    echo "[4/5] Starting Backup Backend (mars_nav.py)..."
    "$VENV_PY" "$SCRIPT_DIR/mars_nav.py" 3 &
    BACKEND_PID=$!
else
    echo "[4/5] Starting Main Backend (mars_backend_server.py) on http://localhost:5000 ..."
    "$VENV_PY" "$SCRIPT_DIR/mars_backend_server.py" &
    BACKEND_PID=$!
fi

echo "      Waiting for backend server on port 5000..."
for _ in $(seq 1 25); do
    if "$VENV_PY" -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/state', timeout=1)" >/dev/null 2>&1; then
        echo "      Backend is ONLINE at http://localhost:5000 !"
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        if [ -f "$SCRIPT_DIR/mars_nav.py" ]; then
            echo "[WARN] Main backend exited early. Failing over to backup mars_nav.py..."
            "$VENV_PY" "$SCRIPT_DIR/mars_nav.py" 3 &
            BACKEND_PID=$!
            sleep 2
        fi
        break
    fi
    sleep 1
done

# ------------------------------------------------------------------------------
# 5. LAUNCH GRADLE 3D FRONTEND
# ------------------------------------------------------------------------------
cd "$FRONTEND_DIR"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        GRADLE_CMD="./gradlew.bat"
        ;;
    *)
        chmod +x ./gradlew
        GRADLE_CMD="./gradlew"
        ;;
esac

echo "[5/5] Pulling Java/LibGDX dependencies & launching 3D Gale Crater Frontend ($MODE)..."
echo "      Controls in 3D Window:"
echo "        - Right-Click (RMB):        Set Point A (Start)"
echo "        - Shift+RMB / Middle-Click: Set Point B (Target)"
echo "        - Keys [1] [2] [3]:         Switch NASA Preset Targets"
echo "        - Keys [R][S][D][T][A][C]:  Toggle Wokwi Hazards Live"
echo "============================================================"

if [ "$MODE" = "wasm" ]; then
    echo ">>> Opening TeaVM WebAssembly release on http://localhost:8181 ..."
    "$GRADLE_CMD" :teavm:gdx_teavm_web_wasm_release_run
elif [ "$MODE" = "js" ]; then
    echo ">>> Opening TeaVM JS Dev Server on http://localhost:8080 ..."
    "$GRADLE_CMD" :teavm:gdx_teavm_web_js_run
else
    "$GRADLE_CMD" :lwjgl3:run
fi
