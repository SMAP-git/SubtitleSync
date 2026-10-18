#!/usr/bin/env bash
# Run with: bash Start_Subtitle_Sync.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

fail() { printf '\n%s\n' "$*" >&2; exit 1; }
guidance() {
    echo "Linux prerequisites: a graphical desktop and the X11/OpenGL system libraries."
    echo "For system Python on Debian/Ubuntu: sudo apt install python3 python3-venv python3-tk"
    echo "Fedora: sudo dnf install python3 python3-tkinter"
    echo "No system packages or drivers will be installed by this launcher."
}
usable() {
    "$1" -c 'import sys, tkinter, venv, ensurepip; assert sys.version_info >= (3, 10); import _tkinter' >/dev/null 2>&1
}

if [[ -f PORTABLE ]]; then
    export SUBTITLE_SYNC_SETTINGS_DIR="$PWD/.config"
    export SUBTITLE_SYNC_MODEL_CACHE="$PWD/.model-cache"
fi

if [[ ! -x .venv-linux/bin/python ]]; then
    python=""
    if [[ -x .python/linux/python/bin/python3 ]] && usable "$PWD/.python/linux/python/bin/python3"; then
        python="$PWD/.python/linux/python/bin/python3"
    elif command -v python3 >/dev/null 2>&1 && usable "$(command -v python3)"; then
        python="$(command -v python3)"
    else
        guidance
        [[ "$(uname -m)" == x86_64 && "$(uname -s)" == Linux ]] ||
            fail "Automatic private Python currently supports Linux x86_64 only. Install system Python with Tkinter/venv."
        command -v curl >/dev/null 2>&1 || fail "Install curl to download private Python, or install system Python with Tkinter/venv."
        for tool in tar sha256sum mktemp; do
            command -v "$tool" >/dev/null 2>&1 || fail "Required tool missing: $tool"
        done
        echo "No suitable Python with Tkinter/venv found."
        echo "Download a private CPython 3.12.15 runtime (about 33 MB) from astral-sh/python-build-standalone into .python?"
        read -r -p "Download? [y/N] " answer || fail "No confirmation received. Run this launcher in a terminal."
        [[ "$answer" == y || "$answer" == Y ]] || fail "Cancelled. No Python was installed."
        mkdir -p .python
        stage="$(mktemp -d "$PWD/.python/download-XXXXXX")"
        trap 'rm -f "$stage/runtime.tar.gz"; rmdir "$stage" 2>/dev/null || true' EXIT
        url="https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15%2B20261013-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
        curl --fail --location --retry 2 --connect-timeout 30 --output "$stage/runtime.tar.gz" "$url"
        echo "178d3d83b6e8e99285a65d3fce9c16b87952cc5efdb4d1a0e3ff3be18ff0fc9c  $stage/runtime.tar.gz" | sha256sum --check -
        mkdir -p .python/linux
        tar -xzf "$stage/runtime.tar.gz" -C .python/linux
        rm -f "$stage/runtime.tar.gz"
        rmdir "$stage"
        trap - EXIT
        python="$PWD/.python/linux/python/bin/python3"
        usable "$python" || { guidance; fail "Private Python cannot run on this distribution. Check missing system libraries above."; }
    fi
    echo "Creating the app's private environment..."
    "$python" -m venv .venv-linux || fail "Cannot create environment. Check folder permissions and Python venv support."
fi

usable "$PWD/.venv-linux/bin/python" || { guidance; fail "The existing environment cannot run. Check its Python/Tkinter dependencies or extract a fresh release folder."; }
echo "Checking app dependencies (first launch requires internet)..."
if ! .venv-linux/bin/python -c 'import fastembed, onnxruntime; assert fastembed.__version__ == "0.9.0"'; then
    .venv-linux/bin/python -m pip install --disable-pip-version-check -r requirements.txt ||
        fail "Dependency installation failed. Check internet access and the errors above."
fi
if ! .venv-linux/bin/python -c 'import imageio_ffmpeg'; then
    .venv-linux/bin/python -m pip install --disable-pip-version-check --no-deps -r requirements.txt ||
        fail "Could not install bundled audio support."
fi
if [[ "${1:-}" == --restore-libraries ]]; then
    .venv-linux/bin/python library_updates.py restore
fi
if [[ "${1:-}" == --restore-components ]]; then
    .venv-linux/bin/python library_updates.py restore-components
fi
app_python="$(.venv-linux/bin/python library_updates.py launch-python)" ||
    fail "Cannot select app libraries. Review the error above; try --restore-libraries."
exec "$app_python" subtitle_sync_app.pyw
