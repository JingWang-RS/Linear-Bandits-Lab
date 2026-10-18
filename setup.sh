#!/usr/bin/env bash
# Install a local, pinned runtime. Run from any directory; no global installs.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION=3.12.14
UV_VERSION=0.13.1
export UV_CACHE_DIR="$ROOT/.setup/cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.setup/python"
export UV_PYTHON_BIN_DIR="$ROOT/.setup/bin"

uv_matches() {
    local reported
    reported="$("$1" --version)"
    [[ "$reported" == "uv $UV_VERSION" || "$reported" == "uv $UV_VERSION "* ]]
}

if command -v uv >/dev/null 2>&1 && uv_matches uv; then
    UV="$(command -v uv)"
else
    UV="$ROOT/.setup/uv/uv"
    if [[ ! -x "$UV" ]] || ! uv_matches "$UV"; then
        mkdir -p "$ROOT/.setup"
        INSTALLER="$ROOT/.setup/uv-install.sh"
        URL="https://astral.sh/uv/$UV_VERSION/install.sh"
        if command -v curl >/dev/null 2>&1; then
            curl --proto '=https' --tlsv1.2 -fLsS --connect-timeout 15 --max-time 120 --retry 2 "$URL" -o "$INSTALLER"
        elif command -v wget >/dev/null 2>&1; then
            wget --https-only --timeout=30 --tries=2 -qO "$INSTALLER" "$URL"
        else
            echo "Setup needs curl or wget (or uv $UV_VERSION), and internet access." >&2
            exit 1
        fi
        # Official versioned installer; do not change shell profiles or global tools.
        UV_INSTALL_DIR="$ROOT/.setup/uv" UV_UNMANAGED_INSTALL="$ROOT/.setup/uv" sh "$INSTALLER"
        uv_matches "$UV" || { echo "Unexpected uv version after installation." >&2; exit 1; }
    fi
fi

if [[ -e "$ROOT/.venv" ]]; then
    if [[ ! -x "$ROOT/.venv/bin/python" ]] || \
       [[ "$("$ROOT/.venv/bin/python" -c 'import platform; print(platform.python_version())')" != "$PYTHON_VERSION" ]]; then
        echo "Existing .venv is not Python $PYTHON_VERSION. Rename it, then rerun setup.sh." >&2
        exit 1
    fi
else
    "$UV" venv --python "$PYTHON_VERSION" "$ROOT/.venv"
fi

"$UV" pip sync --python "$ROOT/.venv/bin/python" --only-binary=:all: "$ROOT/requirements.txt"
"$UV" pip check --python "$ROOT/.venv/bin/python"
printf '\nReady. From the repository root, run:\n  source .venv/bin/activate\n  python exp_nonstationary_LB.py\n  python exp_heavy-tailed_LB.py --workers 4\n'
