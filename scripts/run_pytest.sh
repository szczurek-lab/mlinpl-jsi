#!/bin/bash
# Wrapper script for pytest.
#
# Always run pytest via mamba/micromamba to ensure a consistent environment.
# This avoids hardcoding a Python executable path and avoids relying on the
# currently-activated shell environment.

set -euo pipefail

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/run_pytest.sh --prefix /absolute/path/to/conda-env [-- <pytest args...>]

Notes:
  - This script MUST run pytest via mamba/micromamba (not via an activated shell).
  - For Cursor "Source Control" commits, the simplest persistent option is:
      git config joint-improvement.envPrefix /absolute/path/to/conda-env
    Then you can run this script with no --prefix.
EOF
}

resolve_mamba() {
    if command -v micromamba >/dev/null 2>&1; then
        echo "micromamba"
        return 0
    fi
    if command -v mamba >/dev/null 2>&1; then
        echo "mamba"
        return 0
    fi
    return 1
}

MAMBA_BIN="$(resolve_mamba || true)"
if [ -z "${MAMBA_BIN}" ]; then
    echo "Error: micromamba/mamba not found on PATH." >&2
    echo "Fix: install micromamba or mamba, then retry." >&2
    exit 1
fi

ENV_PREFIX=""
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prefix)
            if [ -z "${2:-}" ]; then
                echo "Error: --prefix/-p requires a path argument." >&2
                usage
                exit 2
            fi
            ENV_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PYTEST_ARGS+=("$@")
            break
            ;;
        *)
            # Forward any other args to pytest (allows e.g. -k, -q, etc.)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "${ENV_PREFIX}" ]; then
    # Allow manual configuration without editing tracked files:
    # - environment variable, or
    # - repo-local git config (works well for Cursor Source Control commits)
    ENV_PREFIX="${JOINT_IMPROVEMENT_ENV_PREFIX:-}"
fi
if [ -z "${ENV_PREFIX}" ]; then
    ENV_PREFIX="$(git -C "${PROJECT_ROOT}" config --get joint-improvement.envPrefix || true)"
fi

if [ -z "${ENV_PREFIX}" ]; then
    echo "Error: No env prefix specified." >&2
    usage
    exit 2
fi
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "Error: joint-improvement env prefix not found: ${ENV_PREFIX}" >&2
    echo "Create it with:" >&2
    echo "  bash scripts/setup_env.sh --prefix \"${ENV_PREFIX}\"" >&2
    exit 1
fi
if [ ! -d "${ENV_PREFIX}/conda-meta" ]; then
    echo "Error: '${ENV_PREFIX}' exists but does not look like a conda env (missing conda-meta/)." >&2
    exit 1
fi

echo "Using ${MAMBA_BIN} env prefix: ${ENV_PREFIX}" >&2
echo "Python version: $(${MAMBA_BIN} run -p "${ENV_PREFIX}" python --version 2>&1)" >&2
echo "Pytest version: $(${MAMBA_BIN} run -p "${ENV_PREFIX}" python -m pytest --version 2>&1)" >&2

# Run pytest with the environment's Python
export PYTHONPATH="$PROJECT_ROOT/src"
exec "${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pytest "$PROJECT_ROOT/tests/" -x --tb=short -p no:xdist "${PYTEST_ARGS[@]}"

