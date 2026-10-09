#!/usr/bin/env bash
set -Eeuo pipefail

USAGE="Usage: $0 [--prefix|-p PATH] [--extra NAME]... [--cpu]
  --prefix, -p PATH   Optional: Path where the environment should be created
  --extra NAME        Optional, repeatable: extra from pyproject.toml to install
                      (dev, notebook, chemprop, docking-cpu, docking-gpu, peptides)
  --cpu               Optional: install CPU-only PyTorch on Linux (default: CUDA 12.6 if an NVIDIA GPU is found)"

# ---------------------------------------------------------------------------
# Output helpers (colors only on a terminal; respects NO_COLOR)
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

STEP=0
TOTAL_STEPS=5
STEP_TITLE="parsing arguments"
STEP_START=0

fmt_duration() {
  local secs=$1
  if [ "$secs" -ge 60 ]; then
    printf '%dm %ds' $((secs / 60)) $((secs % 60))
  else
    printf '%ds' "$secs"
  fi
}

banner() {
  printf '\n%s==========================================%s\n' "$BOLD" "$RESET"
  printf '%s %s%s\n' "$BOLD" "$1" "$RESET"
  printf '%s==========================================%s\n' "$BOLD" "$RESET"
}

step() {
  STEP=$((STEP + 1))
  STEP_TITLE="$1"
  STEP_START=$SECONDS
  printf '\n%s[%d/%d] %s%s\n' "$BOLD$BLUE" "$STEP" "$TOTAL_STEPS" "$STEP_TITLE" "$RESET"
}

step_done() {
  printf '  %sdone in %s%s\n' "$DIM" "$(fmt_duration $((SECONDS - STEP_START)))" "$RESET"
}

info() { printf '  %s\n' "$*"; }
ok() { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }

# Print a command (dimmed) and then run it
run() {
  printf '  %s$ %s%s\n' "$DIM" "$*" "$RESET"
  "$@"
}

# fail MESSAGE [HINT_LINE]...: print an error with optional hints to stderr and exit
fail() {
  local message=$1
  shift
  printf '\n%s✗ Error:%s %s\n' "$RED$BOLD" "$RESET" "$message" >&2
  local line
  for line in "$@"; do
    printf '  %s\n' "$line" >&2
  done
  exit 1
}

usage_error() {
  printf '%s✗ Error:%s %s\n\n%s\n' "$RED$BOLD" "$RESET" "$1" "$USAGE" >&2
  exit 1
}

# Any unexpected command failure ends up here (errexit keeps the original exit code)
trap 'printf "\n%s✗ Setup failed during step %d/%d: %s%s\n  See the output above for details.\n" "$RED$BOLD" "$STEP" "$TOTAL_STEPS" "$STEP_TITLE" "$RESET" >&2' ERR

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
ENV_DIR=""
FORCE_CPU=false
EXTRAS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--prefix)
      if [ -z "${2:-}" ]; then
        usage_error "--prefix/-p requires a path argument"
      fi
      ENV_DIR="$2"
      shift 2
      ;;
    --extra)
      if [ -z "${2:-}" ]; then
        usage_error "--extra requires a name argument"
      fi
      EXTRAS+=("$2")
      shift 2
      ;;
    --cpu)
      FORCE_CPU=true
      shift
      ;;
    *)
      usage_error "Unknown option: $1"
      ;;
  esac
done

# Validate requested extras and note which need special handling
WITH_DEV=false
WITH_DOCKING=false
WITH_DOCKING_GPU=false
for extra in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do
  case "$extra" in
    dev) WITH_DEV=true ;;
    notebook|chemprop|peptides) ;;
    docking-cpu) WITH_DOCKING=true ;;
    docking-gpu) WITH_DOCKING=true; WITH_DOCKING_GPU=true ;;
    *)
      usage_error "unknown extra '$extra'"
      ;;
  esac
done

# Extras string for pip (e.g. "[notebook,dev]") and for display
if [ ${#EXTRAS[@]} -gt 0 ]; then
  EXTRAS_SPEC="[$(IFS=,; echo "${EXTRAS[*]}")]"
  EXTRAS_DESC="$(IFS=,; echo "${EXTRAS[*]}")"
else
  EXTRAS_SPEC=""
  EXTRAS_DESC="none (core only)"
fi

# Steps: prerequisites, environment, PyTorch, project, verification (+ Boost for docking-gpu)
if [ "$WITH_DOCKING_GPU" = true ]; then
  TOTAL_STEPS=6
fi

banner "Joint Self-Improvement: environment setup"

# ---------------------------------------------------------------------------
# Step: prerequisites
# ---------------------------------------------------------------------------
step "Checking prerequisites"

# Detect platform and pick the matching PyTorch build
OS="$(uname -s)"
ARCH="$(uname -m)"
GPU_NAME=""
if [ "$OS" = "Darwin" ]; then
  # macOS wheels (Apple Silicon, MPS) come from the default index
  TORCH_INDEX_ARGS=()
  TORCH_DESC="default macOS build"
elif [ "$FORCE_CPU" = false ] && command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  TORCH_INDEX_ARGS=(--index-url https://download.pytorch.org/whl/cu126)
  TORCH_DESC="CUDA 12.6 build"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
else
  TORCH_INDEX_ARGS=(--index-url https://download.pytorch.org/whl/cpu)
  TORCH_DESC="CPU-only build"
fi
ok "Platform: $OS ($ARCH)"
if [ -n "$GPU_NAME" ]; then
  ok "NVIDIA GPU detected: $GPU_NAME"
elif [ "$FORCE_CPU" = true ]; then
  info "GPU ignored (--cpu)"
fi

if [ "$WITH_DOCKING" = true ] && [ "$OS" != "Linux" ]; then
  fail "docking extras are only supported on Linux (the bundled QuickVina2 binary is Linux x86-64)."
fi
if [ "$WITH_DOCKING_GPU" = true ] && [ "$FORCE_CPU" = true ]; then
  fail "--extra docking-gpu cannot be combined with --cpu."
fi

# Verify environment.yml exists
ENV_YML="environment.yml"
if [ ! -f "$ENV_YML" ]; then
  fail "environment.yml not found at: $ENV_YML" "Run this script from the repository root."
fi
ok "Found $ENV_YML"

# Extract environment name from environment.yml (used when NOT using --prefix)
ENV_NAME="$(grep -E "^name:" "$ENV_YML" | sed 's/^name:[[:space:]]*//' | sed 's/[[:space:]]*$//')"
if [ -z "$ENV_NAME" ]; then
  fail "Could not extract environment name from environment.yml" "Please ensure environment.yml contains a 'name:' field."
fi
if [[ "$ENV_NAME" == *"/"* ]]; then
  fail "Invalid environment name in environment.yml: '$ENV_NAME'"
fi

# Verify mamba is installed and available
if ! command -v mamba &> /dev/null; then
  fail "mamba is not installed or not in PATH." "Please install mamba from: https://mamba.readthedocs.io/en/latest/installation.html"
fi
MAMBA_VERSION="$(mamba --version 2>/dev/null | head -n 1 || true)"
ok "Found mamba${MAMBA_VERSION:+ ($MAMBA_VERSION)}"

# Set environment specification: prefix if provided, otherwise use name from environment.yml
if [ -n "$ENV_DIR" ]; then
  ENV_SPEC="-p"
  ENV_LOCATION="$ENV_DIR"
  if [ -d "$ENV_DIR" ]; then
    # micromamba refuses to create into an existing non-conda directory.
    # Treat an existing directory as an env only if it has conda metadata.
    if [ -d "$ENV_DIR/conda-meta" ]; then
      ENV_EXISTS=true
    else
      fail "Prefix path '$ENV_DIR' already exists but is not a conda environment (missing conda-meta/)." \
        "Please remove it (e.g. rm -rf '$ENV_DIR') or choose a different --prefix."
    fi
  else
    ENV_EXISTS=false
  fi
else
  ENV_SPEC="-n"
  ENV_LOCATION="$ENV_NAME"
  if mamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ENV_EXISTS=true
  else
    ENV_EXISTS=false
  fi
fi

# Check for C compiler (required for PyTorch compilation with torch.compile)
COMPILER_PATH="$(command -v gcc || command -v clang || true)"
if [ -z "$COMPILER_PATH" ]; then
  fail "C compiler (gcc or clang) not found." \
    "A C compiler is required for PyTorch compilation (torch.compile) to work." \
    "Please install a C compiler:" \
    "  Ubuntu/Debian: sudo apt-get install build-essential" \
    "  CentOS/RHEL: sudo yum install gcc gcc-c++ make" \
    "  macOS: xcode-select --install"
fi
ok "Found C compiler: $COMPILER_PATH"

if [ "$ENV_EXISTS" = true ]; then
  ENV_STATE="exists, will be reused"
else
  ENV_STATE="will be created"
fi

printf '\n  %sPlan%s\n' "$BOLD" "$RESET"
printf '    %-13s %s\n' "Environment:" "$ENV_LOCATION ($ENV_STATE)"
printf '    %-13s %s\n' "PyTorch:" "$TORCH_DESC"
printf '    %-13s %s\n' "Extras:" "$EXTRAS_DESC"
step_done

# ---------------------------------------------------------------------------
# Step: environment
# ---------------------------------------------------------------------------
step "Creating conda environment"
if [ "$ENV_EXISTS" = false ]; then
  if [ -n "$ENV_DIR" ]; then
    # Only ensure the parent directory exists; the prefix dir itself must not exist.
    mkdir -p "$(dirname "$ENV_DIR")"
    # Prefix mode: create the environment at the provided directory
    run mamba create --yes --prefix "$ENV_DIR" -f "$ENV_YML"
  else
    # Name mode: create using the name from environment.yml
    run mamba env create --yes -f "$ENV_YML"
  fi
  ok "Created environment: $ENV_LOCATION"
else
  info "Environment '$ENV_LOCATION' already exists, skipping creation (installing into it)"
fi
step_done

# ---------------------------------------------------------------------------
# Step: Boost (docking-gpu only)
# ---------------------------------------------------------------------------
if [ "$WITH_DOCKING_GPU" = true ]; then
  step "Installing Docking GPU dependencies (Boost)"
  run mamba install --yes "$ENV_SPEC" "$ENV_LOCATION" -c conda-forge "libboost=1.84.0" "boost-cpp=1.84.0"
  ok "Installed Boost 1.84.0"
  step_done
fi

# ---------------------------------------------------------------------------
# Step: PyTorch (installed FIRST, before packages that depend on it)
# ---------------------------------------------------------------------------
step "Installing PyTorch ($TORCH_DESC)"
run mamba run "$ENV_SPEC" "$ENV_LOCATION" pip install torch torchvision ${TORCH_INDEX_ARGS[@]+"${TORCH_INDEX_ARGS[@]}"}
ok "Installed PyTorch"
step_done

# ---------------------------------------------------------------------------
# Step: project and requested extras (pip will see PyTorch is already installed)
# ---------------------------------------------------------------------------
step "Installing joint-improvement (editable${EXTRAS_SPEC:+, extras: $EXTRAS_DESC})"
run mamba run "$ENV_SPEC" "$ENV_LOCATION" pip install -e ".${EXTRAS_SPEC}"
ok "Installed project"
if [ "$WITH_DEV" = true ]; then
  run mamba run "$ENV_SPEC" "$ENV_LOCATION" pre-commit install
  ok "Installed pre-commit hooks"
fi
step_done

# ---------------------------------------------------------------------------
# Step: verification
# ---------------------------------------------------------------------------
step "Verifying installation"
if PYTHONIOENCODING=utf-8 mamba run "$ENV_SPEC" "$ENV_LOCATION" python - <<'PY'; then
import platform
import sys


def ok(message):
    print(f"  ✓ {message}")


def fail(message):
    print(f"  ✗ {message}")
    sys.exit(1)


ok(f"Python {platform.python_version()}")

# Check package installation
try:
    import joint_improvement  # noqa: F401

    ok("joint-improvement package imports")
except ImportError as e:
    fail(f"Failed to import joint-improvement: {e}")

# Check PyTorch
try:
    import torch  # noqa: F401

    if torch.cuda.is_available():
        device = f"CUDA {torch.version.cuda}"
    elif torch.backends.mps.is_available():
        device = "MPS (Apple Silicon)"
    else:
        device = "no GPU, CPU-only mode"
    ok(f"PyTorch {torch.__version__} ({device})")
except ImportError as e:
    fail(f"Failed to import PyTorch: {e}")

# Check RDKit-based calculations
try:
    from joint_improvement.utils.chemistry import calculate_logp, calculate_validity

    assert calculate_validity("CCO")
    ok(f"RDKit works (logP of ethanol: {calculate_logp('CCO'):.3f})")
except Exception as e:  # noqa: BLE001
    fail(f"RDKit check failed: {e}")
PY
  step_done
  banner "Installation completed successfully!"
  printf '\n  Total time: %s\n' "$(fmt_duration "$SECONDS")"
  printf '\n  %sNext step%s\n' "$BOLD" "$RESET"
  printf '    Activate the environment:  mamba activate %s\n\n' "$ENV_LOCATION"
else
  banner "Installation verification failed!"
  exit 1
fi
