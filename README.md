# Joint Self-Improvement

[![arXiv](https://img.shields.io/badge/arXiv-2602.10984-b31b1b.svg)](https://arxiv.org/abs/2602.10984)

Official repository for the paper [Sample Efficient Generative Molecular Optimization with Joint Self-Improvement](https://arxiv.org/abs/2602.10984).

## Installation

To install the project environment using [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), run the following from the repository root:

```bash
# install core requirements only
./scripts/setup_env.sh
```
```bash
# custom environment location
./scripts/setup_env.sh --prefix /custom/path/to/env
```
```bash
# extra dependencies, e.g. notebooks and GPU docking
./scripts/setup_env.sh --extra notebook --extra docking-gpu
```

> [!NOTE]
> A C compiler (gcc or clang) is required for `torch.compile`. Install it with `sudo apt-get install build-essential` on Ubuntu/Debian or `xcode-select --install` on macOS.

The setup script creates a conda environment from `environment.yml`, installs PyTorch, installs the project with the requested extras, and verifies the installation.

### Additional dependencies

Install extras with `--extra NAME` in the script, or `pip install -e ".[NAME]"` manually. With pip, combine them with commas: `".[notebook,dev]"`.

| Extra | Adds | Platforms |
| --- | --- | --- |
| *(default)* | Core requirements and RDKit | Linux, Apple Silicon |
| `docking-cpu` | `openbabel-wheel` for docking scores on CPU | Linux only |
| `docking-gpu` | `openbabel-wheel` and Boost, plus a manual QuickVina2-GPU build | Linux with NVIDIA GPU only |
| `peptides` | Placeholder, nothing yet | Linux, Apple Silicon |
| `notebook` | `ipykernel`, `matplotlib` for running notebooks | Linux, Apple Silicon |
| `dev` | `ruff`, `pre-commit`, `nbstripout` | Linux, Apple Silicon |


#### Docking (GPU)

Install with the `docking-gpu` extra, using the script:

```bash
./scripts/setup_env.sh --extra docking-gpu
```

Next, clone and build the upstream project (follow their README for your system):
```bash
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git
```

You should end up with a QuickVina2-GPU executable. Export the QuickVina2-GPU binary path and call `calculate_docking`:
```bash
export QUICKVINA2_GPU_BINARY=/full/path/to/QuickVina2-GPU-2-1
python -c "from joint_improvement.utils.chemistry.docking import calculate_docking; print(calculate_docking('CCO', target='braf', device='gpu'))"
```

### Experiments

to be added. 