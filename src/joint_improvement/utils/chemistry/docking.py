"""Docking score calculation utilities for molecular properties."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from rdkit import Chem, RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    Chem = None
    RDLogger = None

try:
    from joint_improvement.utils.chemistry.docking_utils.quickvina2 import DockingVina

    _CPU_DOCKING_AVAILABLE = True
    _CPU_DOCKING_IMPORT_ERROR: str | None = None
except ImportError as e:
    _CPU_DOCKING_AVAILABLE = False
    DockingVina = None
    _CPU_DOCKING_IMPORT_ERROR = str(e)


# Docking score thresholds for each target (lower scores indicate better binding)
# https://github.com/SeulLee05/GEAM/blob/main/eval.py
DOCKING_THRESHOLDS: dict[str, float] = {
    "braf": -10.3,
    "parp1": -10.0,
    "fa7": -8.5,
    "jak2": -9.1,
    "5ht1b": -8.7845,
}


class _DockingPredictor(Protocol):
    def predict(self, smiles_list: list[str]) -> list[float]: ...


def _normalize_device(device: str) -> str:
    """Normalize device strings to the internal set {'cpu','gpu'}.

    Supported inputs:
    - 'cpu'
    - 'gpu'
    - 'cuda'
    - 'cuda:N' (e.g. 'cuda:0')
    """
    d = device.strip().lower()
    if d == "cpu":
        return "cpu"
    if d == "gpu" or d == "cuda" or d.startswith("cuda:"):
        return "gpu"
    raise ValueError("device must be 'cpu', 'gpu', 'cuda', or 'cuda:N'")


def _get_quickvina2_gpu_binary() -> str:
    """Return the QuickVina2-GPU binary path from environment variables.

    We keep this out of function signatures to avoid breaking existing callers.
    """
    for key in ("QUICKVINA2_GPU_BINARY", "QVINA_GPU_BINARY"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    raise ImportError(
        "GPU docking requested but QuickVina2-GPU binary path is not configured.\n"
        "Set environment variable QUICKVINA2_GPU_BINARY (or QVINA_GPU_BINARY) to the full path of the executable."
    )


def _make_gpu_predictor(target: str) -> _DockingPredictor:
    """Create a GPU docking predictor using QuickVina2_GPU.

    This adapts the existing `QuickVina2_GPU` oracle component to a `.predict(smiles_list)` interface
    without modifying `quickvina2.py` or `quickvina2_gpu.py`.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for GPU docking.")
    if not _CPU_DOCKING_AVAILABLE or DockingVina is None:
        msg = "CPU docking utilities are required to configure GPU docking targets."
        if _CPU_DOCKING_IMPORT_ERROR:
            msg += f"\nImport error: {_CPU_DOCKING_IMPORT_ERROR}"
        raise ImportError(msg)

    # Import lazily so CPU-only users don't need the GPU module import to succeed.
    from joint_improvement.utils.chemistry.docking_utils.dataclass import OracleComponentParameters
    from joint_improvement.utils.chemistry.docking_utils.quickvina2_gpu import QuickVina2_GPU

    # Reuse the same target definitions as CPU docking (box + receptor).
    cpu = DockingVina(target)

    binary = _get_quickvina2_gpu_binary()
    # Use local temp directory to avoid Lustre I/O overhead
    # Prefer /dev/shm (RAM disk) if available, otherwise /tmp
    local_temp_base = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    results_dir = tempfile.mkdtemp(prefix=f"quickvina2_gpu_{target}_", dir=local_temp_base)
    atexit.register(shutil.rmtree, results_dir, ignore_errors=True)
    thread = int(os.environ.get("QUICKVINA2_GPU_THREAD", os.environ.get("QVINA_GPU_THREAD", "5000")))

    params = OracleComponentParameters(
        name="quickvina2_gpu",
        reward_shaping_function_parameters={"transformation_function": "no_transformation", "parameters": {}},
        specific_parameters={
            "binary": binary,
            "receptor": cpu.receptor_file,
            "results_dir": results_dir,
            "box_center": cpu.box_center,
            "box_size": cpu.box_size,
            # Sampling effort (QuickVina2-GPU calls this `--thread`)
            "thread": thread,
            "verbose": False,
            "raise_on_failure": True,
        },
    )

    oracle = QuickVina2_GPU(params)

    class _Predictor:
        def predict(self, smiles_list: list[str]) -> list[float]:
            mols = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)  # type: ignore[union-attr]
                mols.append(mol)
            valid_mols = [m for m in mols if m is not None]
            if not valid_mols:
                return [float("nan") for _ in smiles_list]

            scores = oracle(np.array(valid_mols, dtype=object), oracle_calls=0)
            # Map back to input order (invalid mols -> NaN).
            out: list[float] = []
            j = 0
            for m in mols:
                if m is None:
                    out.append(float("nan"))
                else:
                    out.append(float(scores[j]))
                    j += 1
            return out

    return _Predictor()


# Singleton instances for docking oracle (one per device)
_docking_instances: dict[str, _DockingPredictor] = {}
_docking_targets: dict[str, str] = {}


def calculate_docking(smiles: str, target: str = "fa7", device: str = "cpu") -> float:
    """Calculate the docking score of a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    target : str, optional
        Target protein for docking. Supported targets: 'fa7', 'parp1', '5ht1b', 'jak2', 'braf'.
        Default is 'fa7'.
    device : str, optional
        Device to use for docking. Options: 'cpu' (default) or 'gpu'.
        Note: GPU support requires QuickVina2-GPU with additional setup.

    Returns
    -------
    float
        Docking score from QuickVina2. Returns NaN if the molecule cannot be
        parsed or if an error occurs during docking calculation.

    Raises
    ------
    ImportError
        If RDKit or docking utilities are not installed.

    Notes
    -----
    RDKit and docking utilities are optional dependencies. Install RDKit with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for docking calculation. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    device_norm = _normalize_device(device)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")

        global _docking_instances, _docking_targets
        instance_key = f"{device_norm}_{target}"
        if instance_key not in _docking_instances:
            if device_norm == "gpu":
                _docking_instances[instance_key] = _make_gpu_predictor(target)
            else:
                if not _CPU_DOCKING_AVAILABLE or DockingVina is None:
                    error_msg = "CPU docking utilities are required for docking calculation."
                    if _CPU_DOCKING_IMPORT_ERROR:
                        error_msg += f"\nImport error: {_CPU_DOCKING_IMPORT_ERROR}"
                    raise ImportError(error_msg)
                _docking_instances[instance_key] = DockingVina(target)
            _docking_targets[instance_key] = target

        scores = _docking_instances[instance_key].predict([smiles])
        if scores is None or len(scores) == 0:
            return float("nan")

        return float(scores[0])
    except Exception as e:
        # Re-raise ImportError and ValueError to surface configuration issues
        if isinstance(e, (ImportError, ValueError)):
            raise
        return float("nan")


def calculate_docking_batch(
    smiles_list: list[str] | NDArray[np.str_], target: str, device: str = "cpu"
) -> NDArray[np.float64]:
    """Calculate docking scores for a batch of molecules.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES string representations of molecules.
    target : str
        Target protein for docking. Supported targets: 'fa7', 'parp1', '5ht1b', 'jak2', 'braf'.
    device : str, optional
        Device to use for docking. Options: 'cpu' (default) or 'gpu'.
        Note: GPU support requires QuickVina2-GPU with additional setup.

    Returns
    -------
    list[float]
        List of docking scores from QuickVina2. Returns NaN for molecules that
        cannot be parsed or if an error occurs during docking calculation.

    Raises
    ------
    ImportError
        If RDKit or docking utilities are not installed.

    Notes
    -----
    RDKit and docking utilities are optional dependencies. Install RDKit with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for docking calculation. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    device_norm = _normalize_device(device)

    global _docking_instances, _docking_targets
    instance_key = f"{device_norm}_{target}"
    if instance_key not in _docking_instances:
        if device_norm == "gpu":
            _docking_instances[instance_key] = _make_gpu_predictor(target)
        else:
            if not _CPU_DOCKING_AVAILABLE or DockingVina is None:
                error_msg = "CPU docking utilities are required for docking calculation."
                if _CPU_DOCKING_IMPORT_ERROR:
                    error_msg += f"\nImport error: {_CPU_DOCKING_IMPORT_ERROR}"
                raise ImportError(error_msg)
            _docking_instances[instance_key] = DockingVina(target)
        _docking_targets[instance_key] = target

    if isinstance(smiles_list, np.ndarray):
        smiles: list[str] = [str(s) for s in smiles_list.tolist()]
    else:
        smiles = list(smiles_list)

    scores = _docking_instances[instance_key].predict(smiles)
    return np.asarray(scores, dtype=np.float64)
