"""Synthetic Accessibility (SA) score utilities."""

import numpy as np
from numpy.typing import NDArray

try:
    from rdkit import Chem, RDLogger

    from joint_improvement.utils.chemistry.synthesizability_utils.sascorer import (
        calculateScore,
    )

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    Chem = None
    RDLogger = None
    calculateScore = None


def calculate_sa(smiles: str) -> float:
    """Calculate the Synthetic Accessibility (SA) score of a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    float
        SA score on a scale of ~1 (easy to synthesize) to 10 (hard to synthesize).
        Returns NaN if the molecule cannot be parsed or if an error occurs during
        calculation.

    Raises
    ------
    ImportError
        If RDKit or the SA scorer is not installed.

    Notes
    -----
    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE or calculateScore is None:
        raise ImportError(
            "RDKit and the SA scorer are required for SA calculation. "
            "Install RDKit with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")

        mol.UpdatePropertyCache(strict=False)
        return float(calculateScore(mol))  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def calculate_sa_batch(smiles_list: list[str] | NDArray[np.str_]) -> NDArray[np.float64]:
    """Calculate SA scores for a batch of molecules.

    Parameters
    ----------
    smiles_list : list[str] | np.ndarray[str]
        List or array of SMILES string representations of molecules.

    Returns
    -------
    np.ndarray[float]
        Array of SA scores. Returns NaN for molecules that cannot be parsed or if an
        error occurs during calculation.

    Raises
    ------
    ImportError
        If RDKit or the SA scorer is not installed.

    Notes
    -----
    This function uses numpy.vectorize to efficiently process multiple molecules.
    It is more efficient than calling calculate_sa multiple times in a loop.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE or calculateScore is None:
        raise ImportError(
            "RDKit and the SA scorer are required for SA calculation. "
            "Install RDKit with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    vectorized_sa = np.vectorize(calculate_sa, otypes=[float])
    return vectorized_sa(smiles_list)
