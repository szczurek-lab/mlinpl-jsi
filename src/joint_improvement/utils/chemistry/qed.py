"""QED (Quantitative Estimate of Drug-likeness) calculation utility."""

import numpy as np
from numpy.typing import NDArray

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    Chem = None
    Descriptors = None
    RDLogger = None


def calculate_qed(smiles: str) -> float:
    """Calculate the QED (quantitative estimate of drug-likeness) of a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    float
        QED score in the range [0, 1]. Returns NaN if the molecule cannot
        be parsed or if an error occurs during calculation.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for QED calculation. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")

        mol.UpdatePropertyCache(strict=False)
        return float(Descriptors.qed(mol))  # type: ignore[attr-defined]
    except Exception:
        return float("nan")


def calculate_qed_batch(smiles_list: list[str] | NDArray[np.str_]) -> NDArray[np.float64]:
    """Calculate QED (quantitative estimate of drug-likeness) for a batch of molecules.

    Parameters
    ----------
    smiles_list : list[str] | np.ndarray[str]
        List or array of SMILES string representations of molecules.

    Returns
    -------
    np.ndarray[float]
        Array of QED scores in the range [0, 1]. Returns NaN for molecules that
        cannot be parsed or if an error occurs during calculation.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function uses numpy.vectorize to efficiently process multiple molecules.
    It is more efficient than calling calculate_qed multiple times in a loop.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for QED calculation. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    vectorized_qed = np.vectorize(calculate_qed, otypes=[float])
    return vectorized_qed(smiles_list)
