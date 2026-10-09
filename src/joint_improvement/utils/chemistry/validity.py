"""Validity check utilities for molecular properties."""

import numpy as np
from numpy.typing import NDArray

try:
    from rdkit import Chem, RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDLogger = None
    _RDKIT_AVAILABLE = False


def calculate_validity(smiles: str) -> bool:
    """
    Check if a SMILES string corresponds to a valid molecule.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    bool
        True if the SMILES string corresponds to a valid, non-empty molecule.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function uses RDKit's MolFromSmiles method to parse the SMILES string.
    It checks if the molecule is valid (non-empty) and has at least one atom.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for validity check. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )
    try:
        mol = Chem.MolFromSmiles(smiles)
        return smiles != "" and mol is not None and mol.GetNumAtoms() > 0
    except:  # noqa: E722
        return False


def calculate_validity_batch(smiles_list: list[str] | NDArray[np.str_]) -> NDArray[np.bool_]:
    """Check validity for a batch of SMILES strings.

    Parameters
    ----------
    smiles_list : list[str] | np.ndarray[str]
        List or array of SMILES string representations of molecules.

    Returns
    -------
    np.ndarray[bool]
        Array of boolean values indicating validity. True if the SMILES string
        corresponds to a valid, non-empty molecule, False otherwise.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function uses numpy.vectorize to efficiently process multiple molecules.
    It is more efficient than calling calculate_validity multiple times in a loop.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for validity check. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    vectorized_validity = np.vectorize(calculate_validity, otypes=[bool])
    return vectorized_validity(smiles_list)
