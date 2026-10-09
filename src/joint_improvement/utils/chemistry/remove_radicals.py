"""Radical detection and removal utilities for molecular properties."""

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


def has_radicals(smiles: str) -> bool:
    """Check if a SMILES string corresponds to a molecule with radicals.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    bool
        True if the molecule has radicals (unpaired electrons), False otherwise.
        Returns True if the molecule cannot be parsed.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function uses RDKit to parse the SMILES string and checks if any atom
    has radical electrons (unpaired electrons). Molecules with radicals are often
    unstable and may need to be filtered out in certain applications.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for radical detection. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True
        return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())
    except Exception:
        return True


def remove_molecules_with_radicals(smiles_batch: list[str] | NDArray[np.str_]) -> NDArray[np.str_]:
    """Remove molecules with radicals from a batch of SMILES strings.

    Parameters
    ----------
    smiles_batch : list[str] | np.ndarray[str]
        List or array of SMILES string representations of molecules.

    Returns
    -------
    np.ndarray[str]
        Array of SMILES strings with molecules containing radicals removed.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function filters out molecules that contain radicals (unpaired electrons).
    It is more efficient than calling has_radicals multiple times in a loop.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for radical detection. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    return np.array([smiles for smiles in smiles_batch if not has_radicals(smiles)])
