"""SMILES randomization utilities for molecular properties."""

import random

try:
    from rdkit import Chem, RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    Chem = None
    RDLogger = None


def randomize_smiles(smiles: str) -> str:
    """Shuffle atom numbering to generate a randomized SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    str
        Randomized SMILES string. Returns original SMILES string if the molecule
        cannot be parsed or if an error occurs during randomization.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function shuffles the atom numbering of a molecule to generate a
    randomized SMILES representation. This is useful for data augmentation
    in machine learning applications.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for SMILES randomization. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        new_atom_order = list(range(mol.GetNumHeavyAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=True)
    except Exception:
        return smiles
