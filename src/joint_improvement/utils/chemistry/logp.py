"""LogP calculation utilities for molecular properties."""

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Crippen

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    Crippen = None
    RDLogger = None


def calculate_logp(smiles: str) -> float:
    """Calculate the logP (partition coefficient) of a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    float
        LogP value. Returns NaN if the molecule cannot be parsed or
        if an error occurs during calculation.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function uses RDKit's Crippen method for logP calculation.
    The molecule is Kekulized before calculation to ensure proper
    aromaticity handling.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for logP calculation. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")

        try:
            AllChem.Kekulize(mol, clearAromaticFlags=True)  # type: ignore[attr-defined]
        except Exception:
            # Kekulization may fail for some molecules, continue anyway
            pass

        mol.UpdatePropertyCache(strict=False)
        return float(Crippen.MolLogP(mol))  # type: ignore[attr-defined]
    except Exception:
        return float("nan")
