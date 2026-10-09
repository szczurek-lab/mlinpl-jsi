"""IntDiv1 metric for evaluating molecular diversity.

IntDiv1 measures the internal diversity of a set of molecules by calculating
the average pairwise Tanimoto distance between molecular fingerprints.

This implementation follows SATURN's IntDiv1 calculation methodology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

if TYPE_CHECKING:
    from collections.abc import Sequence


def calculate_intdiv1(smiles_list: np.ndarray | Sequence[str]) -> float:
    """Calculate IntDiv1 metric: average pairwise Tanimoto distance between Morgan fingerprints.

    IntDiv1 measures internal diversity as the average pairwise Tanimoto distance
    between molecular fingerprints. The formula is:

    IntDiv1 = mean(1 - Tanimoto_similarity) for all pairs of molecules

    Parameters
    ----------
    smiles_list : np.ndarray | Sequence[str]
        List of SMILES strings. Invalid SMILES are automatically filtered out.

    Returns
    -------
    float
        IntDiv1 score (higher = more diverse, range: 0.0 to 1.0).
        Returns 0.0 if there are fewer than 2 valid molecules.

    References
    ----------
    SATURN: Sample-efficient Generative Molecular Design
    https://openreview.net/pdf/557b757e0909cb07740adac04b0d67a01855a40e.pdf
    """
    smiles_array = cast("np.ndarray", np.asarray(smiles_list, dtype=str))
    if smiles_array.ndim != 1:
        raise ValueError("SMILES list must be 1D.")

    # Filter to valid molecules only
    valid_mols = []
    for smi in smiles_array:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)

    if len(valid_mols) < 2:
        return 0.0

    # Generate Morgan fingerprints (radius 2, 2048 bits)
    fps = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for mol in valid_mols
    ]

    # Calculate pairwise Tanimoto similarities
    similarities = []
    n = len(fps)
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    if len(similarities) == 0:
        return 0.0

    # Convert similarity to distance (1 - similarity) and take mean
    distances = [1.0 - sim for sim in similarities]
    intdiv1 = np.mean(distances)

    return float(intdiv1)
