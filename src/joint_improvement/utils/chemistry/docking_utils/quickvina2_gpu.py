"""
Workflow based on: https://arxiv.org/abs/2406.08506 Appendix C

Workflow Overview:
* Input canonical SMILES
* Convert to RDKit Mol
* Protonate
* Generate 1 (lowest energy) conformer using RDKit ETKDG and minimize using RDKit UFF
* Convert conformer to PDBQT file
* Dock using QuickVina2-GPU version 2.1

# ETKDG: https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654
"""
import os
import subprocess
import tempfile
import shutil
import numpy as np
import re
from typing import Mapping, Sequence, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from joint_improvement.utils.chemistry.docking_utils.oracle_component import OracleComponent
from joint_improvement.utils.chemistry.docking_utils.dataclass import OracleComponentParameters


class QuickVina2_GPU(OracleComponent):
    """
    Executes QuickVina2-GPU version 2.1.

    References:
    1. https://www.biorxiv.org/content/early/2023/11/05/2023.11.04.565429
    2. https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # QuickVina2-GPU-2.1 binary
        self.binary = self.parameters.specific_parameters.get("binary", None)
        assert self.binary is not None, "Please provide the path to the QuickVina2-GPU binary."
        # Ensure the binary is executable for the user
        subprocess.run(["chmod", "u+x", self.binary])

        # Force-field for ligand energy minimization
        force_field_id = self.parameters.specific_parameters.get("force_field", "uff").lower()
        assert force_field_id in ["uff", "mmff94"], "force_field must be either 'uff' or 'mmff94'."
        self.force_field = AllChem.UFFOptimizeMolecule if force_field_id == "uff" else AllChem.MMFFOptimizeMolecule

        # Receptor path
        self.receptor = self.parameters.specific_parameters.get("receptor", None)
        assert self.receptor is not None and self.receptor.endswith(".pdbqt"), "Please provide the path to the receptor PDBQT file."

        # Docking box: either provide directly, or derive from a reference ligand.
        box_center = self.parameters.specific_parameters.get("box_center", None)
        box_size = self.parameters.specific_parameters.get("box_size", None)
        if box_center is not None and box_size is not None:
            self.box_center = self._validate_xyz_tuple(box_center, key_name="box_center")
            self.box_size = self._validate_xyz_tuple(box_size, key_name="box_size")
            self.reference_ligand = None
        else:
            # Reference ligand path
            self.reference_ligand = self.parameters.specific_parameters.get("reference_ligand", None)
            assert (
                self.reference_ligand is not None and self.reference_ligand.endswith(".pdb")
            ), "Please provide the path to the reference ligand PDB file (or set box_center and box_size)."
            # Setup docking box
            self._setup_docking_box()

        # Exhaustiveness (known as "Thread" in QuickVina2-GPU)
        self.thread = self.parameters.specific_parameters.get("thread", 5000)  # Default to 5000 (same as source code default)

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Debug / strictness controls
        # - raise_on_failure: raise with stderr/stdout if obabel or docking fails (recommended True)
        # - verbose: print subprocess stdout/stderr for debugging
        self.raise_on_failure = bool(self.parameters.specific_parameters.get("raise_on_failure", True))
        self.verbose = bool(self.parameters.specific_parameters.get("verbose", False))
        # Extra library paths to prepend to LD_LIBRARY_PATH when launching the docking binary.
        # Useful on clusters where the Jupyter kernel doesn't inherit conda's runtime linker setup.
        self.ld_library_path_prefix = self.parameters.specific_parameters.get("ld_library_path_prefix", None)

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        return self._compute_property(mols, oracle_calls)
    
    def _compute_property(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        """
        Execute QuickVina2-GPU-2.1 as a subprocess.
        """
        # 1. Make temporary files to store the input and output
        # Use local temp directory to avoid Lustre I/O overhead
        # Prefer /dev/shm (RAM disk) if available, otherwise /tmp
        local_temp_base = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
        temp_input_sdf_dir = tempfile.mkdtemp(dir=local_temp_base)
        temp_input_pdbqt_dir = tempfile.mkdtemp(dir=local_temp_base)
        temp_output_dir = tempfile.mkdtemp(dir=local_temp_base)

        env = self._build_subprocess_env()

        # 2. Convert RDKit Mols to *canonical* SMILES
        canonical_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]

        # 3. Convert back to RDKit Mols
        # NOTE: This is likely redundant but is done to match the original workflow
        mols = [Chem.MolFromSmiles(smiles) for smiles in canonical_smiles]

        # 4. Protonate Mols
        mols = [Chem.AddHs(mol) for mol in mols]

        # 5. Generate 1 (lowest energy) conformer using RDKit ETKDG and force-field (UFF or MMFF94) minimize
        for idx, mol in enumerate(mols):
        # Skip molecules that fail to embed
            try:
                # Generate conformer with ETKDG
                AllChem.EmbedMolecule(mol, ETversion=2, randomSeed=0)
                # Minimize conformer
                self.force_field(mol)
            except Exception:
                continue
            # Write out the minimized conformer in SDF format
            sdf_file = os.path.join(temp_input_sdf_dir, f"ligand_{idx+1}.sdf")
            writer = Chem.SDWriter(sdf_file)
            writer.write(mol)
            writer.flush()
            writer.close()
            # Convert SDF to PDBQT with OpenBabel
            pdbqt_file = os.path.join(temp_input_pdbqt_dir, f"ligand_{idx+1}.pdbqt")
            obabel_proc = subprocess.run(
                ["obabel", sdf_file, "-opdbqt", "-O", pdbqt_file],
                capture_output=True,
                text=True,
                env=env,
            )
            if self.verbose and (obabel_proc.stdout or obabel_proc.stderr):
                print(f"[QuickVina2_GPU] obabel stdout:\n{obabel_proc.stdout}")
                print(f"[QuickVina2_GPU] obabel stderr:\n{obabel_proc.stderr}")
            if self.raise_on_failure and obabel_proc.returncode != 0:
                raise RuntimeError(
                    "OpenBabel (obabel) failed converting SDF -> PDBQT.\n"
                    f"returncode={obabel_proc.returncode}\n"
                    f"stdout:\n{obabel_proc.stdout}\n"
                    f"stderr:\n{obabel_proc.stderr}\n"
                    f"sdf_file={sdf_file}\n"
                    f"pdbqt_file={pdbqt_file}\n"
                )

        # 6. Run QuickVina2-GPU-2.1
        # TODO: Could be paralellized but GPU docking should be fast enough as large libraries are not expected
            
        # Subprocess call should occur in the directory of the binary due to needing to read kernel files
        # Based on this GitHub issue: https://github.com/DeltaGroupNJUPT/Vina-GPU/issues/12
        binary_dir = os.path.dirname(self.binary)
        docking_proc = subprocess.run(
            [
                self.binary,
                "--receptor",
                self.receptor,
                "--ligand_directory",
                temp_input_pdbqt_dir,
                "--output_directory",
                temp_output_dir,
                # Exhaustiveness
                "--thread",
                str(self.thread),
                # Docking box
                "--center_x",
                str(self.box_center[0]),
                "--center_y",
                str(self.box_center[1]),
                "--center_z",
                str(self.box_center[2]),
                "--size_x",
                str(self.box_size[0]),
                "--size_y",
                str(self.box_size[1]),
                "--size_z",
                str(self.box_size[2]),
                # Output only the lowest docking score pose
                "--num_modes",
                "1",
                # Fix seed
                # TODO: Expose parameter to user
                "--seed",
                "0",
            ],
            cwd=binary_dir,
            capture_output=True,
            text=True,
            env=env,
        )
        if self.verbose and (docking_proc.stdout or docking_proc.stderr):
            print(f"[QuickVina2_GPU] docking stdout:\n{docking_proc.stdout}")
            print(f"[QuickVina2_GPU] docking stderr:\n{docking_proc.stderr}")
        if self.raise_on_failure and docking_proc.returncode != 0:
            error_msg = (
                "QuickVina2-GPU docking failed.\n"
                f"returncode={docking_proc.returncode}\n"
                f"cwd={binary_dir}\n"
            )
            if "CL_OUT_OF_HOST_MEMORY" in docking_proc.stdout or "CL_OUT_OF_HOST_MEMORY" in docking_proc.stderr:
                error_msg += (
                    "\nGPU/OpenCL out of memory error detected. "
                    "Try reducing the batch size or freeing GPU memory.\n"
                )
            error_msg += f"stdout:\n{docking_proc.stdout}\nstderr:\n{docking_proc.stderr}\n"
            raise RuntimeError(error_msg)

        # 7. Copy and save the docking output
        # NOTE: copy the *contents* of the temp output dir, not the temp dir name (avoids results_*/tmpXXXX/ nesting).
        results_dir = os.path.join(self.output_dir, f"results_{oracle_calls}")
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        shutil.copytree(temp_output_dir, results_dir)

        # 8. Parse the docking scores
        docking_scores = np.zeros(len(mols))
        docked_output = os.listdir(temp_output_dir)
        if self.raise_on_failure and len(docked_output) == 0:
            raise RuntimeError(
                "QuickVina2-GPU produced no output files.\n"
                f"temp_output_dir={temp_output_dir}\n"
                f"cwd={binary_dir}\n"
                f"stdout:\n{docking_proc.stdout}\n"
                f"stderr:\n{docking_proc.stderr}\n"
            )
        # NOTE: docking_scores is populated below - ligands that failed to embed would already have an assigned score = 0.0
        for file in docked_output:
            match = re.search(r"ligand_(\d+)", file)
            if match is None:
                continue
            ligand_idx = int(match.group(1)) - 1
            with open(os.path.join(temp_output_dir, file), "r") as f:
                for line in f.readlines():
                    # Extract the docking score
                    if "REMARK VINA RESULT" in line:
                        try:
                            docking_scores[ligand_idx] = (float(line.split()[3]))
                        except Exception:
                            docking_scores[ligand_idx] = 0.0
       
        # 9. Delete the temporary folders
        shutil.rmtree(temp_input_sdf_dir)
        shutil.rmtree(temp_input_pdbqt_dir)
        shutil.rmtree(temp_output_dir)

        return docking_scores

    def _setup_docking_box(self):
        """
        Setup the docking box for the target receptor based on the reference ligand.

        Follows the protocol from https://arxiv.org/abs/2406.08506 Appendix C:

        * Centroids are the average position of the reference ligand atoms

        * "Box sizes individually determined to encompass each target binding"
           Unclear how this is done - box size is set to 20 Å x 20 Å x 20 Å instead which is a common default
        """
        # Get the average coordinates of the reference ligand atoms
        ref_mol = Chem.MolFromPDBFile(self.reference_ligand)
        ref_conformer = ref_mol.GetConformer()
        ref_coords = ref_conformer.GetPositions()
        ref_center = tuple(np.mean(ref_coords, axis=0)) 

        # Set the box center and size
        self.box_center = ref_center  # Tuple[float, float, float]
        self.box_size = (20.0, 20.0, 20.0)

    @staticmethod
    def _validate_xyz_tuple(value: Sequence[float], key_name: str) -> Tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError(f"{key_name} must have length 3 (x, y, z). Got length {len(value)}.")
        try:
            x, y, z = (float(value[0]), float(value[1]), float(value[2]))
        except Exception as e:
            raise ValueError(f"{key_name} must contain numeric values.") from e
        return (x, y, z)

    def _build_subprocess_env(self) -> Mapping[str, str]:
        env = dict(os.environ)

        # Best-effort: ensure conda runtime libs are discoverable.
        conda_prefix = env.get("CONDA_PREFIX", "")
        conda_lib = os.path.join(conda_prefix, "lib") if conda_prefix else ""
        ld_library_path_parts: list[str] = []

        prefix = self.ld_library_path_prefix
        if isinstance(prefix, str) and prefix:
            ld_library_path_parts.append(prefix)
        elif isinstance(prefix, (list, tuple)):
            ld_library_path_parts.extend([str(p) for p in prefix if str(p)])

        if conda_lib and os.path.isdir(conda_lib):
            ld_library_path_parts.append(conda_lib)

        existing = env.get("LD_LIBRARY_PATH", "")
        if ld_library_path_parts:
            env["LD_LIBRARY_PATH"] = ":".join(ld_library_path_parts + ([existing] if existing else []))

        return env
        