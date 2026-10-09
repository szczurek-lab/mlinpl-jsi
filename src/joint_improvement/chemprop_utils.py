"""
Chemprop Training Utilities

A simple module for training Chemprop models for docking score prediction (regression).

Usage:
    from chemprop import (
        load_data_from_npz,
        load_data_from_csv,
        train_chemprop_regressor,
        predict_chemprop,
        evaluate_chemprop
    )
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models, nn
from chemprop.nn import metrics
from chemprop.models import MPNN


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data_from_npz(
    npz_path: Union[str, Path],
    smiles_key: str = "smiles",
    target_key: str = "targets",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load molecular data from an NPZ file.
    
    Parameters
    ----------
    npz_path : str or Path
        Path to the NPZ file containing SMILES and target values.
    smiles_key : str, default="smiles"
        Key name for SMILES strings in the NPZ archive.
    target_key : str, default="targets"
        Key name for target values in the NPZ archive.
    verbose : bool, default=True
        If True, print summary information about loaded data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'smiles' and 'target' columns.
    
    Examples
    --------
    >>> df = load_data_from_npz("data.npz", smiles_key="smiles", target_key="docking_scores")
    >>> print(df.head())
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load the NPZ file
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # Check available keys
    available_keys = list(npz_data.keys())
    if verbose:
        print(f"Available keys in NPZ: {available_keys}")
    
    # Validate keys exist
    if smiles_key not in npz_data:
        raise KeyError(f"SMILES key '{smiles_key}' not found. Available keys: {available_keys}")
    if target_key not in npz_data:
        raise KeyError(f"Target key '{target_key}' not found. Available keys: {available_keys}")
    
    # Extract data
    smiles = npz_data[smiles_key]
    targets = npz_data[target_key]
    
    # Handle different array types
    if isinstance(smiles, np.ndarray):
        smiles = smiles.tolist() if smiles.ndim > 0 else [smiles.item()]
    if isinstance(targets, np.ndarray):
        targets = targets.tolist() if targets.ndim > 0 else [targets.item()]
    
    # Create DataFrame
    df = pd.DataFrame({
        'smiles': smiles,
        'target': targets
    })
    
    # Remove rows with missing SMILES
    df = df.dropna(subset=['smiles'])
    
    if verbose:
        print(f"Loaded {len(df)} molecules from {npz_path}")
        print(f"  Target range: [{df['target'].min():.4f}, {df['target'].max():.4f}]")
        print(f"  Target mean: {df['target'].mean():.4f}")
    
    return df


def load_data_from_csv(
    csv_path: Union[str, Path],
    smiles_col: str = "smiles",
    target_col: str = "target",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load molecular data from a CSV file.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    smiles_col : str, default="smiles"
        Column name containing SMILES strings.
    target_col : str, default="target"
        Column name containing target values.
    verbose : bool, default=True
        If True, print summary information about loaded data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'smiles' and 'target' columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Rename columns to standard names
    df = df.rename(columns={smiles_col: 'smiles', target_col: 'target'})
    
    # Remove rows with missing SMILES
    df = df.dropna(subset=['smiles'])
    
    if verbose:
        print(f"Loaded {len(df)} molecules from {csv_path}")
        print(f"  Target range: [{df['target'].min():.4f}, {df['target'].max():.4f}]")
        print(f"  Target mean: {df['target'].mean():.4f}")
    
    return df


# =============================================================================
# Data Preparation Helpers
# =============================================================================

def _create_molecule_datapoints(
    smiles: np.ndarray,
    targets: np.ndarray
) -> List[data.MoleculeDatapoint]:
    """Create MoleculeDatapoint objects from SMILES and targets."""
    datapoints = []
    for smi, target in zip(smiles, targets):
        try:
            # Ensure target is in the right format [value]
            if isinstance(target, (int, float)):
                target_val = [float(target)]
            else:
                target_val = [float(target)]
            dp = data.MoleculeDatapoint.from_smi(smi, target_val)
            if dp is not None:
                datapoints.append(dp)
        except Exception:
            continue
    return datapoints


# =============================================================================
# Regression Training (for Docking Predictions)
# =============================================================================

def train_chemprop_regressor(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    output_dir: Union[str, Path] = "chemprop_regression_model",
    smiles_col: str = "smiles",
    target_col: str = "target",
    batch_size: int = 50,
    max_epochs: int = 30,
    val_split: float = 0.2,
    normalize_targets: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a Chemprop regression model for docking score prediction.
    
    The training data is split internally into train/validation sets.
    Test data (if provided) is kept separate and not used during training.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with SMILES and target columns. Will be split into train/val.
    test_df : pd.DataFrame, optional
        Test data (kept separate, not used during training).
    output_dir : str or Path, default="chemprop_regression_model"
        Directory to save model checkpoints.
    smiles_col : str, default="smiles"
        Column name for SMILES strings.
    target_col : str, default="target"
        Column name for target values (docking scores).
    batch_size : int, default=50
        Batch size for training.
    max_epochs : int, default=30
        Maximum number of training epochs.
    val_split : float, default=0.2
        Fraction of train_df to use for validation.
    normalize_targets : bool, default=True
        Whether to normalize target values (recommended for regression).
    verbose : bool, default=True
        Print training progress.
    
    Returns
    -------
    dict
        Training results including model path, metrics, scaler, and test_df info.
    
    Examples
    --------
    >>> train_df = load_data_from_npz("train.npz")
    >>> test_df = load_data_from_npz("test.npz")
    >>> results = train_chemprop_regressor(train_df, test_df, output_dir="my_model")
    >>> print(f"Model saved to: {results['checkpoint_path']}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split train_df into train and validation
    from sklearn.model_selection import train_test_split
    
    train_smiles = train_df[smiles_col].values
    train_targets = train_df[target_col].values
    
    train_smiles_split, val_smiles_split, train_targets_split, val_targets_split = train_test_split(
        train_smiles, train_targets, test_size=val_split, random_state=42
    )
    
    if verbose:
        print(f"Training samples: {len(train_smiles_split)} (from {len(train_df)} total)")
        print(f"Validation samples: {len(val_smiles_split)} ({val_split*100:.1f}% of training data)")
        if test_df is not None:
            print(f"Test samples: {len(test_df)} (kept separate, not used during training)")
    
    # Create datapoints
    train_datapoints = _create_molecule_datapoints(train_smiles_split, train_targets_split)
    val_datapoints = _create_molecule_datapoints(val_smiles_split, val_targets_split)
    
    if verbose:
        print(f"Valid training molecules: {len(train_datapoints)}")
        print(f"Valid validation molecules: {len(val_datapoints)}")
    
    if len(train_datapoints) == 0 or len(val_datapoints) == 0:
        raise ValueError("No valid molecules found in the data")
    
    # Create datasets
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dataset = data.MoleculeDataset(train_datapoints, featurizer)
    val_dataset = data.MoleculeDataset(val_datapoints, featurizer)
    
    # Normalize targets if requested
    scaler = None
    if normalize_targets:
        scaler = StandardScaler().fit(train_dataset.Y)
        train_dataset.normalize_targets(scaler)
        val_dataset.normalize_targets(scaler)
        
        # Save scaler for later use
        scaler_path = output_dir / "scaler.npz"
        np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
        if verbose:
            print(f"Target scaler saved to: {scaler_path}")
    
    # Create data loaders
    train_loader = data.build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    message_passing = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    
    # Create output transform for unscaling predictions
    if normalize_targets and scaler is not None:
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = nn.RegressionFFN(
            input_dim=message_passing.output_dim,
            output_transform=output_transform
        )
    else:
        ffn = nn.RegressionFFN(input_dim=message_passing.output_dim)
    
    # Metrics for regression
    metric_list = [
        metrics.RMSE(),
        metrics.MAE(),
        metrics.MSE(),
        metrics.R2Score()
    ]
    
    model = MPNN(
        message_passing=message_passing,
        agg=agg,
        predictor=ffn,
        metrics=metric_list
    )
    
    # Configure checkpointing
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=False
    )
    
    # Trainer
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        enable_progress_bar=verbose,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=str(output_dir)
    )
    
    # Train
    if verbose:
        print(f"\nStarting training for {max_epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
    
    # Get best checkpoint path
    best_checkpoint = checkpoint_callback.best_model_path
    if best_checkpoint:
        final_checkpoint = Path(os.path.realpath(best_checkpoint))
    else:
        final_checkpoint = checkpoint_dir / "best_model.ckpt"
        trainer.save_checkpoint(str(final_checkpoint))
    
    if verbose:
        print(f"\n✓ Training completed!")
        print(f"  Best model saved to: {final_checkpoint}")
        if checkpoint_callback.best_model_score is not None:
            print(f"  Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    
    return {
        "status": "trained",
        "checkpoint_path": str(final_checkpoint),
        "output_dir": str(output_dir),
        "scaler_path": str(output_dir / "scaler.npz") if normalize_targets else None,
        "best_val_loss": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None
    }


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_chemprop(
    smiles: Union[List[str], np.ndarray, pd.Series],
    model_path: Union[str, Path],
    batch_size: int = 50,
    return_dataframe: bool = True
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Make predictions using a trained Chemprop model.
    
    Parameters
    ----------
    smiles : list, array, or Series
        SMILES strings to predict on.
    model_path : str or Path
        Path to the model checkpoint (.ckpt file).
    batch_size : int, default=50
        Batch size for prediction.
    return_dataframe : bool, default=True
        If True, return DataFrame with SMILES and predictions.
        If False, return numpy array of predictions.
    
    Returns
    -------
    pd.DataFrame or np.ndarray
        Predictions for each molecule.
    
    Examples
    --------
    >>> predictions = predict_chemprop(
    ...     smiles=["CCO", "CCCO", "CCCCO"],
    ...     model_path="model/checkpoints/best_model.ckpt"
    ... )
    >>> print(predictions)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Convert to list if needed
    if isinstance(smiles, pd.Series):
        smiles = smiles.tolist()
    elif isinstance(smiles, np.ndarray):
        smiles = smiles.tolist()
    
    # Load model
    model = MPNN.load_from_checkpoint(str(model_path))
    model.eval()
    
    # Create datapoints (with dummy targets)
    datapoints = []
    valid_indices = []
    for i, smi in enumerate(smiles):
        try:
            dp = data.MoleculeDatapoint.from_smi(smi, [0.0])  # Dummy target
            if dp is not None:
                datapoints.append(dp)
                valid_indices.append(i)
        except Exception:
            continue
    
    if len(datapoints) == 0:
        raise ValueError("No valid molecules found in the input SMILES")
    
    # Create dataset and loader
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer)
    loader = data.build_dataloader(dataset, batch_size=batch_size, shuffle=False)
    
    # Make predictions
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1
    )
    
    predictions_list = trainer.predict(model, loader)
    
    # Flatten predictions
    predictions = np.concatenate([p.numpy() for p in predictions_list], axis=0)
    predictions = predictions.flatten()
    
    if return_dataframe:
        # Create DataFrame with valid SMILES and predictions
        valid_smiles = [smiles[i] for i in valid_indices]
        result_df = pd.DataFrame({
            'smiles': valid_smiles,
            'prediction': predictions
        })
        return result_df
    else:
        # Return full array with NaN for invalid molecules
        full_predictions = np.full(len(smiles), np.nan)
        full_predictions[valid_indices] = predictions
        return full_predictions


def evaluate_chemprop(
    test_df: pd.DataFrame,
    model_path: Union[str, Path],
    smiles_col: str = "smiles",
    target_col: str = "target"
) -> Dict[str, float]:
    """
    Evaluate a trained Chemprop regression model on test data.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with SMILES and target columns.
    model_path : str or Path
        Path to the model checkpoint.
    smiles_col : str, default="smiles"
        Column name for SMILES.
    target_col : str, default="target"
        Column name for targets (docking scores).
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics (RMSE, MAE, R2, n_samples).
    
    Examples
    --------
    >>> metrics = evaluate_chemprop(test_df, model_path="model/checkpoints/best_model.ckpt")
    >>> print(f"R2 Score: {metrics['r2']:.4f}")
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Get predictions
    predictions_df = predict_chemprop(
        smiles=test_df[smiles_col],
        model_path=model_path,
        return_dataframe=True
    )
    
    # Merge with targets (only for molecules that got predictions)
    merged = predictions_df.merge(
        test_df[[smiles_col, target_col]].rename(columns={smiles_col: 'smiles'}),
        on='smiles',
        how='inner'
    )
    
    y_true = merged[target_col].values
    y_pred = merged['prediction'].values
    
    results = {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'n_samples': len(y_true)
    }
    
    return results

