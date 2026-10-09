"""Base configuration class for all configs."""

from __future__ import annotations

import json
from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BaseConfig(ABC):  # noqa: B024
    """Base configuration class providing common serialization methods.

    All config classes should inherit from this base class to ensure
    consistent interface for loading and saving configurations.
    """

    @classmethod
    def from_json(cls, path: str | Path) -> BaseConfig:
        """Load configuration from JSON file.

        Parameters
        ----------
        path : str | Path
            Path to JSON configuration file.

        Returns
        -------
        BaseConfig
            Configuration instance loaded from JSON.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        json.JSONDecodeError
            If the JSON file is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> BaseConfig:
        """Load configuration from JSON file (alias for from_json).

        This method provides a consistent interface with other model classes
        that use `from_pretrained` for loading configurations.

        Parameters
        ----------
        path : str | Path
            Path to JSON configuration file.

        Returns
        -------
        BaseConfig
            Configuration instance loaded from JSON.
        """
        return cls.from_json(path)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Parameters
        ----------
        path : str | Path
            Path to save JSON configuration file.

        Raises
        ------
        ValueError
            If the config cannot be serialized to JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return asdict(self)
