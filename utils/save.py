import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional
import json


class DataPathConfig:
    """Configuration class for managing data paths."""

    def __init__(self, base_data_dir: Optional[Path] = None):
        """
        Initialize path configuration.

        :param base_data_dir: Base directory for data storage. 
                             If None, uses environment variable or default.
        """
        if base_data_dir is None:
            # Try environment variable first, then default
            env_path = os.getenv('SYNTHETIC_DATA_DIR')
            if env_path:
                self.base_data_dir = Path(env_path)
            else:
                # Default to a directory relative to the current working directory
                self.base_data_dir = Path.cwd() / "generatedData"
        else:
            self.base_data_dir = Path(base_data_dir)

    def get_dataset_path(self, dataset_name: str, generator_name: str) -> Path:
        """Get the full path for a specific dataset and generator combination."""
        return self.base_data_dir / dataset_name / generator_name

    @classmethod
    def from_config_file(cls, config_path: str) -> 'DataPathConfig':
        """Create configuration from a JSON config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(base_data_dir=Path(config.get('base_data_dir')))


# Global default configuration - can be overridden
_default_config = DataPathConfig()


def set_default_data_path(base_data_dir: Path) -> None:
    """Set the default data path for all operations."""
    global _default_config
    _default_config = DataPathConfig(base_data_dir)


def save_synthetic_data(
        synth_data: np.ndarray,
        dataset_name: str,
        generator_name: str,
        class_label: str,
        epochs: int,
        config: Optional[DataPathConfig] = None
) -> None:
    """
    Saves synthetic data to file.

    :param synth_data: The synthetic data array to save
    :param dataset_name: Name of the dataset
    :param generator_name: Name of the generator
    :param class_label: Class label for the data
    :param epochs: Number of training epochs
    :param config: Optional path configuration. Uses default if None.
    """
    if config is None:
        config = _default_config

    folder = config.get_dataset_path(dataset_name, generator_name)
    folder.mkdir(parents=True, exist_ok=True)

    file_path = folder / f"{class_label}_{epochs}.npy"
    np.save(file_path, synth_data)


def load_synthetic_data(
        dataset_name: str,
        generator_name: str,
        epochs: int,
        config: Optional[DataPathConfig] = None
) -> Dict[str, np.ndarray]:
    """
    Loads the synthetic data from file.

    :param dataset_name: Name of the dataset
    :param generator_name: Name of the generator
    :param epochs: Number of training epochs
    :param config: Optional path configuration. Uses default if None.
    :return: Dict[class_label, synthetic_data]
    """
    if config is None:
        config = _default_config

    folder = config.get_dataset_path(dataset_name, generator_name)

    if not folder.exists():
        return {}

    all_synth_data = {}
    for file_path in folder.glob("*.npy"):
        if str(epochs) in file_path.name:
            class_label = file_path.name.split("_")[0]
            file_epochs = file_path.name.split("_")[1].split(".")[0]

            # Verify epochs match exactly
            if file_epochs == str(epochs):
                array = np.load(file_path)
                all_synth_data[class_label] = array

    return all_synth_data


def load_all_synthetic_data(
        dataset_name: str,
        generator_name: str,
        config: Optional[DataPathConfig] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads all synthetic data for a dataset/generator combination.

    :param dataset_name: Name of the dataset
    :param generator_name: Name of the generator
    :param config: Optional path configuration. Uses default if None.
    :return: Dict[epochs, Dict[class_label, synthetic_data]]
    """
    if config is None:
        config = _default_config

    folder = config.get_dataset_path(dataset_name, generator_name)

    if not folder.exists():
        return {}

    all_data = {}
    for file_path in folder.glob("*.npy"):
        parts = file_path.stem.split("_")
        if len(parts) >= 2:
            class_label = parts[0]
            epochs = parts[1]

            if epochs not in all_data:
                all_data[epochs] = {}

            array = np.load(file_path)
            all_data[epochs][class_label] = array

    return all_data
