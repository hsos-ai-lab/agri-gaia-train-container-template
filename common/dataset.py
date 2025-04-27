# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
#
# SPDX-License-Identifier: MIT

import os

from pathlib import Path
from typing import List, Dict

MOUNTED_DATASETS_ROOT_PATH = "/minio/datasets"
DATASETS_ROOT_PATH = "/datasets"

DATA_DIRS: Dict[str, Path] = {}


def get_dataset_path() -> Path:
    dataset_path = os.path.join(
        MOUNTED_DATASETS_ROOT_PATH, os.environ.get("DATASET_ID")
    )
    assert os.path.isdir(dataset_path), f"Dataset directory '{dataset_path}' not found."
    return Path(dataset_path)


def load_dataset_filepaths() -> List[Path]:
    dataset_path = get_dataset_path()
    return list(dataset_path.glob("*.*"))
