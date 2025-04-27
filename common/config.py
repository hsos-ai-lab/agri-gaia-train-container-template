# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
#
# SPDX-License-Identifier: MIT

import json
from typing import Dict

TRAIN_CONFIG_FILEPATH = "./train_config.json"
PRESETS_FILEPATH = "./presets.json"


def load_config() -> Dict:
    with open(TRAIN_CONFIG_FILEPATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def finalize_config(config: Dict, custom: Dict = {}):
    with open(PRESETS_FILEPATH, "r", encoding="utf-8") as fh:
        presets = json.load(fh)
    config = {**config, **presets, **custom}
    with open(TRAIN_CONFIG_FILEPATH, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)
