# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
#
# SPDX-License-Identifier: MIT

import json

with open("train_config.json", "r", encoding="utf-8") as fh:
    train_config = json.load(fh)

reserved = {"train-split", "test-split"}

args = []
for option, value in train_config.items():
    if value is None or value == "" or option in reserved:
        continue
    if isinstance(value, bool):
        if value:
            args.append(f"--{option}")
    elif isinstance(value, list):
        if value:
            args.append(f"--{option} {' '.join(map(str, value))}")
    else:
        args.append(f"--{option} {value}")

print(" ".join(args))
