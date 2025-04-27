#!/usr/bin/env python

# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

from args import get_args
from export import export_model

if __name__ == "__main__":
    ARGS = get_args()

    """
        Exemplary training procedure:

        1. Download pretrained weights from a server (FTP, HTTP, ...).
        2. Load images and labels from ARGS.images_dir and ARGS.labels_dir.
        3. Initialize architecture with pretrained weights.
        4. Configure training: Train/Test loops, Optimizer, Learning rate scheduler, ...
        5. Train model on train split and print train metrics like train_loss.
        6. Test model on test split and print test metrics.
        7. Store trained model as file in ARGS.output_dir.
    """

    print(f"Executing your training code in {__file__}::{__name__}")

    """
    Save your model to a file by either:
        1. Using your own implementation of some "save_model" function:
            save_model(model=trained_model, model_filepath=model_filepath)
        2. Or using the predefined "export_model" function from "export.py":
            export_model(
                model=trained_model,
                model_filepath=model_filepath,
                model_format="pytorch",
                default_model_export_func=save_model,
            )
    """