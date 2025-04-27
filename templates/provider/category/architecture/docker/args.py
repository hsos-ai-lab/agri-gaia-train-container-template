#!/usr/bin/env python

# SPDX-FileCopyrightText: 2024 Osnabrück University of Applied Sciences
# SPDX-FileContributor: Andreas Schliebitz
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train SomeAwesomeNet.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Size of a training batch.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=300,
        help="Size used for resizing input images.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimization algorithm used while training.",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="ReduceLROnPlateau",
        help="Learning rate scheduler used while training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate used while training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing training and validation images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Directory containing training and validation labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the trained model.",
    )
    return parser.parse_args()
