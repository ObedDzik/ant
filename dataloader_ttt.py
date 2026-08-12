import os
import copy
import random
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.transforms.prostnfound_transform import ProstNFoundTransform
from src.datasets.nct2013.cohort_selection import (
    get_parser as get_cohort_selection_parser,
    select_cohort_from_args,
)
from src.datasets.nct2013.data_access import data_accessor
from src.datasets.nct2013.bmode_dataset import BModeDatasetV1
from projects.seg_ttt.needle_trace_dataset_ttt import NeedleTraceImageFramesDataset


def get_val_transform(args, mode: str):
    return ProstNFoundTransform(
        augment="none",
        image_size=args.image_size,
        mask_size=args.mask_size,
        mean=args.mean,
        std=args.std,
        crop_to_prostate=args.crop_to_prostate,
        first_downsample_size=args.first_downsample_size,
        return_raw_images=mode != "train",
        grade_group_for_positive_label=vars(args).get(
            "grade_group_for_positive_label", 1
        ),
        flip_ud=args.get("flip_ud", False),
    )


def get_train_transform(args, mode: str):
    return ProstNFoundTransform(
        augment=args.augmentations,
        image_size=args.image_size,
        mask_size=args.mask_size,
        mean=args.mean,
        std=args.std,
        crop_to_prostate=args.crop_to_prostate,
        first_downsample_size=args.first_downsample_size,
        return_raw_images=mode != "train",
        grade_group_for_positive_label=vars(args).get(
            "grade_group_for_positive_label", 1
        ),
        flip_ud=args.get("flip_ud", False),
    )


def get_test_loaders(args):
    """
    Returns only the validation/test loader from the OPTIMUM samples.json.
    Used during inference (test.py) — no training loader is constructed.

    Args:
        args: Config object with at least:
            - samples_json (str): Path to samples.json
            - splits_json (str, optional): Path to splits.json
            - optimum_center (str, optional): Center prefix to filter by (e.g. 'OL')
            - batch_size (int)
            - num_workers (int)

    Returns:
        dict with key 'val' containing the DataLoader
    """
    val_transform = get_val_transform(args, mode="test")

    # Optionally filter by center prefix
    case_ids = None
    if hasattr(args, "optimum_center") and args.optimum_center:
        import json
        with open(args.samples_json, "r") as f:
            samples = json.load(f)
        case_ids = [
            cid for cid in samples
            if cid.startswith(args.optimum_center)
        ]

    val_dataset = NeedleTraceImageFramesDataset(
        samples_json=args.samples_json,
        splits_json=getattr(args, "splits_json", None),
        case_ids=case_ids,
        transform=val_transform,
        out_fmt="pil",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dict(val=val_loader)


def get_dataloaders_from_args(
    args,
    mode: Literal["train", "test", "heatmap"] = "test"
):
    """
    Returns train and val loaders.
    Used during training (train.py) — both loaders are constructed.

    The training loader uses the NCT2013/BModeDataset pipeline.
    The validation loader uses NeedleTraceImageFramesDataset with samples.json.

    Args:
        args: Config object
        mode: 'train', 'test', or 'heatmap'

    Returns:
        dict with keys 'train' and 'val'
    """
    train_transform = get_train_transform(args, mode)
    val_transform = get_val_transform(args, mode)

    # --- Training loader (NCT2013 pipeline) ---
    train_cores, val_cores, test_cores = select_cohort_from_args(args)
    train_dataset = BModeDatasetV1(
        train_cores,
        train_transform,
        rf_as_bmode=args.rf_as_bmode,
        include_rf=args.include_rf,
        flip_ud=args.flip_ud,
        frames=args.frames,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Validation loader (OPTIMUM samples.json pipeline) ---
    case_ids = None
    if hasattr(args, "optimum_center") and args.optimum_center:
        import json
        with open(args.samples_json, "r") as f:
            samples = json.load(f)
        case_ids = [
            cid for cid in samples
            if cid.startswith(args.optimum_center)
        ]

    val_dataset = NeedleTraceImageFramesDataset(
        samples_json=args.samples_json,
        splits_json=getattr(args, "splits_json", None),
        case_ids=case_ids,
        transform=val_transform,
        out_fmt="pil",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dict(train=train_loader, val=val_loader)


def get_val_loader_with_seed(args, val_loader, seed, mode="test"):
    """
    Returns a new DataLoader with the validation dataset shuffled
    according to the given random seed. Used for ordering sensitivity analysis.

    Args:
        args: Config object
        val_loader: Existing DataLoader wrapping a NeedleTraceImageFramesDataset
        seed (int): Random seed for shuffling
        mode (str): 'train' or 'test' — determines batch size

    Returns:
        DataLoader with shuffled dataset
    """
    rng = random.Random(seed)
    shuffled_data = val_loader.dataset.data.copy()
    rng.shuffle(shuffled_data)

    shuffled_dataset = copy.copy(val_loader.dataset)
    shuffled_dataset.data = shuffled_data

    return DataLoader(
        shuffled_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )