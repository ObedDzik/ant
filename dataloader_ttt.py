from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import json
import os
from typing import Literal
import random
import numpy as np
import sklearn
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from tqdm import tqdm
import copy
from src.datasets.nct2013.bmode_dataset import BModeDatasetV1
from src.transforms.crop_to_mask import CropToMask
from src.transforms.pixel_augmentations import RandomContrast, RandomGamma
from src.datasets.nct2013.cohort_selection import (
    get_parser as get_cohort_selection_parser,
    select_cohort_from_args,
)
from src.datasets.nct2013.data_access import data_accessor
from src.transforms.prostnfound_transform import ProstNFoundTransform
from typing import List, Optional
import random
from projects.seg_ttt.needle_trace_dataset_ttt import NeedleTraceImageFramesDataset


def get_dataloaders_from_args(args, mode: Literal["train", "test", "heatmap"] = "test"):
    choose_params = args.get('choose_params', False)
    if args.flip_ud:
        transform_flip_ud = True
    else: 
        transform_flip_ud = False
    case_ids = None

    #TODO: The prostate mask is turned on and off in the prostnfoundtransform class
    train_transform = ProstNFoundTransform(
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
        flip_ud=transform_flip_ud,
    )
    val_transform = ProstNFoundTransform(
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
        flip_ud=transform_flip_ud,
    )
        
    train_cores, val_cores, test_cores = select_cohort_from_args(args) #train only, the rest are empty!
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

    if args.optimum_center == 'UA' and args.root_dir == '/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe':
        print(f'Inside UA leg. Optimum center is {args.optimum_center}')
        raise ValueError(f'Root dir must be /datasets/exactvu_pca/OPTIMUM/processed/UA_annotated_needles but got {args.root_dir}')

    if args.optimum_center == ('OL' or 'PU') and args.root_dir == '/datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe':
        case_ids = [
            p
            for p in os.listdir(args.root_dir)
            if os.path.isdir(os.path.join(args.root_dir, p)) and p[:2]==args.optimum_center
        ]
    if args.optimum_center == ('OL' or 'PU') and args.root_dir == '/datasets/exactvu_pca/OPTIMUM/processed/UA_annotated_needles':
        print(f'Inside PU and OL leg. Optimum center is {args.optimum_center}')
        raise ValueError(f'Root dir must be /datasets/exactvu_pca/OPTIMUM/UA_OL_PU_annotated_needles_multiframe but got {args.root_dir}')

    val_dataset = NeedleTraceImageFramesDataset(
        root_dir=args.root_dir,
        transform=val_transform,
        case_ids = case_ids,
        needle_mask_fname=("needle_mask.png" if mode != "heatmap" else "needle_mask_full.png"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size if mode == "train" else 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dict(train=train_loader, val=val_loader)

def get_val_loader_with_seed(args, val_loader, seed, mode='train '):
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