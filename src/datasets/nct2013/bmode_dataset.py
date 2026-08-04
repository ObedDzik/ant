import typing as tp
from enum import StrEnum

import numpy as np
from PIL import Image
import skimage
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import torch
import os
import random
from src.datasets.nct2013.utils import apply_colormap
from src.datasets.registry import register_dataset
from .data_access import data_accessor


class DataKeys(StrEnum):
    BMODE = "bmode"
    PROSTATE_MASK = "prostate_mask"
    NEEDLE_MASK = "needle_mask"
    CENTER = "center"
    RF = "rf"
    CORE_ID = "core_id"
    FRAME_IDX = "frame_idx"
    PSA = "psa"
    PRIMARY_GRADE = "primary_grade"
    SECONDARY_GRADE = "secondary_grade"
    AGE = "age"
    FAMILY_HISTORY = "family_history"


class BModeDatasetV1(Dataset):
    """Dataset for B-mode images.

    Samples are dictionaries with the following keys:
        bmode_frames (np.ndarray): Shape (H, W) uint8
        prostate_mask (np.ndarray): Shape (H, W)
        needle_mask (np.ndarray): Shape (H, W)
        core_id (str): Core id of the sample.
        frame_idx (int): Frame index of the sample.
        psa (float): PSA value of the sample.
        primary_grade (int): Primary grade of the sample.
        secondary_grade (int): Secondary grade of the sample.
        age (int): Age of the sample.
        family_history (int): Family history of the sample.
        ... etc (other metadata)

    Args:
        core_ids (list): List of core ids to include in the dataset.
        transform (callable): Transform to apply to each sample.
        frames (str): If 'first', only the first frame is returned. If 'all', all frames are returned.
    """

    def __init__(
        self,
        core_ids=None,
        transform=None,
        frames: tp.Literal["first", "all", "all_concat"] = "first",
        include_rf=False,
        rf_as_bmode=False,
        api_compatibility=None,
        apply_colormap=True,
        flip_ud=False,
        core_selection_kw={}, 
        split=None,
        output_fmt='numpy',
    ):
        if core_ids is None:
            if split is not None: 
                from src.datasets.nct2013.cohort_selection import select_cohort
                train, val, test = select_cohort(**core_selection_kw)
                if split == "train":
                    core_ids = train
                elif split == "val":
                    core_ids = val
                elif split == "test":
                    core_ids = test
            else:
                print("Using all cores.")
                core_ids = data_accessor.get_metadata_table().core_id.unique().tolist()
        self.metadata = data_accessor.get_metadata_table().copy()
        self.metadata = self.metadata[self.metadata.core_id.isin(core_ids)]
        self._indices = []
        self.core_ids = core_ids
        for i, core_id in enumerate(tqdm(self.core_ids, desc="Loading dataset")):
            if frames == "first":
                self._indices.append((i, 0))
            elif frames == "all":
                n = data_accessor.get_num_frames(core_id)
                self._indices.extend([(i, j) for j in range(n)])
            elif frames == "all_concat": 
                n = data_accessor.get_num_frames(core_id)
                self._indices.append((i, np.arange(n)))
        self.transform = transform
        self.include_rf = include_rf
        self.rf_as_bmode = rf_as_bmode
        self.api_compatibility = api_compatibility
        self.apply_colormap = apply_colormap
        self.flip_ud = flip_ud
        self.frames = frames
        self.output_fmt = output_fmt

    def __len__(self):
        return len(self._indices)

    def get_patient_id(self, idx):
        """
        Extract patient ID from core_id.
        core_id format: "center-patientID-coreNumber"
        e.g. "UVA-00123-4" -> patient_id = "UVA-00123"
        Adjust the split logic to match your actual core_id format.
        """
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]
        patient_id = core_id.split("_")[0]
        return patient_id

    def __getitem__(self, idx):
        core_idx, frame_idx = self._indices[idx]
        core_id = self.core_ids[core_idx]
        if not self.rf_as_bmode:
            bmode = data_accessor.get_bmode_image(core_id, frame_idx)
        else:
            bmode = data_accessor.get_rf_image(core_id, frame_idx)
        h, w, *_ = bmode.shape
        if self.apply_colormap: 
            from .utils import apply_colormap
            bmode = apply_colormap(bmode)

        prostate_mask = data_accessor.get_prostate_mask(core_id)
        prostate_mask = skimage.transform.resize(
            prostate_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        needle_mask = data_accessor.get_needle_mask(core_id)
        needle_mask = skimage.transform.resize(
            needle_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

        if self.flip_ud: 
            bmode = np.flipud(bmode)
            prostate_mask = np.flipud(prostate_mask)
            needle_mask = np.flipud(needle_mask)
        metadata = self.metadata[self.metadata.core_id == core_id].iloc[0].to_dict()
        metadata["frame_idx"] = frame_idx
        output = {
            "bmode": bmode,
            "prostate_mask": prostate_mask,
            "needle_mask": needle_mask,
            "orientation": "anterior_is_top" if self.flip_ud else "anterior_is_bottom",
            **metadata,
        }

        if self.output_fmt == 'pil': 
            output["bmode"] = Image.fromarray(output["bmode"]).convert("RGB")
            output["prostate_mask"] = Image.fromarray(output["prostate_mask"])
            output["needle_mask"] = Image.fromarray(output["needle_mask"])
        if self.include_rf:
            rf = data_accessor.get_rf_image(core_id, frame_idx)
            output["rf"] = rf
        if self.api_compatibility == "torchvision":
            image = Image.fromarray(output["bmode"]).convert("RGB")
            label = int(output["grade"] != "Benign")
            if self.transform:
                image = self.transform(image)
            return image, label
        if self.transform:
            output = self.transform(output)
        return output

@register_dataset
def nct2013_bmode_dataset(split=None, **kwargs): 
    if split: 
        ...