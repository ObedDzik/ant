import os
import json
from typing import Literal
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class NeedleTraceImageFramesDataset(Dataset):
    def __init__(
        self,
        samples_json: str,
        split: str | None = None,
        splits_json: str | None = None,
        case_ids: list | None = None,
        cine_ids: list | None = None,
        transform=None,
        out_fmt: Literal["pil", "np"] = "pil",
    ):
        """
        PyTorch Dataset for loading needle annotation data from a samples.json file.

        Args:
            samples_json (str): Path to the samples.json file.
            split (str, optional): Which split to use ('train', 'val', 'test').
                                   Requires splits_json to be provided.
            splits_json (str, optional): Path to splits.json mapping split names
                                         to lists of case IDs.
            case_ids (list, optional): Explicit list of case IDs to include.
                                        Overrides split-based selection.
            cine_ids (list, optional): List of cine IDs to include.
            transform: Transform applied to each sample dict.
            out_fmt (str): Output format — 'pil' or 'np'.
        """
        self.transform = transform
        self.out_fmt = out_fmt
        self.data = []

        # Load samples
        with open(samples_json, "r") as f:
            samples = json.load(f)

        # Resolve case_ids from split file if provided
        if case_ids is None and split is not None:
            if splits_json is None:
                raise ValueError(
                    "splits_json must be provided when using the split argument."
                )
            with open(splits_json, "r") as f:
                splits_data = json.load(f)
            split_id = list(splits_data.keys())[0]
            if split not in splits_data[split_id]:
                raise ValueError(
                    f"Split '{split}' not found. "
                    f"Available: {list(splits_data[split_id].keys())}"
                )
            case_ids = splits_data[split_id][split]

        # Build flat list of cine samples
        for case_id, case_data in samples.items():

            if case_ids is not None and case_id not in case_ids:
                continue

            for cine in case_data.get("cines", []):
                cine_id = cine["cine_id"]

                if cine_ids is not None and cine_id not in cine_ids:
                    continue

                # Image path — use last path in list per README spec
                image_paths = cine.get("image_path", [])
                if not image_paths:
                    continue
                image_path = image_paths[-1]

                if not os.path.exists(image_path):
                    continue

                needle_mask_path = cine.get("needle_mask_path", None)

                # MicroSegNet mask stored alongside the image
                microsegnet_mask_path = os.path.join(
                    os.path.dirname(image_path),
                    "micro_seg_net_prostate_mask.png"
                )

                # Build info dict matching what _ProstNFoundDatasetAdapterOptimum expects
                info = {
                    "case": case_id,
                    "cine_id": cine_id,
                    "center": case_data.get("center", ""),
                    "age": case_data.get("age", np.nan),
                    "psa": case_data.get("psa", np.nan),
                    "GG": cine.get("GG", 0.0),
                    "% Cancer": cine.get("% Cancer", np.nan),
                    "PRI-MUS": cine.get("PRI-MUS", None),
                    "Diagnosis": "Benign" if (cine.get("GG", 0) == 0) else "Cancer",
                    "clinically_significant": bool(
                        cine.get("GG", 0) is not None and
                        not (isinstance(cine.get("GG", 0), float) and
                             np.isnan(float(cine.get("GG", 0)))) and
                        float(cine.get("GG", 0)) >= 2
                    ),
                    "all_cores_benign": float(cine.get("GG", 0) or 0) == 0,
                    "Sample ID": cine_id,
                }

                self.data.append({
                    "image_path": image_path,
                    "needle_mask_path": needle_mask_path,
                    "microsegnet_prostate_mask_path": microsegnet_mask_path,
                    "info": info,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.out_fmt == "np":
            image = np.array(image)

        # Load needle mask
        if sample["needle_mask_path"] and \
           os.path.exists(sample["needle_mask_path"]):
            needle_mask = Image.open(sample["needle_mask_path"]).convert("L")
        else:
            w, h = image.size if hasattr(image, "size") else (image.shape[1], image.shape[0])
            needle_mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        if self.out_fmt == "np":
            needle_mask = np.array(needle_mask)

        out = {
            "image": image,
            "needle_mask": needle_mask,
            "path": sample["image_path"],
            "info": sample["info"],
        }

        # Load MicroSegNet prostate mask if available
        if os.path.exists(sample["microsegnet_prostate_mask_path"]):
            prostate_mask = Image.open(
                sample["microsegnet_prostate_mask_path"]
            ).convert("L")
            if self.out_fmt == "np":
                prostate_mask = np.array(prostate_mask)
            out["prostate_mask"] = prostate_mask

        if self.transform:
            out = self.transform(out)

        return out

    def list_indices_by_patient_ids(self):
        """Returns dict mapping case_id to sorted list of dataset indices."""
        outputs = {}
        for index, sample in enumerate(self.data):
            case_id = sample["info"]["case"]
            cine_id = sample["info"]["cine_id"]
            outputs.setdefault(case_id, []).append((index, cine_id))
        for case_id, index_info in outputs.items():
            outputs[case_id] = [
                i for i, _ in sorted(index_info, key=lambda x: x[1])
            ]
        return outputs