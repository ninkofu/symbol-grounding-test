"""Evaluate disentangled VAE checkpoints on synthetic shapes."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.synthetic_shapes import SyntheticShapesConfig, SyntheticShapesDataset
from ..data.image_folder import ImageFolderConfig, ImageFolderDataset
from ..disentangled_vae import DisentangledVAE, DisentangledVAEConfig


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda"):
        return torch.device(requested if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _collate_batch(batch: list[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {"image": images, "labels": labels}


def _labels_to_targets(
    labels: list[Optional[list[Dict[str, object]]]],
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    shape_ids = []
    color_ids = []
    bbox_targets = []
    for label_list in labels:
        if not label_list:
            shape_ids.append(0)
            color_ids.append(0)
            bbox_targets.append([0.0, 0.0, 1.0, 1.0])
            continue
        label = label_list[0]
        shape_ids.append(int(label["shape_idx"]))
        color_ids.append(int(label["color_idx"]))
        bbox_targets.append(list(label["bbox"]))
    if not shape_ids:
        return None
    return {
        "shape": torch.tensor(shape_ids, device=device, dtype=torch.long),
        "color": torch.tensor(color_ids, device=device, dtype=torch.long),
        "bbox": torch.tensor(bbox_targets, device=device, dtype=torch.float32),
    }


def evaluate(checkpoint_path: str, config_path: str, output_path: Optional[str] = None) -> Dict[str, float]:
    config = _load_config(config_path)
    device = _select_device(config.get("device", "auto"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg_dict = checkpoint.get("config", config.get("model", {}))
    model_cfg = DisentangledVAEConfig(**model_cfg_dict)
    model = DisentangledVAE(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_type = config.get("data", {}).get("type", "synthetic")
    if data_type == "synthetic":
        data_cfg = SyntheticShapesConfig(**config.get("data", {}))
        dataset = SyntheticShapesDataset(data_cfg)
    elif data_type == "image_folder":
        data_cfg = ImageFolderConfig(**config.get("data", {}))
        dataset = ImageFolderDataset(data_cfg)
    else:
        raise ValueError(f"Unknown data.type: {data_type}")
    loader = DataLoader(
        dataset,
        batch_size=config.get("eval", {}).get("batch_size", 64),
        shuffle=False,
        num_workers=config.get("eval", {}).get("num_workers", 0),
        collate_fn=_collate_batch,
    )

    total = 0
    recon_loss = 0.0
    shape_correct = 0
    color_correct = 0
    pos_loss = 0.0
    labeled = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = _labels_to_targets(batch["labels"], device)

            output = model(images)
            preds = model.predict_attributes(output["z"])

            recon_loss += F.mse_loss(output["recon"], images, reduction="sum").item()
            if labels is not None:
                pos_loss += F.mse_loss(preds["position"], labels["bbox"], reduction="sum").item()
                shape_preds = preds["shape_logits"].argmax(dim=1)
                color_preds = preds["color_logits"].argmax(dim=1)
                shape_correct += (shape_preds == labels["shape"]).sum().item()
                color_correct += (color_preds == labels["color"]).sum().item()
                labeled += images.size(0)
            total += images.size(0)

    metrics = {
        "recon_mse": recon_loss / (total * images.shape[1] * images.shape[2] * images.shape[3]),
        "shape_accuracy": shape_correct / labeled if labeled else None,
        "color_accuracy": color_correct / labeled if labeled else None,
        "position_mse": pos_loss / (labeled * 4) if labeled else None,
        "num_samples": float(total),
    }

    if output_path:
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate disentangled VAE checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="Path to config JSON (data/eval settings)")
    parser.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, args.config, output_path=args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
