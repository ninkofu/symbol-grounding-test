"""Train Slot Attention autoencoder on synthetic shapes."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

from ..data.synthetic_shapes import SyntheticShapesConfig, SyntheticShapesDataset
from ..slot_attention import SlotAttentionAutoEncoder, SlotAttentionConfig


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return requested


def _tensor_to_pil(img: torch.Tensor) -> "Image.Image":
    img = img.detach().cpu().clamp(0.0, 1.0)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def _save_recon_and_masks(
    output_dir: str,
    step: int,
    originals: torch.Tensor,
    recons: torch.Tensor,
    masks: torch.Tensor,
) -> None:
    if not _PIL_AVAILABLE:
        return

    os.makedirs(output_dir, exist_ok=True)
    orig = _tensor_to_pil(originals[0])
    recon = _tensor_to_pil(recons[0])

    width, height = orig.size
    canvas = Image.new("RGB", (width * 2, height), color="white")
    canvas.paste(orig, (0, 0))
    canvas.paste(recon, (width, 0))
    canvas.save(os.path.join(output_dir, f"recon_{step:06d}.png"))

    slot_masks = masks[0].detach().cpu()
    num_slots = slot_masks.shape[0]
    mask_canvas = Image.new("L", (width * num_slots, height), color=0)
    for i in range(num_slots):
        mask = slot_masks[i, 0]
        mask_img = (mask * 255.0).clamp(0, 255).byte().numpy()
        mask_pil = Image.fromarray(mask_img, mode="L")
        mask_canvas.paste(mask_pil, (i * width, 0))
    mask_canvas.save(os.path.join(output_dir, f"masks_{step:06d}.png"))


def _collate_batch(batch: list[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {"image": images, "labels": labels}


def train(config_path: str, output_dir: Optional[str] = None) -> None:
    config = _load_config(config_path)

    device = _select_device(config.get("device", "auto"))
    torch.manual_seed(config.get("seed", 0))

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})

    model = SlotAttentionAutoEncoder(SlotAttentionConfig(**model_cfg)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-4))

    dataset = SyntheticShapesDataset(SyntheticShapesConfig(**data_cfg))
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        drop_last=True,
        collate_fn=_collate_batch,
    )

    outputs_root = output_dir or config.get("outputs_dir", "outputs/slot_ae")
    sample_dir = os.path.join(outputs_root, "samples")
    log_every = train_cfg.get("log_every", 50)
    sample_every = train_cfg.get("sample_every", 200)
    epochs = train_cfg.get("epochs", 10)

    global_step = 0
    for epoch in range(epochs):
        for batch in loader:
            images = batch["image"].to(device)
            recons, info = model(images)
            loss = F.mse_loss(recons, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

            if global_step % sample_every == 0:
                _save_recon_and_masks(sample_dir, global_step, images, recons, info["masks"])

            global_step += 1


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Slot Attention autoencoder.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args(argv)

    try:
        train(args.config, output_dir=args.output_dir)
    except ImportError as exc:
        print(f"[ERROR] {exc}")
    except Exception as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
