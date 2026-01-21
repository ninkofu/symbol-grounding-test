"""Train a disentangled beta-VAE on synthetic shapes."""
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
from ..disentangled_vae import DisentangledVAE, DisentangledVAEConfig, beta_vae_loss


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


def _save_recon(output_dir: str, step: int, originals: torch.Tensor, recons: torch.Tensor) -> None:
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


def _save_traversal(
    output_dir: str,
    step: int,
    model: DisentangledVAE,
    z: torch.Tensor,
    dim_index: int,
    span: float = 2.0,
    steps: int = 7,
    label: str = "shape",
) -> None:
    if not _PIL_AVAILABLE:
        return
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        z_base = z[0:1].clone()
        values = torch.linspace(-span, span, steps=steps, device=z.device)
        images = []
        for val in values:
            z_mod = z_base.clone()
            z_mod[:, dim_index] = val
            recon = model.decode(z_mod)[0]
            images.append(_tensor_to_pil(recon))

    width, height = images[0].size
    canvas = Image.new("RGB", (width * steps, height), color="white")
    for i, img in enumerate(images):
        canvas.paste(img, (i * width, 0))
    canvas.save(os.path.join(output_dir, f"traversal_{label}_{step:06d}.png"))


def _collate_batch(batch: list[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["labels"] for item in batch]
    return {"image": images, "labels": labels}


def _labels_to_targets(labels: list[list[Dict[str, object]]], device: torch.device) -> Dict[str, torch.Tensor]:
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
    return {
        "shape": torch.tensor(shape_ids, device=device, dtype=torch.long),
        "color": torch.tensor(color_ids, device=device, dtype=torch.long),
        "bbox": torch.tensor(bbox_targets, device=device, dtype=torch.float32),
    }


def train(config_path: str, output_dir: Optional[str] = None) -> None:
    config = _load_config(config_path)

    device = torch.device(_select_device(config.get("device", "auto")))
    torch.manual_seed(config.get("seed", 0))

    model_cfg = DisentangledVAEConfig(**config.get("model", {}))
    data_cfg = SyntheticShapesConfig(**config.get("data", {}))
    train_cfg = config.get("train", {})

    model = DisentangledVAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-4))

    dataset = SyntheticShapesDataset(data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        drop_last=True,
        collate_fn=_collate_batch,
    )

    outputs_root = output_dir or config.get("outputs_dir", "outputs/disentangled_vae")
    sample_dir = os.path.join(outputs_root, "samples")
    checkpoint_dir = os.path.join(outputs_root, "checkpoints")
    log_every = train_cfg.get("log_every", 50)
    sample_every = train_cfg.get("sample_every", 200)
    save_every = train_cfg.get("save_every", 500)
    epochs = train_cfg.get("epochs", 10)

    beta = train_cfg.get("beta", 4.0)
    shape_weight = train_cfg.get("shape_weight", 1.0)
    color_weight = train_cfg.get("color_weight", 1.0)
    position_weight = train_cfg.get("position_weight", 1.0)

    global_step = 0
    for epoch in range(epochs):
        for batch in loader:
            images = batch["image"].to(device)
            labels = _labels_to_targets(batch["labels"], device)

            output = model(images)
            loss_terms = beta_vae_loss(output["recon"], images, output["mu"], output["logvar"], beta=beta)
            preds = model.predict_attributes(output["z"])

            shape_loss = F.cross_entropy(preds["shape_logits"], labels["shape"])
            color_loss = F.cross_entropy(preds["color_logits"], labels["color"])
            position_loss = F.mse_loss(preds["position"], labels["bbox"])

            loss = (
                loss_terms["loss"]
                + shape_weight * shape_loss
                + color_weight * color_loss
                + position_weight * position_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % log_every == 0:
                print(
                    "epoch={epoch} step={step} loss={loss:.4f} recon={recon:.4f} kl={kl:.4f} "
                    "shape={shape:.4f} color={color:.4f} pos={pos:.4f}".format(
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item(),
                        recon=loss_terms["recon"].item(),
                        kl=loss_terms["kl"].item(),
                        shape=shape_loss.item(),
                        color=color_loss.item(),
                        pos=position_loss.item(),
                    )
                )

            if global_step % sample_every == 0:
                _save_recon(sample_dir, global_step, images, output["recon"])
                cfg = model.config
                _save_traversal(sample_dir, global_step, model, output["z"], 0, label="shape")
                _save_traversal(sample_dir, global_step, model, output["z"], cfg.shape_dim, label="color")

            if global_step % save_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": model_cfg.__dict__,
                        "step": global_step,
                    },
                    os.path.join(checkpoint_dir, f"checkpoint_{global_step:06d}.pt"),
                )

            global_step += 1

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model_cfg.__dict__,
            "step": global_step,
        },
        os.path.join(checkpoint_dir, "model_final.pt"),
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a disentangled beta-VAE.")
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
