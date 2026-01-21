"""Train a multimodal Slot Attention autoencoder with text conditioning."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image  # type: ignore

from ..data.captioned_shapes import CaptionedShapesConfig, CaptionedShapesDataset
from ..multimodal_slot_attention import MultiModalSlotAttentionAutoEncoder, MultiModalSlotAttentionConfig
from ..slot_attention import SlotAttentionConfig
from ..text_encoder import SimpleTextEncoder, TextEncoderConfig, build_shape_color_vocab


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


def _save_recon(
    output_dir: str,
    step: int,
    originals: torch.Tensor,
    recons: torch.Tensor,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    orig = _tensor_to_pil(originals[0])
    recon = _tensor_to_pil(recons[0])
    width, height = orig.size
    canvas = Image.new("RGB", (width * 2, height), color="white")
    canvas.paste(orig, (0, 0))
    canvas.paste(recon, (width, 0))
    canvas.save(os.path.join(output_dir, f"recon_{step:06d}.png"))


def _collate_batch(batch: list[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {"image": images, "caption": captions, "labels": labels}


def train(config_path: str, output_dir: Optional[str] = None) -> None:
    config = _load_config(config_path)

    device = _select_device(config.get("device", "auto"))
    torch.manual_seed(config.get("seed", 0))

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    text_cfg = config.get("text", {})

    slot_config = SlotAttentionConfig(**model_cfg.get("slot", {}))
    text_vocab = text_cfg.get("vocab", build_shape_color_vocab())
    text_config = TextEncoderConfig(
        vocab=text_vocab,
        embedding_dim=text_cfg.get("embedding_dim", 64),
    )
    text_encoder = SimpleTextEncoder(text_config).to(device)

    mm_config = MultiModalSlotAttentionConfig(
        slot_config=slot_config,
        text_dim=text_config.embedding_dim,
        conditioning_scale=model_cfg.get("conditioning_scale", 1.0),
    )
    model = MultiModalSlotAttentionAutoEncoder(mm_config).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(text_encoder.parameters()),
        lr=train_cfg.get("lr", 1e-4),
    )

    dataset = CaptionedShapesDataset(CaptionedShapesConfig(**data_cfg))
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        drop_last=True,
        collate_fn=_collate_batch,
    )

    outputs_root = output_dir or config.get("outputs_dir", "outputs/multimodal_slot_ae")
    sample_dir = os.path.join(outputs_root, "samples")
    log_every = train_cfg.get("log_every", 50)
    sample_every = train_cfg.get("sample_every", 200)
    epochs = train_cfg.get("epochs", 10)

    global_step = 0
    for epoch in range(epochs):
        for batch in loader:
            images = batch["image"].to(device)
            captions = batch["caption"]
            text_emb = text_encoder(captions)

            recons, _info = model(images, text_emb)
            loss = F.mse_loss(recons, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

            if global_step % sample_every == 0:
                _save_recon(sample_dir, global_step, images, recons)

            global_step += 1


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a multimodal Slot Attention autoencoder.")
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
