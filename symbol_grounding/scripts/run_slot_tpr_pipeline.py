"""CLI demo for the Slot + Typed Slots + TPR pipeline."""
from __future__ import annotations

import argparse
from typing import Optional

import torch

from ..slot_attention import SlotAttentionConfig
from ..slot_tpr_pipeline import SlotTPRConfig, SlotTPRPipeline


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Slot-TPR pipeline on random inputs.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--image-size", type=int, default=64, help="Image size")
    parser.add_argument("--num-slots", type=int, default=7, help="Number of slots")
    args = parser.parse_args(argv)

    slot_cfg = SlotAttentionConfig(image_size=args.image_size, num_slots=args.num_slots)
    tpr_cfg = SlotTPRConfig()
    pipeline = SlotTPRPipeline(slot_config=slot_cfg, tpr_config=tpr_cfg)

    images = torch.rand(args.batch, 3, args.image_size, args.image_size)
    output = pipeline(images, verb="move_right")
    print("recon", tuple(output["recon"].shape))
    print("slots", tuple(output["slots"].shape))
    print("bound", tuple(output["bound"].shape))


if __name__ == "__main__":
    main()
