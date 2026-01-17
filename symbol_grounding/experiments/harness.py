"""Experiment harness for batch generation and editing."""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from ..diffusion import DiffusionConfig
from ..editing import EditRequest, InpaintEditor
from ..eval.locality import evaluate_locality
from ..layout import generate_layout, layout_to_mask
from ..pipeline_diffusion import generate_diffusion_image
from ..scene_graph import parse_text


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_experiment(
    config_path: str,
    output_root: str,
    skip_generate: bool = False,
    skip_edit: bool = False,
    skip_eval: bool = False,
) -> str:
    """Run experiment config and return the output directory."""
    config = _load_config(config_path)
    name = config.get("name", "experiment")
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_root, f"{name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    model_cfg = config.get("model", {})
    control_cfg = config.get("controlnet", {})
    eval_cfg = config.get("eval", {})

    seed_list = model_cfg.get("seed_list", [0])
    if not seed_list:
        seed_list = [0]

    results_path = os.path.join(exp_dir, "results.jsonl")

    cases = config.get("cases", [])
    for case_idx, case in enumerate(cases):
        base_prompt = case.get("base_prompt")
        if not base_prompt:
            raise ValueError("Each case must include base_prompt.")

        scene_graph = parse_text(base_prompt)
        layout = generate_layout(scene_graph)

        for seed in seed_list:
            case_dir = os.path.join(exp_dir, f"case_{case_idx:03d}_seed{seed:03d}")
            base_dir = os.path.join(case_dir, "base")
            edits_dir = os.path.join(case_dir, "edits")
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(edits_dir, exist_ok=True)

            base_image_path = os.path.join(base_dir, "image.png")
            control_image_path = os.path.join(base_dir, "control.png")

            base_meta = {
                "base_prompt": base_prompt,
                "seed": seed,
                "model": {
                    "model_id": model_cfg.get("model_id", "runwayml/stable-diffusion-v1-5"),
                    "steps": model_cfg.get("steps", 30),
                    "guidance_scale": model_cfg.get("guidance_scale", 7.5),
                    "device": model_cfg.get("device", "auto"),
                },
                "controlnet": {
                    "enabled": bool(control_cfg.get("enabled", False)),
                    "controlnet_model_id": control_cfg.get("controlnet_model_id"),
                    "controlnet_scale": control_cfg.get("controlnet_scale", 1.0),
                    "control_mode": control_cfg.get("control_mode", "scribble"),
                },
            }

            if skip_generate:
                base_source = case.get("base_image")
                if not base_source:
                    raise ValueError("skip_generate is set but base_image is missing in case config.")
                shutil.copy(base_source, base_image_path)
                control_image = None
                try:
                    from ..diffusion import layout_to_control_image

                    control_image = layout_to_control_image(
                        layout, image_size=512, mode=control_cfg.get("control_mode", "scribble")
                    )
                except Exception:
                    control_image = None
                if control_image is not None:
                    control_image.save(control_image_path)
            else:
                config_obj = DiffusionConfig(
                    model_id=model_cfg.get("model_id", "runwayml/stable-diffusion-v1-5"),
                    height=512,
                    width=512,
                    num_inference_steps=model_cfg.get("steps", 30),
                    guidance_scale=model_cfg.get("guidance_scale", 7.5),
                    seed=seed,
                    device=model_cfg.get("device", "auto"),
                    negative_prompt=model_cfg.get("negative_prompt"),
                    use_controlnet=bool(control_cfg.get("enabled", False)),
                    controlnet_model_id=control_cfg.get("controlnet_model_id"),
                    controlnet_conditioning_scale=control_cfg.get("controlnet_scale", 1.0),
                )

                result = generate_diffusion_image(
                    prompt=base_prompt,
                    output_dir=base_dir,
                    config=config_obj,
                    control_mode=control_cfg.get("control_mode", "scribble"),
                    scene_graph=scene_graph,
                    layout=layout,
                )
                shutil.copy(result.image_path, base_image_path)
                shutil.copy(result.control_path, control_image_path)
                base_meta["generated_paths"] = {
                    "image": result.image_path,
                    "control": result.control_path,
                }

            _write_json(os.path.join(base_dir, "meta.json"), base_meta)

            edits = case.get("edits", [])
            editor = InpaintEditor()
            for edit_idx, edit in enumerate(edits):
                edit_dir = os.path.join(edits_dir, f"edit_{edit_idx:03d}")
                os.makedirs(edit_dir, exist_ok=True)

                target = edit.get("target")
                if not target:
                    raise ValueError("Each edit must include target (object id).")

                mask = layout_to_mask(
                    layout,
                    target_id=target,
                    image_size=512,
                    pad_px=int(edit.get("mask_pad_px", 0)),
                    blur=float(edit.get("mask_blur", 0.0)),
                )
                mask_path = os.path.join(edit_dir, "mask.png")
                mask.save(mask_path)

                edit_meta = {
                    "base_prompt": base_prompt,
                    "edit_prompt": edit.get("edit_prompt"),
                    "negative_prompt": edit.get("negative_prompt"),
                    "target": target,
                    "mask_pad_px": edit.get("mask_pad_px", 0),
                    "mask_blur": edit.get("mask_blur", 0.0),
                    "seed": edit.get("seed", seed),
                }

                edited_path = os.path.join(edit_dir, "edited.png")
                metrics_path = os.path.join(edit_dir, "metrics.json")

                if not skip_edit:
                    request = EditRequest(
                        image_path=base_image_path,
                        mask_path=None,
                        mask_image=mask,
                        edit_prompt=edit.get("edit_prompt", ""),
                        base_prompt=base_prompt,
                        negative_prompt=edit.get("negative_prompt"),
                        seed=edit.get("seed", seed),
                        steps=model_cfg.get("steps", 30),
                        guidance_scale=model_cfg.get("guidance_scale", 7.5),
                        height=512,
                        width=512,
                        model_id=edit.get("model_id", config.get("inpaint_model_id", "runwayml/stable-diffusion-inpainting")),
                        device=model_cfg.get("device", "auto"),
                    )

                    result = editor.edit(request)
                    result.image.save(edited_path)
                else:
                    edited_path = None

                _write_json(os.path.join(edit_dir, "meta.json"), edit_meta)

                metrics = None
                if not skip_eval and edited_path:
                    metrics = evaluate_locality(
                        before_path=base_image_path,
                        after_path=edited_path,
                        mask_path=mask_path,
                        threshold=eval_cfg.get("threshold", 10 / 255),
                        out_path=metrics_path,
                    )

                _append_jsonl(
                    results_path,
                    {
                        "case": case_idx,
                        "seed": seed,
                        "base_prompt": base_prompt,
                        "edit_index": edit_idx,
                        "target": target,
                        "paths": {
                            "base_image": base_image_path,
                            "control_image": control_image_path,
                            "mask": mask_path,
                            "edited": edited_path,
                            "metrics": metrics_path if metrics else None,
                        },
                        "metrics": metrics,
                    },
                )

    return exp_dir


__all__ = ["run_experiment"]
