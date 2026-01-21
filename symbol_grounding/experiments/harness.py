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


def _strength_from_edit(edit: Dict[str, Any]) -> float:
    raw = edit.get("strength", 0.75)
    if raw is None:
        raw = 0.75
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid strength value: {raw}") from exc
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"strength must be between 0 and 1: {value}")
    return value


def _strength_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _compute_semantic_metrics(
    after_path: str,
    mask_path: str,
    text: Optional[str],
    neg_text: Optional[str],
    out_path: str,
    semantic_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    from ..eval.semantic import evaluate_semantic

    pos_text = text if text is not None else ""
    neg_text = neg_text if neg_text is not None else None

    metrics = evaluate_semantic(
        after_path=after_path,
        mask_path=mask_path,
        text=pos_text,
        out_path=None,
        model_id=semantic_cfg.get("model_id", "openai/clip-vit-base-patch32"),
        device=semantic_cfg.get("device", "auto"),
        pad_px=int(semantic_cfg.get("pad_px", 8)),
    )

    if neg_text:
        neg_metrics = evaluate_semantic(
            after_path=after_path,
            mask_path=mask_path,
            text=neg_text,
            out_path=None,
            model_id=semantic_cfg.get("model_id", "openai/clip-vit-base-patch32"),
            device=semantic_cfg.get("device", "auto"),
            pad_px=int(semantic_cfg.get("pad_px", 8)),
        )
        metrics["neg_text"] = neg_text
        metrics["clip_similarity_neg"] = float(neg_metrics["clip_similarity"])
        metrics["clip_margin"] = float(metrics["clip_similarity"]) - float(neg_metrics["clip_similarity"])

    _write_json(out_path, metrics)
    return metrics


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
    semantic_cfg = eval_cfg.get("semantic", {}) if isinstance(eval_cfg.get("semantic", {}), dict) else {}

    seed_list = model_cfg.get("seed_list", [0])
    if not seed_list:
        seed_list = [0]

    results_path = os.path.join(exp_dir, "results.jsonl")

    cases = config.get("cases", [])
    for case_idx, case in enumerate(cases):
        base_prompt = case.get("base_prompt")
        if not base_prompt:
            raise ValueError("Each case must include base_prompt.")

        generate_prompt = case.get("generate_prompt", base_prompt)
        edit_base_prompt = case.get("edit_base_prompt", base_prompt)
        layout_prompt = case.get("layout_prompt", generate_prompt or base_prompt)

        scene_graph = parse_text(layout_prompt)
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
                "generate_prompt": generate_prompt,
                "edit_base_prompt": edit_base_prompt,
                "layout_prompt": layout_prompt,
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
                    prompt=generate_prompt,
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
                strength_list = edit.get("strength_list")
                if strength_list is None:
                    strength_values = [_strength_from_edit(edit)]
                else:
                    strength_values = [_strength_from_edit({"strength": s}) for s in strength_list]

                target_id_spec = edit.get("target")
                target_noun_spec = edit.get("target_noun")
                target_id_resolved = None
                target_noun_resolved = None

                if target_id_spec:
                    target_id_resolved = target_id_spec
                    obj = scene_graph.get_object(target_id_spec)
                    if obj is not None:
                        target_noun_resolved = obj.noun
                elif target_noun_spec:
                    noun_key = str(target_noun_spec).strip().lower()
                    for obj in scene_graph.objects:
                        if obj.noun.strip().lower() == noun_key:
                            target_id_resolved = obj.id
                            target_noun_resolved = obj.noun
                            break

                for strength_value in strength_values:
                    slug = _strength_slug(strength_value)
                    edit_dir = os.path.join(edits_dir, f"edit_{edit_idx:03d}_strength{slug}")
                    os.makedirs(edit_dir, exist_ok=True)

                    if target_id_resolved is None:
                        _append_jsonl(
                            results_path,
                            {
                                "case": case_idx,
                                "seed": seed,
                                "base_prompt": base_prompt,
                                "generate_prompt": generate_prompt,
                                "edit_base_prompt": edit_base_prompt,
                                "layout_prompt": layout_prompt,
                                "edit_index": edit_idx,
                                "strength": strength_value,
                                "target": target_id_resolved,
                                "target_id_spec": target_id_spec,
                                "target_noun_spec": target_noun_spec,
                                "target_id_resolved": target_id_resolved,
                                "target_noun_resolved": target_noun_resolved,
                                "skipped_reason": "target_not_found",
                                "paths": {
                                    "base_image": base_image_path,
                                    "control_image": control_image_path,
                                    "mask": None,
                                    "edited": None,
                                    "metrics": None,
                                },
                            },
                        )
                        continue

                    if target_id_resolved not in layout.boxes:
                        _append_jsonl(
                            results_path,
                            {
                                "case": case_idx,
                                "seed": seed,
                                "base_prompt": base_prompt,
                                "generate_prompt": generate_prompt,
                                "edit_base_prompt": edit_base_prompt,
                                "layout_prompt": layout_prompt,
                                "edit_index": edit_idx,
                                "strength": strength_value,
                                "target": target_id_resolved,
                                "target_id_spec": target_id_spec,
                                "target_noun_spec": target_noun_spec,
                                "target_id_resolved": target_id_resolved,
                                "target_noun_resolved": target_noun_resolved,
                                "skipped_reason": "target_not_in_layout",
                                "paths": {
                                    "base_image": base_image_path,
                                    "control_image": control_image_path,
                                    "mask": None,
                                    "edited": None,
                                    "metrics": None,
                                },
                            },
                        )
                        continue

                    mask = layout_to_mask(
                        layout,
                        target_id=target_id_resolved,
                        image_size=512,
                        pad_px=int(edit.get("mask_pad_px", 0)),
                        blur=float(edit.get("mask_blur", 0.0)),
                    )
                    mask_path = os.path.join(edit_dir, "mask.png")
                    mask.save(mask_path)

                    edit_meta = {
                        "base_prompt": base_prompt,
                        "edit_base_prompt": edit_base_prompt,
                        "generate_prompt": generate_prompt,
                        "layout_prompt": layout_prompt,
                        "edit_prompt": edit.get("edit_prompt"),
                        "negative_prompt": edit.get("negative_prompt"),
                        "target": target_id_spec,
                        "target_noun": target_noun_spec,
                        "target_id_spec": target_id_spec,
                        "target_noun_spec": target_noun_spec,
                        "target_id_resolved": target_id_resolved,
                        "target_noun_resolved": target_noun_resolved,
                        "mask_pad_px": edit.get("mask_pad_px", 0),
                        "mask_blur": edit.get("mask_blur", 0.0),
                        "seed": edit.get("seed", seed),
                        "strength": strength_value,
                    }
                    resolved_semantic_text = edit.get("semantic_text")
                    if resolved_semantic_text is None:
                        resolved_semantic_text = edit.get("edit_prompt", "")
                    edit_meta["semantic_text"] = resolved_semantic_text
                    edit_meta["semantic_neg_text"] = edit.get("semantic_neg_text")

                    edited_path = os.path.join(edit_dir, "edited.png")
                    metrics_path = os.path.join(edit_dir, "metrics.json")
                    null_dir = os.path.join(edit_dir, "null")
                    null_edited_path = os.path.join(null_dir, "edited.png")
                    null_metrics_path = os.path.join(null_dir, "metrics.json")

                    if not skip_edit:
                        request = EditRequest(
                            image_path=base_image_path,
                            mask_path=None,
                            mask_image=mask,
                            edit_prompt=edit.get("edit_prompt", ""),
                            base_prompt=edit_base_prompt,
                            negative_prompt=edit.get("negative_prompt"),
                            strength=strength_value,
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
                        save_viz = bool(eval_cfg.get("save_viz", False))
                        diff_path = os.path.join(edit_dir, "diff.png") if save_viz else None
                        overlay_path = os.path.join(edit_dir, "overlay.png") if save_viz else None
                        metrics = evaluate_locality(
                            before_path=base_image_path,
                            after_path=edited_path,
                            mask_path=mask_path,
                            threshold=eval_cfg.get("threshold", 10 / 255),
                            out_path=metrics_path,
                            save_diff=diff_path,
                            save_overlay=overlay_path,
                        )

                    semantic_metrics = None
                    if semantic_cfg.get("enabled", False) and edited_path:
                        semantic_path = os.path.join(edit_dir, "semantic.json")
                        semantic_text = edit.get("semantic_text")
                        if semantic_text is None:
                            semantic_text = edit.get("edit_prompt", "")
                        semantic_neg_text = edit.get("semantic_neg_text")
                        semantic_text = (semantic_text or "").strip()
                        if semantic_text:
                            semantic_metrics = _compute_semantic_metrics(
                                after_path=edited_path,
                                mask_path=mask_path,
                                text=semantic_text,
                                neg_text=semantic_neg_text,
                                out_path=semantic_path,
                                semantic_cfg=semantic_cfg,
                            )

                    null_metrics = None
                    null_semantic_metrics = None
                    if not skip_edit:
                        os.makedirs(null_dir, exist_ok=True)
                        null_request = EditRequest(
                            image_path=base_image_path,
                            mask_path=None,
                            mask_image=mask,
                            edit_prompt="",
                            base_prompt=edit_base_prompt,
                            negative_prompt=edit.get("negative_prompt"),
                            strength=strength_value,
                            seed=edit.get("seed", seed),
                            steps=model_cfg.get("steps", 30),
                            guidance_scale=model_cfg.get("guidance_scale", 7.5),
                            height=512,
                            width=512,
                            model_id=edit.get("model_id", config.get("inpaint_model_id", "runwayml/stable-diffusion-inpainting")),
                            device=model_cfg.get("device", "auto"),
                        )
                        null_result = editor.edit(null_request)
                        null_result.image.save(null_edited_path)

                        if not skip_eval:
                            save_viz = bool(eval_cfg.get("save_viz", False))
                            null_diff_path = os.path.join(null_dir, "diff.png") if save_viz else None
                            null_overlay_path = os.path.join(null_dir, "overlay.png") if save_viz else None
                            null_metrics = evaluate_locality(
                                before_path=base_image_path,
                                after_path=null_edited_path,
                                mask_path=mask_path,
                                threshold=eval_cfg.get("threshold", 10 / 255),
                                out_path=null_metrics_path,
                                save_diff=null_diff_path,
                                save_overlay=null_overlay_path,
                            )

                        if semantic_cfg.get("enabled", False):
                            null_semantic_path = os.path.join(null_dir, "semantic.json")
                            semantic_text = edit.get("semantic_text")
                            if semantic_text is None:
                                semantic_text = edit.get("edit_prompt", "")
                            semantic_neg_text = edit.get("semantic_neg_text")
                            semantic_text = (semantic_text or "").strip()
                            if semantic_text:
                                null_semantic_metrics = _compute_semantic_metrics(
                                    after_path=null_edited_path,
                                    mask_path=mask_path,
                                    text=semantic_text,
                                    neg_text=semantic_neg_text,
                                    out_path=null_semantic_path,
                                    semantic_cfg=semantic_cfg,
                                )

                    if semantic_metrics and null_semantic_metrics:
                        if "clip_margin" in semantic_metrics and "clip_margin" in null_semantic_metrics:
                            semantic_metrics["clip_margin_adj"] = float(semantic_metrics["clip_margin"]) - float(
                                null_semantic_metrics["clip_margin"]
                            )

                    _append_jsonl(
                        results_path,
                        {
                            "case": case_idx,
                            "seed": seed,
                            "base_prompt": base_prompt,
                            "generate_prompt": generate_prompt,
                            "edit_base_prompt": edit_base_prompt,
                            "layout_prompt": layout_prompt,
                            "edit_index": edit_idx,
                            "target": target_id_resolved,
                            "target_id_spec": target_id_spec,
                            "target_noun_spec": target_noun_spec,
                            "target_id_resolved": target_id_resolved,
                            "target_noun_resolved": target_noun_resolved,
                            "strength": strength_value,
                            "semantic_text": resolved_semantic_text,
                            "semantic_neg_text": edit.get("semantic_neg_text"),
                            "paths": {
                                "base_image": base_image_path,
                                "control_image": control_image_path,
                                "mask": mask_path,
                                "edited": edited_path,
                                "metrics": metrics_path if metrics else None,
                            },
                            "metrics": metrics,
                            "null": {
                                "edited": null_edited_path if not skip_edit else None,
                                "metrics": null_metrics_path if null_metrics else None,
                            },
                            "null_metrics": null_metrics,
                            "semantic_metrics": semantic_metrics,
                            "null_semantic_metrics": null_semantic_metrics,
                        },
                    )

    return exp_dir


__all__ = ["run_experiment"]
