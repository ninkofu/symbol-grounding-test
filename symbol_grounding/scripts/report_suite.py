"""Generate a visual markdown report from suite_index.json."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a markdown report from suite results.")
    parser.add_argument("--suite-index", required=True, help="Path to suite_index.json")
    parser.add_argument("--out", required=True, help="Output directory for the report")
    parser.add_argument("--topk", type=int, default=5, help="Top-k samples per category")
    return parser


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _experiment_label(entry: Dict[str, Any]) -> str:
    name = entry.get("experiment_name")
    if name:
        return str(name)
    exp_dir = entry.get("experiment_dir")
    if exp_dir:
        return Path(exp_dir).name
    config_path = entry.get("config_path")
    if config_path:
        return Path(config_path).stem
    return "experiment"


def _semantic_delta(row: Dict[str, Any]) -> Optional[float]:
    margin = row.get("clip_margin_delta_raw")
    if margin is not None:
        return float(margin)
    sim = row.get("clip_similarity_delta_raw")
    if sim is not None:
        return float(sim)
    return None


def _outside_adj(row: Dict[str, Any]) -> Optional[float]:
    value = row.get("outside_mean_abs_diff_adj")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _sort_key_pareto(row: Dict[str, Any]) -> Tuple[float, float]:
    outside = _outside_adj(row)
    semantic = _semantic_delta(row)
    if outside is None or semantic is None:
        return (float("inf"), float("-inf"))
    return (outside, -semantic)


def _select_topk_pareto(rows: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    valid = [r for r in rows if _outside_adj(r) is not None and _semantic_delta(r) is not None]
    return sorted(valid, key=_sort_key_pareto)[: max(0, int(topk))]


def _select_topk_worst_leakage(rows: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    valid = [r for r in rows if _outside_adj(r) is not None]
    return sorted(valid, key=lambda r: _outside_adj(r) or 0.0, reverse=True)[: max(0, int(topk))]


def _select_topk_best_semantic(rows: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    valid = [r for r in rows if _semantic_delta(r) is not None]
    return sorted(valid, key=lambda r: _semantic_delta(r) or 0.0, reverse=True)[: max(0, int(topk))]


def _resolve_paths(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    paths = row.get("paths") if isinstance(row.get("paths"), dict) else {}
    base_image = row.get("base_image") or paths.get("base_image")
    edited = row.get("edited") or paths.get("edited")
    null_edited = row.get("null_edited") or paths.get("null_edited")
    mask = row.get("mask") or paths.get("mask")
    if not mask and edited:
        mask = str(Path(edited).parent / "mask.png")
    return {
        "base_image": base_image,
        "edited": edited,
        "null_edited": null_edited,
        "mask": mask,
    }


def _copy_if_exists(src: Optional[str], dest: Path) -> Optional[str]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return str(dest)


def _find_extra_images(edited_path: Optional[str]) -> Dict[str, Optional[str]]:
    if not edited_path:
        return {"diff": None, "overlay": None, "heatmap": None}
    base_dir = Path(edited_path).parent
    candidates = {
        "diff": base_dir / "diff.png",
        "overlay": base_dir / "overlay.png",
        "heatmap": base_dir / "heatmap.png",
    }
    found: Dict[str, Optional[str]] = {}
    for key, path in candidates.items():
        found[key] = str(path) if path.exists() else None
    return found


def _load_edit_meta(edited_path: Optional[str]) -> Dict[str, Any]:
    if not edited_path:
        return {}
    meta_path = Path(edited_path).parent / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return _read_json(meta_path)
    except Exception:
        return {}


def _write_report_section(
    f,
    title: str,
    rows: List[Dict[str, Any]],
    assets_dir: Path,
    exp_name: str,
    category: str,
) -> None:
    f.write(f"### {title}\n\n")
    if not rows:
        f.write("_No samples found._\n\n")
        return

    for idx, row in enumerate(rows):
        sample_id = f"{category}_{idx:03d}"
        sample_dir = assets_dir / exp_name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        meta_paths = _resolve_paths(row)
        extra_paths = _find_extra_images(meta_paths.get("edited"))
        meta = _load_edit_meta(meta_paths.get("edited"))

        base_prompt = meta.get("base_prompt") or row.get("base_prompt")
        generate_prompt = meta.get("generate_prompt") or row.get("generate_prompt")
        edit_base_prompt = meta.get("edit_base_prompt") or row.get("edit_base_prompt")
        edit_prompt = meta.get("edit_prompt") or row.get("edit_prompt")
        semantic_text = meta.get("semantic_text") or row.get("semantic_text")
        semantic_neg_text = meta.get("semantic_neg_text") or row.get("semantic_neg_text")
        mask_pad_px = meta.get("mask_pad_px") if "mask_pad_px" in meta else row.get("mask_pad_px")
        mask_blur = meta.get("mask_blur") if "mask_blur" in meta else row.get("mask_blur")
        strength = meta.get("strength") if "strength" in meta else row.get("strength")
        negative_prompt = meta.get("negative_prompt") or row.get("negative_prompt")
        seed = meta.get("seed") if "seed" in meta else row.get("seed")
        target_id_spec = meta.get("target_id_spec") if "target_id_spec" in meta else row.get("target_id_spec")
        target_noun_spec = meta.get("target_noun_spec") if "target_noun_spec" in meta else row.get("target_noun_spec")
        target_id_resolved = meta.get("target_id_resolved") if "target_id_resolved" in meta else row.get("target_id_resolved")
        target_noun_resolved = meta.get("target_noun_resolved") if "target_noun_resolved" in meta else row.get("target_noun_resolved")
        skipped_reason = row.get("skipped_reason")
        outside_mean_abs_raw = row.get("outside_mean_abs_diff_raw")
        outside_mean_abs_null = row.get("outside_mean_abs_diff_null")
        outside_mean_abs_delta_raw = row.get("outside_mean_abs_diff_delta_raw")
        outside_mean_abs_delta_clipped = row.get("outside_mean_abs_diff_delta_clipped")
        outside_frac_raw = row.get("outside_frac_changed_raw")
        outside_frac_null = row.get("outside_frac_changed_null")
        outside_frac_delta_raw = row.get("outside_frac_changed_delta_raw")
        outside_frac_delta_clipped = row.get("outside_frac_changed_delta_clipped")
        clip_margin = row.get("clip_margin")
        null_clip_margin = row.get("null_clip_margin")
        clip_margin_delta_raw = row.get("clip_margin_delta_raw")

        copied: Dict[str, Optional[str]] = {}
        copied["base_image"] = _copy_if_exists(meta_paths.get("base_image"), sample_dir / "base.png")
        copied["edited"] = _copy_if_exists(meta_paths.get("edited"), sample_dir / "edited.png")
        copied["null_edited"] = _copy_if_exists(meta_paths.get("null_edited"), sample_dir / "null_edited.png")
        copied["mask"] = _copy_if_exists(meta_paths.get("mask"), sample_dir / "mask.png")
        copied["diff"] = _copy_if_exists(extra_paths.get("diff"), sample_dir / "diff.png")
        copied["overlay"] = _copy_if_exists(extra_paths.get("overlay"), sample_dir / "overlay.png")
        copied["heatmap"] = _copy_if_exists(extra_paths.get("heatmap"), sample_dir / "heatmap.png")

        f.write(f"#### {sample_id}\n\n")
        f.write(f"- case: {row.get('case')}\n")
        f.write(f"- seed: {row.get('seed')}\n")
        f.write(f"- edit_index: {row.get('edit_index')}\n")
        f.write(f"- target: {row.get('target')}\n")
        f.write(f"- outside_mean_abs_diff_adj: {row.get('outside_mean_abs_diff_adj')}\n")
        f.write(f"- semantic_delta_raw: {_semantic_delta(row)}\n\n")

        f.write("Conditions:\n\n")
        if base_prompt is not None:
            f.write(f"- base_prompt: {base_prompt}\n")
        if generate_prompt is not None:
            f.write(f"- generate_prompt: {generate_prompt}\n")
        if edit_base_prompt is not None:
            f.write(f"- edit_base_prompt: {edit_base_prompt}\n")
        if target_id_spec is not None:
            f.write(f"- target_id_spec: {target_id_spec}\n")
        if target_noun_spec is not None:
            f.write(f"- target_noun_spec: {target_noun_spec}\n")
        if target_id_resolved is not None:
            f.write(f"- target_id_resolved: {target_id_resolved}\n")
        if target_noun_resolved is not None:
            f.write(f"- target_noun_resolved: {target_noun_resolved}\n")
        if edit_prompt is not None:
            f.write(f"- edit_prompt: {edit_prompt}\n")
        if semantic_text is not None:
            f.write(f"- semantic_text: {semantic_text}\n")
        if semantic_neg_text is not None:
            f.write(f"- semantic_neg_text: {semantic_neg_text}\n")
        if mask_pad_px is not None:
            f.write(f"- mask_pad_px: {mask_pad_px}\n")
        if mask_blur is not None:
            f.write(f"- mask_blur: {mask_blur}\n")
        if strength is not None:
            f.write(f"- strength: {strength}\n")
        if negative_prompt is not None:
            f.write(f"- negative_prompt: {negative_prompt}\n")
        if seed is not None:
            f.write(f"- seed: {seed}\n")
        if skipped_reason is not None:
            f.write(f"- skipped_reason: {skipped_reason}\n")
        if outside_mean_abs_raw is not None:
            f.write(f"- outside_mean_abs_diff_raw: {outside_mean_abs_raw}\n")
        if outside_mean_abs_null is not None:
            f.write(f"- outside_mean_abs_diff_null: {outside_mean_abs_null}\n")
        if outside_mean_abs_delta_raw is not None:
            f.write(f"- outside_mean_abs_diff_delta_raw: {outside_mean_abs_delta_raw}\n")
        if outside_mean_abs_delta_clipped is not None:
            f.write(f"- outside_mean_abs_diff_delta_clipped: {outside_mean_abs_delta_clipped}\n")
        if outside_frac_raw is not None:
            f.write(f"- outside_frac_changed_raw: {outside_frac_raw}\n")
        if outside_frac_null is not None:
            f.write(f"- outside_frac_changed_null: {outside_frac_null}\n")
        if outside_frac_delta_raw is not None:
            f.write(f"- outside_frac_changed_delta_raw: {outside_frac_delta_raw}\n")
        if outside_frac_delta_clipped is not None:
            f.write(f"- outside_frac_changed_delta_clipped: {outside_frac_delta_clipped}\n")
        if clip_margin is not None:
            f.write(f"- clip_margin: {clip_margin}\n")
        if null_clip_margin is not None:
            f.write(f"- null_clip_margin: {null_clip_margin}\n")
        if clip_margin_delta_raw is not None:
            f.write(f"- clip_margin_delta_raw: {clip_margin_delta_raw}\n")
        f.write("\n")

        for label in ["base_image", "edited", "null_edited", "mask", "diff", "overlay", "heatmap"]:
            path = copied.get(label)
            if path:
                rel_path = os.path.relpath(path, assets_dir.parent)
                f.write(f"{label}:\n\n![]({rel_path})\n\n")
            else:
                missing_path = meta_paths.get(label) or extra_paths.get(label)
                if missing_path:
                    f.write(f"{label}: (missing) {missing_path}\n\n")
        f.write("\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    suite_index_path = Path(args.suite_index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = _read_json(suite_index_path)
    entries = suite.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("suite_index.json entries must be a list")

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Suite Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"Suite index: {suite_index_path}\n\n")

        for entry in entries:
            if entry.get("status") != "ok":
                continue
            summary_path = entry.get("summary_json")
            if not summary_path or not Path(summary_path).exists():
                f.write(f"## Experiment: {_experiment_label(entry)}\n\n")
                f.write("_Missing locality_summary.json; skipped._\n\n")
                continue

            payload = _read_json(Path(summary_path))
            summary = payload.get("summary", {})
            rows = payload.get("rows", [])
            pareto_rows = summary.get("pareto_front", [])

            exp_name = _experiment_label(entry)
            f.write(f"## Experiment: {exp_name}\n\n")

            pareto_topk = _select_topk_pareto(pareto_rows, args.topk)
            worst_topk = _select_topk_worst_leakage(rows, args.topk)
            best_topk = _select_topk_best_semantic(rows, args.topk)

            _write_report_section(f, "Pareto topk", pareto_topk, assets_dir, exp_name, "pareto")
            _write_report_section(f, "Worst leakage topk", worst_topk, assets_dir, exp_name, "worst")
            _write_report_section(f, "Best semantic topk", best_topk, assets_dir, exp_name, "best")

    print(f"[DONE] Report written to {report_path}")


if __name__ == "__main__":
    main()
