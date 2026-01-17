"""Aggregate locality metrics from an experiment manifest (results.jsonl).

This script NEVER glob-matches images. It only uses explicit paths from results.jsonl
produced by `symbol_grounding.scripts.run_experiment`.

It also validates that:
- base_image is under .../case_XXX_seedYYY/base/
- edited/mask/metrics are under the SAME case_XXX_seedYYY/ directory
So cross-case mixing is structurally prevented.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {path}") from exc


def _is_within(path: Path, root: Path) -> bool:
    """True if path is inside root (after resolve)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": float(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _default_out_paths(results_jsonl: Path) -> Tuple[Path, Path]:
    out_csv = results_jsonl.with_name("locality_summary.csv")
    out_json = results_jsonl.with_name("locality_summary.json")
    return out_csv, out_json


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate locality metrics from results.jsonl (no glob mixing).")
    p.add_argument(
        "--results",
        required=True,
        help="Path to results.jsonl OR an experiment directory that contains results.jsonl",
    )
    p.add_argument("--out-csv", default=None, help="Output CSV path (default: alongside results.jsonl)")
    p.add_argument("--out-json", default=None, help="Output JSON path (default: alongside results.jsonl)")
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute metrics from images instead of reading metrics.json / embedded metrics",
    )
    p.add_argument(
        "--write-metrics",
        action="store_true",
        help="When recomputing, write metrics.json to the path specified in results.jsonl (if any).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold (0..1). If omitted, uses metrics.json value if present, else default 10/255.",
    )
    p.add_argument("--topk", type=int, default=10, help="Show top-k worst leakage rows in stdout.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    results_path = Path(args.results)
    if results_path.is_dir():
        results_jsonl = results_path / "results.jsonl"
    else:
        results_jsonl = results_path

    if not results_jsonl.exists():
        print(f"[ERROR] results.jsonl not found: {results_jsonl}", file=sys.stderr)
        sys.exit(1)

    out_csv, out_json = _default_out_paths(results_jsonl)
    if args.out_csv:
        out_csv = Path(args.out_csv)
    if args.out_json:
        out_json = Path(args.out_json)

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # Lazy import for recompute mode (so aggregation from existing metrics can work without numpy/PIL).
    evaluate_locality = None
    if args.recompute:
        try:
            from symbol_grounding.eval.locality import evaluate_locality as _evaluate_locality  # type: ignore
        except Exception as exc:
            print(f"[ERROR] Failed to import symbol_grounding.eval.locality.evaluate_locality: {exc}", file=sys.stderr)
            sys.exit(1)
        evaluate_locality = _evaluate_locality

    for rec in _read_jsonl(results_jsonl):
        paths = rec.get("paths") or {}

        base_image = paths.get("base_image")
        edited = paths.get("edited")
        mask = paths.get("mask")
        metrics_path = paths.get("metrics")

        # Skip incomplete records (e.g., --skip-edit / --skip-eval cases)
        if not base_image or not edited or not mask:
            skipped.append(
                {
                    "reason": "missing_paths",
                    "case": rec.get("case"),
                    "seed": rec.get("seed"),
                    "edit_index": rec.get("edit_index"),
                    "paths": paths,
                }
            )
            continue

        base_p = Path(base_image)
        edited_p = Path(edited)
        mask_p = Path(mask)
        metrics_p = Path(metrics_path) if metrics_path else None

        # Existence checks
        missing = [str(p) for p in [base_p, edited_p, mask_p] if not p.exists()]
        if missing:
            skipped.append(
                {
                    "reason": "missing_files",
                    "missing": missing,
                    "case": rec.get("case"),
                    "seed": rec.get("seed"),
                    "edit_index": rec.get("edit_index"),
                }
            )
            continue

        # Structural anti-mixing check:
        # base: .../case_XXX_seedYYY/base/image.png -> case_dir = .../case_XXX_seedYYY
        try:
            base_dir = base_p.parent  # .../base
            case_dir = base_dir.parent  # .../case_XXX_seedYYY
        except Exception:
            skipped.append(
                {
                    "reason": "bad_base_path",
                    "base_image": str(base_p),
                    "case": rec.get("case"),
                    "seed": rec.get("seed"),
                    "edit_index": rec.get("edit_index"),
                }
            )
            continue

        if not _is_within(edited_p, case_dir) or not _is_within(mask_p, case_dir):
            skipped.append(
                {
                    "reason": "pair_mismatch_outside_case_dir",
                    "case_dir": str(case_dir),
                    "base_image": str(base_p),
                    "edited": str(edited_p),
                    "mask": str(mask_p),
                    "case": rec.get("case"),
                    "seed": rec.get("seed"),
                    "edit_index": rec.get("edit_index"),
                }
            )
            continue

        metrics: Optional[Dict[str, Any]] = None

        # Prefer: metrics.json on disk > embedded rec["metrics"]
        if not args.recompute:
            if metrics_p and metrics_p.exists():
                try:
                    metrics = _read_json(metrics_p)
                except Exception:
                    metrics = None
            if metrics is None and isinstance(rec.get("metrics"), dict):
                metrics = rec.get("metrics")

        if metrics is None:
            if evaluate_locality is None:
                skipped.append(
                    {
                        "reason": "no_metrics_and_not_recompute",
                        "case": rec.get("case"),
                        "seed": rec.get("seed"),
                        "edit_index": rec.get("edit_index"),
                        "metrics_path": str(metrics_p) if metrics_p else None,
                    }
                )
                continue

            thr = args.threshold
            if thr is None:
                thr = 10 / 255

            out_path = None
            if args.write_metrics:
                if metrics_p is not None:
                    out_path = str(metrics_p)
                else:
                    # Fallback: place next to edited image
                    out_path = str(edited_p.parent / "metrics.json")

            metrics = evaluate_locality(
                before_path=str(base_p),
                after_path=str(edited_p),
                mask_path=str(mask_p),
                threshold=float(thr),
                out_path=out_path,
            )

        # Flatten key fields for CSV
        row: Dict[str, Any] = {
            "case": rec.get("case"),
            "seed": rec.get("seed"),
            "edit_index": rec.get("edit_index"),
            "target": rec.get("target"),
            "base_prompt": rec.get("base_prompt"),
            "base_image": str(base_p),
            "edited": str(edited_p),
            "mask": str(mask_p),
            "metrics_path": str(metrics_p) if metrics_p else "",
            "outside_mean_abs_diff": _safe_float(metrics.get("outside_mean_abs_diff")),
            "outside_mean_sq_diff": _safe_float(metrics.get("outside_mean_sq_diff")),
            "outside_frac_changed": _safe_float(metrics.get("outside_frac_changed")),
            "inside_mean_abs_diff": _safe_float(metrics.get("inside_mean_abs_diff")),
            "inside_frac_changed": _safe_float(metrics.get("inside_frac_changed")),
            "threshold": _safe_float(metrics.get("threshold")),
        }
        rows.append(row)

    # Write outputs
    os.makedirs(out_csv.parent, exist_ok=True)
    os.makedirs(out_json.parent, exist_ok=True)

    # CSV field order (stable)
    fieldnames = [
        "case",
        "seed",
        "edit_index",
        "target",
        "outside_mean_abs_diff",
        "outside_mean_sq_diff",
        "outside_frac_changed",
        "inside_mean_abs_diff",
        "inside_frac_changed",
        "threshold",
        "base_image",
        "edited",
        "mask",
        "metrics_path",
        "base_prompt",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "results_jsonl": str(results_jsonl),
        "n_rows": len(rows),
        "n_skipped": len(skipped),
        "skipped_reasons": {},
        "outside_mean_abs_diff": _summarize([r["outside_mean_abs_diff"] for r in rows if r["outside_mean_abs_diff"] is not None]),
        "outside_frac_changed": _summarize([r["outside_frac_changed"] for r in rows if r["outside_frac_changed"] is not None]),
        "inside_mean_abs_diff": _summarize([r["inside_mean_abs_diff"] for r in rows if r["inside_mean_abs_diff"] is not None]),
        "inside_frac_changed": _summarize([r["inside_frac_changed"] for r in rows if r["inside_frac_changed"] is not None]),
        "top_worst_by_outside_mean_abs_diff": [],
    }

    # Count skipped reasons
    reasons: Dict[str, int] = {}
    for s in skipped:
        reasons[s.get("reason", "unknown")] = reasons.get(s.get("reason", "unknown"), 0) + 1
    summary["skipped_reasons"] = reasons

    # Worst rows (largest outside_mean_abs_diff)
    worst = sorted(
        [r for r in rows if r["outside_mean_abs_diff"] is not None],
        key=lambda r: float(r["outside_mean_abs_diff"]),
        reverse=True,
    )[: max(0, int(args.topk))]
    summary["top_worst_by_outside_mean_abs_diff"] = [
        {
            "case": r["case"],
            "seed": r["seed"],
            "edit_index": r["edit_index"],
            "target": r["target"],
            "outside_mean_abs_diff": r["outside_mean_abs_diff"],
            "outside_frac_changed": r["outside_frac_changed"],
            "base_image": r["base_image"],
            "edited": r["edited"],
        }
        for r in worst
    ]

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "rows": rows,
                "skipped": skipped,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Print concise stdout summary
    print("=== locality aggregation ===")
    print(f"results:  {results_jsonl}")
    print(f"rows:     {len(rows)}")
    print(f"skipped:  {len(skipped)}  reasons={reasons}")
    print(f"out_csv:  {out_csv}")
    print(f"out_json: {out_json}")
    print("")
    print("=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
