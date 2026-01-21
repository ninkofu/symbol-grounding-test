"""Plot suite results and emit CSV summaries."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot suite pareto/strength sweep and emit CSVs.")
    parser.add_argument("--suite-index", required=True, help="Path to suite_index.json")
    parser.add_argument("--out", required=True, help="Output directory for plots/CSVs")
    return parser


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _semantic_delta(row: Dict[str, Any]) -> Optional[float]:
    margin = row.get("clip_margin_delta_raw")
    if margin is not None:
        return float(margin)
    sim = row.get("clip_similarity_delta_raw")
    if sim is not None:
        return float(sim)
    return None


def _summary_strength_delta(summary_by_strength: Dict[str, Any]) -> Optional[float]:
    margin = summary_by_strength.get("clip_margin_delta_raw")
    if margin and margin.get("n", 0) > 0:
        return float(margin.get("mean", 0.0))
    sim = summary_by_strength.get("clip_similarity_delta_raw")
    if sim and sim.get("n", 0) > 0:
        return float(sim.get("mean", 0.0))
    return None


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(path.parent, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _try_import_matplotlib() -> Optional[Any]:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    suite_index = Path(args.suite_index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = _read_json(suite_index)
    entries = index.get("entries", [])
    if not isinstance(entries, list):
        print("[ERROR] Invalid suite_index.json: entries must be a list", file=sys.stderr)
        sys.exit(1)

    plt = _try_import_matplotlib()
    if plt is None:
        print("[WARN] matplotlib not available; plots will be skipped (CSV only).")

    suite_pareto_rows: List[Dict[str, Any]] = []

    for entry in entries:
        if entry.get("status") != "ok":
            continue
        summary_json_path = entry.get("summary_json")
        if not summary_json_path:
            continue
        summary_path = Path(summary_json_path)
        if not summary_path.exists():
            continue

        payload = _read_json(summary_path)
        summary = payload.get("summary", {})
        pareto = summary.get("pareto_front", [])
        summary_by_strength = summary.get("summary_by_strength", {})

        name = _experiment_label(entry)

        # Pareto CSV
        pareto_rows: List[Dict[str, Any]] = []
        for row in pareto:
            x = row.get("outside_mean_abs_diff_adj")
            y = _semantic_delta(row)
            if x is None or y is None:
                continue
            pareto_rows.append(
                {
                    "experiment": name,
                    "case": row.get("case"),
                    "seed": row.get("seed"),
                    "edit_index": row.get("edit_index"),
                    "target": row.get("target"),
                    "strength": row.get("strength"),
                    "outside_mean_abs_diff_adj": x,
                    "outside_frac_changed_adj": row.get("outside_frac_changed_adj"),
                    "semantic_delta_raw": y,
                    "base_image": (row.get("paths") or {}).get("base_image"),
                    "edited": (row.get("paths") or {}).get("edited"),
                    "null_edited": (row.get("paths") or {}).get("null_edited"),
                }
            )
        if pareto_rows:
            pareto_csv = out_dir / f"pareto_{name}.csv"
            _write_csv(pareto_csv, pareto_rows)
            suite_pareto_rows.extend(pareto_rows)

        # Strength sweep CSV
        strength_rows: List[Dict[str, Any]] = []
        if isinstance(summary_by_strength, dict):
            for strength_key, stats in summary_by_strength.items():
                if not isinstance(stats, dict):
                    continue
                outside_stats = stats.get("outside_mean_abs_diff_adj", {})
                if outside_stats.get("n", 0) == 0:
                    continue
                semantic_mean = _summary_strength_delta(stats)
                strength_rows.append(
                    {
                        "experiment": name,
                        "strength": strength_key,
                        "outside_mean_abs_diff_adj_mean": outside_stats.get("mean", 0.0),
                        "semantic_delta_mean": semantic_mean,
                    }
                )
        if strength_rows:
            strength_csv = out_dir / f"strength_{name}.csv"
            _write_csv(strength_csv, strength_rows)

        if plt is None:
            continue

        # Pareto scatter plot
        if pareto_rows:
            xs = [r["outside_mean_abs_diff_adj"] for r in pareto_rows]
            ys = [r["semantic_delta_raw"] for r in pareto_rows]
            plt.figure(figsize=(5, 4))
            plt.scatter(xs, ys, s=24)
            for r in pareto_rows:
                strength = r.get("strength")
                if strength is not None:
                    plt.annotate(f"s={strength}", (r["outside_mean_abs_diff_adj"], r["semantic_delta_raw"]), fontsize=6)
            plt.xlabel("outside_mean_abs_diff_adj (lower is better)")
            plt.ylabel("semantic delta raw (higher is better)")
            plt.title(f"Pareto: {name}")
            plt.tight_layout()
            plt.savefig(out_dir / f"pareto_{name}.png")
            plt.close()

        # Strength sweep plot
        if strength_rows:
            strength_rows_sorted = sorted(strength_rows, key=lambda r: float(r["strength"]))
            xs = [float(r["strength"]) for r in strength_rows_sorted]
            ys_outside = [r["outside_mean_abs_diff_adj_mean"] for r in strength_rows_sorted]
            ys_semantic = [r["semantic_delta_mean"] for r in strength_rows_sorted]

            plt.figure(figsize=(5, 4))
            plt.plot(xs, ys_outside, marker="o", label="outside_mean_abs_diff_adj mean")
            if any(v is not None for v in ys_semantic):
                ys_sem = [v if v is not None else 0.0 for v in ys_semantic]
                plt.plot(xs, ys_sem, marker="o", label="semantic delta mean")
            plt.xlabel("strength")
            plt.ylabel("metric")
            plt.title(f"Strength sweep: {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"strength_{name}.png")
            plt.close()

    # Suite-wide pareto CSV
    if suite_pareto_rows:
        suite_pareto_csv = out_dir / "suite_pareto.csv"
        _write_csv(suite_pareto_csv, suite_pareto_rows)

    print(f"[DONE] Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
