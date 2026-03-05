"""
Section 13 — Batch Logger & Enhanced Structured Logging.

Takes sweep results from SweepRunner and exports:
  1. Per-run summary CSV  (one row per condition×seed)
  2. Per-generation CSV    (one row per condition×seed×generation)
  3. Condition summary CSV (one row per condition, aggregated)
  4. Full results JSON     (complete machine-readable output)
  5. LaTeX summary table   (for direct paper inclusion)

All files include metadata columns (condition name, experiment type,
seed, isolation parameters) for downstream statistical analysis.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np


class BatchLogger:
    """
    Exports structured sweep results to publication-ready formats.
    """

    def __init__(self, output_dir: str = "data/exports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Export from SweepRunner results
    # ------------------------------------------------------------------

    def export_sweep(
        self,
        sweep_results: Dict[str, Any],
        prefix: str = "sweep",
    ) -> Dict[str, str]:
        """
        Export complete sweep results to multiple files.

        Parameters
        ----------
        sweep_results : output of SweepRunner.run()
        prefix : filename prefix

        Returns dict of {file_type: filepath}.
        """
        all_results = sweep_results.get("all_results", [])
        summary = sweep_results.get("sweep_summary", {})

        exported = {}

        # 1. Per-run summary CSV
        path = self._export_per_run_csv(all_results, prefix)
        exported["per_run_csv"] = path

        # 2. Per-generation CSV
        path = self._export_per_generation_csv(all_results, prefix)
        exported["per_generation_csv"] = path

        # 3. Condition summary CSV
        path = self._export_condition_summary_csv(summary, prefix)
        exported["condition_summary_csv"] = path

        # 4. Full JSON
        path = self._export_full_json(sweep_results, prefix)
        exported["full_json"] = path

        # 5. LaTeX table
        path = self._export_latex_table(summary, prefix)
        exported["latex_table"] = path

        return exported

    # ------------------------------------------------------------------
    # Per-run CSV
    # ------------------------------------------------------------------

    def _export_per_run_csv(
        self, results: List[Dict[str, Any]], prefix: str
    ) -> str:
        """One row per condition × seed."""
        path = os.path.join(self.output_dir, f"{prefix}_per_run.csv")

        fieldnames = [
            "condition_name", "experiment_type", "seed",
            "selection_criteria", "isolation_fraction", "isolation_duration",
            "isolation_frequency", "no_return",
            "num_generations", "steps_per_generation",
            "food_initial", "food_max",
            "ctrl_avg_fitness", "treat_avg_fitness", "fitness_impact",
            "ctrl_alive_final", "treat_alive_final", "treat_extinct",
            "ctrl_extinction_step", "treat_extinction_step",
            "ctrl_total_food", "treat_total_food", "food_reduction_pct",
            "ctrl_diversity_final", "treat_diversity_final",
            "total_isolations", "total_returns",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for r in results:
                cond = r.get("condition", {})
                row = {
                    "condition_name": cond.get("name", ""),
                    "experiment_type": cond.get("experiment_type", ""),
                    "seed": r.get("seed", ""),
                    "selection_criteria": cond.get("selection_criteria", ""),
                    "isolation_fraction": cond.get("isolation_fraction", ""),
                    "isolation_duration": cond.get("isolation_duration", ""),
                    "isolation_frequency": cond.get("isolation_frequency", ""),
                    "no_return": cond.get("no_return", False),
                    "num_generations": cond.get("num_generations", ""),
                    "steps_per_generation": cond.get("steps_per_generation", ""),
                    "food_initial": cond.get("food_initial", ""),
                    "food_max": cond.get("food_max", ""),
                    "ctrl_avg_fitness": r.get("ctrl_avg_fitness", 0),
                    "treat_avg_fitness": r.get("treat_avg_fitness", 0),
                    "fitness_impact": r.get("fitness_impact", 0),
                    "ctrl_alive_final": r.get("ctrl_alive_final", 0),
                    "treat_alive_final": r.get("treat_alive_final", 0),
                    "treat_extinct": r.get("treat_extinct", False),
                    "ctrl_extinction_step": r.get("ctrl_extinction_step", ""),
                    "treat_extinction_step": r.get("treat_extinction_step", ""),
                    "ctrl_total_food": r.get("ctrl_total_food", 0),
                    "treat_total_food": r.get("treat_total_food", 0),
                    "food_reduction_pct": round(r.get("food_reduction_pct", 0), 2),
                    "ctrl_diversity_final": r.get("ctrl_diversity_final", 0),
                    "treat_diversity_final": r.get("treat_diversity_final", 0),
                    "total_isolations": r.get("total_isolations", 0),
                    "total_returns": r.get("total_returns", 0),
                }
                writer.writerow(row)

        return path

    # ------------------------------------------------------------------
    # Per-generation CSV
    # ------------------------------------------------------------------

    def _export_per_generation_csv(
        self, results: List[Dict[str, Any]], prefix: str
    ) -> str:
        """One row per condition × seed × generation × group."""
        path = os.path.join(self.output_dir, f"{prefix}_per_generation.csv")

        fieldnames = [
            "condition_name", "experiment_type", "seed",
            "group", "generation",
            "alive_at_end", "avg_fitness", "best_fitness",
            "genome_diversity", "total_food_eaten", "total_offspring",
            "avg_energy",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for r in results:
                cond = r.get("condition", {})
                seed = r.get("seed", "")

                for group, key in [("control", "ctrl_gen_records"),
                                   ("treatment", "treat_gen_records")]:
                    for rec in r.get(key, []):
                        row = {
                            "condition_name": cond.get("name", ""),
                            "experiment_type": cond.get("experiment_type", ""),
                            "seed": seed,
                            "group": group,
                            "generation": rec.get("generation", ""),
                            "alive_at_end": rec.get("alive_at_end", 0),
                            "avg_fitness": rec.get("avg_fitness", 0),
                            "best_fitness": rec.get("best_fitness", 0),
                            "genome_diversity": rec.get("genome_diversity", 0),
                            "total_food_eaten": rec.get("total_food_eaten", 0),
                            "total_offspring": rec.get("total_offspring", 0),
                            "avg_energy": rec.get("avg_energy", 0),
                        }
                        writer.writerow(row)

        return path

    # ------------------------------------------------------------------
    # Condition summary CSV
    # ------------------------------------------------------------------

    def _export_condition_summary_csv(
        self, summary: Dict[str, Any], prefix: str
    ) -> str:
        """One row per condition (aggregated across seeds)."""
        path = os.path.join(self.output_dir, f"{prefix}_summary.csv")

        fieldnames = [
            "condition_name", "experiment_type", "n_seeds",
            "ctrl_fitness_mean", "ctrl_fitness_std",
            "treat_fitness_mean", "treat_fitness_std",
            "fitness_impact_mean", "fitness_impact_std", "fitness_impact_ci95",
            "food_reduction_mean", "food_reduction_std",
            "extinction_rate",
            "extinction_step_mean", "extinction_step_std",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for name, s in summary.items():
                cond = s.get("condition", {})
                row = {
                    "condition_name": name,
                    "experiment_type": cond.get("experiment_type", ""),
                    "n_seeds": s.get("n_seeds", 0),
                    "ctrl_fitness_mean": _r(s.get("ctrl_fitness_mean", 0)),
                    "ctrl_fitness_std": _r(s.get("ctrl_fitness_std", 0)),
                    "treat_fitness_mean": _r(s.get("treat_fitness_mean", 0)),
                    "treat_fitness_std": _r(s.get("treat_fitness_std", 0)),
                    "fitness_impact_mean": _r(s.get("fitness_impact_mean", 0)),
                    "fitness_impact_std": _r(s.get("fitness_impact_std", 0)),
                    "fitness_impact_ci95": _r(s.get("fitness_impact_ci95", 0)),
                    "food_reduction_mean": _r(s.get("food_reduction_mean", 0)),
                    "food_reduction_std": _r(s.get("food_reduction_std", 0)),
                    "extinction_rate": _r(s.get("extinction_rate", 0)),
                    "extinction_step_mean": _r(s.get("extinction_step_mean")),
                    "extinction_step_std": _r(s.get("extinction_step_std")),
                }
                writer.writerow(row)

        return path

    # ------------------------------------------------------------------
    # Full JSON
    # ------------------------------------------------------------------

    def _export_full_json(
        self, sweep_results: Dict[str, Any], prefix: str
    ) -> str:
        """Complete machine-readable output."""
        path = os.path.join(self.output_dir, f"{prefix}_full.json")

        # Strip per-generation records from all_results to reduce size
        compact = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_runs": sweep_results.get("total_runs", 0),
                "elapsed_seconds": sweep_results.get("elapsed_seconds", 0),
            },
            "sweep_summary": sweep_results.get("sweep_summary", {}),
            "per_run": [
                {k: v for k, v in r.items()
                 if k not in ("ctrl_gen_records", "treat_gen_records")}
                for r in sweep_results.get("all_results", [])
            ],
        }

        with open(path, "w") as f:
            json.dump(compact, f, indent=2, default=str)

        return path

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------

    def _export_latex_table(
        self, summary: Dict[str, Any], prefix: str
    ) -> str:
        """Generate a LaTeX-formatted summary table."""
        path = os.path.join(self.output_dir, f"{prefix}_table.tex")

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Sweep Results Summary}")
        lines.append(r"\label{tab:sweep_results}")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"Condition & $n$ & Ctrl Fitness & Treat Fitness & "
            r"Impact & Extinction & Food Red. \\"
        )
        lines.append(r"\midrule")

        for name, s in sorted(summary.items()):
            n = s.get("n_seeds", 0)
            cf = s.get("ctrl_fitness_mean", 0)
            tf = s.get("treat_fitness_mean", 0)
            fi = s.get("fitness_impact_mean", 0)
            ci = s.get("fitness_impact_ci95", 0)
            ext = s.get("extinction_rate", 0)
            fr = s.get("food_reduction_mean", 0)

            # Escape underscores for LaTeX
            name_tex = name.replace("_", r"\_")

            lines.append(
                f"  {name_tex} & {n} & "
                f"{cf:.4f} & {tf:.4f} & "
                f"${fi:+.4f} \\pm {ci:.4f}$ & "
                f"{ext:.0%} & {fr:.1f}\\% \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path


def _r(v, n=4):
    """Round helper that handles None."""
    if v is None:
        return ""
    return round(v, n)
