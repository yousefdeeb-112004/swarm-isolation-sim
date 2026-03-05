"""
Section 14 — Statistical Analysis Module.

Loads sweep results and computes:
  1. Descriptive statistics with 95% CIs
  2. Independent t-tests (treatment vs control)
  3. One-way ANOVA across conditions
  4. Cohen's d effect sizes
  5. Kaplan–Meier survival analysis (extinction time)
  6. Mediation analysis (inner states → extinction)
  7. Summary tables in CSV and LaTeX

All functions accept sweep results dicts (from SweepRunner or BatchLogger JSON).
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ======================================================================
# 1. Descriptive Statistics
# ======================================================================

def descriptive_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, SEM, 95% CI, median, min, max."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0, "std": 0, "sem": 0, "ci95": 0,
                "median": 0, "min": 0, "max": 0}
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    ci95 = 1.96 * sem
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "sem": round(sem, 6),
        "ci95": round(ci95, 6),
        "median": round(float(np.median(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


# ======================================================================
# 2. T-tests
# ======================================================================

def paired_comparison(
    ctrl_values: List[float],
    treat_values: List[float],
    label: str = "",
) -> Dict[str, Any]:
    """
    Independent samples t-test + Cohen's d.

    Tests H0: ctrl_mean == treat_mean.
    """
    ctrl = np.array(ctrl_values, dtype=float)
    treat = np.array(treat_values, dtype=float)

    ctrl_desc = descriptive_stats(ctrl_values)
    treat_desc = descriptive_stats(treat_values)

    # t-test (Welch's — unequal variance)
    if len(ctrl) >= 2 and len(treat) >= 2:
        t_stat, p_value = sp_stats.ttest_ind(ctrl, treat, equal_var=False)
    else:
        t_stat, p_value = 0.0, 1.0

    # Cohen's d
    d = cohens_d(ctrl_values, treat_values)

    return {
        "label": label,
        "ctrl": ctrl_desc,
        "treat": treat_desc,
        "difference": round(treat_desc["mean"] - ctrl_desc["mean"], 6),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "cohens_d": round(d, 4),
        "effect_size_label": _effect_label(d),
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Cohen's d effect size (pooled standard deviation).

    |d| < 0.2: negligible
    0.2 ≤ |d| < 0.5: small
    0.5 ≤ |d| < 0.8: medium
    |d| ≥ 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    a1, a2 = np.array(group1), np.array(group2)
    m1, m2 = np.mean(a1), np.mean(a2)
    s1, s2 = np.std(a1, ddof=1), np.std(a2, ddof=1)

    # Pooled standard deviation
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return float((m1 - m2) / sp)


def _effect_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ======================================================================
# 3. ANOVA
# ======================================================================

def one_way_anova(
    groups: Dict[str, List[float]],
    metric_name: str = "",
) -> Dict[str, Any]:
    """
    One-way ANOVA across multiple conditions.

    Parameters
    ----------
    groups : {condition_name: [values]} for each condition
    """
    group_names = sorted(groups.keys())
    arrays = [np.array(groups[name], dtype=float)
              for name in group_names if len(groups[name]) > 0]

    if len(arrays) < 2:
        return {"metric": metric_name, "f_statistic": 0, "p_value": 1.0,
                "significant": False, "n_groups": len(arrays)}

    f_stat, p_value = sp_stats.f_oneway(*arrays)

    # Pairwise comparisons (post-hoc)
    pairwise = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1 = groups[group_names[i]]
            g2 = groups[group_names[j]]
            if len(g1) >= 2 and len(g2) >= 2:
                t, p = sp_stats.ttest_ind(g1, g2, equal_var=False)
                d = cohens_d(g1, g2)
                pairwise.append({
                    "group_a": group_names[i],
                    "group_b": group_names[j],
                    "t_statistic": round(float(t), 4),
                    "p_value": round(float(p), 6),
                    "cohens_d": round(d, 4),
                })

    return {
        "metric": metric_name,
        "n_groups": len(arrays),
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": p_value < 0.05,
        "pairwise": pairwise,
        "group_descriptives": {
            name: descriptive_stats(groups[name]) for name in group_names
        },
    }


# ======================================================================
# 4. Kaplan–Meier Survival Analysis
# ======================================================================

def kaplan_meier(
    extinction_steps: List[Optional[int]],
    max_step: int,
    label: str = "",
) -> Dict[str, Any]:
    """
    Kaplan–Meier survival curve for swarm extinction.

    Parameters
    ----------
    extinction_steps : list of extinction step per run (None = survived)
    max_step : total steps in experiment (censoring time)

    Returns times, survival probabilities, and confidence intervals.
    """
    n = len(extinction_steps)
    if n == 0:
        return {"label": label, "n": 0, "n_events": 0, "n_censored": 0,
                "times": [], "survival": [],
                "ci_lower": [], "ci_upper": [], "median_survival": None}

    # Separate events and censored
    events = []
    for es in extinction_steps:
        if es is not None:
            events.append((es, 1))   # event at step es
        else:
            events.append((max_step, 0))  # censored at max_step

    events.sort(key=lambda x: x[0])

    # Build KM curve
    times = [0]
    survival = [1.0]
    ci_lower = [1.0]
    ci_upper = [1.0]

    at_risk = n
    cum_survival = 1.0
    greenwood_sum = 0.0

    prev_time = -1
    for t, event in events:
        if event == 1:  # death
            if t != prev_time:
                times.append(t)
                d = sum(1 for tt, e in events if tt == t and e == 1)
                cum_survival *= (at_risk - d) / max(at_risk, 1)

                if at_risk > d and at_risk > 0:
                    greenwood_sum += d / (at_risk * (at_risk - d))

                se = cum_survival * math.sqrt(greenwood_sum) if greenwood_sum >= 0 else 0
                ci_lo = max(0, cum_survival - 1.96 * se)
                ci_hi = min(1, cum_survival + 1.96 * se)

                survival.append(round(cum_survival, 6))
                ci_lower.append(round(ci_lo, 6))
                ci_upper.append(round(ci_hi, 6))

                prev_time = t
            at_risk -= 1
        else:
            at_risk -= 1

    # Median survival
    median = None
    for i, s in enumerate(survival):
        if s <= 0.5:
            median = times[i]
            break

    return {
        "label": label,
        "n": n,
        "n_events": sum(1 for es in extinction_steps if es is not None),
        "n_censored": sum(1 for es in extinction_steps if es is None),
        "times": times,
        "survival": survival,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median_survival": median,
    }


def log_rank_test(
    group1_steps: List[Optional[int]],
    group2_steps: List[Optional[int]],
    max_step: int,
) -> Dict[str, float]:
    """
    Log-rank test comparing two survival curves.

    Returns chi-square statistic and p-value.
    """
    # Combine all unique event times
    all_events = set()
    for s in group1_steps + group2_steps:
        if s is not None:
            all_events.add(s)
    all_events = sorted(all_events)

    if not all_events:
        return {"chi2": 0.0, "p_value": 1.0}

    # Count events and at-risk at each time
    O1, E1 = 0.0, 0.0
    n1_alive = len(group1_steps)
    n2_alive = len(group2_steps)

    # Track who's still alive (simple approach)
    g1_events = sorted([s for s in group1_steps if s is not None])
    g2_events = sorted([s for s in group2_steps if s is not None])
    g1_idx, g2_idx = 0, 0

    for t in all_events:
        d1 = sum(1 for s in g1_events[g1_idx:] if s == t)
        d2 = sum(1 for s in g2_events[g2_idx:] if s == t)
        d = d1 + d2
        n = n1_alive + n2_alive

        if n > 0:
            E1 += n1_alive * d / n
            O1 += d1

        n1_alive -= d1
        n2_alive -= d2
        g1_idx += d1
        g2_idx += d2

    # Chi-square statistic
    var = max(E1 * (1 - E1 / max(len(group1_steps) + len(group2_steps), 1)), 0.001)
    chi2 = (O1 - E1) ** 2 / var
    p_value = 1.0 - sp_stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
    }


# ======================================================================
# 5. Mediation Analysis (simplified Baron & Kenny)
# ======================================================================

def mediation_test(
    x: List[float],
    mediator: List[float],
    y: List[float],
    labels: Tuple[str, str, str] = ("X", "M", "Y"),
) -> Dict[str, Any]:
    """
    Simplified mediation analysis (Baron & Kenny, 1986).

    Tests whether M mediates the effect of X on Y.

    Steps:
    1. X → Y (total effect, path c)
    2. X → M (path a)
    3. X + M → Y (direct effect c', and M→Y path b)
    4. Indirect effect = a × b
    5. Sobel test for significance of indirect effect

    Parameters
    ----------
    x : independent variable (e.g., isolation_fraction)
    mediator : proposed mediator (e.g., aggression change)
    y : outcome (e.g., fitness impact)
    """
    n = min(len(x), len(mediator), len(y))
    if n < 5:
        return {"error": "insufficient data", "n": n}

    X = np.array(x[:n], dtype=float)
    M = np.array(mediator[:n], dtype=float)
    Y = np.array(y[:n], dtype=float)

    # Step 1: X → Y (total effect c)
    slope_c, intercept_c, r_c, p_c, se_c = sp_stats.linregress(X, Y)

    # Step 2: X → M (path a)
    slope_a, intercept_a, r_a, p_a, se_a = sp_stats.linregress(X, M)

    # Step 3: X + M → Y (partial regression)
    # Use multiple regression: Y = c' * X + b * M + intercept
    # Via OLS with design matrix [X, M, 1]
    A = np.column_stack([X, M, np.ones(n)])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, Y, rcond=None)
        c_prime = coeffs[0]  # direct effect
        b = coeffs[1]        # M → Y path
    except np.linalg.LinAlgError:
        return {"error": "regression failed", "n": n}

    # Indirect effect = a * b
    indirect = slope_a * b

    # Sobel test
    se_indirect = math.sqrt(slope_a**2 * se_c**2 + b**2 * se_a**2)
    if se_indirect > 0:
        z_sobel = indirect / se_indirect
        p_sobel = 2 * (1 - sp_stats.norm.cdf(abs(z_sobel)))
    else:
        z_sobel = 0.0
        p_sobel = 1.0

    # Proportion mediated
    prop_mediated = (indirect / slope_c * 100) if abs(slope_c) > 1e-10 else 0.0

    return {
        "n": n,
        "labels": {"x": labels[0], "mediator": labels[1], "y": labels[2]},
        "total_effect_c": round(float(slope_c), 6),
        "total_effect_p": round(float(p_c), 6),
        "path_a": round(float(slope_a), 6),
        "path_a_p": round(float(p_a), 6),
        "path_b": round(float(b), 6),
        "direct_effect_c_prime": round(float(c_prime), 6),
        "indirect_effect_ab": round(float(indirect), 6),
        "sobel_z": round(float(z_sobel), 4),
        "sobel_p": round(float(p_sobel), 6),
        "mediation_significant": p_sobel < 0.05,
        "proportion_mediated_pct": round(float(prop_mediated), 2),
    }


# ======================================================================
# 6. Analyze Sweep Results (Main Entry Point)
# ======================================================================

def analyze_sweep(
    sweep_results: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete statistical analysis on sweep results.

    Parameters
    ----------
    sweep_results : output of SweepRunner.run()
    output_dir : if provided, export CSV/LaTeX tables

    Returns comprehensive analysis dict.
    """
    all_results = sweep_results.get("all_results", [])
    summary = sweep_results.get("sweep_summary", {})

    analysis = {
        "n_total_runs": len(all_results),
        "n_conditions": len(summary),
    }

    # Group results by condition
    by_condition: Dict[str, List[Dict]] = {}
    for r in all_results:
        name = r["condition"]["name"]
        by_condition.setdefault(name, []).append(r)

    # --- T-tests: control vs treatment within each condition ---
    t_tests = {}
    for name, results in by_condition.items():
        ctrl_fits = [r["ctrl_avg_fitness"] for r in results]
        treat_fits = [r["treat_avg_fitness"] for r in results]
        t_tests[name] = paired_comparison(ctrl_fits, treat_fits,
                                          label=f"{name}: ctrl vs treat fitness")
    analysis["t_tests"] = t_tests

    # --- ANOVA: fitness impact across conditions ---
    groups_fitness = {name: [r["fitness_impact"] for r in results]
                      for name, results in by_condition.items()}
    analysis["anova_fitness_impact"] = one_way_anova(
        groups_fitness, metric_name="fitness_impact"
    )

    groups_food_red = {name: [r["food_reduction_pct"] for r in results]
                       for name, results in by_condition.items()}
    analysis["anova_food_reduction"] = one_way_anova(
        groups_food_red, metric_name="food_reduction_pct"
    )

    # --- Kaplan-Meier survival ---
    km_curves = {}
    for name, results in by_condition.items():
        cond = results[0]["condition"]
        total_steps = cond["num_generations"] * cond["steps_per_generation"]
        ext_steps = [r["treat_extinction_step"] for r in results]
        km_curves[name] = kaplan_meier(ext_steps, total_steps, label=name)
    analysis["kaplan_meier"] = km_curves

    # --- Condition descriptives ---
    condition_stats = {}
    for name, results in by_condition.items():
        condition_stats[name] = {
            "fitness_impact": descriptive_stats([r["fitness_impact"] for r in results]),
            "food_reduction": descriptive_stats([r["food_reduction_pct"] for r in results]),
            "extinction_rate": sum(1 for r in results if r["treat_extinct"]) / len(results),
        }
    analysis["condition_stats"] = condition_stats

    # --- Export tables if output_dir given ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _export_stats_csv(analysis, output_dir)
        _export_stats_latex(analysis, output_dir)

    return analysis


def _export_stats_csv(analysis: Dict[str, Any], output_dir: str) -> str:
    """Export t-test results as CSV."""
    path = os.path.join(output_dir, "statistical_tests.csv")

    fieldnames = [
        "condition", "ctrl_mean", "ctrl_std", "treat_mean", "treat_std",
        "difference", "t_statistic", "p_value", "cohens_d", "effect_size",
        "significant_005",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, t in analysis.get("t_tests", {}).items():
            writer.writerow({
                "condition": name,
                "ctrl_mean": t["ctrl"]["mean"],
                "ctrl_std": t["ctrl"]["std"],
                "treat_mean": t["treat"]["mean"],
                "treat_std": t["treat"]["std"],
                "difference": t["difference"],
                "t_statistic": t["t_statistic"],
                "p_value": t["p_value"],
                "cohens_d": t["cohens_d"],
                "effect_size": t["effect_size_label"],
                "significant_005": t["significant_005"],
            })

    return path


def _export_stats_latex(analysis: Dict[str, Any], output_dir: str) -> str:
    """Export statistical results as LaTeX table."""
    path = os.path.join(output_dir, "statistical_tests.tex")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical Comparisons (Control vs Treatment)}")
    lines.append(r"\label{tab:stats}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Condition & Ctrl ($\bar{x}$) & Treat ($\bar{x}$) & "
        r"$t$ & $p$ & $d$ & Sig. \\"
    )
    lines.append(r"\midrule")

    for name, t in sorted(analysis.get("t_tests", {}).items()):
        name_tex = name.replace("_", r"\_")
        sig = r"$***$" if t["significant_001"] else (r"$*$" if t["significant_005"] else "")
        lines.append(
            f"  {name_tex} & {t['ctrl']['mean']:.4f} & "
            f"{t['treat']['mean']:.4f} & "
            f"{t['t_statistic']:.2f} & {t['p_value']:.4f} & "
            f"{t['cohens_d']:.2f} & {sig} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return path
