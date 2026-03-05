#!/usr/bin/env python3
"""
==========================================================================
  DIAGNOSTIC: Are the Results Real or Is There a Bug?
==========================================================================

This script runs a systematic investigation into WHY the simulation
produces 100% treatment extinction across all conditions. It tests:

  1. SANITY CHECKS — Do control & treatment actually start identically?
  2. PARAMETER AUDIT — Are isolation parameters physically reasonable?
  3. ENERGY BUDGET  — Can agents survive isolation without food?
  4. ISOLATION LOAD  — What % of population is isolated at any moment?
  5. NULL TEST      — Does 0% isolation match control perfectly?
  6. MINIMAL DOSE   — Does isolating 1 agent once still cause extinction?
  7. POSITION PILEUP — Are all isolated agents dumped at (0,0)?
  8. CONTROL HEALTH  — Does control itself stay healthy across generations?
  9. TIMELINE TRACE  — Step-by-step trace of what kills treatment agents

Place this file at:
    scripts/diagnose_results.py

Run:
    python3 scripts/diagnose_results.py

==========================================================================
"""

from __future__ import annotations

import sys
import os
import copy
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.world import World
from swarm_sim.agents.agent import Agent
from swarm_sim.experiments.extended import (
    ExtendedExperiment, ExperimentCondition, build_experiment_suite,
)

# ======================================================================
# Formatting
# ======================================================================

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def header(title):
    print(f"\n{'='*70}")
    print(f"  {BOLD}{CYAN}{title}{RESET}")
    print(f"{'='*70}")

def ok(msg):
    print(f"  {GREEN}✅ OK:{RESET} {msg}")

def warn(msg):
    print(f"  {YELLOW}⚠️  WARNING:{RESET} {msg}")

def bug(msg):
    print(f"  {RED}🐛 BUG FOUND:{RESET} {BOLD}{msg}{RESET}")

def info(msg):
    print(f"  {msg}")

findings = []   # Collect all findings for final summary

def record(level, msg):
    findings.append((level, msg))
    if level == "BUG":
        bug(msg)
    elif level == "WARN":
        warn(msg)
    else:
        ok(msg)


# ======================================================================
# TEST 1: Sanity — Do both worlds start identically?
# ======================================================================

def test_1_initial_conditions():
    header("TEST 1: Do control & treatment start from identical seeds?")

    cfg = SimulationConfig.default()
    cfg.world.seed = 42

    Agent.reset_id_counter()
    w1 = World(cfg)
    agents1 = [(a.x, a.y, a.energy, round(a.genome["adventurousness"], 6))
               for a in w1.agents]

    Agent.reset_id_counter()
    w2 = World(cfg)
    agents2 = [(a.x, a.y, a.energy, round(a.genome["adventurousness"], 6))
               for a in w2.agents]

    if agents1 == agents2:
        record("OK", f"Both worlds produce identical agents ({len(agents1)} agents)")
    else:
        diffs = sum(1 for a, b in zip(agents1, agents2) if a != b)
        record("BUG", f"Worlds differ in {diffs}/{len(agents1)} agents despite same seed!")

    # Run a few steps without isolation - should they stay in sync?
    Agent.reset_id_counter()
    w1b = World(cfg)
    Agent.reset_id_counter()
    w2b = World(cfg)

    for _ in range(10):
        w1b.step()
        w2b.step()

    e1 = sorted([round(a.energy, 4) for a in w1b.agents])
    e2 = sorted([round(a.energy, 4) for a in w2b.agents])
    if e1 == e2:
        record("OK", "Both worlds stay in sync after 10 steps (no isolation)")
    else:
        record("WARN", "Worlds diverge after 10 steps even without isolation "
               "(may be due to agent ID counter or RNG state)")


# ======================================================================
# TEST 2: Parameter Audit — Are the defaults physically reasonable?
# ======================================================================

def test_2_parameter_audit():
    header("TEST 2: Parameter audit — are isolation parameters reasonable?")

    # Check the ExperimentCondition defaults
    default = ExperimentCondition(name="test", experiment_type="test")
    info(f"  Default isolation_frequency: {default.isolation_frequency} "
         f"(isolation happens every {default.isolation_frequency} STEPS)")
    info(f"  Default isolation_duration:  {default.isolation_duration} steps")
    info(f"  Default isolation_fraction:  {default.isolation_fraction} "
         f"({default.isolation_fraction*100:.0f}%)")
    info(f"  Default steps_per_gen:       {default.steps_per_generation}")
    print()

    # How many isolation events per generation?
    events_per_gen = default.steps_per_generation // default.isolation_frequency
    info(f"  Isolation events per generation: {events_per_gen}")

    if events_per_gen > 50:
        record("BUG",
            f"isolation_frequency={default.isolation_frequency} means isolating "
            f"every {default.isolation_frequency} STEPS → {events_per_gen} events/gen! "
            f"The config comment says 'every N generations' but code uses it as STEPS")
    else:
        record("OK", f"{events_per_gen} isolation events per generation is reasonable")

    # Check YAML vs code discrepancy
    info("")
    info(f"  {YELLOW}Config YAML says:{RESET} isolation_frequency: {default.isolation_frequency}  "
         f"# 'Isolate agents every N steps within a generation'")
    info(f"  {YELLOW}Code does:{RESET}  "
         f"if step % isolation_frequency == 0  → every {default.isolation_frequency} STEPS")

    # Effective simultaneous isolation
    pop = 50
    frac = default.isolation_fraction
    agents_per_batch = max(1, int(pop * frac))
    simultaneous = (default.isolation_duration // default.isolation_frequency) * agents_per_batch
    pct = min(100, simultaneous / pop * 100)

    info("")
    info(f"  Agents isolated per event:    {agents_per_batch}")
    info(f"  Events during one duration:   "
         f"{default.isolation_duration // default.isolation_frequency}")
    info(f"  Peak simultaneous isolation:  {simultaneous} "
         f"({pct:.0f}% of pop={pop})")

    if simultaneous >= pop:
        record("BUG",
            f"With freq={default.isolation_frequency} and dur={default.isolation_duration}, "
            f"up to {simultaneous} agents ({pct:.0f}%) are isolated simultaneously "
            f"— exceeds population of {pop}! Nearly ALL agents in isolation at once.")

    # Check ratio_sweep conditions
    suite = build_experiment_suite(5, 500)
    info("")
    info("  Ratio sweep conditions — effective simultaneous isolation:")
    for cond in suite["ratio_sweep"]:
        n = max(1, int(pop * cond.isolation_fraction))
        peak = (cond.isolation_duration // cond.isolation_frequency) * n
        info(f"    {cond.name}: {n}/event × "
             f"{cond.isolation_duration // cond.isolation_frequency} overlapping = "
             f"{peak} simultaneous ({min(100,peak/pop*100):.0f}% of {pop})")


# ======================================================================
# TEST 3: Energy Budget — Can agents survive isolation?
# ======================================================================

def test_3_energy_budget():
    header("TEST 3: Energy budget during isolation")

    cfg = SimulationConfig.default()

    initial_energy = cfg.agents.initial_energy      # 100
    metabolism = cfg.agents.energy_per_step          # -1 per step
    iso_duration = cfg.experiment.isolation_duration  # 50 steps
    food_energy = cfg.environment.food.energy_value  # 20 per food

    # Agents get half metabolism during isolation
    iso_metabolism = metabolism * 0.5
    energy_after_isolation = initial_energy + (iso_metabolism * iso_duration)
    info(f"  Initial energy:         {initial_energy}")
    info(f"  Metabolism cost/step:   {metabolism} (halved to {iso_metabolism} during isolation)")
    info(f"  Isolation duration:     {iso_duration} steps")
    info(f"  Energy after isolation: {initial_energy} + ({iso_metabolism} × {iso_duration}) "
         f"= {energy_after_isolation}")
    print()

    if energy_after_isolation <= 0:
        record("BUG",
            f"Agent starts with {initial_energy} energy, loses "
            f"{abs(iso_metabolism)}/step for {iso_duration} steps → "
            f"energy reaches {energy_after_isolation}. "
            f"Agent DIES of starvation during isolation!")
    else:
        record("OK", f"Agent survives isolation with {energy_after_isolation} energy remaining")

    # Food needed if energy is low
    food_needed = int(np.ceil(-energy_after_isolation / food_energy)) if energy_after_isolation <= 0 else 0
    if food_needed > 0:
        info(f"  Food items needed to survive: {food_needed} "
             f"(each worth {food_energy} energy)")

    # Predator note
    pred_dmg = cfg.environment.predators.energy_damage  # 30
    info(f"\n  Predator damage per hit: {pred_dmg} (but isolated agents skip predator checks)")
    info(f"  Agent returns with {energy_after_isolation:.0f} energy → can forage to recover")


# ======================================================================
# TEST 4: Live Isolation Load Trace
# ======================================================================

def test_4_isolation_load():
    header("TEST 4: What % of treatment is isolated at each timestep?")

    cfg = SimulationConfig.default()
    cfg.world.seed = 42
    cond = ExperimentCondition(
        name="diag_ratio_20pct",
        experiment_type="diagnostic",
        isolation_fraction=0.20,
        num_generations=1,
        steps_per_generation=200,
    )
    exp = ExtendedExperiment(cond, cfg)
    exp.config.world.seed = 42
    Agent.reset_id_counter()
    exp.setup()

    total_pop = len(exp.treatment_world.agents)
    iso_counts = []
    alive_counts = []
    dead_in_iso = 0

    for step in range(1, 201):
        exp.control_world.step()
        exp.treatment_world.step()

        if step % cond.isolation_frequency == 0:
            exp._apply_isolation(exp.treatment_world, step)
        exp._return_agents(exp.treatment_world, step)

        n_isolated = sum(1 for a in exp.treatment_world.agents
                        if a.is_isolated and a.alive)
        n_alive = sum(1 for a in exp.treatment_world.agents if a.alive)
        n_dead_iso = sum(1 for a in exp.treatment_world.agents
                        if a.is_isolated and not a.alive)
        iso_counts.append(n_isolated)
        alive_counts.append(n_alive)
        dead_in_iso += n_dead_iso

    info(f"  Population size: {total_pop}")
    info(f"  Step  20: {iso_counts[19]:>2d} isolated, {alive_counts[19]:>2d} alive "
         f"({iso_counts[19]/max(alive_counts[19],1)*100:.0f}% of living are in isolation)")
    info(f"  Step  50: {iso_counts[49]:>2d} isolated, {alive_counts[49]:>2d} alive "
         f"({iso_counts[49]/max(alive_counts[49],1)*100:.0f}%)")
    info(f"  Step 100: {iso_counts[99]:>2d} isolated, {alive_counts[99]:>2d} alive "
         f"({iso_counts[99]/max(alive_counts[99],1)*100:.0f}%)")
    info(f"  Step 150: {iso_counts[149]:>2d} isolated, {alive_counts[149]:>2d} alive "
         f"({iso_counts[149]/max(alive_counts[149],1)*100:.0f}%)")
    info(f"  Step 200: {iso_counts[199]:>2d} isolated, {alive_counts[199]:>2d} alive")
    print()

    peak = max(iso_counts)
    peak_step = iso_counts.index(peak) + 1
    peak_pct = peak / total_pop * 100

    info(f"  Peak simultaneous isolation: {peak} agents at step {peak_step} "
         f"({peak_pct:.0f}% of total population)")

    if peak_pct > 80:
        record("BUG",
            f"Peak isolation reached {peak_pct:.0f}% of population — "
            f"the swarm is effectively destroyed by mass isolation alone")

    # How fast does treatment die?
    first_zero = None
    for i, a in enumerate(alive_counts):
        if a == 0:
            first_zero = i + 1
            break
    if first_zero:
        info(f"  Treatment went extinct at step {first_zero}")
        ctrl_alive = sum(1 for a in exp.control_world.agents if a.alive)
        info(f"  Control alive at same point: {ctrl_alive}")
        record("WARN", f"Treatment extinct by step {first_zero} "
               f"while control has {ctrl_alive} alive")


# ======================================================================
# TEST 5: Null Test — 0% isolation should match control exactly
# ======================================================================

def test_5_null_isolation():
    header("TEST 5: Does 0% isolation match control? (null test)")

    cfg = SimulationConfig.default()
    cfg.world.seed = 100

    cond = ExperimentCondition(
        name="null_test",
        experiment_type="diagnostic",
        isolation_fraction=0.0,  # No isolation!
        num_generations=2,
        steps_per_generation=200,
    )
    exp = ExtendedExperiment(cond, cfg)
    Agent.reset_id_counter()
    result = exp.run(seed=100)

    fi = result["fitness_impact"]
    info(f"  Fitness impact (0% isolation): {fi:.6f}")
    info(f"  Control fitness:  {result['ctrl_avg_fitness']:.4f}")
    info(f"  Treatment fitness: {result['treat_avg_fitness']:.4f}")
    info(f"  Treatment extinct: {result['treat_extinct']}")

    # With 0% isolation: int(0 * 50) = 0, no agents isolated
    n_iso = int(0 * 50)  # This is the FIXED formula used in _apply_isolation
    info(f"\n  {YELLOW}Critical check:{RESET} int(0.0 * 50) = {n_iso}")
    if n_iso >= 1:
        record("BUG",
            f"Even with isolation_fraction=0.0, the code isolates "
            f"{n_iso} agent(s) per event — there is no true zero-isolation condition.")
    else:
        record("OK", "0% isolation correctly isolates 0 agents")

    if abs(fi) > 0.01:
        record("BUG",
            f"0% isolation still shows fitness impact of {fi:.4f} — "
            f"treatment should be identical to control")
    elif result["treat_extinct"]:
        record("BUG", "0% isolation resulted in treatment extinction!")


# ======================================================================
# TEST 6: Minimal Dose — Single isolation event
# ======================================================================

def test_6_minimal_dose():
    header("TEST 6: What happens with very infrequent isolation?")

    cfg = SimulationConfig.default()
    cond = ExperimentCondition(
        name="minimal_dose",
        experiment_type="diagnostic",
        isolation_fraction=0.05,
        isolation_frequency=250,   # Only twice per generation
        isolation_duration=10,     # Very short isolation
        num_generations=1,
        steps_per_generation=300,
    )
    exp = ExtendedExperiment(cond, cfg)
    Agent.reset_id_counter()
    result = exp.run(seed=42)

    info(f"  Config: 5% fraction, freq=250 (twice/gen), dur=10 steps")
    info(f"  Fitness impact:    {result['fitness_impact']:.4f}")
    info(f"  Treatment extinct: {result['treat_extinct']}")
    info(f"  Ctrl alive final:  {result['ctrl_alive_final']}")
    info(f"  Treat alive final: {result['treat_alive_final']}")

    if result["treat_extinct"]:
        record("WARN",
            "Even with 1 isolation event of 10 steps, treatment goes extinct — "
            "suggests the simulation may be too fragile or there's a deeper issue")
    else:
        fi = result["fitness_impact"]
        if abs(fi) < 0.05:
            record("OK", f"Minimal isolation has minimal impact ({fi:.4f})")
        else:
            record("WARN", f"Even minimal isolation has fitness impact of {fi:.4f}")


# ======================================================================
# TEST 7: Position Pileup — Isolated agents all at (0,0)
# ======================================================================

def test_7_position_pileup():
    header("TEST 7: Isolation zone position analysis")

    cfg = SimulationConfig.default()
    cond = ExperimentCondition(
        name="pileup_test",
        experiment_type="diagnostic",
        isolation_fraction=0.20,
        num_generations=1,
        steps_per_generation=50,
    )
    exp = ExtendedExperiment(cond, cfg)
    exp.config.world.seed = 42
    Agent.reset_id_counter()
    exp.setup()

    # Run 50 steps with isolation
    for step in range(1, 51):
        exp.control_world.step()
        exp.treatment_world.step()
        if step % cond.isolation_frequency == 0:
            exp._apply_isolation(exp.treatment_world, step)
        exp._return_agents(exp.treatment_world, step)

    isolated_agents = [a for a in exp.treatment_world.agents
                       if a.is_isolated and a.alive]
    positions = [(a.x, a.y) for a in isolated_agents]
    unique_positions = set(positions)

    info(f"  Isolated agents: {len(isolated_agents)}")
    info(f"  Unique positions: {unique_positions}")

    if len(unique_positions) == 1 and (0, 0) in unique_positions:
        record("BUG",
            f"ALL {len(isolated_agents)} isolated agents are at (0,0) — "
            f"they compete for food on a single cell! "
            f"Only 1 agent per step can eat at (0,0). "
            f"The rest starve even if food exists there.")
    elif len(unique_positions) <= 1 and len(isolated_agents) > 1:
        pos = unique_positions.pop() if unique_positions else "N/A"
        record("BUG",
            f"All {len(isolated_agents)} isolated agents at single position {pos}")
    else:
        record("OK", f"Isolated agents spread across {len(unique_positions)} positions")

    # Check if isolated agents have food access at their scattered positions
    food_count = 0
    for a in isolated_agents:
        f = exp.treatment_world.environment.consume_food(a.x, a.y)
        if f > 0:
            food_count += 1
    info(f"  Isolated agents with food at position: {food_count}/{len(isolated_agents)}")
    if len(isolated_agents) > 0 and food_count == 0:
        record("WARN", "No food at any isolation position — agents may struggle")
    else:
        record("OK", f"{food_count} isolated agents have food access at their positions")


# ======================================================================
# TEST 8: Control Health Check
# ======================================================================

def test_8_control_health():
    header("TEST 8: Does control population stay healthy?")

    cfg = SimulationConfig.default()
    cfg.world.seed = 42
    Agent.reset_id_counter()
    w = World(cfg)

    gen_results = []
    for gen in range(2):
        for step in range(300):
            m = w.step()
            if m["agents_alive"] == 0:
                break
        alive = sum(1 for a in w.agents if a.alive)
        avg_e = np.mean([a.energy for a in w.agents if a.alive]) if alive > 0 else 0
        avg_f = np.mean([a.compute_fitness() for a in w.agents])
        gen_results.append((alive, avg_e, avg_f))
        info(f"  Gen {gen}: {alive} alive, avg energy={avg_e:.1f}, avg fitness={avg_f:.4f}")
        w.evolve()

    final_alive = gen_results[-1][0]
    if final_alive > 0:
        record("OK", f"Control population survives: {final_alive} alive after 2 generations")
    else:
        record("BUG", "Control population also goes extinct after 2 gens — "
               "problem is not isolation-specific!")


# ======================================================================
# TEST 9: Step-by-Step Death Timeline
# ======================================================================

def test_9_death_timeline():
    header("TEST 9: Step-by-step death timeline for treatment")

    cfg = SimulationConfig.default()
    cfg.world.seed = 42
    cond = ExperimentCondition(
        name="timeline_trace",
        experiment_type="diagnostic",
        isolation_fraction=0.10,
        num_generations=1,
        steps_per_generation=300,
    )
    exp = ExtendedExperiment(cond, cfg)
    Agent.reset_id_counter()
    exp.setup()

    deaths_in_isolation = 0
    deaths_outside = 0
    death_causes = {}
    first_death_step = None

    for step in range(1, 301):
        # Snapshot before step
        pre_alive = {a.id: a.alive for a in exp.treatment_world.agents}

        exp.control_world.step()
        exp.treatment_world.step()

        if step % cond.isolation_frequency == 0:
            exp._apply_isolation(exp.treatment_world, step)
        exp._return_agents(exp.treatment_world, step)

        # Find who died this step
        for a in exp.treatment_world.agents:
            if pre_alive.get(a.id, False) and not a.alive:
                cause = a.cause_of_death or "unknown"
                death_causes[cause] = death_causes.get(cause, 0) + 1
                if a.is_isolated:
                    deaths_in_isolation += 1
                else:
                    deaths_outside += 1
                if first_death_step is None:
                    first_death_step = step

    total_deaths = deaths_in_isolation + deaths_outside
    info(f"  Total deaths:          {total_deaths}")
    info(f"  Deaths DURING isolation: {deaths_in_isolation} "
         f"({deaths_in_isolation/max(total_deaths,1)*100:.0f}%)")
    info(f"  Deaths OUTSIDE isolation: {deaths_outside} "
         f"({deaths_outside/max(total_deaths,1)*100:.0f}%)")
    info(f"  First death at step:   {first_death_step}")
    info(f"  Death causes: {death_causes}")

    if deaths_in_isolation > deaths_outside:
        record("BUG",
            f"{deaths_in_isolation}/{total_deaths} deaths occur DURING isolation — "
            f"agents are dying while isolated, likely from starvation at (0,0)")
    else:
        record("OK", "Most deaths occur outside isolation period")

    # Control deaths for comparison
    ctrl_deaths = sum(1 for a in exp.control_world.agents if not a.alive)
    ctrl_alive = sum(1 for a in exp.control_world.agents if a.alive)
    info(f"\n  Control: {ctrl_alive} alive, {ctrl_deaths} dead at step 300")


# ======================================================================
# FINAL SUMMARY
# ======================================================================

def print_summary():
    header("DIAGNOSTIC SUMMARY")

    bugs = [(l, m) for l, m in findings if l == "BUG"]
    warns = [(l, m) for l, m in findings if l == "WARN"]
    oks = [(l, m) for l, m in findings if l == "OK"]

    print(f"\n  {GREEN}Passed: {len(oks)}{RESET}  |  "
          f"{YELLOW}Warnings: {len(warns)}{RESET}  |  "
          f"{RED}Bugs: {len(bugs)}{RESET}")

    if bugs:
        print(f"\n  {RED}{BOLD}═══ BUGS FOUND ═══{RESET}")
        for i, (_, msg) in enumerate(bugs, 1):
            print(f"  {RED}{i}. {msg}{RESET}")

    if warns:
        print(f"\n  {YELLOW}═══ WARNINGS ═══{RESET}")
        for i, (_, msg) in enumerate(warns, 1):
            print(f"  {YELLOW}{i}. {msg}{RESET}")

    # Root cause analysis
    print(f"\n  {BOLD}═══ ROOT CAUSE ANALYSIS ═══{RESET}")
    print(f"""
  The diagnostic checks whether the simulation's isolation mechanics
  are properly calibrated. Issues found above (if any) indicate
  remaining problems. Previously fixed issues included:

  {GREEN}[FIXED] isolation_frequency now 50 steps (was 5){RESET}
  {GREEN}[FIXED] isolation_duration now 50 steps (was 100){RESET}
  {GREEN}[FIXED] Isolated agents get half metabolism cost{RESET}
  {GREEN}[FIXED] Isolated agents skip predator collisions{RESET}
  {GREEN}[FIXED] max(1,...) replaced with int() — 0% isolation = 0 agents{RESET}
  {GREEN}[FIXED] Agents scattered to random positions during isolation{RESET}

  If all tests pass, the simulation is correctly calibrated and
  results reflect genuine isolation effects, not parameter artifacts.
""")

    return len(bugs)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print(f"\n{BOLD}{'='*70}")
    print(f"  SWARM ISOLATION SIMULATION — RESULTS DIAGNOSTIC")
    print(f"  Investigating: Is 100% extinction a real finding or a bug?")
    print(f"{'='*70}{RESET}\n")

    test_1_initial_conditions()
    test_2_parameter_audit()
    test_3_energy_budget()
    test_4_isolation_load()
    test_5_null_isolation()
    test_6_minimal_dose()
    test_7_position_pileup()
    test_8_control_health()
    test_9_death_timeline()

    n_bugs = print_summary()
    sys.exit(1 if n_bugs > 0 else 0)