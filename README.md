# Swarm Isolation Simulator

**Agent-Based Simulation of Individual Isolation's Impact on Heterogeneous Swarm Dynamics with Genomic Evolution**

A Python simulation framework investigating how individual isolation shapes collective swarm dynamics. Combines genetic algorithms, Bayesian networks, reinforcement learning, and neural networks to simulate emergent intelligence through interaction between inherited traits (genomes) and individual learning.

> **Key Finding:** After fixing 7 critical bugs identified by the diagnostic system, the simulation reveals a counterintuitive "Isolation Paradox" — higher isolation ratios *improve* population fitness and survival (ANOVA F=7.69, p=0.0001), contradicting the initial (buggy) finding of universal extinction.

---

## Table of Contents

- [Key Results](#key-results)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Diagnostic & Validation](#diagnostic--validation)
- [Configuration](#configuration)
- [Development Sections](#development-sections)
- [License](#license)

---

## Key Results

### Corrected Results (30 seeds × 4 conditions = 120 runs)

| Condition | Fitness Impact | Extinction Rate | Cohen's d | KM Median Survival |
|-----------|---------------|-----------------|-----------|-------------------|
| 5% isolation | +0.012 | 97% | -0.07 (ns) | 1,910 steps |
| 10% isolation | +0.061 | 90% | -0.33 (ns) | 1,960 steps |
| 20% isolation | +0.153 | 80% | -0.86 (p=0.002) | 3,805 steps |
| 30% isolation | +0.273 | 53% | -1.71 (p<0.0001) | 4,940 steps |

**ANOVA:** F=7.69, p=0.0001 — strong dose-response relationship confirmed.

### The Isolation Paradox

More isolation leads to *better* outcomes. At 30%, treatment fitness is double the control (0.53 vs 0.26) with 47% of populations surviving. This occurs because isolated agents receive reduced metabolic costs and skip predator damage, creating a "rotating shelter" effect that improves population-level energy management.

### Previous (Buggy) Results — Now Invalidated

The original run showed 100% extinction across all conditions with uniform fitness impact of -0.27 (ANOVA p=0.998). This was caused by 7 bugs (see `scripts/diagnose_results.py`) that made isolation maximally lethal regardless of dosage. The full comparison is documented in `docs/bugfix_comparison_analysis.docx`.

---

## Features

- **2D Grid Environment** with food sources (depletion/regeneration), obstacles, and predators
- **Genomic Agents** with 10+ heritable traits (adventurousness, plasticity, xenophobia, etc.)
- **Bayesian Belief Networks** for probabilistic environment modeling
- **Neural Network Inner States** representing subjective experience (hunger, fear, curiosity, loneliness, aggression)
- **Reinforcement Learning Policies** (tabular Q-learning) for action selection
- **Multi-Agent Communication** and social learning with message passing
- **Genetic Algorithm Evolution** with tournament selection, crossover, mutation, and elitism
- **Isolation Experiments** with control vs treatment comparison across 6 sweep types
- **Parallel Execution** using multiprocessing (`--workers 0` for all CPU cores)
- **Full Analysis Pipeline** — batch logging (5 formats), statistical tests (t-test, ANOVA, Kaplan-Meier, Cohen's d), and publication-quality figures (7 types, 300 DPI)
- **Diagnostic System** — 9-test validation suite to detect parameter bugs
- **815 Unit Tests** covering all modules

---

## Project Structure

```
swarm-isolation-sim/
├── swarm_sim/                    # Main package
│   ├── core/                     # Sections 1-2: Environment & world
│   │   ├── environment.py        # 2D grid, food, obstacles, predators
│   │   ├── world.py              # World manager, simulation loop, evolution
│   │   └── config.py             # Configuration dataclasses
│   ├── agents/                   # Sections 2-6: Agent architecture
│   │   ├── genome.py             # Genome encoding, crossover, mutation
│   │   ├── agent.py              # Full agent: observe → decide → act
│   │   ├── bayesian.py           # Bayesian belief network (Section 3)
│   │   ├── neural.py             # Neural network inner states (Section 4)
│   │   ├── policy.py             # RL Q-learning policy (Section 5)
│   │   └── interaction.py        # Communication & reproduction (Section 6)
│   ├── evolution/                # Section 7: Genetic algorithm
│   │   └── evolution.py          # Selection, crossover, mutation, elitism
│   ├── experiments/              # Sections 8, 11-12: Experiments
│   │   ├── isolation.py          # Core isolation experiment (control vs treatment)
│   │   ├── research.py           # Deep research analysis (Section 11)
│   │   └── extended.py           # 6-type parameter sweep suite (Section 12)
│   ├── analysis/                 # Sections 14-15: Statistical analysis
│   │   └── stats_analysis.py     # t-tests, ANOVA, Kaplan-Meier, Cohen's d
│   └── utils/                    # Sections 9-10, 13, 15: Utilities
│       ├── data_collector.py     # Data collection (Section 9)
│       ├── visualization.py      # Real-time visualization (Section 10)
│       ├── batch_logger.py       # 5-format export: CSV, JSON, LaTeX (Section 13)
│       └── pub_visualization.py  # Publication figures, 300 DPI (Section 15)
├── scripts/
│   ├── run_simulation.py         # Main CLI entry point
│   ├── run_tests.py              # Full test suite (815 tests)
│   ├── run_all_experiments.sh    # Master parallel launcher (all 6 sweeps)
│   └── diagnose_results.py       # 9-test diagnostic for result validation
├── tests/
│   ├── test_environment.py       # Environment unit tests
│   └── conftest.py               # Shared fixtures
├── configs/
│   └── default.yaml              # Simulation configuration
├── docs/
│   └── bugfix_comparison_analysis.docx  # Pre-fix vs post-fix comparison
├── data/                         # Output directory (gitignored)
│   ├── results/                  # Sweep results
│   └── logs/                     # Run logs
├── requirements.txt
├── setup.py
├── pyproject.toml
├── LICENSE
└── .gitignore
```

---

## Installation

### Prerequisites

- **OS:** Linux / macOS / Windows (WSL2)
- **Python:** 3.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/swarm-isolation-sim.git
cd swarm-isolation-sim

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Quick Start

```bash
# Basic run (default config, 1 generation)
python3 scripts/run_simulation.py

# Multi-generation evolution
python3 scripts/run_simulation.py --generations 5 --steps 500

# With visualization output
python3 scripts/run_simulation.py --steps 200 --plot data/plots

# Run test suite (815 tests)
python3 scripts/run_tests.py
```

---

## Running Experiments

### Single Sweep (recommended first run)

```bash
# Ratio sweep — the key experiment (5%, 10%, 20%, 30% isolation)
python3 scripts/run_simulation.py \
  --sweep ratio_sweep \
  --generations 5 \
  --runs 30 \
  --workers 0 \
  --analyze \
  --plot data/results/ratio_sweep
```

### All 6 Sweeps in Parallel

```bash
# Full suite — 21 conditions × 30 seeds = 630 runs
bash scripts/run_all_experiments.sh --runs 30 --gens 5 --outdir data/results

# Quick test (fewer seeds)
bash scripts/run_all_experiments.sh --runs 5 --gens 3 --outdir data/test

# Sequential mode (saves RAM)
bash scripts/run_all_experiments.sh --runs 30 --gens 5 --serial
```

### Available Sweep Types

| Sweep | Conditions | What It Tests |
|-------|-----------|---------------|
| `ratio_sweep` | 4 | 5%, 10%, 20%, 30% isolation fraction |
| `duration_sweep` | 4 | 50, 100, 200, 500 step isolation |
| `selection_criteria` | 4 | Random, fitness, age, genome-based selection |
| `resource_variation` | 3 | Scarce (50), normal (150), rich (300) food |
| `generation_length` | 4 | 250, 500, 1000, 2000 steps per generation |
| `no_return` | 2 | Permanent isolation vs return |

### Output Per Sweep

Each sweep produces in its output directory:

- **Batch logs:** `sweep_*_per_run.csv`, `*_summary.csv`, `*_full.json`, `*_table.tex`
- **Statistics:** `statistical_tests.csv`, `statistical_tests.tex`
- **Figures:** 7 PNG + 1 PDF (condition comparison, Kaplan-Meier, forest plot, heatmap, fitness trajectories, extinction distribution, 4-panel summary)

---

## Diagnostic & Validation

### Run the Diagnostic

```bash
python3 scripts/diagnose_results.py
```

This runs 9 automated tests to verify the simulation is working correctly:

| Test | What It Checks |
|------|---------------|
| 1 | Control & treatment start from identical seeds |
| 2 | Isolation parameters are physically reasonable |
| 3 | Agents can survive isolation (energy budget) |
| 4 | Simultaneous isolation doesn't exceed population |
| 5 | 0% isolation matches control exactly (null test) |
| 6 | Minimal isolation has minimal impact |
| 7 | Isolated agents aren't all at the same position |
| 8 | Control population stays healthy across generations |
| 9 | Deaths occur outside isolation, not during it |

Expected output: **9 passed, 0 bugs**.

### Bugs Fixed (v2.0)

The diagnostic identified and fixed 7 bugs in the original code:

1. **Units mismatch** — `isolation_frequency` used as steps instead of generations (5→50)
2. **Mass simultaneous isolation** — 98% of population isolated at peak (duration 100→50)
3. **Guaranteed starvation** — agents died during isolation (added half metabolism)
4. **No predator protection** — isolated agents took full predator damage (now skipped)
5. **max(1,...) floor** — 0% isolation still isolated 1 agent (fixed to `int()`)
6. **Position pileup** — all isolated agents at (0,0) (now scattered randomly)
7. **58% deaths during isolation** — post-fix: 0% deaths during isolation

---

## Configuration

Edit `configs/default.yaml`:

```yaml
world:
  width: 100
  height: 100
  max_steps: 1000
  seed: 42

environment:
  food:
    initial_count: 150
    energy_value: 20
    regeneration_rate: 0.02
    max_food: 200

agents:
  population_size: 50
  sensor_range: 7
  initial_energy: 100
  energy_per_step: -1        # Metabolism cost (halved during isolation)

experiment:
  isolation_duration: 50     # Steps in isolation (survivable)
  isolation_frequency: 50    # Isolate every N steps
  selection_criteria: "adventurousness"
```

---

## Development Sections

| Section | Description | Tests | Status |
|---------|-------------|-------|--------|
| 1 | Environment & World Foundation | 142 | ✅ Complete |
| 2 | Agent Genome & Inner States | 89 | ✅ Complete |
| 3 | Bayesian Belief Network | 67 | ✅ Complete |
| 4 | Neural Network Inner State Model | 58 | ✅ Complete |
| 5 | Reinforcement Learning Policy | 72 | ✅ Complete |
| 6 | Multi-Agent Interaction | 63 | ✅ Complete |
| 7 | Genetic Algorithm & Evolution | 54 | ✅ Complete |
| 8 | Isolation Experiment Workflow | 48 | ✅ Complete |
| 9 | Data Collection & Structured Logging | 35 | ✅ Complete |
| 10 | Visualization & Live Display | 28 | ✅ Complete |
| 11 | Deep Research Analysis | 41 | ✅ Complete |
| 12 | Extended Experiment Suite (6 sweeps) | 38 | ✅ Complete |
| 13 | Batch Logging (CSV, JSON, LaTeX) | 32 | ✅ Complete |
| 14 | Statistical Analysis (t-test, ANOVA, KM) | 26 | ✅ Complete |
| 15 | Publication-Quality Visualization | 22 | ✅ Complete |
| 16 | Research Paper Generation | — | ✅ Complete |
| — | **Total** | **815** | ✅ All passing |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Yousef Deeb** — Computer Science Student
