"""
Data Collection, Logging & Export — persistent data pipeline.

Provides:
  - DataCollector: hooks into World/Experiment to capture structured data
  - CSV export: per-step metrics, per-agent snapshots, generation summaries
  - JSON export: full experiment results
  - SimLogger: structured console + file logging with configurable verbosity

All data is collected in memory and flushed to disk on demand or at the
end of a run. Exports go to `data/exports/` by default.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from swarm_sim.agents.agent import Agent


class DataCollector:
    """
    Collects and stores simulation data for export.

    Attaches to a World or IsolationExperiment and records:
      - Step-level metrics (population, energy, food, deaths, births)
      - Agent snapshots (per-agent state at configurable intervals)
      - Generation summaries (fitness, diversity, gene averages)
      - Isolation events (isolate/return with agent metadata)
    """

    def __init__(
        self,
        snapshot_interval: int = 50,
        collect_agent_snapshots: bool = True,
    ):
        self.snapshot_interval = max(1, snapshot_interval)
        self.collect_agent_snapshots = collect_agent_snapshots

        # Storage
        self.step_records: List[Dict[str, Any]] = []
        self.agent_snapshots: List[Dict[str, Any]] = []
        self.generation_records: List[Dict[str, Any]] = []
        self.isolation_events: List[Dict[str, Any]] = []
        self.experiment_config: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

        # Timing
        self._start_time: Optional[float] = None

    def start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start data collection session."""
        self._start_time = time.time()
        self.metadata = metadata or {}
        self.metadata["start_time"] = datetime.now().isoformat()

    def stop(self) -> None:
        """Stop data collection and record duration."""
        if self._start_time:
            self.metadata["duration_seconds"] = round(
                time.time() - self._start_time, 2
            )
        self.metadata["stop_time"] = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        generation: int,
        metrics: Dict[str, Any],
        label: str = "default",
    ) -> None:
        """Record step-level metrics."""
        record = {
            "step": step,
            "generation": generation,
            "label": label,
            **{k: v for k, v in metrics.items()
               if isinstance(v, (int, float, str, bool, type(None)))},
        }
        self.step_records.append(record)

    def record_agent_snapshot(
        self,
        step: int,
        generation: int,
        agents: List[Agent],
        label: str = "default",
    ) -> None:
        """Record per-agent state snapshots."""
        if not self.collect_agent_snapshots:
            return
        if step % self.snapshot_interval != 0:
            return

        for agent in agents:
            snapshot = {
                "step": step,
                "generation": generation,
                "label": label,
                "agent_id": agent.id,
                "alive": agent.alive,
                "x": agent.x,
                "y": agent.y,
                "energy": round(agent.energy, 2),
                "age": agent.age,
                "is_isolated": agent.is_isolated,
                "times_isolated": agent.times_isolated,
                "total_food_eaten": agent.total_food_eaten,
                "num_offspring": agent.num_offspring,
                "fitness": round(agent.compute_fitness(), 4),
                "cause_of_death": agent.cause_of_death,
            }
            # Inner state
            for state_name, value in agent.inner_state.items():
                snapshot[f"inner_{state_name}"] = round(float(value), 4)

            # Key genome genes
            for gene_name in agent.genome.genes:
                snapshot[f"gene_{gene_name}"] = round(
                    float(agent.genome[gene_name]), 4
                )

            self.agent_snapshots.append(snapshot)

    def record_generation(
        self,
        record: Dict[str, Any],
        label: str = "default",
    ) -> None:
        """Record a generation summary."""
        gen_record = {"label": label}
        # Flatten avg_genes into the record
        avg_genes = record.pop("avg_genes", {})
        gen_record.update(record)
        for gene_name, value in avg_genes.items():
            gen_record[f"avg_gene_{gene_name}"] = value
        self.generation_records.append(gen_record)

    def record_isolation_event(self, event: Dict[str, Any]) -> None:
        """Record an isolation/return event."""
        self.isolation_events.append(event)

    def record_experiment_config(self, config: Dict[str, Any]) -> None:
        """Store experiment configuration."""
        self.experiment_config = config

    # ------------------------------------------------------------------
    # Export — CSV
    # ------------------------------------------------------------------

    def export_csv(
        self,
        output_dir: str = "data/exports",
        prefix: str = "",
    ) -> Dict[str, str]:
        """
        Export all collected data to CSV files.

        Returns dict of {data_type: filepath}.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            prefix = f"{prefix}_"

        exported = {}

        # Step records
        if self.step_records:
            path = os.path.join(output_dir, f"{prefix}steps_{timestamp}.csv")
            self._write_csv(path, self.step_records)
            exported["steps"] = path

        # Agent snapshots
        if self.agent_snapshots:
            path = os.path.join(
                output_dir, f"{prefix}agents_{timestamp}.csv"
            )
            self._write_csv(path, self.agent_snapshots)
            exported["agents"] = path

        # Generation records
        if self.generation_records:
            path = os.path.join(
                output_dir, f"{prefix}generations_{timestamp}.csv"
            )
            self._write_csv(path, self.generation_records)
            exported["generations"] = path

        # Isolation events
        if self.isolation_events:
            path = os.path.join(
                output_dir, f"{prefix}isolation_{timestamp}.csv"
            )
            self._write_csv(path, self.isolation_events)
            exported["isolation"] = path

        return exported

    def _write_csv(
        self, path: str, records: List[Dict[str, Any]]
    ) -> None:
        """Write a list of dicts to CSV."""
        if not records:
            return
        # Collect all keys across all records
        all_keys = []
        seen = set()
        for record in records:
            for key in record.keys():
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    # ------------------------------------------------------------------
    # Export — JSON
    # ------------------------------------------------------------------

    def export_json(
        self,
        output_dir: str = "data/exports",
        prefix: str = "",
    ) -> str:
        """
        Export full dataset as a single JSON file.

        Returns filepath.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            prefix = f"{prefix}_"

        path = os.path.join(output_dir, f"{prefix}results_{timestamp}.json")

        data = {
            "metadata": self.metadata,
            "experiment_config": self.experiment_config,
            "step_records": self.step_records,
            "agent_snapshots": self.agent_snapshots,
            "generation_records": self.generation_records,
            "isolation_events": self.isolation_events,
            "summary": self.get_summary(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all collected data."""
        return {
            "total_step_records": len(self.step_records),
            "total_agent_snapshots": len(self.agent_snapshots),
            "total_generation_records": len(self.generation_records),
            "total_isolation_events": len(self.isolation_events),
            "unique_labels": list(set(
                r.get("label", "default") for r in self.step_records
            )),
            "duration_seconds": self.metadata.get("duration_seconds", None),
        }

    def clear(self) -> None:
        """Clear all collected data."""
        self.step_records.clear()
        self.agent_snapshots.clear()
        self.generation_records.clear()
        self.isolation_events.clear()
        self.experiment_config.clear()
        self.metadata.clear()

    def __repr__(self) -> str:
        return (
            f"DataCollector("
            f"steps={len(self.step_records)}, "
            f"snapshots={len(self.agent_snapshots)}, "
            f"gens={len(self.generation_records)}, "
            f"iso_events={len(self.isolation_events)})"
        )


class SimLogger:
    """
    Structured logging for simulation runs.

    Levels: 0=silent, 1=summary, 2=generations, 3=steps, 4=debug
    """

    SILENT = 0
    SUMMARY = 1
    GENERATIONS = 2
    STEPS = 3
    DEBUG = 4

    def __init__(
        self,
        level: int = 2,
        log_file: Optional[str] = None,
    ):
        self.level = level
        self.log_file = log_file
        self._file_handle = None

        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self._file_handle = open(log_file, "w")

    def log(self, message: str, level: int = 1) -> None:
        """Log a message at the given level."""
        if level <= self.level:
            print(message)
        if self._file_handle:
            self._file_handle.write(message + "\n")
            self._file_handle.flush()

    def step(
        self, step: int, metrics: Dict[str, Any], label: str = ""
    ) -> None:
        """Log a step summary."""
        prefix = f"[{label}] " if label else ""
        msg = (
            f"  {prefix}Step {step:>5d} | "
            f"Alive: {metrics.get('agents_alive', '?'):>3} | "
            f"Food: {metrics.get('food_count', '?'):>4} | "
            f"AvgE: {metrics.get('avg_energy', 0):>6.1f} | "
            f"AvgAge: {metrics.get('avg_age', 0):>6.1f}"
        )
        born = metrics.get("agents_born_this_step", 0)
        comms = metrics.get("communications", 0)
        if born:
            msg += f" Born: {born}"
        if comms:
            msg += f" Comms: {comms}"
        self.log(msg, level=self.STEPS)

    def generation(self, record: Dict[str, Any], label: str = "") -> None:
        """Log a generation summary."""
        prefix = f"[{label}] " if label else ""
        msg = (
            f"  {prefix}Gen {record.get('generation', '?'):>3d} | "
            f"Steps: {record.get('steps_survived', '?'):>4} | "
            f"Alive: {record.get('alive_at_end', '?'):>3} | "
            f"BestFit: {record.get('best_fitness', 0):.4f} | "
            f"AvgFit: {record.get('avg_fitness', 0):.4f} | "
            f"Diversity: {record.get('genome_diversity', 0):.4f}"
        )
        self.log(msg, level=self.GENERATIONS)

    def header(self, message: str) -> None:
        """Log a header message."""
        self.log(f"\n{'='*70}", level=self.SUMMARY)
        self.log(f"  {message}", level=self.SUMMARY)
        self.log(f"{'='*70}", level=self.SUMMARY)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(f"  {message}", level=self.SUMMARY)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(f"  [DEBUG] {message}", level=self.DEBUG)

    def close(self) -> None:
        """Close log file if open."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return f"SimLogger(level={self.level}, file={self.log_file})"


def collect_from_world(
    world,
    collector: DataCollector,
    label: str = "default",
) -> None:
    """
    Convenience: collect current state from a World into a DataCollector.

    Call this after each world.step() for step records, or at snapshot
    intervals for agent data.
    """
    step = world.current_step
    gen = world.current_generation

    # Step metrics from history
    if world.step_history:
        metrics = world.step_history[-1]
        collector.record_step(step, gen, metrics, label=label)

    # Agent snapshots
    collector.record_agent_snapshot(
        step, gen, world.agents, label=label,
    )


def collect_from_experiment(
    experiment,
    collector: DataCollector,
) -> None:
    """
    Convenience: collect all data from a completed IsolationExperiment.
    """
    results = experiment.get_results()

    collector.record_experiment_config(results.get("config", {}))

    # Control generation records
    for record in results.get("control_generations", []):
        collector.record_generation(dict(record), label="control")

    # Treatment generation records
    for record in results.get("treatment_generations", []):
        collector.record_generation(dict(record), label="treatment")

    # Step histories
    for i, metrics in enumerate(experiment.control_history):
        collector.record_step(i + 1, 0, metrics, label="control")

    for i, metrics in enumerate(experiment.treatment_history):
        collector.record_step(i + 1, 0, metrics, label="treatment")

    # Isolation events
    for event in experiment.isolation_events:
        collector.record_isolation_event(event)
