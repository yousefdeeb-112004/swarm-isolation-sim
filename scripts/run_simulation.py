#!/usr/bin/env python3
"""
Swarm Isolation Simulator — main entry point.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --config configs/default.yaml
    python scripts/run_simulation.py --steps 500
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path so we can import swarm_sim
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.world import World
from swarm_sim.experiments.isolation import IsolationExperiment
from swarm_sim.utils.data_collector import (
    DataCollector, SimLogger, collect_from_world, collect_from_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Swarm Isolation Simulator"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: built-in defaults)"
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override number of steps to run"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output"
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Run evolutionary mode for N generations"
    )
    parser.add_argument(
        "--experiment", action="store_true",
        help="Run isolation experiment (control vs treatment)"
    )
    parser.add_argument(
        "--criteria", type=str, default="adventurousness",
        choices=["adventurousness", "affiliation_need", "random",
                 "best_fitness", "worst_fitness", "xenophobia"],
        help="Selection criteria for isolation (default: adventurousness)"
    )
    parser.add_argument(
        "--export", type=str, default=None, metavar="DIR",
        help="Export collected data to CSV/JSON in DIR (e.g., data/exports)"
    )
    parser.add_argument(
        "--plot", type=str, default=None, metavar="DIR",
        help="Generate analysis plots in DIR (e.g., data/plots)"
    )
    parser.add_argument(
        "--research", action="store_true",
        help="Run full research analysis (Section 11) with deep profiling"
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Number of independent runs for statistical analysis (with --research or --sweep)"
    )
    parser.add_argument(
        "--sweep", type=str, default=None, metavar="TYPE",
        choices=["selection_criteria", "ratio_sweep", "duration_sweep",
                 "resource_variation", "no_return", "generation_length", "all"],
        help="Run parameter sweep experiment (Section 12)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers for --sweep (0=all cores, default=serial)"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run full analysis pipeline after sweep (Sections 13-15: logs, stats, figures)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    if args.config:
        config = SimulationConfig.from_yaml(args.config)
        print(f"[*] Loaded config from: {args.config}")
    else:
        config = SimulationConfig.default()
        print("[*] Using default configuration")

    # Apply CLI overrides
    if args.seed is not None:
        config.world.seed = args.seed
    if args.steps is not None:
        config.world.max_steps = args.steps

    # Sweep mode (Section 12)
    if args.sweep:
        from swarm_sim.experiments.extended import (
            build_experiment_suite, SweepRunner,
        )
        import json

        num_gen = args.generations or 5
        steps = config.world.max_steps
        num_seeds = args.runs or 10

        if args.sweep == "all":
            suite = build_experiment_suite(num_gen, steps)
            conditions = [c for conds in suite.values() for c in conds]
            print(f"[*] Sweep mode: ALL experiments ({len(conditions)} conditions "
                  f"× {num_seeds} seeds = {len(conditions)*num_seeds} runs)")
        else:
            suite = build_experiment_suite(num_gen, steps)
            conditions = suite[args.sweep]
            print(f"[*] Sweep mode: {args.sweep} ({len(conditions)} conditions "
                  f"× {num_seeds} seeds = {len(conditions)*num_seeds} runs)")

        runner = SweepRunner(conditions, config, num_seeds=num_seeds)
        workers = args.workers
        if workers is not None:
            import multiprocessing
            actual = multiprocessing.cpu_count() if workers == 0 else workers
            print(f"[*] Parallel mode: {actual} workers")
        sweep_results = runner.run(verbose=True, workers=workers)

        # Save results
        out_dir = args.plot or "data/sweep"
        os.makedirs(out_dir, exist_ok=True)

        summary = sweep_results["sweep_summary"]
        print(f"\n{'='*70}")
        print(f"  SWEEP RESULTS SUMMARY")
        print(f"{'='*70}")
        for name, s in summary.items():
            print(f"\n  {name}:")
            print(f"    Ctrl fitness:   {s['ctrl_fitness_mean']:.4f} ± {s['ctrl_fitness_std']:.4f}")
            print(f"    Treat fitness:  {s['treat_fitness_mean']:.4f} ± {s['treat_fitness_std']:.4f}")
            print(f"    Fitness impact: {s['fitness_impact_mean']:+.4f} ± {s['fitness_impact_ci95']:.4f}")
            print(f"    Extinction rate: {s['extinction_rate']:.0%}")
            print(f"    Food reduction:  {s['food_reduction_mean']:.1f}% ± {s['food_reduction_std']:.1f}%")

        # Save summary JSON
        summary_path = os.path.join(out_dir, f"sweep_{args.sweep}_summary.json")
        serializable = {}
        for name, s in summary.items():
            serializable[name] = {k: v for k, v in s.items()
                                   if not isinstance(v, dict) or k == "condition"}
        with open(summary_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n[*] Summary saved to {summary_path}")
        print(f"[*] Total time: {sweep_results['elapsed_seconds']:.1f}s")

        # Full analysis pipeline (Sections 13-15)
        if args.analyze:
            from swarm_sim.utils.batch_logger import BatchLogger
            from swarm_sim.analysis.stats_analysis import analyze_sweep
            from swarm_sim.utils.pub_visualization import generate_publication_figures

            print(f"\n{'='*70}")
            print(f"  ANALYSIS PIPELINE (Sections 13-15)")
            print(f"{'='*70}")

            # Section 13: Batch logging
            print("\n[*] Section 13: Exporting structured logs...")
            logger = BatchLogger(out_dir)
            exported = logger.export_sweep(sweep_results, prefix=f"sweep_{args.sweep}")
            for ftype, fpath in exported.items():
                fsize = os.path.getsize(fpath)
                print(f"    {ftype}: {os.path.basename(fpath)} ({fsize:,} bytes)")

            # Section 14: Statistical analysis
            print("\n[*] Section 14: Running statistical analysis...")
            stats_dir = os.path.join(out_dir, "stats")
            analysis = analyze_sweep(sweep_results, output_dir=stats_dir)

            # Print key results
            print(f"    T-tests ({len(analysis.get('t_tests', {}))} conditions):")
            for name, t in sorted(analysis.get("t_tests", {}).items()):
                sig = "***" if t["significant_001"] else ("*" if t["significant_005"] else "ns")
                print(f"      {name}: d={t['cohens_d']:.2f} ({t['effect_size_label']}), "
                      f"p={t['p_value']:.4f} {sig}")

            anova = analysis.get("anova_fitness_impact", {})
            print(f"    ANOVA: F={anova.get('f_statistic', 0):.4f}, "
                  f"p={anova.get('p_value', 1):.4f}")

            km = analysis.get("kaplan_meier", {})
            print(f"    Kaplan-Meier ({len(km)} curves):")
            for name, k in sorted(km.items()):
                med = k.get("median_survival")
                med_str = f"median={med}" if med else "no median (survived)"
                print(f"      {name}: events={k['n_events']}/{k['n']}, {med_str}")

            # Section 15: Publication figures
            print("\n[*] Section 15: Generating publication figures...")
            fig_dir = os.path.join(out_dir, "figures")
            figures = generate_publication_figures(
                sweep_results, analysis, output_dir=fig_dir,
                prefix=f"pub_{args.sweep}"
            )
            for fname, fpath in figures.items():
                print(f"    {fname}: {os.path.basename(fpath)}")

            print(f"\n[*] Analysis complete. Output in: {out_dir}/")

        return

    # Research mode (Section 11)
    if args.research:
        from swarm_sim.experiments.research import (
            ResearchExperiment, MultiRunAnalysis, generate_research_report,
        )
        from swarm_sim.utils.visualization import generate_research_plots

        num_gen = args.generations or config.experiment.num_generations
        steps = config.world.max_steps

        if args.runs and args.runs > 1:
            # Multi-run statistical analysis
            print(f"[*] Research mode: {args.runs} runs × {num_gen} generations "
                  f"× {steps} steps")
            print(f"[*] Selection criteria: {args.criteria}")

            multi = MultiRunAnalysis(
                base_config=config,
                num_runs=args.runs,
                isolation_fraction=0.2,
                isolation_duration=config.experiment.isolation_duration,
                isolation_frequency=config.experiment.isolation_frequency,
                selection_criteria=args.criteria,
            )
            results = multi.run(
                num_generations=num_gen,
                steps_per_generation=steps,
                verbose=True,
            )
            report = generate_research_report(
                results, multi_run=True,
                filepath=os.path.join(
                    args.plot or "data/research", "research_report.txt"
                ),
            )
            print(report)
        else:
            # Single-run deep profiling
            print(f"[*] Research mode: {num_gen} generations × {steps} steps")
            print(f"[*] Selection criteria: {args.criteria}")

            exp = ResearchExperiment(
                config=config,
                isolation_fraction=0.2,
                isolation_duration=config.experiment.isolation_duration,
                isolation_frequency=config.experiment.isolation_frequency,
                selection_criteria=args.criteria,
            )
            results = exp.run(
                num_generations=num_gen,
                steps_per_generation=steps,
                verbose=True,
            )

            # Generate report
            report_dir = args.plot or "data/research"
            report = generate_research_report(
                results,
                filepath=os.path.join(report_dir, "research_report.txt"),
            )
            print(report)

            # Generate plots
            plots = generate_research_plots(
                results, output_dir=report_dir, prefix="research",
            )
            if plots:
                print(f"\n[*] Research plots generated in {report_dir}/:")
                for name, path in sorted(plots.items()):
                    print(f"    {name}: {path}")

        return

    # Experiment mode
    if args.experiment:
        num_gen = args.generations or config.experiment.num_generations
        steps = config.world.max_steps
        print(f"[*] Running isolation experiment: {num_gen} generations x {steps} steps")
        print(f"[*] Selection criteria: {args.criteria}")

        exp = IsolationExperiment(
            config=config,
            isolation_fraction=0.2,
            isolation_duration=config.experiment.isolation_duration,
            isolation_frequency=config.experiment.isolation_frequency,
            selection_criteria=args.criteria,
        )
        results = exp.run_experiment(
            num_generations=num_gen,
            steps_per_generation=steps,
            verbose=True,
        )
        exp.print_summary()

        # Export if requested
        if args.export:
            collector = DataCollector(snapshot_interval=config.logging.log_interval)
            collector.start({"mode": "experiment", "criteria": args.criteria})
            collect_from_experiment(exp, collector)
            collector.stop()
            csv_files = collector.export_csv(output_dir=args.export, prefix="experiment")
            json_path = collector.export_json(output_dir=args.export, prefix="experiment")
            print(f"\n[*] Data exported to {args.export}/:")
            for name, path in sorted(csv_files.items()):
                print(f"    {name}: {path}")
            print(f"    json: {json_path}")

        # Generate plots if requested
        if args.plot:
            from swarm_sim.utils.visualization import visualize_experiment
            plots = visualize_experiment(exp, output_dir=args.plot, prefix="experiment")
            print(f"\n[*] Plots generated in {args.plot}/:")
            for name, path in sorted(plots.items()):
                print(f"    {name}: {path}")

        return

    # Initialize world
    world = World(config)
    print(f"[*] World initialized: {world}")
    print(f"[*] Environment: {world.environment}")
    print(f"[*] Agents: {len(world.agents)} spawned")

    # Evolutionary mode
    if args.generations:
        print(f"[*] Running evolution for {args.generations} generations, "
              f"{config.world.max_steps} steps each...\n")
        records = world.run_evolution(
            num_generations=args.generations,
            steps_per_generation=config.world.max_steps,
            verbose=True,
        )
        print(f"\n{'='*70}")
        print(f"Evolution complete: {len(records)} generations")
        evo_stats = world.evolution_manager.get_stats()
        print(f"  Best fitness ever:   {evo_stats['best_fitness_ever']}")
        print(f"  Latest avg fitness:  {evo_stats['latest_avg_fitness']}")
        print(f"  Latest diversity:    {evo_stats['latest_diversity']}")
        trend = world.evolution_manager.get_fitness_trend()
        if trend['best_fitness']:
            print(f"  Fitness trend:       "
                  f"{trend['best_fitness'][0]:.4f} -> {trend['best_fitness'][-1]:.4f}")
        print(f"{'='*70}")

        # Generate plots if requested
        if args.plot:
            from swarm_sim.utils.visualization import visualize_evolution
            plots = visualize_evolution(world, output_dir=args.plot, prefix="evolution")
            print(f"\n[*] Plots generated in {args.plot}/:")
            for name, path in sorted(plots.items()):
                print(f"    {name}: {path}")

        return

    print(f"[*] Running for {config.world.max_steps} steps...\n")

    # Data collection (if --export specified)
    collector = None
    if args.export:
        collector = DataCollector(
            snapshot_interval=config.logging.log_interval,
        )
        collector.start({"mode": "simulation", "steps": config.world.max_steps})
        print(f"[*] Data collection enabled → {args.export}")

    # Run simulation
    log_interval = config.logging.log_interval
    for step_num in range(1, config.world.max_steps + 1):
        metrics = world.step()

        # Collect data
        if collector is not None:
            collect_from_world(world, collector, label="simulation")

        if not args.quiet and step_num % log_interval == 0:
            alive = metrics["agents_alive"]
            born = metrics.get("agents_born_this_step", 0)
            comms = metrics.get("communications", 0)
            born_str = f" Born:{born:>2d}" if born > 0 else ""
            comm_str = f" Comms:{comms:>2d}" if comms > 0 else ""
            print(
                f"  Step {step_num:>5d} | "
                f"Alive: {alive:>3d} | "
                f"Food: {metrics['total_food']:>4d} | "
                f"Eaten: {metrics['food_eaten_this_step']:>3d} | "
                f"Died: {metrics['agents_died_this_step']:>2d} | "
                f"AvgE: {metrics.get('avg_energy', 0):>6.1f} | "
                f"AvgAge: {metrics.get('avg_age', 0):>6.1f}"
                f"{born_str}{comm_str}"
            )

        # Stop early if all dead
        if metrics["agents_alive"] == 0:
            print(f"\n  *** All agents dead at step {step_num} ***")
            break

    # Final summary
    print(f"\n{'='*70}")
    print(f"Simulation complete: {world.current_step} steps")
    stats = world.environment.get_stats()
    pop = world.get_population_stats()
    print(f"  Final food count:    {stats['total_food']}")
    print(f"  Obstacles:           {stats['num_obstacles']}")
    print(f"  Predators:           {stats['num_predators']}")
    print(f"  Agents alive:        {pop.get('alive', 0)}")
    print(f"  Agents dead:         {pop.get('dead', 0)}")
    if pop.get('alive', 0) > 0:
        print(f"  Avg energy:          {pop['avg_energy']}")
        print(f"  Avg age:             {pop['avg_age']}")
        print(f"  Avg fitness:         {pop['avg_fitness']}")
        print(f"  Genome diversity:    {pop['genome_diversity']}")
        print(f"  Total food eaten:    {pop['total_food_eaten']}")
        print(f"  Total offspring:     {pop['total_offspring']}")
    print(f"  Total agents ever:   {len(world.agents)}")
    im = world.interaction_manager.get_stats()
    print(f"  Communications:      {im['total_communications']}")
    print(f"  Reproductions:       {im['total_reproductions']}")
    print(f"  Social pairs:        {im['unique_interactions']}")
    print(f"{'='*70}")

    # Export data
    if collector is not None:
        collector.stop()
        csv_files = collector.export_csv(output_dir=args.export, prefix="simulation")
        json_path = collector.export_json(output_dir=args.export, prefix="simulation")
        print(f"\n[*] Data exported to {args.export}/:")
        for name, path in sorted(csv_files.items()):
            print(f"    {name}: {path}")
        print(f"    json: {json_path}")
        summary = collector.get_summary()
        print(f"    ({summary['total_step_records']} steps, "
              f"{summary['total_agent_snapshots']} snapshots)")

    # Generate plots if requested
    if args.plot:
        from swarm_sim.utils.visualization import (
            plot_population_dynamics, plot_environment_snapshot,
            plot_inner_state_distribution, plot_genome_heatmap,
        )
        import matplotlib.pyplot as plt
        os.makedirs(args.plot, exist_ok=True)
        generated = {}

        # Population dynamics from step history
        if world.step_history:
            steps = [d.get("step", i+1) for i, d in enumerate(world.step_history)]
            alive = [d.get("agents_alive", 0) for d in world.step_history]
            energy = [d.get("avg_energy", 0) for d in world.step_history]
            food = [d.get("total_food", 0) for d in world.step_history]
            fp = os.path.join(args.plot, "simulation_population.png")
            plot_population_dynamics(steps, alive, energy, food, filepath=fp)
            plt.close()
            generated["population"] = fp

        # Environment snapshot
        fp = os.path.join(args.plot, "simulation_environment.png")
        agent_data = [{"x": a.x, "y": a.y, "alive": a.alive,
                       "is_isolated": a.is_isolated} for a in world.agents]
        plot_environment_snapshot(world.environment.grid, agent_data, filepath=fp)
        plt.close()
        generated["environment"] = fp

        # Inner states of living agents
        living = [a for a in world.agents if a.alive]
        if living:
            snapshots = []
            for a in living:
                snap = {f"inner_{k}": float(v) for k, v in a.inner_state.items()}
                for g in a.genome.genes:
                    snap[f"gene_{g}"] = float(a.genome[g])
                snapshots.append(snap)
            fp = os.path.join(args.plot, "simulation_inner_states.png")
            plot_inner_state_distribution(snapshots, filepath=fp)
            plt.close()
            generated["inner_states"] = fp

            fp = os.path.join(args.plot, "simulation_genome_heatmap.png")
            plot_genome_heatmap(snapshots, filepath=fp)
            plt.close()
            generated["genome_heatmap"] = fp

        print(f"\n[*] Plots generated in {args.plot}/:")
        for name, path in sorted(generated.items()):
            print(f"    {name}: {path}")



if __name__ == "__main__":
    main()
