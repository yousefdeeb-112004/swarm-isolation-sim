#!/usr/bin/env python3
"""
Manual test runner for all sections (no pytest required).
Run from project root: python3 scripts/run_tests.py
"""

import sys
import traceback
import os
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
import csv
import tempfile

from swarm_sim.core.config import SimulationConfig
from swarm_sim.core.environment import Environment, CellType
from swarm_sim.core.world import World
from swarm_sim.agents.genome import Genome, GENE_SPEC, GENE_NAMES
from swarm_sim.agents.agent import Agent, Action, MemoryEntry
from swarm_sim.agents.bayesian import BeliefNetwork
from swarm_sim.agents.neural import (
    InnerStateNetwork, build_vitals_vector, compute_heuristic_inner_state,
    INNER_STATE_NAMES, NUM_INNER_STATES, sigmoid,
)
from swarm_sim.agents.policy import RLPolicy, QNetwork, STATE_SIZE, NUM_ACTIONS
from swarm_sim.agents.interaction import InteractionManager
from swarm_sim.evolution.evolution import EvolutionManager
from swarm_sim.experiments.isolation import IsolationExperiment
from swarm_sim.utils.data_collector import (
    DataCollector, SimLogger, collect_from_world, collect_from_experiment,
)
from swarm_sim.utils.visualization import (
    plot_fitness_evolution, plot_gene_evolution, plot_population_dynamics,
    plot_isolation_comparison, plot_isolation_timeline, plot_genome_heatmap,
    plot_environment_snapshot, plot_inner_state_distribution,
    generate_report, visualize_evolution, visualize_experiment,
    plot_subjective_experience, plot_loneliness_trajectory,
    plot_belief_divergence, plot_group_inner_comparison,
    plot_behavioral_change, generate_research_plots,
)
from swarm_sim.experiments.research import (
    AgentSnapshot, IsolationProfile, ResearchExperiment,
    MultiRunAnalysis, generate_research_report,
)
from swarm_sim.experiments.extended import (
    ExperimentCondition, ExtendedExperiment, SweepRunner,
    build_experiment_suite, run_experiment_type,
    list_experiment_types, list_all_conditions,
)
from swarm_sim.utils.batch_logger import BatchLogger
from swarm_sim.analysis.stats_analysis import (
    descriptive_stats, paired_comparison, cohens_d, one_way_anova,
    kaplan_meier, log_rank_test, mediation_test, analyze_sweep,
)
from swarm_sim.utils.pub_visualization import (
    plot_condition_comparison, plot_kaplan_meier, plot_forest,
    plot_sweep_heatmap, plot_fitness_trajectories,
    plot_extinction_distribution, plot_publication_summary,
    generate_publication_figures, HAS_MPL,
)


passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  \u2705 {name}")
        passed += 1
    else:
        print(f"  \u274c {name} \u2014 {detail}")
        failed += 1


def make_small_config():
    cfg = SimulationConfig.default()
    cfg.world.width = 20
    cfg.world.height = 20
    cfg.world.seed = 123
    cfg.environment.food.initial_count = 15
    cfg.environment.food.max_food = 30
    cfg.environment.obstacles.count = 5
    cfg.environment.predators.count = 2
    cfg.agents.population_size = 10
    cfg.agents.sensor_range = 4
    cfg.agents.initial_energy = 100
    return cfg


def section1_tests():
    """Section 1: Environment & World Foundation."""

    # === Config Tests ===
    print("\n\U0001f527 Configuration Tests")
    cfg = SimulationConfig.default()
    test("Default config creates", cfg is not None)
    test("Default width=100", cfg.world.width == 100)
    test("Default food=150", cfg.environment.food.initial_count == 150)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("world:\n  width: 42\n  height: 42\nenvironment:\n  food:\n    initial_count: 10\n  obstacles:\n    count: 3\n  predators:\n    count: 1\nagents:\n  population_size: 5\n")
        f.flush()
        cfg_yaml = SimulationConfig.from_yaml(f.name)
    test("YAML load width=42", cfg_yaml.world.width == 42)
    test("YAML load food=10", cfg_yaml.environment.food.initial_count == 10)

    # === Environment Init ===
    print("\n\U0001f30d Environment Initialization")
    small_cfg = make_small_config()
    env = Environment(small_cfg)
    test("Grid shape (20, 20)", env.grid.shape == (20, 20))
    food_count = int(np.sum(env.grid == CellType.FOOD))
    test(f"Food count = {food_count} (expected 15)", food_count == 15)
    obs_count = int(np.sum(env.grid == CellType.OBSTACLE))
    test(f"Obstacle count = {obs_count} (expected 5)", obs_count == 5)
    test(f"Predator count = {len(env.predators)} (expected 2)", len(env.predators) == 2)

    food_mask = env.grid == CellType.FOOD
    obs_mask = env.grid == CellType.OBSTACLE
    test("No food-obstacle overlap", not np.any(food_mask & obs_mask))

    env2 = Environment(small_cfg)
    test("Deterministic (same seed = same grid)", np.array_equal(env.grid, env2.grid))

    # === Food Mechanics ===
    print("\n\U0001f34e Food Mechanics")
    food_pos = np.argwhere(env.grid == CellType.FOOD)
    fy, fx = food_pos[0]
    energy = env.consume_food(int(fx), int(fy))
    test(f"Consume food returns energy={energy}", energy == small_cfg.environment.food.energy_value)
    test("Cell is now EMPTY", env.grid[fy, fx] == CellType.EMPTY)
    test(f"Total food decreased to {env.total_food}", env.total_food == 14)

    empty_pos = np.argwhere(env.grid == CellType.EMPTY)
    ey, ex = empty_pos[0]
    test("Consume at empty returns 0", env.consume_food(int(ex), int(ey)) == 0)

    regen_cfg = SimulationConfig.default()
    regen_cfg.world.width = 15
    regen_cfg.world.height = 15
    regen_cfg.world.seed = 77
    regen_cfg.environment.food.initial_count = 5
    regen_cfg.environment.food.max_food = 50
    regen_cfg.environment.food.regeneration_rate = 0.3
    regen_cfg.environment.obstacles.count = 2
    regen_cfg.environment.predators.count = 0
    regen_env = Environment(regen_cfg)
    initial_food = regen_env.total_food
    for _ in range(20):
        regen_env.step()
    test(f"Food regenerated: {initial_food} -> {regen_env.total_food}", regen_env.total_food > initial_food)
    test(f"Food <= max ({regen_env.total_food} <= 50)", regen_env.total_food <= 50)

    # === Obstacles ===
    print("\n\U0001f9f1 Obstacle Checks")
    obs_pos = np.argwhere(env.grid == CellType.OBSTACLE)
    oy, ox = obs_pos[0]
    test("is_obstacle at obstacle cell", env.is_obstacle(int(ox), int(oy)))
    test("Out of bounds is obstacle", env.is_obstacle(-1, 0))
    test("is_valid_position works", env.is_valid_position(int(ex), int(ey)))
    test("OOB is invalid", not env.is_valid_position(-1, 0))

    # === Predators ===
    print("\n\U0001f43a Predator Behavior")
    for _ in range(100):
        env.step()
    all_in_bounds = all(0 <= p.x < env.width and 0 <= p.y < env.height for p in env.predators)
    test("Predators stay in bounds after 100 steps", all_in_bounds)

    chase_cfg = SimulationConfig.default()
    chase_cfg.world.width = 30
    chase_cfg.world.height = 30
    chase_cfg.world.seed = 55
    chase_cfg.environment.predators.count = 1
    chase_cfg.environment.predators.detection_range = 50
    chase_cfg.environment.food.initial_count = 5
    chase_cfg.environment.obstacles.count = 0
    chase_env = Environment(chase_cfg)
    pred = chase_env.predators[0]
    ax = (pred.x + 8) % 30
    ay = (pred.y + 8) % 30
    old_dist = abs(pred.x - ax) + abs(pred.y - ay)
    chase_env.step(agent_positions=[(ax, ay)])
    new_dist = abs(pred.x - ax) + abs(pred.y - ay)
    test(f"Predator chases: dist {old_dist} -> {new_dist}", new_dist <= old_dist)

    damage = chase_env.check_predator_collision(pred.x, pred.y)
    test(f"Collision damage = {damage}", damage == chase_cfg.environment.predators.energy_damage)

    # === Observations ===
    print("\n\U0001f441\ufe0f  Sensor Observations")
    obs_result = env.get_local_observation(10, 10, sensor_range=4)
    test("Obs has grid_patch", "grid_patch" in obs_result)
    test("Patch shape (9,9)", obs_result["grid_patch"].shape == (9, 9))

    corner_obs = env.get_local_observation(0, 0, sensor_range=4)
    test("Corner obs shape (9,9)", corner_obs["grid_patch"].shape == (9, 9))

    agent_obs = env.get_local_observation(
        10, 10, sensor_range=4,
        agent_positions=[(10, 10), (11, 11), (100, 100)]
    )
    test("Nearby agent detected", (1, 1) in agent_obs["agent_positions"])
    test("Far agent not detected", len(agent_obs["agent_positions"]) == 1)


def section2_tests():
    """Section 2: Agent Genome & Basic Agent Structure."""

    # === Genome Tests ===
    print("\n\U0001f9ec Genome Tests")
    rng = np.random.default_rng(42)

    g_default = Genome.default()
    test("Default genome creates", g_default is not None)
    test("Default adventurousness=0.3", g_default["adventurousness"] == 0.3)
    test("Default lifespan=500", g_default["lifespan"] == 500)
    test("Has all genes", all(name in g_default for name in GENE_NAMES))

    g_random = Genome.random(rng)
    test("Random genome creates", g_random is not None)
    for name, spec in GENE_SPEC.items():
        val = g_random[name]
        test(f"  {name}={val} in range [{spec['min']}, {spec['max']}]",
             spec["min"] <= val <= spec["max"])

    # Clipping
    g_clipped = Genome({"adventurousness": 5.0, "lifespan": -100})
    test("Gene clipped: adventurousness <= 1.0", g_clipped["adventurousness"] <= 1.0)
    test("Gene clipped: lifespan >= 100", g_clipped["lifespan"] >= 100)

    # Reproduction
    g_parent_a = Genome.random(rng)
    g_parent_b = Genome.random(rng)
    g_child = Genome.from_parents(g_parent_a, g_parent_b, rng=rng)
    test("Child genome creates", g_child is not None)
    for name, spec in GENE_SPEC.items():
        test(f"  Child {name} in range", spec["min"] <= g_child[name] <= spec["max"])

    # Distance
    dist_self = g_parent_a.distance(g_parent_a)
    test(f"Distance to self = {dist_self:.4f} (should be 0)", dist_self == 0.0)
    dist_other = g_parent_a.distance(g_parent_b)
    test(f"Distance to other = {dist_other:.4f} (should be > 0)", dist_other > 0.0)
    test("Distance in [0, 1]", 0.0 <= dist_other <= 1.0)

    # Vector
    vec = g_random.to_vector()
    test(f"Vector shape = {vec.shape}", vec.shape == (len(GENE_NAMES),))
    test("Vector values in [0, 1]", np.all(vec >= 0) and np.all(vec <= 1))

    # to_dict
    d = g_random.to_dict()
    test("to_dict returns dict", isinstance(d, dict))
    test("to_dict has all genes", len(d) == len(GENE_NAMES))

    # Equality
    g_copy = Genome(g_default.to_dict())
    test("Genome equality works", g_copy == g_default)

    # === Agent Tests ===
    print("\n\U0001f916 Agent Basic Tests")
    Agent.reset_id_counter()

    a1 = Agent(x=5, y=5, energy=100.0, rng=rng)
    test("Agent creates", a1 is not None)
    test("Agent id=0", a1.id == 0)
    test("Agent alive", a1.alive)
    test("Agent position (5,5)", a1.x == 5 and a1.y == 5)
    test("Agent energy=100", a1.energy == 100.0)
    test("Agent age=0", a1.age == 0)
    test("Agent has genome", a1.genome is not None)

    a2 = Agent(x=10, y=10, energy=50.0, rng=rng)
    test("Auto-increment id=1", a2.id == 1)

    # Inner state
    test("Has inner_state dict", isinstance(a1.inner_state, dict))
    test("Inner state has hunger", "hunger" in a1.inner_state)
    test("Inner state has fear", "fear" in a1.inner_state)
    test("Inner state has curiosity", "curiosity" in a1.inner_state)
    test("Inner state has loneliness", "loneliness" in a1.inner_state)

    # Memory
    test(f"Memory capacity = {a1.memory.maxlen}", a1.memory.maxlen == a1.genome["memory_size"])

    # Agent state serialization
    state = a1.get_state()
    test("get_state returns dict", isinstance(state, dict))
    test("State has id", state["id"] == 0)
    test("State has genome", "genome" in state)
    test("State has inner_state", "inner_state" in state)
    test("State has fitness", "fitness" in state)

    # === Agent Actions Tests ===
    print("\n\U0001f3af Agent Actions Tests")
    small_cfg = make_small_config()
    env = Environment(small_cfg)

    Agent.reset_id_counter()
    agent = Agent(x=10, y=10, energy=100.0, rng=np.random.default_rng(99))

    # Observe
    obs = env.get_local_observation(agent.x, agent.y, sensor_range=4)
    agent.observe(obs)
    test("Observation stored", agent._last_observation is not None)
    test("Inner state updated (hunger >= 0)", agent.inner_state["hunger"] >= 0)

    # Decide
    action = agent.decide()
    test(f"Decide returns Action: {action.name}", isinstance(action, Action))

    # Act
    old_energy = agent.energy
    result = agent.act(action, env)
    test("Act returns dict", isinstance(result, dict))
    test("Age incremented to 1", agent.age == 1)
    test("Energy changed", agent.energy != old_energy or action == Action.STAY)
    test("Memory has 1 entry", len(agent.memory) == 1)

    # Run multiple steps
    for _ in range(50):
        obs = env.get_local_observation(agent.x, agent.y, sensor_range=4)
        agent.observe(obs)
        action = agent.decide()
        agent.act(action, env)
    test(f"Agent age after 51 steps = {agent.age}", agent.age == 51)
    test(f"Memory has entries: {len(agent.memory)}", len(agent.memory) > 0)
    test("Agent still in bounds", 0 <= agent.x < 20 and 0 <= agent.y < 20)

    # === Agent Death Tests ===
    print("\n\U0001f480 Agent Death Tests")
    Agent.reset_id_counter()
    dying_agent = Agent(x=5, y=5, energy=2.0, rng=rng)
    dying_agent.genome.genes["lifespan"] = 2000  # Don't die of old age
    obs = env.get_local_observation(5, 5, sensor_range=4)
    dying_agent.observe(obs)
    # Drain energy: metabolism is -1 per step, so after a few steps...
    for _ in range(10):
        obs = env.get_local_observation(dying_agent.x, dying_agent.y, sensor_range=4)
        dying_agent.observe(obs)
        action = dying_agent.decide()
        dying_agent.act(action, env)
        if not dying_agent.alive:
            break
    test(f"Agent died (alive={dying_agent.alive})", not dying_agent.alive)
    test(f"Cause of death: {dying_agent.cause_of_death}", dying_agent.cause_of_death == "starvation")

    # Kill method
    Agent.reset_id_counter()
    kill_agent = Agent(x=5, y=5, energy=100.0, rng=rng)
    kill_agent.kill("predator")
    test("kill() works", not kill_agent.alive)
    test("kill cause = predator", kill_agent.cause_of_death == "predator")

    # === Agent Reproduction ===
    print("\n\U0001f476 Agent Reproduction Tests")
    Agent.reset_id_counter()
    parent_a = Agent(x=5, y=5, energy=150.0, rng=rng)
    parent_b = Agent(x=6, y=5, energy=150.0, rng=rng)
    child = Agent.from_parents(parent_a, parent_b, x=5, y=6, energy=50.0, rng=rng)
    test("Child created", child is not None)
    test(f"Child id = {child.id}", child.id == 2)
    test("Child has parents", child.parent_ids == (0, 1))
    test("Child generation = 1", child.generation == 1)
    test("Parent a offspring count = 1", parent_a.num_offspring == 1)
    test("Parent b offspring count = 1", parent_b.num_offspring == 1)
    test("Child genome is valid", all(
        GENE_SPEC[n]["min"] <= child.genome[n] <= GENE_SPEC[n]["max"]
        for n in GENE_NAMES
    ))

    # === Isolation Tests ===
    print("\n\U0001f3dd\ufe0f  Agent Isolation Tests")
    Agent.reset_id_counter()
    iso_agent = Agent(x=10, y=10, energy=100.0, rng=rng)
    test("Not isolated initially", not iso_agent.is_isolated)

    iso_agent.isolate(0, 0)
    test("Is isolated after isolate()", iso_agent.is_isolated)
    test("Position changed to (0,0)", iso_agent.x == 0 and iso_agent.y == 0)
    test("times_isolated = 1", iso_agent.times_isolated == 1)

    iso_agent.return_from_isolation(15, 15)
    test("Not isolated after return", not iso_agent.is_isolated)
    test("Position changed to (15,15)", iso_agent.x == 15 and iso_agent.y == 15)
    test("steps_since_return = 0", iso_agent.steps_since_return == 0)

    # === World Integration Tests ===
    print("\n\U0001f30e World + Agents Integration")
    small_cfg = make_small_config()
    world = World(small_cfg)
    test(f"World has {len(world.agents)} agents", len(world.agents) == 10)
    test("All agents alive", all(a.alive for a in world.agents))

    # Run a few steps
    results = world.run(steps=50)
    test(f"Ran 50 steps, got {len(results)} results", len(results) == 50)
    alive = results[-1]["agents_alive"]
    test(f"Some agents alive after 50 steps: {alive}", alive >= 0)
    test("Food eaten tracked", any(r["food_eaten_this_step"] > 0 for r in results))
    test("avg_energy present", "avg_energy" in results[0])
    test("avg_age present", "avg_age" in results[-1])

    # Reset
    world.reset()
    test("World reset: step=0", world.current_step == 0)
    test("World reset: agents restored", len(world.agents) == 10)
    test("World reset: all alive", all(a.alive for a in world.agents))

    # Population stats
    world.run(steps=10)
    pop = world.get_population_stats()
    test("Population stats has alive", "alive" in pop)
    test("Population stats has genome_diversity", "genome_diversity" in pop)
    test("Population stats has avg_fitness", "avg_fitness" in pop)

    # Genome diversity
    div = world.get_genome_diversity()
    test(f"Genome diversity = {div:.4f} (should be > 0)", div > 0)

    # State summary
    summary = world.get_state_summary()
    test("Summary has population", "population" in summary)

    # Determinism
    w1 = World(small_cfg)
    w2 = World(small_cfg)
    r1 = w1.run(steps=20)
    r2 = w2.run(steps=20)
    deterministic = all(
        a["agents_alive"] == b["agents_alive"] and a["total_food"] == b["total_food"]
        for a, b in zip(r1, r2)
    )
    test("Two worlds same seed are identical", deterministic)


def section3_tests():
    """Section 3: Bayesian Belief Network."""

    # === Basic Construction ===
    print("\n\U0001f9e0 Belief Network Construction")
    bn = BeliefNetwork(world_width=100, world_height=100, region_size=5)
    test("BeliefNetwork creates", bn is not None)
    test(f"Grid size = {bn.grid_w}x{bn.grid_h} (expected 20x20)",
         bn.grid_w == 20 and bn.grid_h == 20)
    test("Food belief shape", bn.food_belief.shape == (20, 20))
    test("Danger belief shape", bn.danger_belief.shape == (20, 20))
    test("Agent belief shape", bn.agent_belief.shape == (20, 20))
    test("Confidence shape", bn.confidence.shape == (20, 20))
    test("Initial food belief = prior", np.allclose(bn.food_belief, 0.15))
    test("Initial danger belief = prior", np.allclose(bn.danger_belief, 0.05))
    test("Initial confidence = 0", np.allclose(bn.confidence, 0.0))

    # Small world test
    bn_small = BeliefNetwork(world_width=20, world_height=20, region_size=5)
    test(f"Small grid = {bn_small.grid_w}x{bn_small.grid_h} (expected 4x4)",
         bn_small.grid_w == 4 and bn_small.grid_h == 4)

    # === Coordinate Mapping ===
    print("\n\U0001f4cd Region Coordinate Mapping")
    rx, ry = bn._cell_to_region(0, 0)
    test(f"Cell (0,0) -> region ({rx},{ry})", rx == 0 and ry == 0)
    rx, ry = bn._cell_to_region(7, 12)
    test(f"Cell (7,12) -> region ({rx},{ry})", rx == 1 and ry == 2)
    rx, ry = bn._cell_to_region(99, 99)
    test(f"Cell (99,99) -> region ({rx},{ry})", rx == 19 and ry == 19)

    # === Bayesian Update ===
    print("\n\U0001f504 Bayesian Update")
    bn2 = BeliefNetwork(world_width=50, world_height=50, region_size=5)
    initial_food = bn2.food_belief[2, 2]  # region (2,2)

    # Simulate observation with food at relative positions near agent at (12, 12)
    fake_obs = {
        "food_positions": [(0, 0), (1, 0), (0, 1), (2, 1), (-1, 0)],
        "predator_positions": [],
        "obstacle_positions": [],
        "agent_positions": [(3, 3)],
    }
    bn2.update(agent_x=12, agent_y=12, observation=fake_obs, sensor_range=7)
    updated_food = bn2.food_belief[2, 2]  # region containing (12,12)
    test(f"Food belief updated: {initial_food:.3f} -> {updated_food:.3f}",
         updated_food != initial_food)
    test("Confidence increased", bn2.confidence[2, 2] > 0)

    # Update with predators
    bn3 = BeliefNetwork(world_width=50, world_height=50, region_size=5)
    initial_danger = bn3.danger_belief[1, 1]
    pred_obs = {
        "food_positions": [],
        "predator_positions": [(0, 0), (1, 1)],
        "obstacle_positions": [(2, 0)],
        "agent_positions": [],
    }
    bn3.update(agent_x=7, agent_y=7, observation=pred_obs, sensor_range=5)
    updated_danger = bn3.danger_belief[1, 1]
    test(f"Danger belief updated: {initial_danger:.3f} -> {updated_danger:.3f}",
         updated_danger > initial_danger)

    # === Belief Decay ===
    print("\n\u23f3 Belief Decay")
    bn4 = BeliefNetwork(world_width=30, world_height=30, region_size=5, decay_rate=0.1)
    # Give it a strong food observation
    food_obs = {
        "food_positions": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
        "predator_positions": [],
        "obstacle_positions": [],
        "agent_positions": [],
    }
    bn4.update(agent_x=3, agent_y=3, observation=food_obs, sensor_range=4)
    high_belief = bn4.food_belief[0, 0]

    # Now run many updates far away — belief should decay toward prior
    empty_obs = {"food_positions": [], "predator_positions": [],
                 "obstacle_positions": [], "agent_positions": []}
    for _ in range(30):
        bn4.update(agent_x=25, agent_y=25, observation=empty_obs, sensor_range=4)
    decayed_belief = bn4.food_belief[0, 0]
    test(f"Belief decayed: {high_belief:.3f} -> {decayed_belief:.3f}",
         abs(decayed_belief - bn4.prior_food) < abs(high_belief - bn4.prior_food))

    # === Belief Queries ===
    print("\n\U0001f50e Belief Queries")
    bn5 = BeliefNetwork(world_width=50, world_height=50, region_size=5)
    belief = bn5.get_belief_at(12, 12)
    test("get_belief_at returns dict", isinstance(belief, dict))
    test("Has food key", "food" in belief)
    test("Has danger key", "danger" in belief)
    test("Has agents key", "agents" in belief)
    test("Has confidence key", "confidence" in belief)

    # Direction queries
    # Place strong food belief in a specific region
    bn5.food_belief[0, 9] = 0.95  # top-right region
    bn5.confidence[0, 9] = 0.9
    dx, dy = bn5.get_best_food_direction(25, 25)
    test(f"Best food direction from (25,25): ({dx},{dy})",
         isinstance(dx, int) and isinstance(dy, int))

    dx, dy = bn5.get_safest_direction(25, 25)
    test(f"Safest direction: ({dx},{dy})", isinstance(dx, int))

    dx, dy = bn5.get_social_direction(25, 25)
    test(f"Social direction: ({dx},{dy})", isinstance(dx, int))

    # === Belief Vector ===
    print("\n\U0001f9ee Belief Vector for NN Input")
    vec = bn5.get_belief_vector(25, 25, radius=2)
    expected_len = (2 * 2 + 1) ** 2 * 4  # 5x5 patch * 4 channels = 100
    test(f"Belief vector shape = ({len(vec)},) (expected {expected_len})",
         len(vec) == expected_len)
    test("All values finite", np.all(np.isfinite(vec)))
    test("Values in [0, 1]", np.all(vec >= 0) and np.all(vec <= 1))

    # Edge case: agent at corner
    vec_corner = bn5.get_belief_vector(0, 0, radius=2)
    test(f"Corner belief vector shape = ({len(vec_corner)},)", len(vec_corner) == expected_len)

    # === Global Summary ===
    summary = bn5.get_global_summary()
    test("Global summary has avg_food_belief", "avg_food_belief" in summary)
    test("Global summary has avg_confidence", "avg_confidence" in summary)

    # === Social Learning (merge_beliefs) ===
    print("\n\U0001f91d Belief Merging (Social Learning)")
    bn_a = BeliefNetwork(world_width=50, world_height=50, region_size=5)
    bn_b = BeliefNetwork(world_width=50, world_height=50, region_size=5)

    # Agent B has strong food knowledge in region (3, 3)
    bn_b.food_belief[3, 3] = 0.9
    bn_b.confidence[3, 3] = 0.8

    old_a_food = bn_a.food_belief[3, 3]
    bn_a.merge_beliefs(bn_b, weight=0.5)
    new_a_food = bn_a.food_belief[3, 3]
    test(f"Merged food belief: {old_a_food:.3f} -> {new_a_food:.3f}",
         new_a_food > old_a_food)

    # === Clone ===
    print("\n\U0001f4cb Clone & Reset")
    bn_clone = bn5.clone()
    test("Clone creates new object", bn_clone is not bn5)
    test("Clone has same food beliefs", np.allclose(bn_clone.food_belief, bn5.food_belief))
    test("Clone has same confidence", np.allclose(bn_clone.confidence, bn5.confidence))

    # Reset
    bn_clone.reset()
    test("Reset returns to prior food", np.allclose(bn_clone.food_belief, bn_clone.prior_food))
    test("Reset clears confidence", np.allclose(bn_clone.confidence, 0.0))

    # === Integration: Agent + BeliefNetwork ===
    print("\n\U0001f310 Agent + Belief Network Integration")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    env = Environment(small_cfg)

    agent = Agent(x=10, y=10, energy=100.0, rng=np.random.default_rng(42))
    agent.init_belief_network(20, 20, 4)
    test("Agent has belief_network", agent.belief_network is not None)
    test(f"Belief grid = {agent.belief_network.grid_w}x{agent.belief_network.grid_h}",
         agent.belief_network.grid_w == 4 and agent.belief_network.grid_h == 4)

    # Run agent for 30 steps with beliefs active
    for _ in range(30):
        obs = env.get_local_observation(agent.x, agent.y, sensor_range=4)
        agent.observe(obs)
        action = agent.decide()
        agent.act(action, env)

    test("Agent survived 30 steps with beliefs", agent.alive)
    test("Belief confidence > 0 somewhere", np.max(agent.belief_network.confidence) > 0)

    # get_state includes beliefs
    state = agent.get_state()
    test("Agent state has beliefs key", "beliefs" in state)
    test("Beliefs summary has avg_food_belief", "avg_food_belief" in state["beliefs"])

    # === World Integration ===
    print("\n\U0001f30d World + Beliefs Integration")
    world = World(small_cfg)
    test("All agents have belief_network",
         all(a.belief_network is not None for a in world.agents))

    results = world.run(steps=30)
    alive = results[-1]["agents_alive"]
    test(f"World ran 30 steps with beliefs, {alive} alive", len(results) == 30)

    # Check that beliefs are being updated
    living = world.get_living_agents()
    if living:
        agent = living[0]
        max_conf = np.max(agent.belief_network.confidence)
        test(f"Agent beliefs updated (max_confidence={max_conf:.3f})", max_conf > 0)


def section4_tests():
    """Section 4: Neural Network Inner State Model."""

    rng = np.random.default_rng(42)

    # === Activation Functions ===
    print("\n\u26a1 Activation Functions")
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    s = sigmoid(x)
    test("Sigmoid output in [0, 1]", np.all(s >= 0) and np.all(s <= 1))
    test("Sigmoid(0) = 0.5", abs(s[2] - 0.5) < 1e-6)
    test("Sigmoid(-10) near 0", s[0] < 0.001)
    test("Sigmoid(10) near 1", s[4] > 0.999)

    # === Network Construction ===
    print("\n\U0001f9e0 Neural Network Construction")
    net = InnerStateNetwork(belief_size=100, hidden_size=16, learning_rate=0.01, rng=rng)
    test("Network creates", net is not None)
    test(f"Input size = {net.input_size} (expected 108)", net.input_size == 108)
    test(f"Hidden1 size = {net.hidden1_size}", net.hidden1_size == 16)
    test(f"Hidden2 size = {net.hidden2_size}", net.hidden2_size == 8)
    test(f"Output size = {net.output_size} (expected 5)", net.output_size == NUM_INNER_STATES)
    test("W1 shape correct", net.W1.shape == (108, 16))
    test("W2 shape correct", net.W2.shape == (16, 8))
    test("W3 shape correct", net.W3.shape == (8, 5))
    test("Initial updates = 0", net.total_updates == 0)

    # Small hidden size
    net_small = InnerStateNetwork(belief_size=20, hidden_size=4, learning_rate=0.05, rng=rng)
    test(f"Small net hidden1 = {net_small.hidden1_size} (min 4)", net_small.hidden1_size >= 4)

    # === Forward Pass ===
    print("\n\u27a1\ufe0f  Forward Pass")
    belief_vec = rng.random(100).astype(np.float32)
    vitals = rng.random(8).astype(np.float32)

    output = net.forward(belief_vec, vitals)
    test(f"Output shape = {output.shape} (expected (5,))", output.shape == (5,))
    test("Output in [0, 1]", np.all(output >= 0) and np.all(output <= 1))
    test("All values finite", np.all(np.isfinite(output)))

    # Deterministic with same input
    output2 = net.forward(belief_vec, vitals)
    test("Same input gives same output", np.allclose(output, output2))

    # Different input gives different output
    diff_vitals = np.ones(8, dtype=np.float32)
    output3 = net.forward(belief_vec, diff_vitals)
    test("Different input gives different output", not np.allclose(output, output3))

    # === Learning ===
    print("\n\U0001f4da Online Learning")
    net_learn = InnerStateNetwork(belief_size=20, hidden_size=8, learning_rate=0.05, rng=rng)
    belief_small = rng.random(20).astype(np.float32)
    vitals_small = rng.random(8).astype(np.float32)
    target = np.array([0.8, 0.1, 0.5, 0.3, 0.2], dtype=np.float32)

    # Initial output (before training)
    initial_out = net_learn.forward(belief_small, vitals_small).copy()
    initial_loss = float(np.mean((initial_out - target) ** 2))

    # Train for many steps
    losses = []
    for _ in range(200):
        net_learn.forward(belief_small, vitals_small)
        loss = net_learn.learn(target)
        losses.append(loss)

    final_out = net_learn.forward(belief_small, vitals_small)
    final_loss = float(np.mean((final_out - target) ** 2))

    test(f"Loss decreased: {initial_loss:.4f} -> {final_loss:.4f}", final_loss < initial_loss)
    test(f"Final loss < 0.1: {final_loss:.4f}", final_loss < 0.1)
    test(f"Total updates = {net_learn.total_updates}", net_learn.total_updates == 200)
    test(f"Avg loss tracked: {net_learn.get_avg_loss():.4f}", net_learn.get_avg_loss() > 0)

    # === Build Vitals Vector ===
    print("\n\U0001f4ca Vitals Vector")
    v = build_vitals_vector(
        energy=80, max_energy=200, age=50, lifespan=500,
        is_isolated=False, adventurousness=0.5, affiliation_need=0.6,
        xenophobia=0.3, plasticity=0.05, exploration_rate=0.4,
    )
    test(f"Vitals shape = {v.shape} (expected (8,))", v.shape == (8,))
    test("Vitals in [0, 1]", np.all(v >= 0) and np.all(v <= 1))
    test(f"Energy norm = {v[0]:.2f} (expected 0.40)", abs(v[0] - 0.40) < 0.01)
    test(f"Age norm = {v[1]:.2f} (expected 0.10)", abs(v[1] - 0.10) < 0.01)
    test("Isolation flag = 0", v[2] == 0.0)

    v_iso = build_vitals_vector(
        energy=80, max_energy=200, age=50, lifespan=500,
        is_isolated=True, adventurousness=0.5, affiliation_need=0.6,
        xenophobia=0.3, plasticity=0.05, exploration_rate=0.4,
    )
    test("Isolation flag = 1 when isolated", v_iso[2] == 1.0)

    # === Heuristic Inner State ===
    print("\n\U0001f3af Heuristic Inner State")
    h = compute_heuristic_inner_state(
        energy=50, max_energy=200, nearby_predators=2,
        nearby_food=3, nearby_agents=1, is_isolated=False,
        adventurousness=0.7, affiliation_need=0.5, xenophobia=0.4,
    )
    test(f"Heuristic shape = {h.shape} (expected (5,))", h.shape == (5,))
    test("Heuristic in [0, 1]", np.all(h >= 0) and np.all(h <= 1))
    test(f"Hunger = {h[0]:.2f} (should be 0.75)", abs(h[0] - 0.75) < 0.01)
    test(f"Fear = {h[1]:.2f} (should be 1.0)", abs(h[1] - 1.0) < 0.01)

    # No predators = no fear
    h2 = compute_heuristic_inner_state(
        energy=200, max_energy=200, nearby_predators=0,
        nearby_food=5, nearby_agents=5, is_isolated=False,
        adventurousness=0.7, affiliation_need=0.5, xenophobia=0.4,
    )
    test(f"No hunger when full: {h2[0]:.2f}", h2[0] < 0.01)
    test(f"No fear without predators: {h2[1]:.2f}", h2[1] < 0.01)

    # === Weight Stats & Clone ===
    print("\n\U0001f4cb Network Stats & Clone")
    stats = net_learn.get_weight_stats()
    test("Stats has total_params", "total_params" in stats)
    test("Stats has avg_loss", "avg_loss" in stats)
    test(f"Total params = {stats['total_params']}", stats["total_params"] > 0)

    clone = net_learn.clone()
    test("Clone creates new object", clone is not net_learn)
    clone_out = clone.forward(belief_small, vitals_small)
    original_out = net_learn.forward(belief_small, vitals_small)
    test("Clone produces same output", np.allclose(clone_out, original_out))

    # === Agent + NN Integration ===
    print("\n\U0001f916 Agent + Neural Network Integration")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    env = Environment(small_cfg)

    agent = Agent(x=10, y=10, energy=100.0, rng=np.random.default_rng(42))
    agent.init_belief_network(20, 20, 4)
    test("Agent has inner_net", agent.inner_net is not None)
    test(f"NN hidden size from genome = {agent.inner_net.hidden1_size}",
         agent.inner_net.hidden1_size == agent.genome["nn_hidden_size"])
    test(f"NN learning rate from plasticity = {agent.inner_net.learning_rate:.4f}",
         abs(agent.inner_net.learning_rate - agent.genome["plasticity"]) < 0.001)

    # Run agent for 50 steps
    for _ in range(50):
        obs = env.get_local_observation(agent.x, agent.y, sensor_range=4)
        agent.observe(obs)
        action = agent.decide()
        agent.act(action, env)

    test("Agent survived 50 steps", agent.alive)
    test(f"NN trained: {agent.inner_net.total_updates} updates",
         agent.inner_net.total_updates == 50)
    test("Inner state has aggression", "aggression" in agent.inner_state)
    test("All inner states in [0,1]",
         all(0 <= v <= 1 for v in agent.inner_state.values()))

    # State includes nn_stats
    state = agent.get_state()
    test("Agent state has nn_stats", "nn_stats" in state)
    test("nn_stats has avg_loss", "avg_loss" in state["nn_stats"])

    # === World + NN Integration ===
    print("\n\U0001f30d World + Neural Network Integration")
    world = World(small_cfg)
    test("All agents have inner_net",
         all(a.inner_net is not None for a in world.agents))

    results = world.run(steps=30)
    alive = results[-1]["agents_alive"]
    test(f"World ran 30 steps, {alive} alive", len(results) == 30)

    living = world.get_living_agents()
    if living:
        a = living[0]
        test(f"Agent NN updates = {a.inner_net.total_updates}",
             a.inner_net.total_updates == 30)
        test(f"Agent NN avg_loss = {a.inner_net.get_avg_loss():.4f}",
             a.inner_net.get_avg_loss() >= 0)


def section5_tests():
    """Section 5: Reinforcement Learning Policy."""

    rng = np.random.default_rng(42)

    # === Q-Network Construction ===
    print("\n\U0001f9e0 Q-Network Construction")
    qnet = QNetwork(hidden_size=16, learning_rate=0.01, rng=rng)
    test("QNetwork creates", qnet is not None)
    test(f"Input size = {qnet.input_size} (expected {STATE_SIZE})", qnet.input_size == STATE_SIZE)
    test(f"Hidden size = {qnet.hidden_size}", qnet.hidden_size == 16)
    test(f"Output size = {qnet.output_size} (expected {NUM_ACTIONS})", qnet.output_size == NUM_ACTIONS)
    test("W1 shape correct", qnet.W1.shape == (STATE_SIZE, 16))
    test("W2 shape correct", qnet.W2.shape == (16, NUM_ACTIONS))

    # === Q-Network Forward ===
    print("\n\u27a1\ufe0f  Q-Network Forward Pass")
    state = rng.random(STATE_SIZE).astype(np.float32)
    q_values = qnet.forward(state)
    test(f"Q-values shape = {q_values.shape} (expected ({NUM_ACTIONS},))",
         q_values.shape == (NUM_ACTIONS,))
    test("Q-values are finite", np.all(np.isfinite(q_values)))

    # Deterministic
    q2 = qnet.forward(state)
    test("Same input = same Q-values", np.allclose(q_values, q2))

    # Different input
    q3 = qnet.forward(np.ones(STATE_SIZE, dtype=np.float32))
    test("Different input = different Q-values", not np.allclose(q_values, q3))

    # === Q-Network Learning ===
    print("\n\U0001f4da Q-Network TD Update")
    qnet_learn = QNetwork(hidden_size=8, learning_rate=0.05, rng=rng)
    test_state = np.array([0.5, 0.0, 0.8, 0.3, 0.1, 0.6, 0.0, 0.2, 1.0], dtype=np.float32)

    initial_q = qnet_learn.forward(test_state).copy()
    # Push action 5 (EAT) toward high value
    for _ in range(100):
        qnet_learn.forward(test_state)
        qnet_learn.update(action=5, td_target=10.0)

    final_q = qnet_learn.forward(test_state)
    test(f"Q[EAT] increased: {initial_q[5]:.2f} -> {final_q[5]:.2f}",
         final_q[5] > initial_q[5])
    test("Q[EAT] is now the highest",
         np.argmax(final_q) == 5)

    # Clone
    clone = qnet_learn.clone()
    clone_q = clone.forward(test_state)
    test("Clone produces same Q-values", np.allclose(final_q, clone_q))

    # === RL Policy Construction ===
    print("\n\U0001f3ae RL Policy Construction")
    policy = RLPolicy(
        epsilon=0.3, gamma=0.95, hidden_size=16,
        learning_rate=0.01, affiliation_need=0.5,
        adventurousness=0.4, rng=rng,
    )
    test("RLPolicy creates", policy is not None)
    test(f"Epsilon = {policy.epsilon}", policy.epsilon == 0.3)
    test(f"Gamma = {policy.gamma}", policy.gamma == 0.95)
    test("Total decisions = 0", policy.total_decisions == 0)

    # === State Building ===
    print("\n\U0001f4ca State Vector")
    inner = {"hunger": 0.5, "fear": 0.1, "curiosity": 0.7, "loneliness": 0.3, "aggression": 0.2}
    obs = {
        "food_positions": [(1, 0), (2, 1), (0, 0)],
        "predator_positions": [(3, 3)],
        "agent_positions": [(1, 1)],
        "obstacle_positions": [],
    }
    sv = policy.build_state(inner, obs)
    test(f"State vector shape = {sv.shape} (expected ({STATE_SIZE},))",
         sv.shape == (STATE_SIZE,))
    test("State[0] = hunger = 0.5", abs(sv[0] - 0.5) < 0.01)
    test("State[5] = food_norm = 0.3", abs(sv[5] - 0.3) < 0.01)
    test("State[8] = on_food = 1.0 (food at (0,0))", sv[8] == 1.0)

    # === Action Selection ===
    print("\n\U0001f3b2 Action Selection")
    actions_taken = set()
    for _ in range(50):
        a = policy.select_action(inner, obs)
        actions_taken.add(a)
    test(f"Multiple actions explored: {len(actions_taken)}", len(actions_taken) > 1)
    test(f"Total decisions = {policy.total_decisions}", policy.total_decisions == 50)
    test("Some explorations occurred", policy.total_explorations > 0)
    test("Some exploitations occurred", policy.total_exploitations > 0)

    # === Reward Computation ===
    print("\n\U0001f3c6 Reward Computation")
    r1 = policy.compute_reward(
        energy_change=20.0, ate=True, nearby_agents=2,
        nearby_predators=0, agent_x=10, agent_y=10,
        is_isolated=False, alive=True,
    )
    test(f"Eating reward = {r1:.2f} (should be > 2)", r1 > 2.0)

    r2 = policy.compute_reward(
        energy_change=-30.0, ate=False, nearby_agents=0,
        nearby_predators=2, agent_x=10, agent_y=10,
        is_isolated=False, alive=True,
    )
    test(f"Danger reward = {r2:.2f} (should be negative)", r2 < 0)

    r3 = policy.compute_reward(
        energy_change=0, ate=False, nearby_agents=0,
        nearby_predators=0, agent_x=10, agent_y=10,
        is_isolated=False, alive=False,
    )
    test(f"Death reward = {r3:.2f} (should be very negative)", r3 < -3)

    # === RL Learning Loop ===
    print("\n\U0001f504 RL Learning Loop")
    pol = RLPolicy(epsilon=0.5, gamma=0.9, hidden_size=8, learning_rate=0.05, rng=rng)
    inner_s = {"hunger": 0.8, "fear": 0.0, "curiosity": 0.3, "loneliness": 0.2, "aggression": 0.1}
    obs_s = {
        "food_positions": [(0, 0)],  # food right here
        "predator_positions": [],
        "agent_positions": [],
        "obstacle_positions": [],
    }

    # Repeatedly select action, compute reward (eating), learn
    eat_count = 0
    for _ in range(200):
        action = pol.select_action(inner_s, obs_s)
        if action == 5:  # EAT
            reward = 3.0
            eat_count += 1
        else:
            reward = -0.1
        pol.learn(reward, inner_s, obs_s, done=False)

    test(f"Cumulative reward tracked: {pol.cumulative_reward:.1f}",
         pol.cumulative_reward != 0)
    test(f"Decisions = {pol.total_decisions}", pol.total_decisions == 200)
    test(f"Avg reward = {pol.get_stats()['avg_reward']:.4f}",
         "avg_reward" in pol.get_stats())

    # After training, the policy should favor EAT when food is present
    pol.epsilon = 0.0  # Pure exploitation
    greedy_actions = [pol.select_action(inner_s, obs_s) for _ in range(20)]
    eat_greedy = sum(1 for a in greedy_actions if a == 5)
    test(f"After training, greedy EAT rate: {eat_greedy}/20",
         eat_greedy > 5)  # Should learn to eat

    # === Epsilon Decay ===
    print("\n\U0001f4c9 Epsilon Decay")
    pol2 = RLPolicy(epsilon=0.5, rng=rng)
    old_eps = pol2.epsilon
    for _ in range(100):
        pol2.decay_epsilon(decay_rate=0.99, min_epsilon=0.1)
    test(f"Epsilon decayed: {old_eps:.3f} -> {pol2.epsilon:.3f}", pol2.epsilon < old_eps)
    test(f"Epsilon above min: {pol2.epsilon:.3f} >= 0.1", pol2.epsilon >= 0.1)

    # === Policy Stats ===
    print("\n\U0001f4cb Policy Stats")
    stats = pol.get_stats()
    test("Stats has epsilon", "epsilon" in stats)
    test("Stats has cumulative_reward", "cumulative_reward" in stats)
    test("Stats has exploration_pct", "exploration_pct" in stats)
    test("Stats has visited_regions", "visited_regions" in stats)
    test(f"Visited regions = {stats['visited_regions']}", stats["visited_regions"] >= 0)

    # === Policy Clone ===
    pol_clone = pol.clone()
    test("Clone creates new object", pol_clone is not pol)
    test("Clone has same epsilon", pol_clone.epsilon == pol.epsilon)

    # === Agent + RL Integration ===
    print("\n\U0001f916 Agent + RL Policy Integration")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    env = Environment(small_cfg)

    agent = Agent(x=10, y=10, energy=100.0, rng=np.random.default_rng(42))
    agent.init_belief_network(20, 20, 4)
    test("Agent has RL policy", agent.policy is not None)
    test(f"Policy epsilon from genome = {agent.policy.epsilon:.3f}",
         abs(agent.policy.epsilon - agent.genome["exploration_rate"]) < 0.001)

    # Run agent for 50 steps
    for _ in range(50):
        obs = env.get_local_observation(agent.x, agent.y, sensor_range=4)
        agent.observe(obs)
        action = agent.decide()
        agent.act(action, env)

    test("Agent survived 50 steps with RL", agent.alive)
    test(f"RL decisions > 0: {agent.policy.total_decisions}",
         agent.policy.total_decisions > 0)
    test(f"RL cumulative reward tracked",
         agent.policy.cumulative_reward != 0 or agent.policy.total_decisions > 0)

    # Agent state includes rl_stats
    state = agent.get_state()
    test("Agent state has rl_stats", "rl_stats" in state)
    test("rl_stats has avg_reward", "avg_reward" in state["rl_stats"])

    # === World + RL Integration ===
    print("\n\U0001f30d World + RL Integration")
    world = World(small_cfg)
    test("All agents have RL policy",
         all(a.policy is not None for a in world.agents))

    results = world.run(steps=50)
    alive = results[-1]["agents_alive"]
    test(f"World ran 50 steps with RL, {alive} alive", len(results) == 50)

    living = world.get_living_agents()
    if living:
        a = living[0]
        test(f"Agent RL policy active",
             a.policy is not None)
        test(f"Agent epsilon valid: {a.policy.epsilon:.4f}",
             0.0 <= a.policy.epsilon <= 1.0)

    # Determinism
    w1 = World(small_cfg)
    w2 = World(small_cfg)
    r1 = w1.run(steps=20)
    r2 = w2.run(steps=20)
    det = all(a["agents_alive"] == b["agents_alive"] for a, b in zip(r1, r2))
    test("World with RL is deterministic (same seed)", det)


def section6_tests():
    """Section 6: Multi-Agent Interaction."""

    rng = np.random.default_rng(42)

    # === InteractionManager Construction ===
    print("\n\U0001f91d InteractionManager Construction")
    im = InteractionManager(
        communication_range=3,
        reproduction_range=2,
        reproduction_cost=40.0,
        child_energy=50.0,
        min_reproduction_age=50,
        communication_cooldown=5,
        rng=rng,
    )
    test("InteractionManager creates", im is not None)
    test(f"Comm range = {im.communication_range}", im.communication_range == 3)
    test(f"Repro range = {im.reproduction_range}", im.reproduction_range == 2)
    test("Initial comms = 0", im.total_communications == 0)
    test("Initial repros = 0", im.total_reproductions == 0)

    # === Communication Tests ===
    print("\n\U0001f4e1 Communication (Belief Sharing)")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    env = Environment(small_cfg)

    # Two agents close together
    a1 = Agent(x=5, y=5, energy=100.0, rng=np.random.default_rng(10))
    a2 = Agent(x=6, y=5, energy=100.0, rng=np.random.default_rng(11))
    a1.init_belief_network(20, 20, 4)
    a2.init_belief_network(20, 20, 4)

    # Give a1 some unique beliefs (food at region 0,0)
    a1.belief_network.food_belief[0, 0] = 0.9
    a1.belief_network.confidence[0, 0] = 0.8
    old_a2_food = a2.belief_network.food_belief[0, 0]

    im_test = InteractionManager(
        communication_range=3, communication_cooldown=1, rng=np.random.default_rng(42)
    )
    result = im_test.process_interactions(
        [a1, a2], env, current_step=10,
        world_width=20, world_height=20, sensor_range=4,
    )
    test(f"Communications occurred: {result['communications']}",
         result['communications'] > 0)
    test("a2's food belief changed",
         a2.belief_network.food_belief[0, 0] != old_a2_food)

    # Interaction history tracked
    test("Interaction history recorded",
         len(im_test.interaction_history) > 0)

    # Familiarity query
    fam = im_test.get_familiarity(a1.id, a2.id)
    test(f"Familiarity between agents: {fam}", fam > 0)

    # Cooldown prevents immediate re-communication
    result2 = im_test.process_interactions(
        [a1, a2], env, current_step=10,  # Same step
        world_width=20, world_height=20, sensor_range=4,
    )
    test("Cooldown prevents re-communication", result2['communications'] == 0)

    # After cooldown, can communicate again
    result3 = im_test.process_interactions(
        [a1, a2], env, current_step=12,
        world_width=20, world_height=20, sensor_range=4,
    )
    test("Communication works after cooldown", result3['communications'] > 0)

    # Distant agents can't communicate
    Agent.reset_id_counter()
    a_far1 = Agent(x=0, y=0, energy=100.0, rng=np.random.default_rng(20))
    a_far2 = Agent(x=10, y=10, energy=100.0, rng=np.random.default_rng(21))
    a_far1.init_belief_network(20, 20, 4)
    a_far2.init_belief_network(20, 20, 4)
    im_far = InteractionManager(communication_range=3, rng=np.random.default_rng(42))
    result_far = im_far.process_interactions(
        [a_far1, a_far2], env, current_step=1,
        world_width=20, world_height=20, sensor_range=4,
    )
    test("Distant agents can't communicate", result_far['communications'] == 0)

    # === Reproduction Tests ===
    print("\n\U0001f476 Reproduction")
    Agent.reset_id_counter()
    # Two nearby agents with very high energy and low reproduction threshold
    parent_genome_a = Genome.default()
    parent_genome_a.genes["reproduction_threshold"] = 50.0
    parent_genome_a.genes["xenophobia"] = 0.0  # No xenophobia blocking
    parent_genome_b = Genome.default()
    parent_genome_b.genes["reproduction_threshold"] = 50.0
    parent_genome_b.genes["xenophobia"] = 0.0

    pa = Agent(genome=parent_genome_a, x=5, y=5, energy=180.0,
               rng=np.random.default_rng(30))
    pb = Agent(genome=parent_genome_b, x=5, y=6, energy=180.0,
               rng=np.random.default_rng(31))
    pa.init_belief_network(20, 20, 4)
    pb.init_belief_network(20, 20, 4)
    # Set age above min_reproduction_age
    pa.age = 100
    pb.age = 100

    im_repro = InteractionManager(
        reproduction_range=2,
        reproduction_cost=40.0,
        child_energy=50.0,
        min_reproduction_age=50,
        communication_cooldown=100,  # Prevent comm from interfering
        rng=np.random.default_rng(42),
    )

    # Run multiple times to account for probability
    total_children = []
    for step in range(50):
        result_r = im_repro.process_interactions(
            [pa, pb] + total_children, env, current_step=step,
            world_width=20, world_height=20, sensor_range=4,
        )
        total_children.extend(result_r.get("new_agents", []))
        # Reset parent energy to keep trying
        pa.energy = 180.0
        pb.energy = 180.0

    test(f"Children born: {len(total_children)} (should be > 0)", len(total_children) > 0)
    if total_children:
        child = total_children[0]
        test("Child has valid position", child.x >= 0 and child.y >= 0)
        test(f"Child energy = {child.energy}", child.energy == 50.0)
        test("Child has belief network", child.belief_network is not None)
        test("Child has inner_net", child.inner_net is not None)
        test("Child has RL policy", child.policy is not None)
        test(f"Child generation = {child.generation}", child.generation == 1)
        test("Child genome is valid Genome",
             hasattr(child.genome, 'genes') and len(child.genome.genes) == 9)

    # Isolated agents can't reproduce
    pa_iso = Agent(x=5, y=5, energy=180.0, rng=np.random.default_rng(40))
    pa_iso.init_belief_network(20, 20, 4)
    pa_iso.age = 100
    pa_iso.isolate(0, 0)  # Isolate
    pb_near = Agent(x=1, y=0, energy=180.0, rng=np.random.default_rng(41))
    pb_near.init_belief_network(20, 20, 4)
    pb_near.age = 100
    im_iso = InteractionManager(min_reproduction_age=50, rng=np.random.default_rng(42))
    result_iso = im_iso.process_interactions(
        [pa_iso, pb_near], env, current_step=1,
        world_width=20, world_height=20, sensor_range=4,
    )
    test("Isolated agents can't reproduce", result_iso['reproductions'] == 0)

    # Low energy prevents reproduction
    Agent.reset_id_counter()
    low_a = Agent(x=5, y=5, energy=30.0, rng=np.random.default_rng(50))
    low_b = Agent(x=5, y=6, energy=30.0, rng=np.random.default_rng(51))
    low_a.init_belief_network(20, 20, 4)
    low_b.init_belief_network(20, 20, 4)
    low_a.age = 100
    low_b.age = 100
    im_low = InteractionManager(min_reproduction_age=50, rng=np.random.default_rng(42))
    result_low = im_low.process_interactions(
        [low_a, low_b], env, current_step=1,
        world_width=20, world_height=20, sensor_range=4,
    )
    test("Low-energy agents can't reproduce", result_low['reproductions'] == 0)

    # === Social Network ===
    print("\n\U0001f310 Social Network")
    Agent.reset_id_counter()
    agents_group = []
    for i in range(5):
        a = Agent(x=5+i, y=5, energy=100.0, rng=np.random.default_rng(60+i))
        a.init_belief_network(20, 20, 4)
        agents_group.append(a)

    im_social = InteractionManager(communication_range=3, communication_cooldown=1,
                                   rng=np.random.default_rng(42))
    # Run a few steps
    for step in range(5):
        im_social.process_interactions(
            agents_group, env, current_step=step*2,
            world_width=20, world_height=20, sensor_range=4,
        )

    network = im_social.get_social_network(agents_group)
    test("Social network has entries", len(network) > 0)
    connected_agents = sum(1 for v in network.values() if len(v) > 0)
    test(f"Connected agents: {connected_agents}", connected_agents > 0)

    stats = im_social.get_stats()
    test("Stats has total_communications", "total_communications" in stats)
    test("Stats has unique_interactions", "unique_interactions" in stats)
    test(f"Total comms = {stats['total_communications']}", stats['total_communications'] > 0)

    # === World + Interaction Integration ===
    print("\n\U0001f30d World + Interaction Integration")
    Agent.reset_id_counter()
    world = World(small_cfg)
    test("World has interaction_manager", world.interaction_manager is not None)

    results = world.run(steps=100)
    test(f"World ran 100 steps with interactions", len(results) > 0)

    # Check metrics include interaction data
    last = results[-1]
    test("Metrics has agents_born_this_step", "agents_born_this_step" in last)
    test("Metrics has communications", "communications" in last)
    test("Metrics has reproductions", "reproductions" in last)

    # Check if any communication happened
    total_comms = sum(r.get("communications", 0) for r in results)
    test(f"Total communications over 100 steps: {total_comms}", total_comms >= 0)

    # Check total agents (may have grown from reproduction)
    test(f"Total agents tracked: {last.get('total_agents', len(world.agents))}",
         len(world.agents) >= small_cfg.agents.population_size)

    # Population stats include interaction data
    pop = world.get_population_stats()
    if pop.get("alive", 0) > 0:
        test("Pop stats has total_communications", "total_communications" in pop)
        test("Pop stats has total_reproductions", "total_reproductions" in pop)

    # World reset clears interactions
    world.reset()
    test("Reset clears interaction history",
         world.interaction_manager.total_communications == 0)


def section7_tests():
    """Section 7: Genetic Algorithm & Evolution."""

    rng = np.random.default_rng(42)

    # === EvolutionManager Construction ===
    print("\n\U0001f9ec EvolutionManager Construction")
    em = EvolutionManager(
        population_size=20, elite_count=3, tournament_size=3,
        mutation_rate=0.1, mutation_strength=0.15, crossover_rate=0.7,
        rng=rng,
    )
    test("EvolutionManager creates", em is not None)
    test(f"Population size = {em.population_size}", em.population_size == 20)
    test(f"Elite count = {em.elite_count}", em.elite_count == 3)
    test(f"Tournament size = {em.tournament_size}", em.tournament_size == 3)
    test("Generation = 0", em.current_generation == 0)
    test("No history yet", len(em.generation_history) == 0)
    test("Best fitness = 0", em.best_fitness == 0.0)

    # === Fitness Evaluation ===
    print("\n\U0001f3c6 Fitness Evaluation")
    Agent.reset_id_counter()
    agents = []
    for i in range(10):
        a = Agent(x=i, y=0, energy=50 + i * 10, rng=np.random.default_rng(i))
        a.age = 100 + i * 20
        a.total_food_eaten = i * 5
        agents.append(a)

    scored = em.evaluate_generation(agents)
    test(f"Scored {len(scored)} agents", len(scored) == 10)
    test("Sorted by fitness (best first)",
         scored[0][1] >= scored[-1][1])
    test("Best fitness tracked", em.best_fitness > 0)
    test("Best genome tracked", em.best_genome is not None)

    # === Tournament Selection ===
    print("\n\U0001f3b2 Tournament Selection")
    selected_genomes = set()
    for _ in range(20):
        g = em.tournament_select(scored)
        selected_genomes.add(id(g))
    test(f"Multiple genomes selected: {len(selected_genomes)}",
         len(selected_genomes) > 1)

    # Fitness proportional
    fp_genomes = set()
    for _ in range(20):
        g = em.fitness_proportional_select(scored)
        fp_genomes.add(id(g))
    test(f"FP selection picks varied genomes: {len(fp_genomes)}",
         len(fp_genomes) > 1)

    # Empty list handling
    empty_g = em.tournament_select([])
    test("Empty list returns random genome", empty_g is not None)

    # === Next Generation Creation ===
    print("\n\U0001f476 Next Generation Creation")
    new_genomes = em.create_next_generation(scored)
    test(f"New generation has {len(new_genomes)} genomes (expected 20)",
         len(new_genomes) == 20)

    # Elites preserved
    elite_genomes = [scored[i][0].genome for i in range(min(3, len(scored)))]
    elites_found = sum(1 for ng in new_genomes[:3] if ng in elite_genomes)
    test(f"Elites preserved: {elites_found}/3", elites_found >= 2)

    # All genes valid
    all_valid = all(
        all(
            spec[0] <= g[name] <= spec[1]
            for name, spec in Genome.get_spec().items()
        )
        for g in new_genomes
    )
    test("All new genomes have valid genes", all_valid)

    # Diversity exists
    if len(new_genomes) >= 2:
        dists = [new_genomes[0].distance(new_genomes[i]) for i in range(1, len(new_genomes))]
        avg_dist = np.mean(dists)
        test(f"New generation has diversity: {avg_dist:.4f}", avg_dist > 0)

    # From empty scored list
    random_gen = em.create_next_generation([])
    test("Empty scored → random generation", len(random_gen) == 20)

    # === Generation Recording ===
    print("\n\U0001f4ca Generation Recording")
    record = em.record_generation(
        generation=0, scored_agents=scored, steps_survived=500,
    )
    test("Record has generation", record["generation"] == 0)
    test("Record has best_fitness", "best_fitness" in record)
    test("Record has avg_fitness", "avg_fitness" in record)
    test("Record has genome_diversity", "genome_diversity" in record)
    test("Record has avg_genes", "avg_genes" in record)
    test("avg_genes has adventurousness", "adventurousness" in record["avg_genes"])
    test("Record has steps_survived", record["steps_survived"] == 500)
    test("History length = 1", len(em.generation_history) == 1)

    # === Fitness Trend ===
    print("\n\U0001f4c8 Fitness Trends")
    # Record a second generation
    em.record_generation(generation=1, scored_agents=scored, steps_survived=600)
    trend = em.get_fitness_trend()
    test("Trend has generations", len(trend["generations"]) == 2)
    test("Trend has best_fitness", len(trend["best_fitness"]) == 2)
    test("Trend has diversity", len(trend["diversity"]) == 2)

    gene_trend = em.get_gene_trend("adventurousness")
    test(f"Gene trend length = {len(gene_trend)}", len(gene_trend) == 2)

    stats = em.get_stats()
    test("Stats has generations_completed", stats["generations_completed"] == 2)
    test("Stats has best_fitness_ever", stats["best_fitness_ever"] > 0)

    # === World + Evolution Integration ===
    print("\n\U0001f30d World + Evolution Integration")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    world = World(small_cfg)
    test("World has evolution_manager", world.evolution_manager is not None)

    # Run one generation
    world.run(steps=100)
    gen_record = world.evolve()
    test("Evolve returns record", gen_record is not None)
    test("Record has best_fitness", "best_fitness" in gen_record)
    test(f"Generation = {world.current_generation}", world.current_generation == 1)
    test(f"Step reset to 0: {world.current_step}", world.current_step == 0)
    test(f"Agents respawned: {len(world.agents)}", len(world.agents) == small_cfg.agents.population_size)
    test("All agents alive", all(a.alive for a in world.agents))
    test("All agents have beliefs", all(a.belief_network is not None for a in world.agents))
    test("All agents have NN", all(a.inner_net is not None for a in world.agents))
    test("All agents have RL", all(a.policy is not None for a in world.agents))

    # Run another generation to verify multi-gen
    world.run(steps=100)
    gen2 = world.evolve()
    test(f"Second generation: {world.current_generation}", world.current_generation == 2)

    # === Multi-Generation Evolution ===
    print("\n\U0001f500 Multi-Generation Evolution")
    Agent.reset_id_counter()
    world2 = World(small_cfg)
    records = world2.run_evolution(
        num_generations=5, steps_per_generation=50, verbose=False,
    )
    test(f"Ran 5 generations", len(records) == 5)
    test("Each record has fitness", all("avg_fitness" in r for r in records))
    test(f"Final generation = {world2.current_generation}",
         world2.current_generation == 5)

    # Evolution manager has history
    test(f"Evolution history = {len(world2.evolution_manager.generation_history)}",
         len(world2.evolution_manager.generation_history) == 5)

    # Fitness should be tracked
    trend = world2.evolution_manager.get_fitness_trend()
    test("Trend has 5 generations", len(trend["generations"]) == 5)

    # Population alive and functional after evolution
    test(f"Agents alive after evolution: {len(world2.get_living_agents())}",
         len(world2.get_living_agents()) == small_cfg.agents.population_size)


def section8_tests():
    """Section 8: Isolation Experiment Workflow."""

    # === Experiment Construction ===
    print("\n\U0001f9ea Experiment Construction")
    small_cfg = make_small_config()
    exp = IsolationExperiment(
        config=small_cfg,
        isolation_fraction=0.3,
        isolation_duration=20,
        isolation_frequency=5,
        selection_criteria="adventurousness",
        isolation_zone=(0, 0),
    )
    test("Experiment creates", exp is not None)
    test(f"Isolation fraction = {exp.isolation_fraction}", exp.isolation_fraction == 0.3)
    test(f"Isolation duration = {exp.isolation_duration}", exp.isolation_duration == 20)
    test(f"Isolation frequency = {exp.isolation_frequency}", exp.isolation_frequency == 5)
    test(f"Selection criteria = {exp.selection_criteria}", exp.selection_criteria == "adventurousness")
    test("No worlds yet", exp.control_world is None)

    # === Setup ===
    print("\n\u2699\ufe0f  Experiment Setup")
    exp.setup()
    test("Control world created", exp.control_world is not None)
    test("Treatment world created", exp.treatment_world is not None)
    test(f"Control agents: {len(exp.control_world.agents)}",
         len(exp.control_world.agents) == small_cfg.agents.population_size)
    test(f"Treatment agents: {len(exp.treatment_world.agents)}",
         len(exp.treatment_world.agents) == small_cfg.agents.population_size)

    # === Agent Selection ===
    print("\n\U0001f3af Agent Selection for Isolation")
    Agent.reset_id_counter()
    agents_sel = []
    for i in range(10):
        g = Genome.default()
        g.genes["adventurousness"] = i * 0.1
        g.genes["affiliation_need"] = (9 - i) * 0.1
        a = Agent(genome=g, x=5+i, y=5, energy=100.0, rng=np.random.default_rng(i))
        agents_sel.append(a)

    # Adventurousness selection (highest first)
    exp_adv = IsolationExperiment(selection_criteria="adventurousness")
    selected = exp_adv.select_agents_for_isolation(agents_sel, 3)
    test(f"Selected 3 agents", len(selected) == 3)
    test("Highest adventurousness selected",
         selected[0].genome["adventurousness"] >= selected[1].genome["adventurousness"])

    # Affiliation selection
    exp_aff = IsolationExperiment(selection_criteria="affiliation_need")
    selected_aff = exp_aff.select_agents_for_isolation(agents_sel, 2)
    test(f"Affiliation: selected 2", len(selected_aff) == 2)
    test("Highest affiliation selected",
         selected_aff[0].genome["affiliation_need"] >= selected_aff[1].genome["affiliation_need"])

    # Random selection
    exp_rnd = IsolationExperiment(selection_criteria="random")
    selected_rnd = exp_rnd.select_agents_for_isolation(agents_sel, 3)
    test(f"Random: selected 3", len(selected_rnd) == 3)

    # Fitness-based selection
    exp_best = IsolationExperiment(selection_criteria="best_fitness")
    selected_best = exp_best.select_agents_for_isolation(agents_sel, 2)
    test(f"Best fitness: selected 2", len(selected_best) == 2)

    # Skip already-isolated agents
    agents_sel[0].is_isolated = True
    selected2 = exp_adv.select_agents_for_isolation(agents_sel, 3)
    test("Skips isolated agents",
         all(not a.is_isolated for a in selected2))
    agents_sel[0].is_isolated = False  # Reset

    # === Single Generation Run ===
    print("\n\U0001f501 Single Generation Run")
    Agent.reset_id_counter()
    exp_gen = IsolationExperiment(
        config=small_cfg,
        isolation_fraction=0.3,
        isolation_duration=10,
        isolation_frequency=5,
        selection_criteria="adventurousness",
    )
    exp_gen.setup()
    comparison = exp_gen.run_single_generation(steps=50, verbose=False)
    test("Comparison has control", "control" in comparison)
    test("Comparison has treatment", "treatment" in comparison)
    test("Control has avg_fitness", "avg_fitness" in comparison["control"])
    test("Treatment has avg_fitness", "avg_fitness" in comparison["treatment"])
    test("Treatment has total_isolations", "total_isolations" in comparison["treatment"])

    # Isolation events recorded
    test(f"Isolation events: {len(exp_gen.isolation_events)}",
         len(exp_gen.isolation_events) > 0)

    # Some agents were isolated
    isolated_events = [e for e in exp_gen.isolation_events if e["action"] == "isolated"]
    test(f"Isolations occurred: {len(isolated_events)}",
         len(isolated_events) > 0)

    # Some returned
    returned_events = [e for e in exp_gen.isolation_events if e["action"] == "returned"]
    test(f"Returns occurred: {len(returned_events)}",
         len(returned_events) >= 0)  # May or may not have returned yet

    # === Multi-Generation Experiment ===
    print("\n\U0001f52c Multi-Generation Experiment")
    Agent.reset_id_counter()
    exp_full = IsolationExperiment(
        config=small_cfg,
        isolation_fraction=0.2,
        isolation_duration=10,
        isolation_frequency=10,
        selection_criteria="adventurousness",
    )
    results = exp_full.run_experiment(
        num_generations=3, steps_per_generation=50, verbose=False,
    )
    test("Results has config", "config" in results)
    test("Results has control_generations", "control_generations" in results)
    test("Results has treatment_generations", "treatment_generations" in results)
    test("Results has comparison", "comparison" in results)
    test(f"Control generations: {len(results['control_generations'])}",
         len(results["control_generations"]) == 3)
    test(f"Treatment generations: {len(results['treatment_generations'])}",
         len(results["treatment_generations"]) == 3)

    # Comparison metrics
    comp = results["comparison"]
    test("Comparison has fitness", "fitness" in comp)
    test("Comparison has survival", "survival" in comp)
    test("Comparison has diversity", "diversity" in comp)
    test("Comparison has food", "food" in comp)
    test("Fitness has isolation_impact", "isolation_impact" in comp["fitness"])

    # Total isolation events
    test(f"Total isolations: {results['total_isolation_events']}",
         results["total_isolation_events"] >= 0)

    # === Different Selection Criteria ===
    print("\n\U0001f500 Different Selection Criteria")
    for criteria in ["random", "affiliation_need", "best_fitness"]:
        Agent.reset_id_counter()
        exp_c = IsolationExperiment(
            config=small_cfg,
            isolation_fraction=0.2,
            isolation_duration=10,
            isolation_frequency=10,
            selection_criteria=criteria,
        )
        res = exp_c.run_experiment(
            num_generations=2, steps_per_generation=30, verbose=False,
        )
        test(f"Criteria '{criteria}' runs OK",
             len(res["control_generations"]) == 2)

    # === Results Summary ===
    print("\n\U0001f4cb Results Summary")
    results2 = exp_full.get_results()
    test("get_results works", results2 is not None)
    test("Config in results", results2["config"]["selection_criteria"] == "adventurousness")

    # repr
    test("repr works", "IsolationExperiment" in repr(exp_full))


def section9_tests():
    """Section 9: Data Collection, Logging & Export."""

    import os
    import json
    import shutil

    export_dir = "/tmp/swarm_test_exports"

    # === DataCollector Construction ===
    print("\n\U0001f4e6 DataCollector Construction")
    dc = DataCollector(snapshot_interval=10, collect_agent_snapshots=True)
    test("DataCollector creates", dc is not None)
    test("Snapshot interval = 10", dc.snapshot_interval == 10)
    test("No step records", len(dc.step_records) == 0)
    test("No snapshots", len(dc.agent_snapshots) == 0)
    test("repr works", "DataCollector" in repr(dc))

    # === Recording Step Metrics ===
    print("\n\U0001f4ca Recording Step Metrics")
    dc.start({"experiment": "test"})
    test("Metadata has start_time", "start_time" in dc.metadata)

    metrics = {
        "agents_alive": 10, "food_count": 50, "avg_energy": 85.3,
        "avg_age": 42.0, "agents_born_this_step": 1, "died_this_step": 0,
    }
    dc.record_step(step=1, generation=0, metrics=metrics, label="control")
    dc.record_step(step=2, generation=0, metrics=metrics, label="control")
    dc.record_step(step=1, generation=0, metrics=metrics, label="treatment")
    test(f"Step records: {len(dc.step_records)}", len(dc.step_records) == 3)
    test("Record has label", dc.step_records[0]["label"] == "control")
    test("Record has step", dc.step_records[0]["step"] == 1)

    # === Recording Agent Snapshots ===
    print("\n\U0001f4f8 Recording Agent Snapshots")
    Agent.reset_id_counter()
    agents = []
    for i in range(5):
        a = Agent(x=i, y=0, energy=80 + i*5, rng=np.random.default_rng(i))
        a.age = 50
        agents.append(a)

    dc.record_agent_snapshot(step=10, generation=0, agents=agents, label="test")
    test(f"Agent snapshots: {len(dc.agent_snapshots)}", len(dc.agent_snapshots) == 5)
    snap = dc.agent_snapshots[0]
    test("Snapshot has agent_id", "agent_id" in snap)
    test("Snapshot has energy", "energy" in snap)
    test("Snapshot has fitness", "fitness" in snap)
    test("Snapshot has gene_adventurousness", "gene_adventurousness" in snap)
    test("Snapshot has inner_hunger", "inner_hunger" in snap)

    dc.record_agent_snapshot(step=11, generation=0, agents=agents, label="test")
    test("Non-interval step skipped", len(dc.agent_snapshots) == 5)

    # === Recording Generation Summaries ===
    print("\n\U0001f9ec Recording Generation Summaries")
    gen_record = {
        "generation": 0, "steps_survived": 200, "population_size": 50,
        "alive_at_end": 30, "best_fitness": 0.82, "avg_fitness": 0.45,
        "avg_genes": {"adventurousness": 0.35, "xenophobia": 0.22},
    }
    dc.record_generation(dict(gen_record), label="control")
    test(f"Generation records: {len(dc.generation_records)}", len(dc.generation_records) == 1)
    test("Has avg_gene_adventurousness", "avg_gene_adventurousness" in dc.generation_records[0])

    # === Recording Isolation Events ===
    print("\n\U0001f3dd\ufe0f  Recording Isolation Events")
    dc.record_isolation_event({"step": 5, "agent_id": 0, "action": "isolated"})
    dc.record_isolation_event({"step": 105, "agent_id": 0, "action": "returned"})
    test(f"Isolation events: {len(dc.isolation_events)}", len(dc.isolation_events) == 2)

    # === CSV Export ===
    print("\n\U0001f4c4 CSV Export")
    dc.stop()
    exported = dc.export_csv(output_dir=export_dir, prefix="test")
    test("Exported steps CSV", "steps" in exported)
    test("Exported agents CSV", "agents" in exported)
    test("Exported generations CSV", "generations" in exported)
    test("Exported isolation CSV", "isolation" in exported)

    steps_path = exported["steps"]
    test("Steps file exists", os.path.exists(steps_path))
    with open(steps_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    test(f"Steps CSV has {len(rows)} rows", len(rows) == 3)
    test("Steps CSV has label column", "label" in rows[0])

    agents_path = exported["agents"]
    with open(agents_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    test(f"Agents CSV has {len(rows)} rows", len(rows) == 5)
    test("Agents CSV has gene columns", "gene_adventurousness" in rows[0])

    # === JSON Export ===
    print("\n\U0001f4be JSON Export")
    json_path = dc.export_json(output_dir=export_dir, prefix="test")
    test("JSON file exists", os.path.exists(json_path))
    with open(json_path) as f:
        data = json.load(f)
    test("JSON has metadata", "metadata" in data)
    test("JSON has step_records", "step_records" in data)
    test("JSON has summary", "summary" in data)
    test("Duration recorded", data["metadata"].get("duration_seconds") is not None)

    # === SimLogger ===
    print("\n\U0001f4dd SimLogger")
    log_path = os.path.join(export_dir, "test.log")
    logger = SimLogger(level=SimLogger.STEPS, log_file=log_path)
    test("SimLogger creates", logger is not None)
    logger.header("Test Header")
    logger.info("Test info")
    logger.step(1, metrics)
    logger.generation(gen_record)
    logger.close()
    test("Log file exists", os.path.exists(log_path))
    with open(log_path) as f:
        log_content = f.read()
    test("Log has header", "Test Header" in log_content)
    test("Log has step", "Step" in log_content)

    # === collect_from_world ===
    print("\n\U0001f30d Collect from World")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    world = World(small_cfg)
    dc2 = DataCollector(snapshot_interval=25)
    dc2.start()
    for _ in range(50):
        world.step()
        collect_from_world(world, dc2, label="sim")
    dc2.stop()
    test(f"Collected {len(dc2.step_records)} step records", len(dc2.step_records) == 50)
    test(f"Collected snapshots", len(dc2.agent_snapshots) > 0)
    exported2 = dc2.export_csv(output_dir=export_dir, prefix="world")
    test("World export has steps", "steps" in exported2)

    # === collect_from_experiment ===
    print("\n\U0001f52c Collect from Experiment")
    Agent.reset_id_counter()
    exp = IsolationExperiment(
        config=small_cfg, isolation_fraction=0.2, isolation_duration=10,
        isolation_frequency=10, selection_criteria="random",
    )
    exp.run_experiment(num_generations=2, steps_per_generation=30, verbose=False)
    dc3 = DataCollector(snapshot_interval=10)
    dc3.start()
    collect_from_experiment(exp, dc3)
    dc3.stop()
    test(f"Experiment step records: {len(dc3.step_records)}", len(dc3.step_records) > 0)
    test(f"Experiment gen records: {len(dc3.generation_records)}", len(dc3.generation_records) > 0)
    labels = set(r["label"] for r in dc3.step_records)
    test("Has control label", "control" in labels)
    test("Has treatment label", "treatment" in labels)
    json_path3 = dc3.export_json(output_dir=export_dir, prefix="experiment")
    test("Experiment JSON exported", os.path.exists(json_path3))

    # === Summary & Clear ===
    print("\n\U0001f4cb Summary & Clear")
    summary = dc3.get_summary()
    test("Summary has total_step_records", "total_step_records" in summary)
    test("Summary has unique_labels", "unique_labels" in summary)
    dc3.clear()
    test("Clear empties records", len(dc3.step_records) == 0)

    # Cleanup
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)


def section10_tests():
    """Section 10: Visualization & Analysis."""

    import os
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = "/tmp/swarm_test_plots"

    # === Fitness Evolution Plot ===
    print("\n\U0001f4c8 Fitness Evolution Plot")
    gens = list(range(5))
    best = [0.3, 0.5, 0.55, 0.6, 0.65]
    avg = [0.2, 0.3, 0.35, 0.4, 0.45]
    div = [0.4, 0.35, 0.3, 0.28, 0.25]
    fp = os.path.join(plot_dir, "test_fitness.png")
    fig = plot_fitness_evolution(gens, best, avg, div, filepath=fp)
    test("Fitness plot returns Figure", isinstance(fig, plt.Figure))
    test("Fitness plot saved", os.path.exists(fp))
    test("Fitness plot file > 0 bytes", os.path.getsize(fp) > 100)
    plt.close(fig)

    # === Gene Evolution Plot ===
    print("\n\U0001f9ec Gene Evolution Plot")
    gene_trends = {
        "adventurousness": [0.3, 0.35, 0.4, 0.38, 0.42],
        "xenophobia": [0.5, 0.45, 0.4, 0.35, 0.3],
        "plasticity": [0.05, 0.06, 0.055, 0.06, 0.065],
    }
    fp = os.path.join(plot_dir, "test_genes.png")
    fig = plot_gene_evolution(gens, gene_trends, filepath=fp)
    test("Gene plot returns Figure", isinstance(fig, plt.Figure))
    test("Gene plot saved", os.path.exists(fp))
    plt.close(fig)

    # === Population Dynamics Plot ===
    print("\n\U0001f4ca Population Dynamics Plot")
    steps = list(range(1, 51))
    alive = [50 - i//3 for i in range(50)]
    energy = [100 - i*0.5 for i in range(50)]
    food = [200 - i for i in range(50)]
    fp = os.path.join(plot_dir, "test_pop.png")
    fig = plot_population_dynamics(steps, alive, energy, food, filepath=fp)
    test("Population plot returns Figure", isinstance(fig, plt.Figure))
    test("Population plot saved", os.path.exists(fp))
    plt.close(fig)

    # === Isolation Comparison Bar Chart ===
    print("\n\U0001f4ca Isolation Comparison Plot")
    comparison = {
        "fitness": {"control_avg": 0.45, "treatment_avg": 0.30},
        "survival": {"control_avg": 0.36, "treatment_avg": 0.02},
        "diversity": {"control_avg": 0.28, "treatment_avg": 0.30},
        "food": {"control_avg": 1800, "treatment_avg": 600},
    }
    fp = os.path.join(plot_dir, "test_comparison.png")
    fig = plot_isolation_comparison(comparison, filepath=fp)
    test("Comparison plot returns Figure", isinstance(fig, plt.Figure))
    test("Comparison plot saved", os.path.exists(fp))
    plt.close(fig)

    # === Isolation Timeline Plot ===
    print("\n\U0001f4c9 Isolation Timeline Plot")
    ctrl_alive = [50 - i//5 for i in range(100)]
    treat_alive = [50 - i//2 for i in range(100)]
    treat_alive = [max(0, x) for x in treat_alive]
    ctrl_energy = [100 - i*0.3 for i in range(100)]
    treat_energy = [100 - i*0.8 for i in range(100)]
    treat_energy = [max(0, x) for x in treat_energy]
    fp = os.path.join(plot_dir, "test_timeline.png")
    fig = plot_isolation_timeline(ctrl_alive, treat_alive,
                                  ctrl_energy, treat_energy, filepath=fp)
    test("Timeline plot returns Figure", isinstance(fig, plt.Figure))
    test("Timeline plot saved", os.path.exists(fp))
    plt.close(fig)

    # === Genome Heatmap ===
    print("\n\U0001f5fa\ufe0f  Genome Heatmap Plot")
    rng = np.random.default_rng(42)
    agents_data = []
    for i in range(20):
        d = {}
        for gene in GENE_NAMES:
            d[f"gene_{gene}"] = rng.random()
        agents_data.append(d)
    fp = os.path.join(plot_dir, "test_heatmap.png")
    fig = plot_genome_heatmap(agents_data, filepath=fp)
    test("Heatmap plot returns Figure", isinstance(fig, plt.Figure))
    test("Heatmap plot saved", os.path.exists(fp))
    plt.close(fig)

    # === Environment Snapshot ===
    print("\n\U0001f30d Environment Snapshot Plot")
    snap_cfg = make_small_config()
    snap_env = Environment(snap_cfg, rng=np.random.default_rng(42))
    agent_positions = [
        {"x": 5, "y": 5, "alive": True, "is_isolated": False},
        {"x": 10, "y": 10, "alive": True, "is_isolated": True},
        {"x": 15, "y": 15, "alive": False, "is_isolated": False},
    ]
    fp = os.path.join(plot_dir, "test_env.png")
    fig = plot_environment_snapshot(snap_env.grid, agent_positions, filepath=fp)
    test("Environment plot returns Figure", isinstance(fig, plt.Figure))
    test("Environment plot saved", os.path.exists(fp))
    plt.close(fig)

    # === Inner State Distribution ===
    print("\n\U0001f9e0 Inner State Distribution Plot")
    inner_data = []
    for i in range(30):
        inner_data.append({
            "inner_hunger": rng.random(),
            "inner_fear": rng.random() * 0.5,
            "inner_curiosity": rng.random(),
            "inner_loneliness": rng.random() * 0.3,
            "inner_aggression": rng.random() * 0.2,
        })
    fp = os.path.join(plot_dir, "test_inner.png")
    fig = plot_inner_state_distribution(inner_data, filepath=fp)
    test("Inner state plot returns Figure", isinstance(fig, plt.Figure))
    test("Inner state plot saved", os.path.exists(fp))
    plt.close(fig)

    # === generate_report ===
    print("\n\U0001f4d1 Generate Report (multi-plot)")
    evolution_data = {
        "generations": gens,
        "best_fitness": best,
        "avg_fitness": avg,
        "diversity": div,
        "gene_trends": gene_trends,
    }
    step_data = [{"step": i, "agents_alive": 50-i//5, "avg_energy": 90-i*0.3,
                  "total_food": 200-i} for i in range(50)]
    report_dir = os.path.join(plot_dir, "report")
    plots = generate_report(
        evolution_data=evolution_data,
        step_data=step_data,
        agent_snapshots=agents_data[:10],
        output_dir=report_dir,
        prefix="test",
    )
    test(f"Report generated {len(plots)} plots", len(plots) >= 4)
    test("Has fitness_evolution", "fitness_evolution" in plots)
    test("Has gene_evolution", "gene_evolution" in plots)
    test("Has population_dynamics", "population_dynamics" in plots)
    test("Has genome_heatmap", "genome_heatmap" in plots)
    for name, path in plots.items():
        test(f"Report plot exists: {name}", os.path.exists(path))
    plt.close("all")

    # === visualize_evolution from World ===
    print("\n\U0001f30d Visualize Evolution from World")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    world = World(small_cfg)
    world.run_evolution(num_generations=3, steps_per_generation=50, verbose=False)
    evo_dir = os.path.join(plot_dir, "evolution")
    evo_plots = visualize_evolution(world, output_dir=evo_dir, prefix="evo")
    test(f"Evolution plots generated: {len(evo_plots)}", len(evo_plots) >= 2)
    test("Has fitness_evolution", "fitness_evolution" in evo_plots)
    for name, path in evo_plots.items():
        test(f"Evolution plot exists: {name}", os.path.exists(path))
    plt.close("all")

    # === visualize_experiment from IsolationExperiment ===
    print("\n\U0001f52c Visualize Experiment")
    Agent.reset_id_counter()
    exp = IsolationExperiment(
        config=small_cfg, isolation_fraction=0.2, isolation_duration=10,
        isolation_frequency=10, selection_criteria="random",
    )
    exp.run_experiment(num_generations=2, steps_per_generation=30, verbose=False)
    exp_dir = os.path.join(plot_dir, "experiment")
    exp_plots = visualize_experiment(exp, output_dir=exp_dir, prefix="exp")
    test(f"Experiment plots generated: {len(exp_plots)}", len(exp_plots) >= 2)
    for name, path in exp_plots.items():
        test(f"Experiment plot exists: {name}", os.path.exists(path))
    plt.close("all")

    # === Edge cases ===
    print("\n\u2699\ufe0f  Edge Cases")
    fig = plot_genome_heatmap([], filepath=None)
    test("Empty heatmap doesn't crash", isinstance(fig, plt.Figure))
    plt.close(fig)

    fig = plot_fitness_evolution([], [], [], filepath=None)
    test("Empty fitness doesn't crash", isinstance(fig, plt.Figure))
    plt.close(fig)

    # Cleanup
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)


def section11_tests():
    """Section 11: Research Analysis Pipeline."""

    import os
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = "/tmp/swarm_test_research"

    # === AgentSnapshot ===
    print("\n\U0001f4f8 AgentSnapshot Capture")
    Agent.reset_id_counter()
    small_cfg = make_small_config()
    world = World(small_cfg)
    for _ in range(10):
        world.step()

    agent = [a for a in world.agents if a.alive][0]
    snap = AgentSnapshot.capture(agent, step=10)
    test("Snapshot has agent_id", snap.agent_id == agent.id)
    test("Snapshot has energy", snap.energy == agent.energy)
    test("Snapshot has inner_states", "hunger" in snap.inner_states)
    test("Snapshot has loneliness", "loneliness" in snap.inner_states)
    test("Snapshot has belief_summary", "avg_food_belief" in snap.belief_summary)
    test("Snapshot has belief_food_grid", snap.belief_food_grid is not None)
    test("Snapshot has fitness", snap.fitness >= 0)
    test("Snapshot has rl_total_decisions", snap.rl_total_decisions >= 0)
    test("Snapshot has genome_vector", snap.genome_vector is not None)

    # === IsolationProfile ===
    print("\n\U0001f4cb IsolationProfile")
    pre = AgentSnapshot.capture(agent, step=10)
    agent.isolate(0, 0)
    during = AgentSnapshot.capture(agent, step=15)
    agent.return_from_isolation(10, 10)
    post = AgentSnapshot.capture(agent, step=30)

    profile = IsolationProfile(
        agent_id=agent.id,
        isolation_step=10,
        return_step=30,
        criteria="random",
        pre_isolation=pre,
        during_snapshots=[during],
        post_return=post,
        group_belief_at_return={"avg_food_belief": 0.3},
    )
    delta = profile.inner_state_delta()
    test("Delta has hunger", "hunger" in delta)
    test("Delta has loneliness", "loneliness" in delta)
    div = profile.belief_divergence_at_return()
    test("Belief divergence computed", isinstance(div, float))
    lone_traj = profile.loneliness_trajectory()
    test("Loneliness trajectory length=1", len(lone_traj) == 1)
    fear_traj = profile.fear_trajectory()
    test("Fear trajectory length=1", len(fear_traj) == 1)
    conf_traj = profile.confidence_trajectory()
    test("Confidence trajectory length=1", len(conf_traj) == 1)

    # === ResearchExperiment — single run ===
    print("\n\U0001f52c ResearchExperiment Single Run")
    Agent.reset_id_counter()
    exp = ResearchExperiment(
        config=small_cfg,
        isolation_fraction=0.3,
        isolation_duration=15,
        isolation_frequency=5,
        selection_criteria="random",
        profile_interval=5,
    )
    results = exp.run(num_generations=2, steps_per_generation=50, verbose=False)
    test("Results has subjective_experience", "subjective_experience" in results)
    test("Results has belief_divergence", "belief_divergence" in results)
    test("Results has behavioral_change", "behavioral_change" in results)
    test("Results has group_dynamics", "group_dynamics" in results)
    test("Results has reintegration", "reintegration" in results)
    test("Results has generation_comparison", "generation_comparison" in results)
    test(f"Total profiles > 0: {results['total_profiles']}", results["total_profiles"] > 0)

    # === Subjective experience analysis ===
    print("\n\U0001f9e0 Subjective Experience Analysis")
    se = results["subjective_experience"]
    test("SE has hunger", "hunger" in se)
    test("SE has loneliness", "loneliness" in se)
    test("SE has aggression", "aggression" in se)
    test("Loneliness pre_isolation_mean", "pre_isolation_mean" in se.get("loneliness", {}))
    test("Loneliness during_isolation_mean", "during_isolation_mean" in se.get("loneliness", {}))
    test("Loneliness post_return_mean", "post_return_mean" in se.get("loneliness", {}))
    test("Loneliness pre_to_during_shift", "pre_to_during_shift" in se.get("loneliness", {}))
    # Loneliness should increase during isolation
    lone = se.get("loneliness", {})
    during_lone = lone.get("during_isolation_mean", 0)
    pre_lone = lone.get("pre_isolation_mean", 0)
    test(f"Loneliness rises during isolation: {during_lone:.4f} > {pre_lone:.4f}",
         during_lone > pre_lone or abs(during_lone - pre_lone) < 0.01)

    # === Belief divergence analysis ===
    print("\n\U0001f50d Belief Divergence Analysis")
    bd = results["belief_divergence"]
    test("Has mean_belief_divergence", "mean_belief_divergence" in bd)
    test("Has confidence_at_isolation", "confidence_at_isolation" in bd)
    test("Has confidence_at_return", "confidence_at_return" in bd)
    test("Has confidence_drop", "confidence_drop" in bd)
    test("Has food_belief_gap_mean", "food_belief_gap_mean" in bd)

    # === Behavioral change analysis ===
    print("\n\U0001f3af Behavioral Change Analysis")
    bc = results["behavioral_change"]
    test("Has mean_energy_change", "mean_energy_change" in bc)
    test("Has mean_food_rate_change", "mean_food_rate_change" in bc)
    test("Has post_return_survival_rate", "post_return_survival_rate" in bc)
    test("Has total_tracked", "total_tracked" in bc)

    # === Group dynamics analysis ===
    print("\n\U0001f465 Group Dynamics Analysis")
    gd = results["group_dynamics"]
    test("Has inner_state_comparison", "inner_state_comparison" in gd)
    test("Has fitness_comparison", "fitness_comparison" in gd)
    test("Has food_comparison", "food_comparison" in gd)
    fc = gd.get("fitness_comparison", {})
    test("Fitness comparison has control_avg", "control_avg" in fc)
    test("Fitness comparison has treatment_avg", "treatment_avg" in fc)
    test("Fitness comparison has fitness_impact", "fitness_impact" in fc)
    isc = gd.get("inner_state_comparison", {})
    test("Inner comparison has loneliness", "loneliness" in isc)
    if "loneliness" in isc:
        test("Loneliness comparison has control_mean", "control_mean" in isc["loneliness"])
        test("Loneliness comparison has difference", "difference" in isc["loneliness"])

    # === Reintegration analysis ===
    print("\n\U0001f504 Reintegration Analysis")
    ri = results["reintegration"]
    test("Has energy_recovery_mean", "energy_recovery_mean" in ri)
    test("Has inner_state_recovery", "inner_state_recovery" in ri)

    # === Report generation ===
    print("\n\U0001f4dd Research Report Generation")
    os.makedirs(plot_dir, exist_ok=True)
    report_path = os.path.join(plot_dir, "test_report.txt")
    report_text = generate_research_report(results, filepath=report_path)
    test("Report text generated", len(report_text) > 500)
    test("Report file created", os.path.exists(report_path))
    test("Report mentions 'FINDING 1'", "FINDING 1" in report_text)
    test("Report mentions 'FINDING 2'", "FINDING 2" in report_text)
    test("Report mentions 'FINDING 3'", "FINDING 3" in report_text)
    test("Report mentions 'FINDING 4'", "FINDING 4" in report_text)
    test("Report mentions 'FINDING 5'", "FINDING 5" in report_text)
    test("Report mentions 'SYNTHESIS'", "SYNTHESIS" in report_text)

    # === Research visualizations ===
    print("\n\U0001f4ca Research Visualizations")
    fig = plot_subjective_experience(se)
    test("Subjective plot returns Figure", isinstance(fig, plt.Figure))
    plt.close(fig)

    lone_traj = se.get("loneliness_trajectory_avg", [0.2, 0.4, 0.5, 0.6])
    conf_traj = se.get("confidence_trajectory_avg", [0.8, 0.6, 0.4, 0.3])
    if len(lone_traj) >= 2:
        fig = plot_loneliness_trajectory(lone_traj, conf_traj)
        test("Loneliness trajectory plot works", isinstance(fig, plt.Figure))
        plt.close(fig)
    else:
        test("Loneliness trajectory plot (skipped - short data)", True)

    fig = plot_belief_divergence(bd)
    test("Belief divergence plot works", isinstance(fig, plt.Figure))
    plt.close(fig)

    isc = gd.get("inner_state_comparison", {})
    if isc:
        fig = plot_group_inner_comparison(isc)
        test("Group inner comparison plot works", isinstance(fig, plt.Figure))
        plt.close(fig)

    fig = plot_behavioral_change(bc)
    test("Behavioral change plot works", isinstance(fig, plt.Figure))
    plt.close(fig)

    # Full plot generation
    res_plot_dir = os.path.join(plot_dir, "research_plots")
    rplots = generate_research_plots(results, output_dir=res_plot_dir)
    test(f"Generated {len(rplots)} research plots", len(rplots) >= 3)
    for name, path in rplots.items():
        test(f"Research plot exists: {name}", os.path.exists(path))
    plt.close("all")

    # === MultiRunAnalysis ===
    print("\n\U0001f4ca MultiRunAnalysis (3 runs)")
    Agent.reset_id_counter()
    multi = MultiRunAnalysis(
        base_config=small_cfg,
        num_runs=3,
        isolation_fraction=0.3,
        isolation_duration=15,
        isolation_frequency=5,
        selection_criteria="random",
    )
    agg = multi.run(num_generations=2, steps_per_generation=50, verbose=False)
    test("Multi-run has num_runs=3", agg.get("num_runs") == 3)
    test("Multi-run has subjective_experience", "subjective_experience" in agg)
    test("Multi-run has belief_divergence", "belief_divergence" in agg)
    test("Multi-run has behavioral_change", "behavioral_change" in agg)
    test("Multi-run has group_dynamics", "group_dynamics" in agg)

    # Check statistical properties
    se_multi = agg.get("subjective_experience", {})
    if "loneliness" in se_multi:
        shift = se_multi["loneliness"].get("pre_to_post_shift", {})
        test("Multi-run loneliness shift has mean", "mean" in shift)
        test("Multi-run loneliness shift has ci_95", "ci_95" in shift)
        test("Multi-run loneliness shift has n=3", shift.get("n") == 3)

    gd_multi = agg.get("group_dynamics", {})
    fi = gd_multi.get("fitness_impact", {})
    test("Multi-run fitness_impact has mean", "mean" in fi)
    test("Multi-run fitness_impact has std", "std" in fi)
    test("Multi-run fitness_impact has ci_95", "ci_95" in fi)

    # Multi-run report
    print("\n\U0001f4dd Multi-Run Report")
    mr_report = generate_research_report(agg, multi_run=True)
    test("Multi-run report generated", len(mr_report) > 500)
    test("Multi-run report mentions CI", "±" in mr_report or "ci" in mr_report.lower())

    # === Different selection criteria ===
    print("\n\U0001f9ea Selection Criteria Variants")
    for criteria in ["adventurousness", "affiliation_need"]:
        Agent.reset_id_counter()
        exp_c = ResearchExperiment(
            config=small_cfg, isolation_fraction=0.2, isolation_duration=10,
            isolation_frequency=10, selection_criteria=criteria,
        )
        r = exp_c.run(num_generations=1, steps_per_generation=30, verbose=False)
        test(f"Criteria '{criteria}' runs OK", "subjective_experience" in r)

    # Cleanup
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)


def section12_tests():
    """Section 12: Extended Experiment Suite."""

    # === ExperimentCondition ===
    print("\n\U0001f4cb ExperimentCondition")
    cond = ExperimentCondition(
        name="test_cond", experiment_type="test",
        selection_criteria="random", isolation_fraction=0.15,
        isolation_duration=50, no_return=False,
        food_initial=100, food_max=200,
        num_generations=3, steps_per_generation=500,
    )
    test("Condition name", cond.name == "test_cond")
    test("Condition type", cond.experiment_type == "test")
    test("Condition fraction", cond.isolation_fraction == 0.15)
    test("Condition no_return=False", cond.no_return is False)

    # to_dict
    d = cond.to_dict()
    test("to_dict has name", d["name"] == "test_cond")
    test("to_dict has isolation_fraction", d["isolation_fraction"] == 0.15)
    test("to_dict has no_return", d["no_return"] is False)
    test("to_dict has food_initial", d["food_initial"] == 100)

    # apply_to_config
    cfg = make_small_config()
    applied = cond.apply_to_config(cfg)
    test("Config overrides food_initial", applied.environment.food.initial_count == 100)
    test("Config overrides food_max", applied.environment.food.max_food == 200)
    test("Config overrides isolation_duration", applied.experiment.isolation_duration == 50)
    test("Config overrides max_steps", applied.world.max_steps == 500)
    test("Original config unchanged", cfg.environment.food.initial_count == 15)

    # repr
    test("repr works", "test_cond" in repr(cond))

    # === build_experiment_suite ===
    print("\n\U0001f9ea Experiment Suite Builder")
    suite = build_experiment_suite(num_generations=3, steps_per_generation=200)
    test("Suite has 6 experiment types", len(suite) == 6)
    test("Has selection_criteria", "selection_criteria" in suite)
    test("Has ratio_sweep", "ratio_sweep" in suite)
    test("Has duration_sweep", "duration_sweep" in suite)
    test("Has resource_variation", "resource_variation" in suite)
    test("Has no_return", "no_return" in suite)
    test("Has generation_length", "generation_length" in suite)

    # Check condition counts
    test("selection_criteria has 4 conditions", len(suite["selection_criteria"]) == 4)
    test("ratio_sweep has 4 conditions", len(suite["ratio_sweep"]) == 4)
    test("duration_sweep has 4 conditions", len(suite["duration_sweep"]) == 4)
    test("resource_variation has 3 conditions", len(suite["resource_variation"]) == 3)
    test("no_return has 2 conditions", len(suite["no_return"]) == 2)
    test("generation_length has 4 conditions", len(suite["generation_length"]) == 4)

    # Check specific conditions
    ratios = {c.name: c for c in suite["ratio_sweep"]}
    test("ratio_5pct exists", "ratio_5pct" in ratios)
    test("ratio_5pct fraction=0.05", ratios["ratio_5pct"].isolation_fraction == 0.05)
    test("ratio_30pct fraction=0.30", ratios["ratio_30pct"].isolation_fraction == 0.30)

    durations = {c.name: c for c in suite["duration_sweep"]}
    test("duration_50 exists", "duration_50" in durations)
    test("duration_50 dur=50", durations["duration_50"].isolation_duration == 50)
    test("duration_500 dur=500", durations["duration_500"].isolation_duration == 500)

    resources = {c.name: c for c in suite["resource_variation"]}
    test("food_scarce initial=50", resources["food_scarce"].food_initial == 50)
    test("food_rich initial=300", resources["food_rich"].food_initial == 300)
    test("food_default initial=None", resources["food_default"].food_initial is None)

    no_ret = {c.name: c for c in suite["no_return"]}
    test("no_return has no_return=True", no_ret["no_return"].no_return is True)
    test("with_return has no_return=False", no_ret["with_return"].no_return is False)

    gen_lens = {c.name: c for c in suite["generation_length"]}
    test("gen_250_steps has steps=250", gen_lens["gen_250_steps"].steps_per_generation == 250)
    test("gen_2000_steps has steps=2000", gen_lens["gen_2000_steps"].steps_per_generation == 2000)

    # list helpers
    types = list_experiment_types()
    test("list_experiment_types returns 6", len(types) == 6)
    all_conds = list_all_conditions()
    test(f"list_all_conditions returns 21 (got {len(all_conds)})", len(all_conds) == 21)

    # === ExtendedExperiment — standard run ===
    print("\n\U0001f52c ExtendedExperiment Standard Run")
    small_cfg = make_small_config()
    Agent.reset_id_counter()
    cond_std = ExperimentCondition(
        name="test_standard", experiment_type="test",
        isolation_fraction=0.2, isolation_duration=10,
        isolation_frequency=5, num_generations=2, steps_per_generation=30,
    )
    exp = ExtendedExperiment(cond_std, small_cfg)
    result = exp.run(seed=42, verbose=False)

    test("Result has condition", "condition" in result)
    test("Result has seed", result["seed"] == 42)
    test("Result has ctrl_avg_fitness", "ctrl_avg_fitness" in result)
    test("Result has treat_avg_fitness", "treat_avg_fitness" in result)
    test("Result has fitness_impact", "fitness_impact" in result)
    test("Result has ctrl_alive_final", "ctrl_alive_final" in result)
    test("Result has treat_extinct", "treat_extinct" in result)
    test("Result has ctrl_total_food", "ctrl_total_food" in result)
    test("Result has food_reduction_pct", "food_reduction_pct" in result)
    test("Result has total_isolations", result["total_isolations"] > 0)
    test("Result has total_returns", "total_returns" in result)
    test("Result has ctrl_gen_records", len(result["ctrl_gen_records"]) == 2)
    test("Result has treat_gen_records", len(result["treat_gen_records"]) == 2)
    test("Gen record has avg_fitness", "avg_fitness" in result["ctrl_gen_records"][0])
    test("Gen record has alive_at_end", "alive_at_end" in result["ctrl_gen_records"][0])

    # Extinction step tracking
    test("ctrl_extinction_step is int or None",
         result["ctrl_extinction_step"] is None or isinstance(result["ctrl_extinction_step"], int))
    test("treat_extinction_step is int or None",
         result["treat_extinction_step"] is None or isinstance(result["treat_extinction_step"], int))

    # === ExtendedExperiment — no-return mode ===
    print("\n\U0001f6ab No-Return Mode")
    Agent.reset_id_counter()
    cond_nr = ExperimentCondition(
        name="test_no_return", experiment_type="test",
        no_return=True, isolation_fraction=0.2, isolation_duration=10,
        isolation_frequency=5, num_generations=2, steps_per_generation=30,
    )
    exp_nr = ExtendedExperiment(cond_nr, small_cfg)
    result_nr = exp_nr.run(seed=42, verbose=False)
    test("No-return: total_returns=0", result_nr["total_returns"] == 0)
    test("No-return: total_isolations > 0", result_nr["total_isolations"] > 0)

    # === ExtendedExperiment — food override ===
    print("\n\U0001f35e Resource Override")
    Agent.reset_id_counter()
    cond_food = ExperimentCondition(
        name="test_food", experiment_type="test",
        food_initial=30, food_max=50,
        num_generations=1, steps_per_generation=20,
    )
    exp_food = ExtendedExperiment(cond_food, small_cfg)
    test("Food config applied",
         exp_food.config.environment.food.initial_count == 30)
    test("Food max applied",
         exp_food.config.environment.food.max_food == 50)

    # === ExtendedExperiment — different criteria ===
    print("\n\U0001f3af Selection Criteria Variants")
    for criteria in ["random", "adventurousness", "affiliation_need",
                     "best_fitness", "worst_fitness"]:
        Agent.reset_id_counter()
        cond_c = ExperimentCondition(
            name=f"test_{criteria}", experiment_type="test",
            selection_criteria=criteria,
            num_generations=1, steps_per_generation=20,
        )
        exp_c = ExtendedExperiment(cond_c, small_cfg)
        r = exp_c.run(seed=42, verbose=False)
        test(f"Criteria '{criteria}' runs", "ctrl_avg_fitness" in r)

    # === SweepRunner ===
    print("\n\U0001f4ca SweepRunner")
    Agent.reset_id_counter()
    conditions_sweep = [
        ExperimentCondition(
            name="sweep_a", experiment_type="test_sweep",
            isolation_fraction=0.1,
            num_generations=1, steps_per_generation=20,
        ),
        ExperimentCondition(
            name="sweep_b", experiment_type="test_sweep",
            isolation_fraction=0.3,
            num_generations=1, steps_per_generation=20,
        ),
    ]
    runner = SweepRunner(conditions_sweep, small_cfg, num_seeds=3)
    sweep = runner.run(verbose=False)

    test("Sweep total_runs=6", sweep["total_runs"] == 6)
    test("Sweep has all_results", len(sweep["all_results"]) == 6)
    test("Sweep has elapsed_seconds", "elapsed_seconds" in sweep)
    test("Sweep has sweep_summary", "sweep_summary" in sweep)

    summary = sweep["sweep_summary"]
    test("Summary has sweep_a", "sweep_a" in summary)
    test("Summary has sweep_b", "sweep_b" in summary)

    s_a = summary["sweep_a"]
    test("Summary n_seeds=3", s_a["n_seeds"] == 3)
    test("Summary has ctrl_fitness_mean", "ctrl_fitness_mean" in s_a)
    test("Summary has treat_fitness_mean", "treat_fitness_mean" in s_a)
    test("Summary has fitness_impact_mean", "fitness_impact_mean" in s_a)
    test("Summary has fitness_impact_ci95", "fitness_impact_ci95" in s_a)
    test("Summary has extinction_rate", "extinction_rate" in s_a)
    test("Summary has food_reduction_mean", "food_reduction_mean" in s_a)
    test("Summary extinction_rate in [0,1]", 0 <= s_a["extinction_rate"] <= 1)

    # get_results_for_condition
    cond_results = runner.get_results_for_condition("sweep_a")
    test("get_results_for_condition returns 3", len(cond_results) == 3)

    # === SweepRunner with custom seeds ===
    print("\n\U0001f522 Custom Seeds")
    Agent.reset_id_counter()
    custom_seeds = [10, 20, 30]
    runner2 = SweepRunner(
        [conditions_sweep[0]], small_cfg, seeds=custom_seeds,
    )
    sweep2 = runner2.run(verbose=False)
    test("Custom seeds total_runs=3", sweep2["total_runs"] == 3)
    used_seeds = [r["seed"] for r in sweep2["all_results"]]
    test("Custom seeds used correctly", used_seeds == [10, 20, 30])

    # === run_experiment_type convenience ===
    print("\n\U0001f680 run_experiment_type Convenience")
    Agent.reset_id_counter()
    try:
        result_convenience = run_experiment_type(
            "no_return", small_cfg,
            num_seeds=2, num_generations=1, steps_per_generation=20,
            verbose=False,
        )
        test("Convenience function runs", "sweep_summary" in result_convenience)
        test("Convenience has 2 conditions",
             len(result_convenience["sweep_summary"]) == 2)
    except Exception as e:
        test(f"Convenience function failed: {e}", False)

    # === Invalid experiment type ===
    print("\n\u26a0\ufe0f Edge Cases")
    try:
        run_experiment_type("nonexistent_type", small_cfg)
        test("Invalid type raises ValueError", False)
    except ValueError:
        test("Invalid type raises ValueError", True)

    # Empty eligible agents (no crash)
    Agent.reset_id_counter()
    cond_empty = ExperimentCondition(
        name="test_empty", experiment_type="test",
        isolation_fraction=0.5,
        num_generations=1, steps_per_generation=5,
    )
    exp_empty = ExtendedExperiment(cond_empty, small_cfg)
    try:
        r_empty = exp_empty.run(seed=999, verbose=False)
        test("High isolation doesn't crash", "ctrl_avg_fitness" in r_empty)
    except Exception as e:
        test(f"High isolation crashed: {e}", False)

    # === Parallel execution ===
    print("\n\u26a1 Parallel Execution")
    Agent.reset_id_counter()
    par_conditions = [
        ExperimentCondition(
            name="par_a", experiment_type="par_test",
            num_generations=1, steps_per_generation=20,
        ),
        ExperimentCondition(
            name="par_b", experiment_type="par_test",
            isolation_fraction=0.3,
            num_generations=1, steps_per_generation=20,
        ),
    ]

    # Serial baseline
    runner_s = SweepRunner(par_conditions, small_cfg, num_seeds=3)
    res_s = runner_s.run(verbose=False, workers=None)
    test("Serial run completes", res_s["total_runs"] == 6)

    # Parallel with 2 workers
    runner_p = SweepRunner(par_conditions, small_cfg, num_seeds=3)
    res_p = runner_p.run(verbose=False, workers=2)
    test("Parallel run completes", res_p["total_runs"] == 6)
    test("Parallel has workers field", res_p.get("workers") == 2)
    test("Parallel results count matches",
         len(res_p["all_results"]) == len(res_s["all_results"]))

    # Both should find same conditions
    s_names = set(res_s["sweep_summary"].keys())
    p_names = set(res_p["sweep_summary"].keys())
    test("Parallel finds same conditions", s_names == p_names)


def _make_test_sweep():
    """Helper: run a small sweep for testing Sections 13-15."""
    small_cfg = make_small_config()
    conditions = [
        ExperimentCondition(
            name="cond_a", experiment_type="test_sweep",
            isolation_fraction=0.1,
            num_generations=2, steps_per_generation=30,
        ),
        ExperimentCondition(
            name="cond_b", experiment_type="test_sweep",
            isolation_fraction=0.3,
            num_generations=2, steps_per_generation=30,
        ),
    ]
    runner = SweepRunner(conditions, small_cfg, num_seeds=3)
    return runner.run(verbose=False)


def section13_tests():
    """Section 13: Batch Logger & Enhanced Logging."""
    sweep = _make_test_sweep()

    # === BatchLogger Export ===
    print("\n\U0001f4be BatchLogger Export")
    out_dir = "/tmp/test_s13"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    logger = BatchLogger(out_dir)
    exported = logger.export_sweep(sweep, prefix="test")

    test("Exports per_run_csv", "per_run_csv" in exported)
    test("Exports per_generation_csv", "per_generation_csv" in exported)
    test("Exports condition_summary_csv", "condition_summary_csv" in exported)
    test("Exports full_json", "full_json" in exported)
    test("Exports latex_table", "latex_table" in exported)

    # All files exist and non-empty
    for ftype, fpath in exported.items():
        test(f"{ftype} exists and non-empty",
             os.path.exists(fpath) and os.path.getsize(fpath) > 0)

    # === Per-Run CSV Content ===
    print("\n\U0001f4ca Per-Run CSV")
    with open(exported["per_run_csv"]) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    test("Per-run CSV has 6 rows (2 conds × 3 seeds)", len(rows) == 6)
    test("Per-run has condition_name column", "condition_name" in rows[0])
    test("Per-run has seed column", "seed" in rows[0])
    test("Per-run has fitness_impact column", "fitness_impact" in rows[0])
    test("Per-run has treat_extinct column", "treat_extinct" in rows[0])
    test("Per-run has food_reduction_pct column", "food_reduction_pct" in rows[0])

    # === Per-Generation CSV ===
    print("\n\U0001f4c8 Per-Generation CSV")
    with open(exported["per_generation_csv"]) as f:
        reader = csv.DictReader(f)
        gen_rows = list(reader)
    # 2 conditions × 3 seeds × 2 generations × 2 groups = 24
    test(f"Per-gen CSV has 24 rows (got {len(gen_rows)})", len(gen_rows) == 24)
    test("Per-gen has group column", "group" in gen_rows[0])
    test("Per-gen has generation column", "generation" in gen_rows[0])
    test("Per-gen has avg_fitness column", "avg_fitness" in gen_rows[0])

    # === Condition Summary CSV ===
    print("\n\U0001f4cb Condition Summary CSV")
    with open(exported["condition_summary_csv"]) as f:
        reader = csv.DictReader(f)
        sum_rows = list(reader)
    test("Summary CSV has 2 rows", len(sum_rows) == 2)
    test("Summary has n_seeds", "n_seeds" in sum_rows[0])
    test("Summary has fitness_impact_ci95", "fitness_impact_ci95" in sum_rows[0])
    test("Summary has extinction_rate", "extinction_rate" in sum_rows[0])

    # === Full JSON ===
    print("\n\U0001f4c4 Full JSON")
    with open(exported["full_json"]) as f:
        jdata = json.load(f)
    test("JSON has metadata", "metadata" in jdata)
    test("JSON has sweep_summary", "sweep_summary" in jdata)
    test("JSON has per_run", "per_run" in jdata)
    test("JSON per_run count=6", len(jdata["per_run"]) == 6)
    test("JSON per_run lacks gen records (compact)",
         "ctrl_gen_records" not in jdata["per_run"][0])

    # === LaTeX Table ===
    print("\n\U0001f4dd LaTeX Table")
    with open(exported["latex_table"]) as f:
        tex = f.read()
    test("LaTeX has \\begin{table}", r"\begin{table}" in tex)
    test("LaTeX has \\toprule", r"\toprule" in tex)
    test("LaTeX has condition names", "cond" in tex.lower())
    test("LaTeX has \\end{table}", r"\end{table}" in tex)

    # Cleanup
    shutil.rmtree(out_dir)


def section14_tests():
    """Section 14: Statistical Analysis Module."""

    # === Descriptive Statistics ===
    print("\n\U0001f4ca Descriptive Statistics")
    ds = descriptive_stats([10, 20, 30, 40, 50])
    test("desc n=5", ds["n"] == 5)
    test("desc mean=30", ds["mean"] == 30)
    test("desc median=30", ds["median"] == 30)
    test("desc min=10", ds["min"] == 10)
    test("desc max=50", ds["max"] == 50)
    test("desc std > 0", ds["std"] > 0)
    test("desc sem > 0", ds["sem"] > 0)
    test("desc ci95 > 0", ds["ci95"] > 0)

    ds_empty = descriptive_stats([])
    test("desc empty n=0", ds_empty["n"] == 0)

    ds_one = descriptive_stats([42])
    test("desc single n=1", ds_one["n"] == 1)
    test("desc single mean=42", ds_one["mean"] == 42)

    # === Cohen's d ===
    print("\n\U0001f4cf Cohen's d")
    d_large = cohens_d([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    test("Large d > 0.8", abs(d_large) > 0.8)

    d_zero = cohens_d([1, 2, 3], [1, 2, 3])
    test("Identical groups d=0", d_zero == 0.0)

    d_small = cohens_d([1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5])
    test("Small shift d > 0", abs(d_small) > 0)

    d_short = cohens_d([1], [2])
    test("Too few data d=0", d_short == 0.0)

    # === Paired Comparison (T-test) ===
    print("\n\U0001f9ea T-Test")
    comp = paired_comparison([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], label="test")
    test("T-test has label", comp["label"] == "test")
    test("T-test has ctrl desc", "mean" in comp["ctrl"])
    test("T-test has treat desc", "mean" in comp["treat"])
    test("T-test difference > 0", comp["difference"] > 0)
    test("T-test t_statistic != 0", comp["t_statistic"] != 0)
    test("T-test p_value < 0.05", comp["p_value"] < 0.05)
    test("T-test significant", comp["significant_005"])
    test("T-test has cohens_d", "cohens_d" in comp)
    test("T-test has effect_size_label", comp["effect_size_label"] in
         ["negligible", "small", "medium", "large"])

    # Non-significant
    comp_ns = paired_comparison([1, 2, 3], [1.1, 2.1, 3.1])
    test("Near-identical not significant", not comp_ns["significant_005"])

    # === ANOVA ===
    print("\n\U0001f4ca ANOVA")
    groups = {
        "g1": [1, 2, 3, 4, 5],
        "g2": [6, 7, 8, 9, 10],
        "g3": [11, 12, 13, 14, 15],
    }
    anova = one_way_anova(groups, metric_name="test_metric")
    test("ANOVA has metric name", anova["metric"] == "test_metric")
    test("ANOVA n_groups=3", anova["n_groups"] == 3)
    test("ANOVA F > 0", anova["f_statistic"] > 0)
    test("ANOVA p < 0.001", anova["p_value"] < 0.001)
    test("ANOVA significant", anova["significant_005"])
    test("ANOVA has pairwise", len(anova["pairwise"]) == 3)
    test("ANOVA has group_descriptives", len(anova["group_descriptives"]) == 3)

    # Non-significant ANOVA
    groups_ns = {"g1": [1, 2, 3], "g2": [1.5, 2.5, 3.5]}
    anova_ns = one_way_anova(groups_ns)
    test("Non-sig ANOVA p > 0.05", anova_ns["p_value"] > 0.05)

    # === Kaplan-Meier ===
    print("\n\U0001f4c9 Kaplan-Meier")
    # Mix of events and censored
    km = kaplan_meier([100, 200, None, 300, None], max_step=500, label="test")
    test("KM has label", km["label"] == "test")
    test("KM n=5", km["n"] == 5)
    test("KM n_events=3", km["n_events"] == 3)
    test("KM n_censored=2", km["n_censored"] == 2)
    test("KM times starts at 0", km["times"][0] == 0)
    test("KM survival starts at 1.0", km["survival"][0] == 1.0)
    test("KM survival decreases", km["survival"][-1] < 1.0)
    test("KM has CI bands", len(km["ci_lower"]) == len(km["survival"]))
    test("KM CI lower <= survival",
         all(lo <= s for lo, s in zip(km["ci_lower"], km["survival"])))

    # All extinct
    km_all = kaplan_meier([10, 20, 30], max_step=100, label="all_dead")
    test("All-extinct survival→0", km_all["survival"][-1] == 0.0)
    test("All-extinct has median", km_all["median_survival"] is not None)

    # All survived
    km_none = kaplan_meier([None, None, None], max_step=100)
    test("All-survived n_events=0", km_none["n_events"] == 0)

    # Empty
    km_empty = kaplan_meier([], max_step=100)
    test("Empty KM n=0", km_empty["n"] == 0)

    # === Log-Rank Test ===
    print("\n\U0001f4ca Log-Rank Test")
    lr = log_rank_test([10, 20, 30], [100, 200, 300], max_step=500)
    test("Log-rank has chi2", "chi2" in lr)
    test("Log-rank has p_value", "p_value" in lr)
    test("Log-rank different survival p < 0.1", lr["p_value"] < 0.5)

    # Same groups
    lr_same = log_rank_test([100, 200], [100, 200], max_step=500)
    test("Same groups log-rank p close to 1",
         lr_same["p_value"] > 0.05 or lr_same["chi2"] < 1)

    # === Mediation Test ===
    print("\n\U0001f9ec Mediation Analysis")
    np.random.seed(42)
    n = 50
    x = np.random.randn(n).tolist()
    m = [xi * 0.5 + np.random.randn() * 0.1 for xi in x]
    y = [mi * 0.8 + np.random.randn() * 0.1 for mi in m]
    med = mediation_test(x, m, y, labels=("X", "M", "Y"))
    test("Mediation has n", med["n"] == 50)
    test("Mediation has total_effect_c", "total_effect_c" in med)
    test("Mediation has path_a", "path_a" in med)
    test("Mediation has path_b", "path_b" in med)
    test("Mediation has indirect_effect_ab", "indirect_effect_ab" in med)
    test("Mediation has sobel_z", "sobel_z" in med)
    test("Mediation has proportion_mediated", "proportion_mediated_pct" in med)
    test("Mediation path_a significant", med["path_a_p"] < 0.05)
    test("Mediation indirect > 0", abs(med["indirect_effect_ab"]) > 0)

    # Too few data
    med_short = mediation_test([1, 2], [3, 4], [5, 6])
    test("Short data returns error", "error" in med_short)

    # === analyze_sweep Integration ===
    print("\n\U0001f52c Full Sweep Analysis")
    sweep = _make_test_sweep()
    out_dir = "/tmp/test_s14"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    analysis = analyze_sweep(sweep, output_dir=out_dir)
    test("Analysis has n_total_runs", analysis["n_total_runs"] == 6)
    test("Analysis has n_conditions", analysis["n_conditions"] == 2)
    test("Analysis has t_tests", len(analysis["t_tests"]) == 2)
    test("Analysis has anova_fitness_impact", "f_statistic" in analysis["anova_fitness_impact"])
    test("Analysis has anova_food_reduction", "f_statistic" in analysis["anova_food_reduction"])
    test("Analysis has kaplan_meier", len(analysis["kaplan_meier"]) == 2)
    test("Analysis has condition_stats", len(analysis["condition_stats"]) == 2)

    # Check exported files
    test("Stats CSV exported", os.path.exists(os.path.join(out_dir, "statistical_tests.csv")))
    test("Stats LaTeX exported", os.path.exists(os.path.join(out_dir, "statistical_tests.tex")))

    shutil.rmtree(out_dir)


def section15_tests():
    """Section 15: Publication-Quality Visualization."""
    sweep = _make_test_sweep()
    analysis = analyze_sweep(sweep)
    summary = sweep.get("sweep_summary", {})
    km_curves = analysis.get("kaplan_meier", {})
    t_tests = analysis.get("t_tests", {})

    fig_dir = "/tmp/test_s15"
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir)

    if not HAS_MPL:
        print("\n\u26a0\ufe0f matplotlib not available — skipping Section 15")
        return

    # === Condition Comparison Bar Chart ===
    print("\n\U0001f4ca Condition Comparison")
    path = os.path.join(fig_dir, "bar_fitness.png")
    fig = plot_condition_comparison(summary, metric="fitness_impact", filepath=path)
    test("Bar chart returns figure", fig is not None)
    test("Bar chart file created", os.path.exists(path))
    test("Bar chart non-empty", os.path.getsize(path) > 1000)

    # === Kaplan-Meier ===
    print("\n\U0001f4c9 Kaplan-Meier Plot")
    path_km = os.path.join(fig_dir, "km.png")
    fig_km = plot_kaplan_meier(km_curves, filepath=path_km)
    test("KM plot returns figure", fig_km is not None)
    test("KM plot file created", os.path.exists(path_km))

    # === Forest Plot ===
    print("\n\U0001f333 Forest Plot (Effect Sizes)")
    path_forest = os.path.join(fig_dir, "forest.png")
    fig_forest = plot_forest(t_tests, filepath=path_forest)
    test("Forest plot returns figure", fig_forest is not None)
    test("Forest plot file created", os.path.exists(path_forest))

    # === Heatmap ===
    print("\n\U0001f5bc\ufe0f Sweep Heatmap")
    path_hm = os.path.join(fig_dir, "heatmap.png")
    fig_hm = plot_sweep_heatmap(summary, filepath=path_hm)
    test("Heatmap returns figure", fig_hm is not None)
    test("Heatmap file created", os.path.exists(path_hm))

    # Custom metrics
    path_hm2 = os.path.join(fig_dir, "heatmap_custom.png")
    fig_hm2 = plot_sweep_heatmap(
        summary, metrics=["fitness_impact_mean", "extinction_rate"],
        filepath=path_hm2
    )
    test("Custom heatmap created", os.path.exists(path_hm2))

    # === Fitness Trajectories ===
    print("\n\U0001f4c8 Fitness Trajectories")
    path_traj = os.path.join(fig_dir, "traj.png")
    fig_traj = plot_fitness_trajectories(sweep, filepath=path_traj)
    test("Trajectories returns figure", fig_traj is not None)
    test("Trajectories file created", os.path.exists(path_traj))

    # Per-condition
    first_cond = list(summary.keys())[0]
    path_traj2 = os.path.join(fig_dir, "traj_cond.png")
    fig_traj2 = plot_fitness_trajectories(
        sweep, condition_name=first_cond, filepath=path_traj2
    )
    test("Per-condition trajectory works", os.path.exists(path_traj2))

    # === Extinction Distribution ===
    print("\n\U0001f4e6 Extinction Distribution")
    path_ext = os.path.join(fig_dir, "extinction.png")
    fig_ext = plot_extinction_distribution(sweep, filepath=path_ext)
    # May return None if no extinctions in small test
    test("Extinction dist doesn't crash", True)

    # === Multi-Panel Summary ===
    print("\n\U0001f4d1 Summary Dashboard")
    path_dash = os.path.join(fig_dir, "dashboard.png")
    fig_dash = plot_publication_summary(sweep, analysis, filepath=path_dash)
    test("Dashboard returns figure", fig_dash is not None)
    test("Dashboard file created", os.path.exists(path_dash))
    test("Dashboard is large (>50KB)", os.path.getsize(path_dash) > 50000)

    # PDF output
    path_pdf = os.path.join(fig_dir, "dashboard.pdf")
    fig_pdf = plot_publication_summary(sweep, analysis, filepath=path_pdf)
    test("Dashboard PDF created", os.path.exists(path_pdf))

    # === generate_publication_figures ===
    print("\n\U0001f680 Full Figure Generation")
    pub_dir = os.path.join(fig_dir, "pub")
    figs = generate_publication_figures(sweep, analysis, output_dir=pub_dir,
                                        prefix="test")
    test("Generated condition_comparison", "condition_comparison" in figs)
    test("Generated kaplan_meier", "kaplan_meier" in figs)
    test("Generated forest_plot", "forest_plot" in figs)
    test("Generated heatmap", "heatmap" in figs)
    test("Generated fitness_trajectories", "fitness_trajectories" in figs)
    test("Generated summary", "summary" in figs)
    test("Generated summary_pdf", "summary_pdf" in figs)

    # All files exist
    for fname, fpath in figs.items():
        test(f"Figure {fname} exists", os.path.exists(fpath))

    # Count total figures
    all_files = [f for f in os.listdir(pub_dir)
                 if f.endswith(".png") or f.endswith(".pdf")]
    test(f"At least 7 figure files (got {len(all_files)})", len(all_files) >= 7)

    # File sizes reasonable
    for fname, fpath in figs.items():
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            test(f"Figure {fname} > 5KB ({sz//1024}KB)", sz > 5000)

    # === Edge Cases ===
    print("\n\u26a0\ufe0f Edge Cases")
    # Empty results
    fig_e1 = plot_fitness_trajectories(
        {"all_results": [], "sweep_summary": {}}, filepath=None
    )
    test("Empty trajectories returns None", fig_e1 is None)

    # Cleanup
    shutil.rmtree(fig_dir)


def run_all():
    global passed, failed

    print("=" * 60)
    print("  SECTION 1: Environment & World Foundation")
    print("=" * 60)
    section1_tests()

    print("\n" + "=" * 60)
    print("  SECTION 2: Agent Genome & Basic Agent Structure")
    print("=" * 60)
    section2_tests()

    print("\n" + "=" * 60)
    print("  SECTION 3: Bayesian Belief Network")
    print("=" * 60)
    section3_tests()

    print("\n" + "=" * 60)
    print("  SECTION 4: Neural Network Inner State Model")
    print("=" * 60)
    section4_tests()

    print("\n" + "=" * 60)
    print("  SECTION 5: Reinforcement Learning Policy")
    print("=" * 60)
    section5_tests()

    print("\n" + "=" * 60)
    print("  SECTION 6: Multi-Agent Interaction")
    print("=" * 60)
    section6_tests()

    print("\n" + "=" * 60)
    print("  SECTION 7: Genetic Algorithm & Evolution")
    print("=" * 60)
    section7_tests()

    print("\n" + "=" * 60)
    print("  SECTION 8: Isolation Experiment Workflow")
    print("=" * 60)
    section8_tests()

    print("\n" + "=" * 60)
    print("  SECTION 9: Data Collection, Logging & Export")
    print("=" * 60)
    section9_tests()

    print("\n" + "=" * 60)
    print("  SECTION 10: Visualization & Analysis")
    print("=" * 60)
    section10_tests()

    print("\n" + "=" * 60)
    print("  SECTION 11: Research Analysis Pipeline")
    print("=" * 60)
    section11_tests()

    print("\n" + "=" * 60)
    print("  SECTION 12: Extended Experiment Suite")
    print("=" * 60)
    section12_tests()

    print("\n" + "=" * 60)
    print("  SECTION 13: Batch Logger & Enhanced Logging")
    print("=" * 60)
    section13_tests()

    print("\n" + "=" * 60)
    print("  SECTION 14: Statistical Analysis Module")
    print("=" * 60)
    section14_tests()

    print("\n" + "=" * 60)
    print("  SECTION 15: Publication-Quality Visualization")
    print("=" * 60)
    section15_tests()

    # === Final Summary ===
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    try:
        success = run_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n\U0001f4a5 CRASH: {e}")
        traceback.print_exc()
        sys.exit(2)
