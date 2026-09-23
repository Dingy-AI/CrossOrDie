from dataclasses import dataclass

import numpy as np
import pymunk

from creatures.body import CreatureBody
from creatures.controller import NeuralController
from creatures.observation import build_observation
from creatures.phenotype import decode_genome
from evolution.genome import Genome


# ============================================================
# Evaluation configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """
    Configuration for one creature evaluation episode.
    """

    # Physics simulation frequency.
    physics_hz: int = 120

    # Neural-controller decision frequency.
    control_hz: int = 30

    # Maximum duration of one episode.
    episode_seconds: float = 10.0

    # World physics.
    gravity: float = 900.0

    # Flat training ground.
    ground_y: float = 650.0

    # Creature starting position.
    spawn_x: float = 350.0
    spawn_y: float = 350.0

    # Temporary locomotion target.
    #
    # This is mainly needed because goal distance is one of the
    # creature's observations.
    goal_x: float = 1350.0

    # If the torso falls below this point, the episode is over.
    death_y: float = 850.0

    # Ground extends far enough that normal creatures should not
    # reach either edge during an episode.
    ground_start_x: float = -2000.0
    ground_end_x: float = 5000.0

    # Flat-ground properties.
    ground_friction: float = 1.2
    ground_elasticity: float = 0.05

    # Pymunk solver iterations.
    solver_iterations: int = 20

    def __post_init__(self):

        if self.physics_hz <= 0:
            raise ValueError(
                "physics_hz must be greater than zero."
            )

        if self.control_hz <= 0:
            raise ValueError(
                "control_hz must be greater than zero."
            )

        if self.physics_hz % self.control_hz != 0:
            raise ValueError(
                "physics_hz must be evenly divisible by "
                "control_hz."
            )

        if self.episode_seconds <= 0.0:
            raise ValueError(
                "episode_seconds must be greater than zero."
            )

        if self.goal_x <= self.spawn_x:
            raise ValueError(
                "goal_x must be greater than spawn_x."
            )

        if self.death_y <= self.spawn_y:
            raise ValueError(
                "death_y must be greater than spawn_y."
            )


DEFAULT_EVALUATION_CONFIG = EvaluationConfig()


# ============================================================
# Evaluation result
# ============================================================

@dataclass(frozen=True, slots=True)
class FitnessResult:
    """
    Result of evaluating one genome.
    """

    fitness: float

    # Maximum positive distance achieved from spawn.
    max_progress: float

    # Furthest absolute X coordinate reached.
    max_x: float

    # Position when episode ended.
    final_x: float
    final_y: float

    # Number of physics steps actually simulated.
    physics_steps: int

    # Number of controller decisions made.
    control_steps: int

    # Simulated episode duration.
    simulated_seconds: float

    # Why the episode ended.
    termination_reason: str
    reached_goal: bool

    @property
    def reached_goal(self) -> bool:
        return self.max_x >= self.goal_x if hasattr(
            self,
            "goal_x",
        ) else False


# ============================================================
# World construction
# ============================================================

def _create_space(
    config: EvaluationConfig,
) -> pymunk.Space:

    space = pymunk.Space()

    space.gravity = (
        0.0,
        config.gravity,
    )

    space.iterations = (
        config.solver_iterations
    )

    ground = pymunk.Segment(
        space.static_body,
        (
            config.ground_start_x,
            config.ground_y,
        ),
        (
            config.ground_end_x,
            config.ground_y,
        ),
        8.0,
    )

    ground.friction = (
        config.ground_friction
    )

    ground.elasticity = (
        config.ground_elasticity
    )

    space.add(
        ground
    )

    return space


# ============================================================
# Validation
# ============================================================

def _creature_state_is_finite(
    creature: CreatureBody,
) -> bool:
    """
    Detect catastrophic numerical instability.
    """

    for body in creature.bodies:

        values = (
            body.position.x,
            body.position.y,
            body.velocity.x,
            body.velocity.y,
            body.angle,
            body.angular_velocity,
        )

        if not np.all(
            np.isfinite(values)
        ):
            return False

    return True


# ============================================================
# Fitness
# ============================================================

def _calculate_fitness(
    max_progress: float,
) -> float:
    """
    Initial CrossOrDie locomotion fitness.

    We intentionally keep this extremely simple:

        fitness = maximum forward distance

    No rewards for:
        - standing upright
        - moving feet
        - looking like walking
        - energy efficiency
        - velocity

    If a cursed rolling Cowie travels farther, it deserves the
    higher fitness.
    """

    return max(
        0.0,
        float(max_progress),
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_genome(
    genome: Genome,
    config: EvaluationConfig = DEFAULT_EVALUATION_CONFIG,
) -> FitnessResult:
    """
    Evaluate one complete creature genome.

    Pipeline:

        Genome
            ↓
        Phenotype
            ↓
        CreatureBody

        Genome.brain
            ↓
        NeuralController

        observation
            ↓
        controller
            ↓
        motor commands
            ↓
        physics
            ↓
        repeat

    The function performs no rendering and makes no mutations.
    """

    if genome.brain is None:
        raise ValueError(
            "Genome must contain a brain chromosome before "
            "it can be evaluated."
        )

    # --------------------------------------------------------
    # Build world
    # --------------------------------------------------------

    space = _create_space(
        config
    )

    spawn_position = (
        config.spawn_x,
        config.spawn_y,
    )

    # --------------------------------------------------------
    # Build creature
    # --------------------------------------------------------

    phenotype = decode_genome(
        genome
    )

    creature = CreatureBody(
        space=space,
        phenotype=phenotype,
        spawn_position=spawn_position,
    )

    controller = (
        NeuralController.from_genome(
            genome
        )
    )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    physics_dt = (
        1.0
        / config.physics_hz
    )

    physics_steps_per_control = (
        config.physics_hz
        // config.control_hz
    )

    max_physics_steps = int(
        round(
            config.episode_seconds
            * config.physics_hz
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    max_x = float(
        creature.position.x
    )

    max_progress = 0.0

    physics_steps = 0
    control_steps = 0

    termination_reason = "time_limit"

    current_action = np.zeros(
        4,
        dtype=np.float32,
    )

    # ========================================================
    # Simulation loop
    # ========================================================

    for step_index in range(
        max_physics_steps
    ):

        # ----------------------------------------------------
        # Neural controller
        # ----------------------------------------------------

        if (
            step_index
            % physics_steps_per_control
            == 0
        ):

            observation = (
                build_observation(
                    creature=creature,
                    spawn_position=spawn_position,
                    goal_x=config.goal_x,
                )
            )

            current_action = controller(
                observation
            )

            creature.apply_motor_commands(
                current_action
            )

            control_steps += 1

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        space.step(
            physics_dt
        )

        physics_steps += 1

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        if not _creature_state_is_finite(
            creature
        ):
            termination_reason = (
                "numerical_failure"
            )
            break

        # ----------------------------------------------------
        # Progress tracking
        # ----------------------------------------------------

        current_x = float(
            creature.position.x
        )

        max_x = max(
            max_x,
            current_x,
        )

        current_progress = (
            current_x
            - config.spawn_x
        )

        max_progress = max(
            max_progress,
            current_progress,
        )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        if current_x >= config.goal_x:

            termination_reason = (
                "goal_reached"
            )

            break

        # ----------------------------------------------------
        # Death
        # ----------------------------------------------------

        if (
            creature.position.y
            >= config.death_y
        ):

            termination_reason = (
                "fell"
            )

            break

    # ========================================================
    # Results
    # ========================================================

    final_x = float(
        creature.position.x
    )

    final_y = float(
        creature.position.y
    )

    fitness = _calculate_fitness(
        max_progress
    )

    simulated_seconds = (
        physics_steps
        * physics_dt
    )

    return FitnessResult(
        fitness=fitness,
        max_progress=max_progress,
        max_x=max_x,
        final_x=final_x,
        final_y=final_y,
        physics_steps=physics_steps,
        control_steps=control_steps,
        simulated_seconds=simulated_seconds,
        termination_reason=termination_reason,
        reached_goal=(
            termination_reason
            == "goal_reached"
        ),
    )
