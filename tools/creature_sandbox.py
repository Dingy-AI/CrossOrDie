import argparse
import sys
from pathlib import Path

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util


# ------------------------------------------------------------
# Allow direct execution from /tools
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from evolution.genome import Genome

from evolution.checkpoint import (
    load_genome_checkpoint,
    evaluation_config_from_checkpoint,
)

from evolution.evaluator import (
    EvaluationConfig,
)

from creatures.phenotype import (
    decode_genome,
)

from creatures.body import (
    CreatureBody,
    MOTOR_COUNT,
)

from creatures.observation import (
    build_observation,
)

from creatures.controller import (
    NeuralController,
    with_random_brain,
)


# ============================================================
# Display configuration
# ============================================================

WIDTH = 1200
HEIGHT = 800

FPS = 60


# ============================================================
# Sandbox
# ============================================================

class CreatureSandbox:

    def __init__(
        self,
        checkpoint_path: Path | None = None,
    ):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                WIDTH,
                HEIGHT,
            )
        )

        pygame.display.set_caption(
            "CrossOrDie - Creature Sandbox"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            20,
        )

        self.small_font = pygame.font.SysFont(
            "consolas",
            16,
        )

        self.draw_options = (
            pymunk.pygame_util.DrawOptions(
                self.screen
            )
        )

        self.rng = np.random.default_rng()

        # ----------------------------------------------------
        # Checkpoint state
        # ----------------------------------------------------

        self.checkpoint_path = checkpoint_path

        self.loaded_checkpoint = None

        self.loaded_generation = None
        self.loaded_fitness = None

        # ----------------------------------------------------
        # Simulation state
        # ----------------------------------------------------

        self.space = None

        self.genome = None
        self.phenotype = None

        self.creature = None
        self.controller = None

        self.observation = None

        self.commands = np.zeros(
            MOTOR_COUNT,
            dtype=np.float32,
        )

        self.physics_step_count = 0

        self.paused = False
        self.autonomous = True

        # ----------------------------------------------------
        # Default evaluation/world config
        # ----------------------------------------------------

        self.evaluation_config = (
            EvaluationConfig()
        )

        # Load checkpoint if supplied.
        if self.checkpoint_path is not None:
            self.load_checkpoint(
                self.checkpoint_path
            )
        else:
            self.create_random_creature()

    # ========================================================
    # Derived timing
    # ========================================================

    @property
    def physics_dt(self) -> float:

        return (
            1.0
            / self.evaluation_config.physics_hz
        )

    @property
    def physics_steps_per_frame(self) -> int:

        return max(
            1,
            round(
                self.evaluation_config.physics_hz
                / FPS
            ),
        )

    @property
    def physics_steps_per_control(self) -> int:

        return (
            self.evaluation_config.physics_hz
            // self.evaluation_config.control_hz
        )

    @property
    def spawn_position(
        self,
    ) -> tuple[float, float]:

        return (
            self.evaluation_config.spawn_x,
            self.evaluation_config.spawn_y,
        )

    @property
    def goal_x(self) -> float:

        return (
            self.evaluation_config.goal_x
        )

    @property
    def ground_y(self) -> float:

        return (
            self.evaluation_config.ground_y
        )

    # ========================================================
    # World
    # ========================================================

    def create_space(self):

        config = self.evaluation_config

        self.space = pymunk.Space()

        self.space.gravity = (
            0.0,
            config.gravity,
        )

        self.space.iterations = (
            config.solver_iterations
        )

        self.create_ground()

    def create_ground(self):

        config = self.evaluation_config

        ground = pymunk.Segment(
            self.space.static_body,
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

        self.space.add(
            ground
        )

    # ========================================================
    # Creature construction
    # ========================================================

    def build_creature_from_genome(
        self,
        genome: Genome,
    ):

        self.create_space()

        self.genome = genome.copy()

        self.phenotype = decode_genome(
            self.genome
        )

        self.creature = CreatureBody(
            space=self.space,
            phenotype=self.phenotype,
            spawn_position=self.spawn_position,
        )

        self.controller = (
            NeuralController.from_genome(
                self.genome
            )
        )

        self.commands = np.zeros(
            MOTOR_COUNT,
            dtype=np.float32,
        )

        self.physics_step_count = 0

        self.observation = (
            build_observation(
                creature=self.creature,
                spawn_position=self.spawn_position,
                goal_x=self.goal_x,
            )
        )

    # ========================================================
    # Random Cowie
    # ========================================================

    def create_random_creature(self):

        self.loaded_checkpoint = None
        self.loaded_generation = None
        self.loaded_fitness = None

        # Return to default world configuration.
        self.evaluation_config = (
            EvaluationConfig()
        )

        genome = Genome.random(
            self.rng
        )

        genome = with_random_brain(
            genome,
            rng=self.rng,
        )

        self.build_creature_from_genome(
            genome
        )

    # ========================================================
    # Checkpoint loading
    # ========================================================

    def load_checkpoint(
        self,
        path: Path,
    ):

        checkpoint = (
            load_genome_checkpoint(
                path
            )
        )

        if checkpoint.genome.brain is None:
            raise ValueError(
                "Checkpoint genome does not contain "
                "a brain chromosome."
            )

        self.loaded_checkpoint = checkpoint

        self.checkpoint_path = path

        self.loaded_generation = (
            checkpoint.generation
        )

        if checkpoint.fitness_result:

            self.loaded_fitness = (
                checkpoint.fitness_result[
                    "fitness"
                ]
            )

        else:

            self.loaded_fitness = None

        # ----------------------------------------------------
        # Reproduce original evaluation environment
        # ----------------------------------------------------

        if (
            checkpoint.evaluation_config
            is not None
        ):

            self.evaluation_config = (
                evaluation_config_from_checkpoint(
                    checkpoint
                )
            )

        else:

            self.evaluation_config = (
                EvaluationConfig()
            )

        self.build_creature_from_genome(
            checkpoint.genome
        )

        print()
        print(
            f"Loaded checkpoint: {path}"
        )

        print(
            f"Generation: "
            f"{self.loaded_generation}"
        )

        if self.loaded_fitness is not None:

            print(
                f"Fitness: "
                f"{self.loaded_fitness:.2f}"
            )

        print()

    # ========================================================
    # Reset
    # ========================================================

    def reset_current_creature(self):
        """
        Restart the currently loaded genome from its original
        spawn state.
        """

        self.build_creature_from_genome(
            self.genome
        )

    # ========================================================
    # Input
    # ========================================================

    def handle_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    return False

                # --------------------------------------------
                # New random Cowie
                # --------------------------------------------

                if event.key == pygame.K_n:
                    self.create_random_creature()

                # --------------------------------------------
                # Restart SAME genome
                # --------------------------------------------

                elif event.key == pygame.K_r:
                    self.reset_current_creature()

                # --------------------------------------------
                # Pause
                # --------------------------------------------

                elif event.key == pygame.K_SPACE:

                    self.paused = (
                        not self.paused
                    )

                # --------------------------------------------
                # Neural/manual control
                # --------------------------------------------

                elif event.key == pygame.K_m:

                    self.autonomous = (
                        not self.autonomous
                    )

                    self.commands[:] = 0.0

                    self.creature.stop_motors()

                # --------------------------------------------
                # Stop motors
                # --------------------------------------------

                elif event.key == pygame.K_x:

                    self.commands[:] = 0.0

                    self.creature.stop_motors()

        return True

    # ========================================================
    # Manual control
    # ========================================================

    def handle_manual_input(self):

        if self.autonomous:
            return

        keys = pygame.key.get_pressed()

        commands = np.zeros(
            MOTOR_COUNT,
            dtype=np.float32,
        )

        # Front hip
        if keys[pygame.K_q]:
            commands[0] = -1.0

        elif keys[pygame.K_w]:
            commands[0] = 1.0

        # Front knee
        if keys[pygame.K_a]:
            commands[1] = -1.0

        elif keys[pygame.K_s]:
            commands[1] = 1.0

        # Rear hip
        if keys[pygame.K_e]:
            commands[2] = -1.0

        elif keys[pygame.K_t]:
            commands[2] = 1.0

        # Rear knee
        if keys[pygame.K_d]:
            commands[3] = -1.0

        elif keys[pygame.K_f]:
            commands[3] = 1.0

        self.commands = commands

        self.creature.apply_motor_commands(
            commands
        )

    # ========================================================
    # Neural control
    # ========================================================

    def update_controller(self):

        self.observation = (
            build_observation(
                creature=self.creature,
                spawn_position=self.spawn_position,
                goal_x=self.goal_x,
            )
        )

        self.commands = (
            self.controller(
                self.observation
            )
        )

        self.creature.apply_motor_commands(
            self.commands
        )

    # ========================================================
    # Simulation
    # ========================================================

    def step(self):

        if self.paused:
            return

        for _ in range(
            self.physics_steps_per_frame
        ):

            if (
                self.autonomous
                and
                self.physics_step_count
                % self.physics_steps_per_control
                == 0
            ):

                self.update_controller()

            self.space.step(
                self.physics_dt
            )

            self.physics_step_count += 1

        if not self.autonomous:

            self.observation = (
                build_observation(
                    creature=self.creature,
                    spawn_position=self.spawn_position,
                    goal_x=self.goal_x,
                )
            )

    # ========================================================
    # Rendering
    # ========================================================

    def draw(self):

        self.screen.fill(
            (
                225,
                235,
                242,
            )
        )

        self.draw_goal()

        self.space.debug_draw(
            self.draw_options
        )

        self.draw_creature_face()

        self.draw_ui()

        pygame.display.flip()

    # ========================================================
    # Goal
    # ========================================================

    def draw_goal(self):

        x = round(
            self.goal_x
        )

        y = round(
            self.ground_y
        )

        pygame.draw.line(
            self.screen,
            (
                50,
                170,
                70,
            ),
            (
                x,
                y - 120,
            ),
            (
                x,
                y,
            ),
            5,
        )

        pygame.draw.polygon(
            self.screen,
            (
                230,
                70,
                70,
            ),
            [
                (
                    x,
                    y - 120,
                ),
                (
                    x + 50,
                    y - 100,
                ),
                (
                    x,
                    y - 80,
                ),
            ],
        )

    # ========================================================
    # Smiley face
    # ========================================================

    def draw_creature_face(self):

        torso = (
            self.creature.torso_body
        )

        body_pheno = (
            self.phenotype.body
        )

        geometry_center = pymunk.Vec2d(
            -body_pheno.balance_offset,
            0.0,
        )

        face_center_local = (
            geometry_center
            + pymunk.Vec2d(
                body_pheno.length * 0.22,
                0.0,
            )
        )

        face_radius = min(
            body_pheno.height * 0.32,
            18.0,
        )

        eye_x = (
            face_radius * 0.35
        )

        eye_y = (
            face_radius * 0.25
        )

        left_eye_local = (
            face_center_local
            + pymunk.Vec2d(
                -eye_x,
                -eye_y,
            )
        )

        right_eye_local = (
            face_center_local
            + pymunk.Vec2d(
                eye_x,
                -eye_y,
            )
        )

        face_center = (
            torso.local_to_world(
                face_center_local
            )
        )

        left_eye = (
            torso.local_to_world(
                left_eye_local
            )
        )

        right_eye = (
            torso.local_to_world(
                right_eye_local
            )
        )

        # Face
        pygame.draw.circle(
            self.screen,
            (
                255,
                225,
                80,
            ),
            (
                round(face_center.x),
                round(face_center.y),
            ),
            round(face_radius),
        )

        pygame.draw.circle(
            self.screen,
            (
                35,
                35,
                35,
            ),
            (
                round(face_center.x),
                round(face_center.y),
            ),
            round(face_radius),
            2,
        )

        # Eyes
        eye_radius = max(
            2,
            round(
                face_radius * 0.12
            ),
        )

        for eye in (
            left_eye,
            right_eye,
        ):

            pygame.draw.circle(
                self.screen,
                (
                    20,
                    20,
                    20,
                ),
                (
                    round(eye.x),
                    round(eye.y),
                ),
                eye_radius,
            )

        # Smile
        smile_points = []

        for t in np.linspace(
            0.15,
            0.85,
            12,
        ):

            x = (
                face_center_local.x
                + (
                    t - 0.5
                )
                * face_radius
                * 1.2
            )

            normalized = (
                (
                    t - 0.5
                )
                / 0.35
            )

            y = (
                face_center_local.y
                + face_radius * 0.15
                + (
                    1.0
                    - normalized ** 2
                )
                * face_radius
                * 0.28
            )

            point = (
                torso.local_to_world(
                    (
                        x,
                        y,
                    )
                )
            )

            smile_points.append(
                (
                    round(point.x),
                    round(point.y),
                )
            )

        pygame.draw.lines(
            self.screen,
            (
                20,
                20,
                20,
            ),
            False,
            smile_points,
            max(
                2,
                round(
                    face_radius * 0.10
                ),
            ),
        )

    # ========================================================
    # UI
    # ========================================================

    def draw_ui(self):

        y = 15

        title = self.font.render(
            "CrossOrDie Creature Sandbox",
            True,
            (
                20,
                20,
                20,
            ),
        )

        self.screen.blit(
            title,
            (
                15,
                y,
            ),
        )

        y += 35

        # ----------------------------------------------------
        # Checkpoint information
        # ----------------------------------------------------

        if self.loaded_checkpoint is not None:

            source_lines = [
                "Source: CHECKPOINT",
                (
                    "Generation: "
                    f"{self.loaded_generation}"
                ),
            ]

            if self.loaded_fitness is not None:

                source_lines.append(
                    "Saved fitness: "
                    f"{self.loaded_fitness:.2f}"
                )

        else:

            source_lines = [
                "Source: RANDOM",
            ]

        for line in source_lines:

            text = (
                self.small_font.render(
                    line,
                    True,
                    (
                        30,
                        30,
                        30,
                    ),
                )
            )

            self.screen.blit(
                text,
                (
                    15,
                    y,
                ),
            )

            y += 21

        y += 8

        mode = (
            "NEURAL"
            if self.autonomous
            else "MANUAL"
        )

        controls = [
            f"Mode:    {mode}",
            "",
            "R       : restart same Cowie",
            "N       : new random Cowie",
            "M       : neural/manual",
            "SPACE   : pause",
            "X       : stop motors",
            "",
            "Manual:",
            "Q / W   : front hip",
            "A / S   : front knee",
            "E / T   : rear hip",
            "D / F   : rear knee",
            "",
            "ESC     : quit",
        ]

        for line in controls:

            text = (
                self.small_font.render(
                    line,
                    True,
                    (
                        30,
                        30,
                        30,
                    ),
                )
            )

            self.screen.blit(
                text,
                (
                    15,
                    y,
                ),
            )

            y += 21

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        y += 8

        motor_text = (
            "Brain: "
            f"{self.commands[0]:+.2f} "
            f"{self.commands[1]:+.2f} "
            f"{self.commands[2]:+.2f} "
            f"{self.commands[3]:+.2f}"
        )

        text = self.small_font.render(
            motor_text,
            True,
            (
                30,
                30,
                30,
            ),
        )

        self.screen.blit(
            text,
            (
                15,
                y,
            ),
        )

        y += 24

        position = (
            self.creature.position
        )

        current_progress = (
            position.x
            - self.evaluation_config.spawn_x
        )

        position_text = (
            f"x={position.x:.1f} "
            f"progress={current_progress:.1f}"
        )

        text = self.small_font.render(
            position_text,
            True,
            (
                30,
                30,
                30,
            ),
        )

        self.screen.blit(
            text,
            (
                15,
                y,
            ),
        )

        if self.paused:

            pause_text = (
                self.font.render(
                    "PAUSED",
                    True,
                    (
                        180,
                        40,
                        40,
                    ),
                )
            )

            self.screen.blit(
                pause_text,
                (
                    WIDTH - 110,
                    15,
                ),
            )

    # ========================================================
    # Main loop
    # ========================================================

    def run(self):

        running = True

        while running:

            running = (
                self.handle_events()
            )

            if not running:
                break

            self.handle_manual_input()

            self.step()

            self.draw()

            self.clock.tick(
                FPS
            )

        pygame.quit()


# ============================================================
# Command-line interface
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "CrossOrDie creature sandbox."
        )
    )

    parser.add_argument(
        "--genome",
        type=Path,
        default=None,
        help=(
            "Path to a saved .npz genome checkpoint."
        ),
    )

    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    sandbox = CreatureSandbox(
        checkpoint_path=args.genome
    )

    sandbox.run()