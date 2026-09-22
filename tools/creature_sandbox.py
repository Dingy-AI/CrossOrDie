import sys
from pathlib import Path

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

# ------------------------------------------------------------
# Allow this script to be run directly from /tools
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from evolution.genome import Genome
from creatures.phenotype import decode_genome
from creatures.body import CreatureBody


# ============================================================
# Configuration
# ============================================================

WIDTH = 1200
HEIGHT = 800

FPS = 60

PHYSICS_HZ = 120
PHYSICS_DT = 1.0 / PHYSICS_HZ

PHYSICS_STEPS_PER_FRAME = (
    PHYSICS_HZ // FPS
)

GRAVITY = 900.0

GROUND_Y = 650

SPAWN_POSITION = (
    350.0,
    350.0,
)


# ============================================================
# Sandbox
# ============================================================

class CreatureSandbox:

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
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

        # Pymunk debug draw helper.
        self.draw_options = (
            pymunk.pygame_util.DrawOptions(
                self.screen
            )
        )

        self.rng = np.random.default_rng()

        self.space = None
        self.creature = None
        self.genome = None
        self.phenotype = None

        self.paused = False

        # Current manual motor commands.
        self.commands = np.zeros(
            4,
            dtype=np.float32,
        )

        self.reset_creature()

    # ========================================================
    # World
    # ========================================================

    def create_space(self):

        self.space = pymunk.Space()

        self.space.gravity = (
            0.0,
            GRAVITY,
        )

        # A few solver iterations help articulated creatures
        # behave more consistently.
        self.space.iterations = 20

        self.create_ground()

    def create_ground(self):

        ground = pymunk.Segment(
            self.space.static_body,
            (50, GROUND_Y),
            (WIDTH - 50, GROUND_Y),
            8,
        )

        ground.friction = 1.2
        ground.elasticity = 0.05

        self.space.add(
            ground
        )

    # ========================================================
    # Creature
    # ========================================================

    def reset_creature(self):

        self.create_space()

        self.genome = Genome.random(
            self.rng
        )

        self.phenotype = decode_genome(
            self.genome
        )

        self.creature = CreatureBody(
            space=self.space,
            phenotype=self.phenotype,
            spawn_position=SPAWN_POSITION,
        )

        self.commands[:] = 0.0

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

                # Generate a completely new creature.
                if event.key == pygame.K_r:
                    self.reset_creature()

                # Pause physics.
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                # Stop all motors.
                elif event.key == pygame.K_x:
                    self.commands[:] = 0.0

        return True

    def handle_motor_input(self):

        keys = pygame.key.get_pressed()

        commands = np.zeros(
            4,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Front leg
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Rear leg
        # ----------------------------------------------------

        # Rear hip
        if keys[pygame.K_e]:
            commands[2] = -1.0

        elif keys[pygame.K_r]:
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
    # Simulation
    # ========================================================

    def step(self):

        if self.paused:
            return

        for _ in range(
            PHYSICS_STEPS_PER_FRAME
        ):
            self.space.step(
                PHYSICS_DT
            )

    # ========================================================
    # Rendering
    # ========================================================

    def draw(self):

        self.screen.fill(
            (225, 235, 242)
        )

        # Pymunk draws bodies, segments and joints.
        self.space.debug_draw(
            self.draw_options
        )

        self.draw_ui()

        pygame.display.flip()

    def draw_ui(self):

        y = 15

        title = self.font.render(
            "CrossOrDie Creature Sandbox",
            True,
            (20, 20, 20),
        )

        self.screen.blit(
            title,
            (15, y),
        )

        y += 35

        controls = [
            "R       : random creature",
            "SPACE   : pause",
            "X       : stop motors",
            "",
            "Q / W   : front hip",
            "A / S   : front knee",
            "E / R   : rear hip",
            "D / F   : rear knee",
            "",
            "ESC     : quit",
        ]

        for line in controls:

            text = self.small_font.render(
                line,
                True,
                (30, 30, 30),
            )

            self.screen.blit(
                text,
                (15, y),
            )

            y += 22

        # ----------------------------------------------------
        # Current motor values
        # ----------------------------------------------------

        y += 10

        motor_text = (
            "Motors: "
            f"{self.commands[0]:+.1f} "
            f"{self.commands[1]:+.1f} "
            f"{self.commands[2]:+.1f} "
            f"{self.commands[3]:+.1f}"
        )

        text = self.small_font.render(
            motor_text,
            True,
            (30, 30, 30),
        )

        self.screen.blit(
            text,
            (15, y),
        )

        # ----------------------------------------------------
        # Phenotype diagnostics
        # ----------------------------------------------------

        y += 35

        body = self.phenotype.body
        front = self.phenotype.front_leg
        rear = self.phenotype.rear_leg

        stats = [
            f"Body length: {body.length:.1f}",
            f"Body height: {body.height:.1f}",
            f"Density:     {body.density_scale:.2f}",
            f"Balance:     {body.balance_offset:.1f}",
            "",
            f"Front upper: {front.upper_length:.1f}",
            f"Front lower: {front.lower_length:.1f}",
            f"Front foot:  {front.foot_length:.1f}",
            "",
            f"Rear upper:  {rear.upper_length:.1f}",
            f"Rear lower:  {rear.lower_length:.1f}",
            f"Rear foot:   {rear.foot_length:.1f}",
        ]

        for line in stats:

            text = self.small_font.render(
                line,
                True,
                (30, 30, 30),
            )

            self.screen.blit(
                text,
                (15, y),
            )

            y += 20

        if self.paused:

            pause_text = self.font.render(
                "PAUSED",
                True,
                (180, 40, 40),
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

            running = self.handle_events()

            if not running:
                break

            self.handle_motor_input()

            self.step()

            self.draw()

            self.clock.tick(
                FPS
            )

        pygame.quit()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    sandbox = CreatureSandbox()

    sandbox.run()