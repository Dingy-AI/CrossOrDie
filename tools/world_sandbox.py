import sys
from pathlib import Path

import pygame
import pymunk
import pymunk.pygame_util


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from envs.world import (
    CrossOrDieWorld,
)


# ============================================================
# Configuration
# ============================================================

WIDTH = 1600
HEIGHT = 800

FPS = 60

SCALE = 0.58

CAMERA_X = -100.0


# ============================================================
# Coordinate conversion
# ============================================================

def world_to_screen(
    point,
):
    x, y = point

    return (
        round(
            (
                x - CAMERA_X
            )
            * SCALE
        ),
        round(
            y * SCALE
            + 150
        ),
    )


# ============================================================
# Drawing
# ============================================================

def draw_surface_land(
    screen,
    points,
    bottom_y,
):

    screen_points = [
        world_to_screen(
            point
        )
        for point in points
    ]

    polygon = (
        screen_points
        + [
            (
                screen_points[-1][0],
                bottom_y,
            ),
            (
                screen_points[0][0],
                bottom_y,
            ),
        ]
    )

    pygame.draw.polygon(
        screen,
        (105, 89, 69),
        polygon,
    )

    pygame.draw.lines(
        screen,
        (83, 132, 72),
        False,
        screen_points,
        10,
    )


def draw_bridge(
    screen,
    world,
):

    wood = (
        142,
        96,
        56,
    )

    dark_wood = (
        78,
        52,
        34,
    )

    rope = (
        89,
        69,
        45,
    )

    # --------------------------------------------------------
    # Planks
    # --------------------------------------------------------

    for start, end in zip(
        world.bridge_points[:-1],
        world.bridge_points[1:],
    ):

        a = world_to_screen(
            start
        )

        b = world_to_screen(
            end
        )

        pygame.draw.line(
            screen,
            dark_wood,
            a,
            b,
            14,
        )

        pygame.draw.line(
            screen,
            wood,
            a,
            b,
            9,
        )

    # --------------------------------------------------------
    # Rope railing
    # --------------------------------------------------------

    rope_points = []

    for x, y in world.bridge_points:

        rope_points.append(
            world_to_screen(
                (
                    x,
                    y - 55,
                )
            )
        )

    pygame.draw.lines(
        screen,
        rope,
        False,
        rope_points,
        3,
    )

    # --------------------------------------------------------
    # Posts
    # --------------------------------------------------------

    for index in range(
        0,
        len(world.bridge_points),
        2,
    ):

        x, y = (
            world.bridge_points[index]
        )

        pygame.draw.line(
            screen,
            dark_wood,
            world_to_screen(
                (
                    x,
                    y,
                )
            ),
            world_to_screen(
                (
                    x,
                    y - 58,
                )
            ),
            4,
        )


def draw_goal(
    screen,
    world,
):

    x = world.goal_x

    # Approximate local surface height.
    y = 640.0

    bottom = world_to_screen(
        (
            x,
            y,
        )
    )

    top = world_to_screen(
        (
            x,
            y - 120,
        )
    )

    pygame.draw.line(
        screen,
        (55, 55, 55),
        bottom,
        top,
        5,
    )

    flag_tip = world_to_screen(
        (
            x + 70,
            y - 95,
        )
    )

    flag_bottom = world_to_screen(
        (
            x,
            y - 70,
        )
    )

    pygame.draw.polygon(
        screen,
        (220, 65, 65),
        [
            top,
            flag_tip,
            flag_bottom,
        ],
    )


# ============================================================
# Main
# ============================================================

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (
            WIDTH,
            HEIGHT,
        )
    )

    pygame.display.set_caption(
        "CrossOrDie - World Preview"
    )

    clock = pygame.time.Clock()

    space = pymunk.Space()

    world = CrossOrDieWorld(
        space
    )

    running = True

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif (
                event.type
                == pygame.KEYDOWN
                and
                event.key
                == pygame.K_ESCAPE
            ):
                running = False

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        screen.fill(
            (
                165,
                210,
                230,
            )
        )

        # Canyon backdrop
        pygame.draw.rect(
            screen,
            (
                71,
                68,
                64,
            ),
            (
                0,
                530,
                WIDTH,
                HEIGHT - 530,
            ),
        )

        # ----------------------------------------------------
        # Land
        # ----------------------------------------------------

        draw_surface_land(
            screen,
            world.left_surface,
            HEIGHT,
        )

        draw_surface_land(
            screen,
            world.right_surface,
            HEIGHT,
        )

        # ----------------------------------------------------
        # Bridge
        # ----------------------------------------------------

        draw_bridge(
            screen,
            world,
        )

        # ----------------------------------------------------
        # Spawn marker
        # ----------------------------------------------------

        spawn = world_to_screen(
            world.spawn_position
        )

        pygame.draw.circle(
            screen,
            (
                255,
                220,
                70,
            ),
            spawn,
            12,
        )

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        draw_goal(
            screen,
            world,
        )

        pygame.display.flip()

        clock.tick(
            FPS
        )

    pygame.quit()


if __name__ == "__main__":
    main()