import pymunk
import pytest

from envs.world import (
    CrossOrDieWorld,
    WorldConfig,
    COLLISION_TERRAIN,
    COLLISION_BRIDGE,
)


# ============================================================
# Helpers
# ============================================================

def make_world():

    space = pymunk.Space()

    world = CrossOrDieWorld(
        space
    )

    return (
        space,
        world,
    )


# ============================================================
# Basic construction
# ============================================================

def test_world_creates_shapes():

    space, world = make_world()

    assert world.shape_count > 0

    assert len(
        space.shapes
    ) == world.shape_count


def test_world_has_terrain():

    _, world = make_world()

    assert len(
        world.terrain_shapes
    ) > 0


def test_world_has_bridge():

    _, world = make_world()

    assert len(
        world.bridge_shapes
    ) == (
        world.config.bridge_segments
    )


# ============================================================
# Collision types
# ============================================================

def test_terrain_collision_type():

    _, world = make_world()

    for shape in world.terrain_shapes:

        assert (
            shape.collision_type
            == COLLISION_TERRAIN
        )


def test_bridge_collision_type():

    _, world = make_world()

    for shape in world.bridge_shapes:

        assert (
            shape.collision_type
            == COLLISION_BRIDGE
        )


# ============================================================
# Bridge geometry
# ============================================================

def test_bridge_starts_at_configured_location():

    _, world = make_world()

    first = (
        world.bridge_points[0]
    )

    assert first[0] == pytest.approx(
        world.config.bridge_start_x
    )

    assert first[1] == pytest.approx(
        world.config.bridge_start_y
    )


def test_bridge_ends_at_configured_location():

    _, world = make_world()

    last = (
        world.bridge_points[-1]
    )

    assert last[0] == pytest.approx(
        world.config.bridge_end_x
    )

    assert last[1] == pytest.approx(
        world.config.bridge_end_y
    )


def test_bridge_sags_downward():

    _, world = make_world()

    endpoint_y = max(
        world.config.bridge_start_y,
        world.config.bridge_end_y,
    )

    assert (
        world.bridge_lowest_y
        > endpoint_y
    )


def test_bridge_center_has_expected_sag():

    _, world = make_world()

    middle_index = (
        len(world.bridge_points)
        // 2
    )

    middle_y = (
        world.bridge_points[
            middle_index
        ][1]
    )

    assert middle_y == pytest.approx(
        world.config.bridge_start_y
        + world.config.bridge_sag
    )


def test_bridge_is_longer_than_horizontal_span():

    _, world = make_world()

    horizontal = (
        world.config.bridge_end_x
        - world.config.bridge_start_x
    )

    assert (
        world.bridge_length
        > horizontal
    )


# ============================================================
# Terrain continuity
# ============================================================

def test_left_terrain_connects_to_bridge():

    _, world = make_world()

    assert (
        world.left_surface[-1]
        == world.bridge_points[0]
    )


def test_right_terrain_connects_to_bridge():

    _, world = make_world()

    assert (
        world.right_surface[0]
        == world.bridge_points[-1]
    )


# ============================================================
# Episode positions
# ============================================================

def test_spawn_is_before_bridge():

    _, world = make_world()

    assert (
        world.spawn_position[0]
        < world.config.bridge_start_x
    )


def test_goal_is_after_bridge():

    _, world = make_world()

    assert (
        world.goal_x
        > world.config.bridge_end_x
    )


def test_course_length():

    _, world = make_world()

    assert world.course_length == pytest.approx(
        world.goal_x
        - world.spawn_position[0]
    )


# ============================================================
# Episode state
# ============================================================

def test_progress_at_spawn_is_zero():

    _, world = make_world()

    position = pymunk.Vec2d(
        *world.spawn_position
    )

    assert world.progress(
        position
    ) == pytest.approx(
        0.0
    )


def test_forward_progress():

    _, world = make_world()

    position = pymunk.Vec2d(
        world.spawn_position[0] + 250.0,
        world.spawn_position[1],
    )

    assert world.progress(
        position
    ) == pytest.approx(
        250.0
    )


def test_goal_detection():

    _, world = make_world()

    before = pymunk.Vec2d(
        world.goal_x - 1.0,
        600.0,
    )

    after = pymunk.Vec2d(
        world.goal_x + 1.0,
        600.0,
    )

    assert not world.reached_goal(
        before
    )

    assert world.reached_goal(
        after
    )


def test_canyon_death_detection():

    _, world = make_world()

    alive = pymunk.Vec2d(
        1000.0,
        world.death_y - 1.0,
    )

    dead = pymunk.Vec2d(
        1000.0,
        world.death_y + 1.0,
    )

    assert not world.fell_into_canyon(
        alive
    )

    assert world.fell_into_canyon(
        dead
    )


# ============================================================
# Configuration validation
# ============================================================

def test_invalid_bridge_width():

    with pytest.raises(ValueError):

        WorldConfig(
            bridge_start_x=1000.0,
            bridge_end_x=900.0,
        )


def test_invalid_bridge_segments():

    with pytest.raises(ValueError):

        WorldConfig(
            bridge_segments=1
        )