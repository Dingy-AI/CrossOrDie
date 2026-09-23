import numpy as np
import pymunk
import pytest

from evolution.genome import Genome
from creatures.phenotype import decode_genome
from creatures.body import CreatureBody
from creatures.observation import (
    build_observation,
    OBSERVATION_SIZE,
    TORSO_ANGLE_SIN,
    TORSO_ANGLE_COS,
    GOAL_DISTANCE,
    FRONT_FOOT_CONTACT,
    REAR_FOOT_CONTACT,
)


SPAWN = (
    300.0,
    200.0,
)

GOAL_X = 1000.0


def make_creature():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    phenotype = decode_genome(
        genome
    )

    space = pymunk.Space()

    space.gravity = (
        0.0,
        900.0,
    )

    creature = CreatureBody(
        space=space,
        phenotype=phenotype,
        spawn_position=SPAWN,
    )

    return (
        space,
        creature,
    )


def test_observation_has_correct_shape():

    _, creature = make_creature()

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert observation.shape == (
        OBSERVATION_SIZE,
    )


def test_observation_is_float32():

    _, creature = make_creature()

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert observation.dtype == np.float32


def test_observation_is_finite():

    _, creature = make_creature()

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert np.all(
        np.isfinite(observation)
    )


def test_observation_stays_normalized():

    _, creature = make_creature()

    creature.torso_body.velocity = (
        1_000_000.0,
        -1_000_000.0,
    )

    creature.torso_body.angular_velocity = (
        1_000_000.0
    )

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert np.all(
        observation >= -1.0
    )

    assert np.all(
        observation <= 1.0
    )


def test_initial_orientation():

    _, creature = make_creature()

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert observation[
        TORSO_ANGLE_SIN
    ] == pytest.approx(
        0.0
    )

    assert observation[
        TORSO_ANGLE_COS
    ] == pytest.approx(
        1.0
    )


def test_goal_distance_is_one_at_spawn():

    _, creature = make_creature()

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert observation[
        GOAL_DISTANCE
    ] == pytest.approx(
        1.0
    )


def test_goal_distance_is_zero_at_goal():

    _, creature = make_creature()

    creature.torso_body.position = (
        GOAL_X,
        creature.torso_body.position.y,
    )

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert observation[
        GOAL_DISTANCE
    ] == pytest.approx(
        0.0
    )


def test_invalid_goal_raises():

    _, creature = make_creature()

    with pytest.raises(ValueError):

        build_observation(
            creature,
            SPAWN,
            SPAWN[0],
        )


def test_foot_contacts_default_to_zero():

    _, creature = make_creature()

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert observation[
        FRONT_FOOT_CONTACT
    ] == 0.0

    assert observation[
        REAR_FOOT_CONTACT
    ] == 0.0


def test_front_foot_can_detect_ground():

    space, creature = make_creature()

    front_leg = creature.front_leg
    phenotype = creature.phenotype.front_leg

    foot_center = (
        front_leg.lower_body.local_to_world(
            (
                0.0,
                phenotype.lower_length / 2.0,
            )
        )
    )

    ground = pymunk.Segment(
        space.static_body,
        (
            foot_center.x - 100,
            foot_center.y,
        ),
        (
            foot_center.x + 100,
            foot_center.y,
        ),
        10.0,
    )

    ground.friction = 1.0

    space.add(
        ground
    )

    observation = build_observation(
        creature,
        SPAWN,
        GOAL_X,
    )

    assert observation[
        FRONT_FOOT_CONTACT
    ] == 1.0