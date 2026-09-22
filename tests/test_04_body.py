import numpy as np
import pymunk
import pytest

from evolution.genome import Genome
from creatures.phenotype import decode_genome
from creatures.body import (
    CreatureBody,
    MotorIndex,
    MOTOR_COUNT,
)


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
        spawn_position=(300.0, 200.0),
    )

    return (
        space,
        phenotype,
        creature,
    )


def test_creature_has_five_bodies():
    _, _, creature = make_creature()

    assert len(creature.bodies) == 5


def test_creature_has_seven_shapes():
    _, _, creature = make_creature()

    assert len(creature.shapes) == 7


def test_creature_has_four_actuated_joints():
    _, _, creature = make_creature()

    assert len(creature.joints) == 4
    assert len(creature.motors) == 4

    assert MOTOR_COUNT == 4


def test_each_joint_has_three_constraints():
    _, _, creature = make_creature()

    # Pivot + limit + motor
    assert len(creature.constraints) == 12


def test_creature_is_added_to_space():
    space, _, creature = make_creature()

    assert len(space.bodies) == 5
    assert len(space.shapes) == 7
    assert len(space.constraints) == 12


def test_torso_spawns_at_requested_position():
    _, _, creature = make_creature()

    assert creature.torso_body.position.x == pytest.approx(
        300.0
    )

    assert creature.torso_body.position.y == pytest.approx(
        200.0
    )


def test_all_body_masses_are_positive():
    _, _, creature = make_creature()

    for body in creature.bodies:
        assert body.mass > 0.0


def test_all_body_moments_are_positive():
    _, _, creature = make_creature()

    for body in creature.bodies:
        assert body.moment > 0.0


def test_shapes_share_self_collision_group():
    _, _, creature = make_creature()

    groups = {
        shape.filter.group
        for shape in creature.shapes
    }

    assert groups == {
        creature.collision_group
    }


def test_motor_strengths_match_phenotype():
    _, phenotype, creature = make_creature()

    assert (
        creature.front_leg.hip.motor.max_force
        == pytest.approx(
            phenotype.front_leg.hip_strength
        )
    )

    assert (
        creature.front_leg.knee.motor.max_force
        == pytest.approx(
            phenotype.front_leg.knee_strength
        )
    )

    assert (
        creature.rear_leg.hip.motor.max_force
        == pytest.approx(
            phenotype.rear_leg.hip_strength
        )
    )

    assert (
        creature.rear_leg.knee.motor.max_force
        == pytest.approx(
            phenotype.rear_leg.knee_strength
        )
    )


def test_motor_command_order():
    _, _, creature = make_creature()

    commands = np.array(
        [
            1.0,
            -1.0,
            0.5,
            -0.5,
        ],
        dtype=np.float32,
    )

    creature.apply_motor_commands(
        commands
    )

    assert (
        creature.motors[
            MotorIndex.FRONT_HIP
        ].rate
        == pytest.approx(
            creature.max_motor_rate
        )
    )

    assert (
        creature.motors[
            MotorIndex.FRONT_KNEE
        ].rate
        == pytest.approx(
            -creature.max_motor_rate
        )
    )

    assert (
        creature.motors[
            MotorIndex.REAR_HIP
        ].rate
        == pytest.approx(
            creature.max_motor_rate * 0.5
        )
    )

    assert (
        creature.motors[
            MotorIndex.REAR_KNEE
        ].rate
        == pytest.approx(
            -creature.max_motor_rate * 0.5
        )
    )


def test_motor_commands_are_clipped():
    _, _, creature = make_creature()

    creature.apply_motor_commands(
        [
            10.0,
            -10.0,
            2.0,
            -2.0,
        ]
    )

    for motor in creature.motors:
        assert abs(motor.rate) <= (
            creature.max_motor_rate
        )


def test_wrong_motor_command_shape_raises():
    _, _, creature = make_creature()

    with pytest.raises(ValueError):
        creature.apply_motor_commands(
            [0.0, 0.0]
        )


def test_nonfinite_motor_command_raises():
    _, _, creature = make_creature()

    with pytest.raises(ValueError):
        creature.apply_motor_commands(
            [
                0.0,
                np.nan,
                0.0,
                0.0,
            ]
        )


def test_stop_motors():
    _, _, creature = make_creature()

    creature.apply_motor_commands(
        [
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )

    creature.stop_motors()

    for motor in creature.motors:
        assert motor.rate == 0.0


def test_short_simulation_remains_finite():
    space, _, creature = make_creature()

    creature.apply_motor_commands(
        [
            0.5,
            -0.5,
            0.25,
            -0.25,
        ]
    )

    dt = 1.0 / 120.0

    for _ in range(120):
        space.step(dt)

    for body in creature.bodies:

        assert np.isfinite(
            body.position.x
        )

        assert np.isfinite(
            body.position.y
        )

        assert np.isfinite(
            body.angle
        )

        assert np.isfinite(
            body.velocity.x
        )

        assert np.isfinite(
            body.velocity.y
        )


def test_remove_from_space():
    space, _, creature = make_creature()

    creature.remove_from_space()

    assert len(space.bodies) == 0
    assert len(space.shapes) == 0
    assert len(space.constraints) == 0