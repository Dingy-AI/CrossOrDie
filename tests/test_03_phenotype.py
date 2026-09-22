import math

import numpy as np

from evolution.genome import Genome
from creatures.phenotype import (
    decode_genome,
    BODY_LENGTH_RANGE,
    BODY_HEIGHT_RANGE,
    BODY_DENSITY_RANGE,
    BODY_FRICTION_RANGE,
    UPPER_LEG_LENGTH_RANGE,
    LOWER_LEG_LENGTH_RANGE,
)


def make_uniform_genome(value: float) -> Genome:
    genome = Genome.random(
        np.random.default_rng(42)
    )

    genome.body[:] = value
    genome.front_leg[:] = value
    genome.rear_leg[:] = value
    genome.joints[:] = value

    return genome


def test_zero_genes_map_to_minimums():
    genome = make_uniform_genome(0.0)

    phenotype = decode_genome(genome)

    assert phenotype.body.length == BODY_LENGTH_RANGE[0]
    assert phenotype.body.height == BODY_HEIGHT_RANGE[0]

    assert (
        phenotype.body.density_scale
        == BODY_DENSITY_RANGE[0]
    )

    assert (
        phenotype.body.friction
        == BODY_FRICTION_RANGE[0]
    )

    assert (
        phenotype.front_leg.upper_length
        == UPPER_LEG_LENGTH_RANGE[0]
    )

    assert (
        phenotype.front_leg.lower_length
        == LOWER_LEG_LENGTH_RANGE[0]
    )


def test_one_genes_map_to_maximums():
    genome = make_uniform_genome(1.0)

    phenotype = decode_genome(genome)

    assert phenotype.body.length == BODY_LENGTH_RANGE[1]
    assert phenotype.body.height == BODY_HEIGHT_RANGE[1]

    assert (
        phenotype.body.density_scale
        == BODY_DENSITY_RANGE[1]
    )

    assert (
        phenotype.body.friction
        == BODY_FRICTION_RANGE[1]
    )

    assert (
        phenotype.front_leg.upper_length
        == UPPER_LEG_LENGTH_RANGE[1]
    )

    assert (
        phenotype.front_leg.lower_length
        == LOWER_LEG_LENGTH_RANGE[1]
    )


def test_half_genes_map_to_midpoints():
    genome = make_uniform_genome(0.5)

    phenotype = decode_genome(genome)

    expected_length = sum(
        BODY_LENGTH_RANGE
    ) / 2

    expected_height = sum(
        BODY_HEIGHT_RANGE
    ) / 2

    assert phenotype.body.length == expected_length
    assert phenotype.body.height == expected_height


def test_front_leg_attaches_in_front():
    genome = make_uniform_genome(0.5)

    phenotype = decode_genome(genome)

    assert (
        phenotype.front_leg.attachment_offset
        > 0
    )


def test_rear_leg_attaches_behind():
    genome = make_uniform_genome(0.5)

    phenotype = decode_genome(genome)

    assert (
        phenotype.rear_leg.attachment_offset
        < 0
    )


def test_balance_offset_stays_inside_body():
    rng = np.random.default_rng(42)

    for _ in range(100):
        genome = Genome.random(rng)

        phenotype = decode_genome(genome)

        half_length = (
            phenotype.body.length / 2
        )

        assert (
            -half_length
            < phenotype.body.balance_offset
            < half_length
        )


def test_joint_ranges_are_positive():
    genome = Genome.random(
        np.random.default_rng(42)
    )

    phenotype = decode_genome(genome)

    assert phenotype.joints.front_hip_range > 0
    assert phenotype.joints.front_knee_range > 0
    assert phenotype.joints.rear_hip_range > 0
    assert phenotype.joints.rear_knee_range > 0


def test_joint_ranges_are_radians():
    genome = make_uniform_genome(1.0)

    phenotype = decode_genome(genome)

    assert phenotype.joints.front_hip_range <= math.pi
    assert phenotype.joints.front_knee_range <= math.pi


def test_decode_does_not_modify_genome():
    genome = Genome.random(
        np.random.default_rng(42)
    )

    before = (
        genome.flatten_morphology().copy()
    )

    decode_genome(genome)

    np.testing.assert_array_equal(
        genome.flatten_morphology(),
        before,
    )