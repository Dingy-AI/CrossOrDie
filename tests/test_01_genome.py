import numpy as np
import pytest

from evolution.genome import (
    Genome,
    BodyGene,
    LegGene,
    JointGene,
    BODY_GENE_COUNT,
    FRONT_LEG_GENE_COUNT,
    REAR_LEG_GENE_COUNT,
    JOINT_GENE_COUNT,
    TOTAL_MORPHOLOGY_GENES,
)


def test_gene_counts():
    assert BODY_GENE_COUNT == len(BodyGene)
    assert FRONT_LEG_GENE_COUNT == len(LegGene)
    assert REAR_LEG_GENE_COUNT == len(LegGene)
    assert JOINT_GENE_COUNT == len(JointGene)

    assert TOTAL_MORPHOLOGY_GENES == 23


def test_random_genome_has_correct_shapes():
    genome = Genome.random(
        np.random.default_rng(42)
    )

    assert genome.body.shape == (BODY_GENE_COUNT,)
    assert genome.front_leg.shape == (FRONT_LEG_GENE_COUNT,)
    assert genome.rear_leg.shape == (REAR_LEG_GENE_COUNT,)
    assert genome.joints.shape == (JOINT_GENE_COUNT,)


def test_random_genome_values_are_normalized():
    genome = Genome.random(
        np.random.default_rng(42)
    )

    genes = genome.flatten_morphology()

    assert np.all(genes >= 0.0)
    assert np.all(genes <= 1.0)


def test_random_genome_is_deterministic_with_seed():
    genome_a = Genome.random(
        np.random.default_rng(42)
    )

    genome_b = Genome.random(
        np.random.default_rng(42)
    )

    np.testing.assert_array_equal(
        genome_a.flatten_morphology(),
        genome_b.flatten_morphology(),
    )


def test_flatten_morphology_has_correct_size():
    genome = Genome.random(
        np.random.default_rng(42)
    )

    flattened = genome.flatten_morphology()

    assert flattened.shape == (
        TOTAL_MORPHOLOGY_GENES,
    )


def test_from_flat_morphology_round_trip():
    original = Genome.random(
        np.random.default_rng(42)
    )

    flattened = original.flatten_morphology()

    rebuilt = Genome.from_flat_morphology(
        flattened
    )

    np.testing.assert_array_equal(
        original.body,
        rebuilt.body,
    )

    np.testing.assert_array_equal(
        original.front_leg,
        rebuilt.front_leg,
    )

    np.testing.assert_array_equal(
        original.rear_leg,
        rebuilt.rear_leg,
    )

    np.testing.assert_array_equal(
        original.joints,
        rebuilt.joints,
    )


def test_copy_is_deep_copy():
    genome = Genome.random(
        np.random.default_rng(42)
    )

    copied = genome.copy()

    copied.body[0] = 0.0

    assert copied.body[0] != genome.body[0]


def test_invalid_gene_range_raises():
    body = np.zeros(
        BODY_GENE_COUNT,
        dtype=np.float32,
    )

    body[0] = 1.5

    with pytest.raises(ValueError):
        Genome(body=body)


def test_invalid_chromosome_shape_raises():
    body = np.zeros(
        BODY_GENE_COUNT + 1,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        Genome(body=body)


def test_non_finite_gene_raises():
    body = np.zeros(
        BODY_GENE_COUNT,
        dtype=np.float32,
    )

    body[0] = np.nan

    with pytest.raises(ValueError):
        Genome(body=body)