import numpy as np

from evolution.genome import Genome
from evolution.genetics import (
    MutationConfig,
    crossover,
    mutate,
    reproduce,
)


def make_parent(value: float) -> Genome:
    """
    Create a genome where every morphology gene has
    the same value.
    """

    genome = Genome.random(
        np.random.default_rng(0)
    )

    genome.body[:] = value
    genome.front_leg[:] = value
    genome.rear_leg[:] = value
    genome.joints[:] = value

    return genome


def test_crossover_does_not_modify_parents():
    parent_a = make_parent(0.0)
    parent_b = make_parent(1.0)

    parent_a_before = (
        parent_a.flatten_morphology().copy()
    )

    parent_b_before = (
        parent_b.flatten_morphology().copy()
    )

    crossover(
        parent_a,
        parent_b,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(
        parent_a.flatten_morphology(),
        parent_a_before,
    )

    np.testing.assert_array_equal(
        parent_b.flatten_morphology(),
        parent_b_before,
    )


def test_crossover_child_only_contains_parent_values():
    parent_a = make_parent(0.0)
    parent_b = make_parent(1.0)

    child = crossover(
        parent_a,
        parent_b,
        rng=np.random.default_rng(42),
    )

    genes = child.flatten_morphology()

    assert np.all(
        np.isin(
            genes,
            [0.0, 1.0],
        )
    )


def test_crossover_inherits_from_both_parents():
    parent_a = make_parent(0.0)
    parent_b = make_parent(1.0)

    child = crossover(
        parent_a,
        parent_b,
        rng=np.random.default_rng(42),
    )

    genes = child.flatten_morphology()

    assert np.any(genes == 0.0)
    assert np.any(genes == 1.0)


def test_crossover_is_deterministic_with_seed():
    parent_a = make_parent(0.0)
    parent_b = make_parent(1.0)

    child_a = crossover(
        parent_a,
        parent_b,
        rng=np.random.default_rng(42),
    )

    child_b = crossover(
        parent_a,
        parent_b,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(
        child_a.flatten_morphology(),
        child_b.flatten_morphology(),
    )


def test_zero_mutation_rate_changes_nothing():
    genome = Genome.random(
        np.random.default_rng(42)
    )

    config = MutationConfig(
        morphology_mutation_rate=0.0
    )

    mutated = mutate(
        genome,
        rng=np.random.default_rng(123),
        config=config,
    )

    np.testing.assert_array_equal(
        genome.flatten_morphology(),
        mutated.flatten_morphology(),
    )


def test_full_mutation_rate_changes_genes():
    genome = make_parent(0.5)

    config = MutationConfig(
        morphology_mutation_rate=1.0,
        morphology_large_mutation_rate=0.0,
        morphology_small_sigma=0.1,
    )

    mutated = mutate(
        genome,
        rng=np.random.default_rng(42),
        config=config,
    )

    assert not np.array_equal(
        genome.flatten_morphology(),
        mutated.flatten_morphology(),
    )


def test_mutation_stays_normalized():
    genome = make_parent(0.5)

    config = MutationConfig(
        morphology_mutation_rate=1.0,
        morphology_large_mutation_rate=1.0,
        morphology_large_sigma=10.0,
    )

    mutated = mutate(
        genome,
        rng=np.random.default_rng(42),
        config=config,
    )

    genes = mutated.flatten_morphology()

    assert np.all(genes >= 0.0)
    assert np.all(genes <= 1.0)


def test_mutation_does_not_modify_original():
    genome = make_parent(0.5)

    before = (
        genome.flatten_morphology().copy()
    )

    config = MutationConfig(
        morphology_mutation_rate=1.0
    )

    mutate(
        genome,
        rng=np.random.default_rng(42),
        config=config,
    )

    np.testing.assert_array_equal(
        genome.flatten_morphology(),
        before,
    )


def test_reproduce_is_deterministic():
    parent_a = make_parent(0.25)
    parent_b = make_parent(0.75)

    child_a = reproduce(
        parent_a,
        parent_b,
        rng=np.random.default_rng(42),
    )

    child_b = reproduce(
        parent_a,
        parent_b,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(
        child_a.flatten_morphology(),
        child_b.flatten_morphology(),
    )


def test_reproduce_produces_valid_genome():
    parent_a = Genome.random(
        np.random.default_rng(1)
    )

    parent_b = Genome.random(
        np.random.default_rng(2)
    )

    child = reproduce(
        parent_a,
        parent_b,
        rng=np.random.default_rng(3),
    )

    genes = child.flatten_morphology()

    assert len(genes) == 23

    assert np.all(genes >= 0.0)
    assert np.all(genes <= 1.0)