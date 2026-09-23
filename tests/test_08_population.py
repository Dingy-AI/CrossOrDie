import numpy as np
import pytest

from evolution.evaluator import (
    EvaluationConfig,
)
from evolution.genetics import (
    MutationConfig,
)
from evolution.population import (
    Population,
    PopulationConfig,
)


# Keep unit tests fast.
TEST_EVALUATION_CONFIG = EvaluationConfig(
    episode_seconds=0.05,
    death_y=10_000.0,
    goal_x=100_000.0,
)


# ============================================================
# Initialization
# ============================================================

def test_random_population_has_correct_size():

    config = PopulationConfig(
        population_size=10,
        elite_count=2,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    assert len(
        population.genomes
    ) == 10


def test_random_population_has_brains():

    config = PopulationConfig(
        population_size=10,
        elite_count=2,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    for genome in population.genomes:

        assert genome.brain is not None


def test_initial_generation_is_zero():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    assert population.generation == 0


# ============================================================
# Determinism
# ============================================================

def test_population_initialization_is_deterministic():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population_a = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    population_b = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    for genome_a, genome_b in zip(
        population_a.genomes,
        population_b.genomes,
    ):

        np.testing.assert_array_equal(
            genome_a.flatten_morphology(),
            genome_b.flatten_morphology(),
        )

        np.testing.assert_array_equal(
            genome_a.brain,
            genome_b.brain,
        )


# ============================================================
# Evaluation
# ============================================================

def test_evaluate_returns_every_individual():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    evaluated = population.evaluate()

    assert len(
        evaluated
    ) == 6


def test_rank_orders_by_fitness():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    evaluated = population.evaluate()

    ranked = population.rank(
        evaluated
    )

    fitnesses = [
        individual.fitness
        for individual in ranked
    ]

    assert fitnesses == sorted(
        fitnesses,
        reverse=True,
    )


# ============================================================
# Reproduction
# ============================================================

def test_next_generation_has_correct_size():

    config = PopulationConfig(
        population_size=8,
        elite_count=2,
        parent_fraction=0.5,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    evaluated = population.evaluate()

    children = (
        population.breed_next_generation(
            evaluated
        )
    )

    assert len(children) == 8


def test_elites_are_preserved_exactly():

    config = PopulationConfig(
        population_size=8,
        elite_count=2,
        parent_fraction=0.5,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    evaluated = population.evaluate()

    ranked = population.rank(
        evaluated
    )

    next_generation = (
        population.breed_next_generation(
            evaluated
        )
    )

    for elite_index in range(2):

        expected = ranked[
            elite_index
        ].genome

        actual = next_generation[
            elite_index
        ]

        np.testing.assert_array_equal(
            actual.flatten_morphology(),
            expected.flatten_morphology(),
        )

        np.testing.assert_array_equal(
            actual.brain,
            expected.brain,
        )


def test_elites_are_copies():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    evaluated = population.evaluate()

    ranked = population.rank(
        evaluated
    )

    next_generation = (
        population.breed_next_generation(
            evaluated
        )
    )

    assert (
        next_generation[0]
        is not ranked[0].genome
    )


# ============================================================
# Generation stepping
# ============================================================

def test_step_advances_generation():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    population.step()

    assert population.generation == 1


def test_step_preserves_population_size():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    population.step()

    assert len(
        population.genomes
    ) == 6


def test_generation_report():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    report = population.step()

    assert report.stats.generation == 0

    assert (
        report.stats.population_size
        == 6
    )

    assert (
        report.stats.best_fitness
        >= report.stats.mean_fitness
    )

    assert (
        report.stats.best_fitness
        >= report.stats.worst_fitness
    )

    assert (
        report.best_genome.brain
        is not None
    )


# ============================================================
# Multiple generations
# ============================================================

def test_run_multiple_generations():

    config = PopulationConfig(
        population_size=6,
        elite_count=1,
    )

    population = Population.random(
        population_config=config,
        evaluation_config=TEST_EVALUATION_CONFIG,
        seed=42,
    )

    reports = population.run(
        generations=3
    )

    assert len(reports) == 3

    assert population.generation == 3

    assert [
        report.stats.generation
        for report in reports
    ] == [
        0,
        1,
        2,
    ]


# ============================================================
# Configuration validation
# ============================================================

def test_invalid_population_size():

    with pytest.raises(ValueError):

        PopulationConfig(
            population_size=1
        )


def test_invalid_elite_count():

    with pytest.raises(ValueError):

        PopulationConfig(
            population_size=10,
            elite_count=10,
        )


def test_invalid_parent_fraction():

    with pytest.raises(ValueError):

        PopulationConfig(
            parent_fraction=0.0
        )


def test_invalid_tournament_size():

    with pytest.raises(ValueError):

        PopulationConfig(
            tournament_size=0
        )