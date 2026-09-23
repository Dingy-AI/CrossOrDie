import numpy as np

from evolution.checkpoint import (
    load_population_checkpoint,
    save_population_checkpoint,
)

from evolution.evaluator import (
    EvaluationConfig,
)

from evolution.population import (
    Population,
    PopulationConfig,
)


def make_population():

    return Population.random(
        population_config=PopulationConfig(
            population_size=8,
            elite_count=2,
            parent_fraction=0.5,
            tournament_size=2,
        ),

        evaluation_config=EvaluationConfig(
            episode_seconds=0.05,
            death_y=10_000.0,
            goal_x=100_000.0,
        ),

        seed=42,
    )


def test_population_checkpoint_round_trip(
    tmp_path,
):

    population = make_population()

    path = (
        tmp_path
        / "population.npz"
    )

    save_population_checkpoint(
        path,
        population,
    )

    loaded = (
        load_population_checkpoint(
            path
        )
    )

    assert loaded.generation == (
        population.generation
    )

    assert len(
        loaded.genomes
    ) == len(
        population.genomes
    )

    for original, restored in zip(
        population.genomes,
        loaded.genomes,
    ):

        np.testing.assert_array_equal(
            original.body,
            restored.body,
        )

        np.testing.assert_array_equal(
            original.front_leg,
            restored.front_leg,
        )

        np.testing.assert_array_equal(
            original.rear_leg,
            restored.rear_leg,
        )

        np.testing.assert_array_equal(
            original.joints,
            restored.joints,
        )

        np.testing.assert_array_equal(
            original.brain,
            restored.brain,
        )


def test_population_configuration_round_trip(
    tmp_path,
):

    population = make_population()

    path = (
        tmp_path
        / "population.npz"
    )

    save_population_checkpoint(
        path,
        population,
    )

    loaded = (
        load_population_checkpoint(
            path
        )
    )

    assert (
        loaded.population_config
        == population.population_config
    )

    assert (
        loaded.evaluation_config
        == population.evaluation_config
    )

    assert (
        loaded.mutation_config
        == population.mutation_config
    )


def test_population_rng_is_restored(
    tmp_path,
):

    population = make_population()

    path = (
        tmp_path
        / "population.npz"
    )

    save_population_checkpoint(
        path,
        population,
    )

    loaded = (
        load_population_checkpoint(
            path
        )
    )

    original_numbers = (
        population.rng.random(
            10
        )
    )

    restored_numbers = (
        loaded.rng.random(
            10
        )
    )

    np.testing.assert_array_equal(
        original_numbers,
        restored_numbers,
    )


def test_population_can_continue_after_loading(
    tmp_path,
):

    population = make_population()

    path = (
        tmp_path
        / "population.npz"
    )

    save_population_checkpoint(
        path,
        population,
    )

    loaded = (
        load_population_checkpoint(
            path
        )
    )

    report = loaded.step()

    assert report.stats.generation == 0

    assert loaded.generation == 1

    assert len(
        loaded.genomes
    ) == 8


def test_generation_is_preserved(
    tmp_path,
):

    population = make_population()

    population.step()
    population.step()

    assert population.generation == 2

    path = (
        tmp_path
        / "population.npz"
    )

    save_population_checkpoint(
        path,
        population,
    )

    loaded = (
        load_population_checkpoint(
            path
        )
    )

    assert loaded.generation == 2