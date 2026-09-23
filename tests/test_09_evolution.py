import numpy as np
import pytest

from creatures.controller import (
    with_random_brain,
)

from evolution.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_genome_checkpoint,
    save_genome_checkpoint,
    evaluation_config_from_checkpoint,
)

from evolution.evaluator import (
    EvaluationConfig,
    evaluate_genome,
)

from evolution.genome import Genome


def make_genome():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    return with_random_brain(
        genome,
        np.random.default_rng(123),
    )


def test_checkpoint_round_trip(
    tmp_path,
):

    genome = make_genome()

    path = (
        tmp_path
        / "champion.npz"
    )

    save_genome_checkpoint(
        path,
        genome,
        generation=12,
    )

    loaded = (
        load_genome_checkpoint(
            path
        )
    )

    np.testing.assert_array_equal(
        loaded.genome.body,
        genome.body,
    )

    np.testing.assert_array_equal(
        loaded.genome.front_leg,
        genome.front_leg,
    )

    np.testing.assert_array_equal(
        loaded.genome.rear_leg,
        genome.rear_leg,
    )

    np.testing.assert_array_equal(
        loaded.genome.joints,
        genome.joints,
    )

    np.testing.assert_array_equal(
        loaded.genome.brain,
        genome.brain,
    )


def test_checkpoint_generation():

    pass


def test_checkpoint_metadata(
    tmp_path,
):

    genome = make_genome()

    path = (
        tmp_path
        / "champion.npz"
    )

    save_genome_checkpoint(
        path,
        genome,
        generation=37,
    )

    loaded = (
        load_genome_checkpoint(
            path
        )
    )

    assert loaded.generation == 37

    assert (
        loaded.format_version
        == CHECKPOINT_FORMAT_VERSION
    )


def test_checkpoint_creates_parent_directory(
    tmp_path,
):

    genome = make_genome()

    path = (
        tmp_path
        / "nested"
        / "folder"
        / "champion.npz"
    )

    save_genome_checkpoint(
        path,
        genome,
    )

    assert path.exists()


def test_evaluation_config_round_trip(
    tmp_path,
):

    genome = make_genome()

    config = EvaluationConfig(
        episode_seconds=7.5,
        goal_x=1500.0,
    )

    path = (
        tmp_path
        / "champion.npz"
    )

    save_genome_checkpoint(
        path,
        genome,
        evaluation_config=config,
    )

    loaded = (
        load_genome_checkpoint(
            path
        )
    )

    rebuilt = (
        evaluation_config_from_checkpoint(
            loaded
        )
    )

    assert (
        rebuilt.episode_seconds
        == config.episode_seconds
    )

    assert rebuilt.goal_x == (
        config.goal_x
    )

    assert rebuilt.physics_hz == (
        config.physics_hz
    )


def test_fitness_result_is_saved(
    tmp_path,
):

    genome = make_genome()

    config = EvaluationConfig(
        episode_seconds=0.1,
    )

    result = evaluate_genome(
        genome,
        config,
    )

    path = (
        tmp_path
        / "champion.npz"
    )

    save_genome_checkpoint(
        path,
        genome,
        generation=5,
        fitness_result=result,
        evaluation_config=config,
    )

    loaded = (
        load_genome_checkpoint(
            path
        )
    )

    assert (
        loaded.fitness_result
        is not None
    )

    assert (
        loaded.fitness_result[
            "fitness"
        ]
        == pytest.approx(
            result.fitness
        )
    )


def test_loaded_genome_reproduces_same_result(
    tmp_path,
):

    genome = make_genome()

    config = EvaluationConfig(
        episode_seconds=0.5,
    )

    original_result = (
        evaluate_genome(
            genome,
            config,
        )
    )

    path = (
        tmp_path
        / "champion.npz"
    )

    save_genome_checkpoint(
        path,
        genome,
        evaluation_config=config,
    )

    loaded = (
        load_genome_checkpoint(
            path
        )
    )

    loaded_config = (
        evaluation_config_from_checkpoint(
            loaded
        )
    )

    loaded_result = (
        evaluate_genome(
            loaded.genome,
            loaded_config,
        )
    )

    assert (
        loaded_result.fitness
        == pytest.approx(
            original_result.fitness
        )
    )

    assert (
        loaded_result.final_x
        == pytest.approx(
            original_result.final_x
        )
    )


def test_missing_checkpoint_raises(
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError
    ):

        load_genome_checkpoint(
            tmp_path
            / "does_not_exist.npz"
        )