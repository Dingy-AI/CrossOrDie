import numpy as np
import pytest

from creatures.controller import (
    BRAIN_GENE_COUNT,
    with_random_brain,
)
from evolution.evaluator import (
    EvaluationConfig,
    FitnessResult,
    evaluate_genome,
)
from evolution.genome import Genome


# ============================================================
# Helpers
# ============================================================

def make_genome(
    morphology_seed=42,
    brain_seed=123,
):

    genome = Genome.random(
        np.random.default_rng(
            morphology_seed
        )
    )

    genome = with_random_brain(
        genome,
        np.random.default_rng(
            brain_seed
        ),
    )

    return genome


# ============================================================
# Basic evaluation
# ============================================================

def test_evaluate_returns_fitness_result():

    genome = make_genome()

    result = evaluate_genome(
        genome,
        EvaluationConfig(
            episode_seconds=0.5,
        ),
    )

    assert isinstance(
        result,
        FitnessResult,
    )


def test_fitness_is_nonnegative():

    genome = make_genome()

    result = evaluate_genome(
        genome,
        EvaluationConfig(
            episode_seconds=0.5,
        ),
    )

    assert result.fitness >= 0.0
    assert result.max_progress >= 0.0


def test_result_values_are_finite():

    genome = make_genome()

    result = evaluate_genome(
        genome,
        EvaluationConfig(
            episode_seconds=0.5,
        ),
    )

    values = [
        result.fitness,
        result.max_progress,
        result.max_x,
        result.final_x,
        result.final_y,
        result.simulated_seconds,
    ]

    assert np.all(
        np.isfinite(values)
    )


# ============================================================
# Determinism
# ============================================================

def test_evaluation_is_deterministic():

    genome = make_genome()

    config = EvaluationConfig(
        episode_seconds=1.0,
    )

    result_a = evaluate_genome(
        genome,
        config,
    )

    result_b = evaluate_genome(
        genome,
        config,
    )

    assert result_a.fitness == pytest.approx(
        result_b.fitness
    )

    assert result_a.max_progress == pytest.approx(
        result_b.max_progress
    )

    assert result_a.final_x == pytest.approx(
        result_b.final_x
    )

    assert result_a.final_y == pytest.approx(
        result_b.final_y
    )


# ============================================================
# Genome safety
# ============================================================

def test_evaluation_does_not_modify_genome():

    genome = make_genome()

    morphology_before = (
        genome.flatten_morphology().copy()
    )

    brain_before = (
        genome.brain.copy()
    )

    evaluate_genome(
        genome,
        EvaluationConfig(
            episode_seconds=0.5,
        ),
    )

    np.testing.assert_array_equal(
        genome.flatten_morphology(),
        morphology_before,
    )

    np.testing.assert_array_equal(
        genome.brain,
        brain_before,
    )


def test_brainless_genome_raises():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    with pytest.raises(ValueError):

        evaluate_genome(
            genome
        )


# ============================================================
# Timing
# ============================================================

def test_episode_respects_time_limit():

    genome = make_genome()

    config = EvaluationConfig(
        physics_hz=120,
        control_hz=30,
        episode_seconds=0.5,
        death_y=10_000.0,
        goal_x=100_000.0,
    )

    result = evaluate_genome(
        genome,
        config,
    )

    assert result.physics_steps == 60

    assert result.control_steps == 15

    assert result.simulated_seconds == pytest.approx(
        0.5
    )

    assert result.termination_reason == (
        "time_limit"
    )


def test_controller_frequency():

    genome = make_genome()

    config = EvaluationConfig(
        physics_hz=120,
        control_hz=30,
        episode_seconds=1.0,
        death_y=10_000.0,
        goal_x=100_000.0,
    )

    result = evaluate_genome(
        genome,
        config,
    )

    assert result.physics_steps == 120
    assert result.control_steps == 30


# ============================================================
# Configuration
# ============================================================

def test_invalid_frequency_ratio_raises():

    with pytest.raises(ValueError):

        EvaluationConfig(
            physics_hz=120,
            control_hz=29,
        )


def test_invalid_goal_raises():

    with pytest.raises(ValueError):

        EvaluationConfig(
            spawn_x=500.0,
            goal_x=400.0,
        )


# ============================================================
# Brain shape sanity
# ============================================================

def test_test_genome_has_correct_brain_size():

    genome = make_genome()

    assert genome.brain.shape == (
        BRAIN_GENE_COUNT,
    )