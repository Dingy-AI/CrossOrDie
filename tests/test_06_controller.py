import numpy as np
import pytest

from creatures.controller import (
    NeuralController,
    random_brain,
    with_random_brain,
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    BRAIN_GENE_COUNT,
    INPUT_HIDDEN_WEIGHT_COUNT,
    HIDDEN_BIAS_COUNT,
    HIDDEN_OUTPUT_WEIGHT_COUNT,
    OUTPUT_BIAS_COUNT,
)
from evolution.genome import Genome


def test_brain_gene_count():

    expected = (
        INPUT_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + HIDDEN_SIZE * OUTPUT_SIZE
        + OUTPUT_SIZE
    )

    assert BRAIN_GENE_COUNT == expected
    assert BRAIN_GENE_COUNT == 180


def test_parameter_counts():

    assert (
        INPUT_HIDDEN_WEIGHT_COUNT
        == 17 * 8
    )

    assert HIDDEN_BIAS_COUNT == 8

    assert (
        HIDDEN_OUTPUT_WEIGHT_COUNT
        == 8 * 4
    )

    assert OUTPUT_BIAS_COUNT == 4


def test_random_brain_has_correct_shape():

    brain = random_brain(
        np.random.default_rng(42)
    )

    assert brain.shape == (
        BRAIN_GENE_COUNT,
    )


def test_random_brain_is_float32():

    brain = random_brain(
        np.random.default_rng(42)
    )

    assert brain.dtype == np.float32


def test_random_brain_is_finite():

    brain = random_brain(
        np.random.default_rng(42)
    )

    assert np.all(
        np.isfinite(brain)
    )


def test_random_brain_is_deterministic():

    brain_a = random_brain(
        np.random.default_rng(42)
    )

    brain_b = random_brain(
        np.random.default_rng(42)
    )

    np.testing.assert_array_equal(
        brain_a,
        brain_b,
    )


def test_with_random_brain_adds_brain():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    assert genome.brain is None

    result = with_random_brain(
        genome,
        np.random.default_rng(123),
    )

    assert result.brain is not None

    assert result.brain.shape == (
        BRAIN_GENE_COUNT,
    )


def test_with_random_brain_does_not_modify_original():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    result = with_random_brain(
        genome,
        np.random.default_rng(123),
    )

    assert genome.brain is None
    assert result.brain is not None


def test_existing_brain_is_preserved():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    genome = with_random_brain(
        genome,
        np.random.default_rng(1),
    )

    original_brain = (
        genome.brain.copy()
    )

    result = with_random_brain(
        genome,
        np.random.default_rng(2),
        overwrite=False,
    )

    np.testing.assert_array_equal(
        result.brain,
        original_brain,
    )


def test_overwrite_replaces_brain():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    genome = with_random_brain(
        genome,
        np.random.default_rng(1),
    )

    original_brain = (
        genome.brain.copy()
    )

    result = with_random_brain(
        genome,
        np.random.default_rng(2),
        overwrite=True,
    )

    assert not np.array_equal(
        result.brain,
        original_brain,
    )


def test_controller_parameter_shapes():

    brain = random_brain(
        np.random.default_rng(42)
    )

    controller = NeuralController(
        brain
    )

    assert controller.w1.shape == (
        HIDDEN_SIZE,
        INPUT_SIZE,
    )

    assert controller.b1.shape == (
        HIDDEN_SIZE,
    )

    assert controller.w2.shape == (
        OUTPUT_SIZE,
        HIDDEN_SIZE,
    )

    assert controller.b2.shape == (
        OUTPUT_SIZE,
    )


def test_controller_output_shape():

    brain = random_brain(
        np.random.default_rng(42)
    )

    controller = NeuralController(
        brain
    )

    observation = np.zeros(
        INPUT_SIZE,
        dtype=np.float32,
    )

    output = controller(
        observation
    )

    assert output.shape == (
        OUTPUT_SIZE,
    )


def test_controller_output_is_float32():

    controller = NeuralController(
        random_brain(
            np.random.default_rng(42)
        )
    )

    observation = np.zeros(
        INPUT_SIZE,
        dtype=np.float32,
    )

    output = controller(
        observation
    )

    assert output.dtype == np.float32


def test_controller_output_is_bounded():

    controller = NeuralController(
        random_brain(
            np.random.default_rng(42)
        )
    )

    observation = np.ones(
        INPUT_SIZE,
        dtype=np.float32,
    ) * 1000.0

    output = controller(
        observation
    )

    assert np.all(
        output >= -1.0
    )

    assert np.all(
        output <= 1.0
    )


def test_controller_is_deterministic():

    brain = random_brain(
        np.random.default_rng(42)
    )

    controller = NeuralController(
        brain
    )

    observation = np.linspace(
        -1.0,
        1.0,
        INPUT_SIZE,
        dtype=np.float32,
    )

    output_a = controller(
        observation
    )

    output_b = controller(
        observation
    )

    np.testing.assert_array_equal(
        output_a,
        output_b,
    )


def test_zero_brain_produces_zero_output():

    brain = np.zeros(
        BRAIN_GENE_COUNT,
        dtype=np.float32,
    )

    controller = NeuralController(
        brain
    )

    observation = np.ones(
        INPUT_SIZE,
        dtype=np.float32,
    )

    output = controller(
        observation
    )

    np.testing.assert_allclose(
        output,
        0.0,
    )


def test_invalid_brain_size_raises():

    brain = np.zeros(
        BRAIN_GENE_COUNT - 1,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):

        NeuralController(
            brain
        )


def test_nonfinite_brain_raises():

    brain = np.zeros(
        BRAIN_GENE_COUNT,
        dtype=np.float32,
    )

    brain[5] = np.nan

    with pytest.raises(ValueError):

        NeuralController(
            brain
        )


def test_invalid_observation_size_raises():

    controller = NeuralController(
        random_brain(
            np.random.default_rng(42)
        )
    )

    observation = np.zeros(
        INPUT_SIZE - 1,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):

        controller(
            observation
        )


def test_nonfinite_observation_raises():

    controller = NeuralController(
        random_brain(
            np.random.default_rng(42)
        )
    )

    observation = np.zeros(
        INPUT_SIZE,
        dtype=np.float32,
    )

    observation[0] = np.nan

    with pytest.raises(ValueError):

        controller(
            observation
        )


def test_controller_from_genome():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    genome = with_random_brain(
        genome,
        np.random.default_rng(123),
    )

    controller = (
        NeuralController.from_genome(
            genome
        )
    )

    assert controller.brain.shape == (
        BRAIN_GENE_COUNT,
    )


def test_controller_from_brainless_genome_raises():

    genome = Genome.random(
        np.random.default_rng(42)
    )

    with pytest.raises(ValueError):

        NeuralController.from_genome(
            genome
        )