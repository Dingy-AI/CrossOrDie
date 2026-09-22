from dataclasses import dataclass

import numpy as np

from evolution.genome import Genome


# ============================================================
# Genetics configuration
# ============================================================

@dataclass(frozen=True)
class MutationConfig:
    """
    Controls mutation behavior.

    Morphology genes:
        - Always remain within [0, 1].
        - Small mutations are common adjustments.
        - Large mutations are rare evolutionary jumps.

    Brain genes:
        - Are neural-network parameters rather than normalized
          morphology values.
        - They are therefore not clipped to [0, 1].
    """

    # Probability that any morphology gene mutates.
    morphology_mutation_rate: float = 0.10

    # Standard deviation for normal small mutations.
    morphology_small_sigma: float = 0.05

    # If a morphology gene mutates, probability that the
    # mutation is a large mutation instead of a small one.
    morphology_large_mutation_rate: float = 0.05

    # Standard deviation for rare large mutations.
    morphology_large_sigma: float = 0.25

    # Future neural-controller mutation settings.
    brain_mutation_rate: float = 0.05
    brain_sigma: float = 0.10

    def __post_init__(self):
        probabilities = {
            "morphology_mutation_rate":
                self.morphology_mutation_rate,

            "morphology_large_mutation_rate":
                self.morphology_large_mutation_rate,

            "brain_mutation_rate":
                self.brain_mutation_rate,
        }

        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be within [0, 1], "
                    f"got {value}."
                )

        if self.morphology_small_sigma < 0.0:
            raise ValueError(
                "morphology_small_sigma must be >= 0."
            )

        if self.morphology_large_sigma < 0.0:
            raise ValueError(
                "morphology_large_sigma must be >= 0."
            )

        if self.brain_sigma < 0.0:
            raise ValueError(
                "brain_sigma must be >= 0."
            )


DEFAULT_MUTATION_CONFIG = MutationConfig()


# ============================================================
# RNG helper
# ============================================================

def _get_rng(
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()

    return rng


# ============================================================
# Crossover
# ============================================================

def _crossover_chromosome(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Perform single-point crossover on one chromosome.

    Example:

        Parent A:
        [A A A A A A A]

        Parent B:
        [B B B B B B B]

                   crossover
                       ↓

        Child:
        [A A A | B B B B]

    Which parent contributes the beginning of the chromosome
    is randomized.
    """

    if parent_a.shape != parent_b.shape:
        raise ValueError(
            "Parent chromosomes must have the same shape. "
            f"Got {parent_a.shape} and {parent_b.shape}."
        )

    size = parent_a.size

    if size == 0:
        return parent_a.copy()

    if size == 1:
        if rng.random() < 0.5:
            return parent_a.copy()

        return parent_b.copy()

    # Crossover must occur between genes, not before the first
    # or after the final gene.
    crossover_point = int(
        rng.integers(
            1,
            size,
        )
    )

    # Randomize which parent contributes the first section.
    if rng.random() < 0.5:
        first = parent_a
        second = parent_b
    else:
        first = parent_b
        second = parent_a

    child = np.concatenate(
        [
            first[:crossover_point],
            second[crossover_point:],
        ]
    )

    return child.astype(
        np.float32,
        copy=False,
    )


def _crossover_brain(
    brain_a: np.ndarray | None,
    brain_b: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """
    Crossover for the future neural-network chromosome.

    If both parents have brains, they must use the same
    controller architecture and therefore have the same
    flattened weight-vector shape.
    """

    if brain_a is None and brain_b is None:
        return None

    if brain_a is None:
        return brain_b.copy()

    if brain_b is None:
        return brain_a.copy()

    if brain_a.shape != brain_b.shape:
        raise ValueError(
            "Parent brain chromosomes must have the same "
            f"shape. Got {brain_a.shape} and {brain_b.shape}."
        )

    return _crossover_chromosome(
        brain_a,
        brain_b,
        rng,
    )


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: np.random.Generator | None = None,
) -> Genome:
    """
    Produce a child by independently crossing each chromosome.

    Chromosomes:

        body
        front_leg
        rear_leg
        joints
        brain

    Each chromosome receives its own crossover point.

    This allows inheritance such as:

        body:
            mostly Parent A

        front legs:
            mostly Parent B

        rear legs:
            mixture

        joints:
            mixture
    """

    rng = _get_rng(rng)

    return Genome(
        body=_crossover_chromosome(
            parent_a.body,
            parent_b.body,
            rng,
        ),
        front_leg=_crossover_chromosome(
            parent_a.front_leg,
            parent_b.front_leg,
            rng,
        ),
        rear_leg=_crossover_chromosome(
            parent_a.rear_leg,
            parent_b.rear_leg,
            rng,
        ),
        joints=_crossover_chromosome(
            parent_a.joints,
            parent_b.joints,
            rng,
        ),
        brain=_crossover_brain(
            parent_a.brain,
            parent_b.brain,
            rng,
        ),
    )


# ============================================================
# Mutation
# ============================================================

def _mutate_morphology_chromosome(
    chromosome: np.ndarray,
    rng: np.random.Generator,
    config: MutationConfig,
) -> np.ndarray:
    """
    Mutate normalized morphology genes.

    Each gene independently has a chance to mutate.

    Most mutations:

        gene += Normal(0, small_sigma)

    Rare large mutations:

        gene += Normal(0, large_sigma)

    All results are clipped back into [0, 1].
    """

    result = chromosome.copy()

    for index in range(result.size):

        # Does this gene mutate at all?
        if (
            rng.random()
            >= config.morphology_mutation_rate
        ):
            continue

        # Determine mutation magnitude.
        large_mutation = (
            rng.random()
            < config.morphology_large_mutation_rate
        )

        if large_mutation:
            sigma = config.morphology_large_sigma
        else:
            sigma = config.morphology_small_sigma

        mutation = rng.normal(
            loc=0.0,
            scale=sigma,
        )

        result[index] += mutation

    # Morphology genes always remain normalized.
    np.clip(
        result,
        0.0,
        1.0,
        out=result,
    )

    return result.astype(
        np.float32,
        copy=False,
    )


def _mutate_brain(
    brain: np.ndarray | None,
    rng: np.random.Generator,
    config: MutationConfig,
) -> np.ndarray | None:
    """
    Mutate neural-network parameters.

    Brain values are NOT normalized and are therefore not
    clipped to [0, 1].
    """

    if brain is None:
        return None

    result = brain.copy()

    mutation_mask = (
        rng.random(result.shape)
        < config.brain_mutation_rate
    )

    mutation_count = int(
        mutation_mask.sum()
    )

    if mutation_count == 0:
        return result

    mutations = rng.normal(
        loc=0.0,
        scale=config.brain_sigma,
        size=mutation_count,
    ).astype(np.float32)

    result[mutation_mask] += mutations

    return result


def mutate(
    genome: Genome,
    rng: np.random.Generator | None = None,
    config: MutationConfig = DEFAULT_MUTATION_CONFIG,
) -> Genome:
    """
    Return a mutated COPY of a genome.

    The supplied genome itself is never modified.
    """

    rng = _get_rng(rng)

    return Genome(
        body=_mutate_morphology_chromosome(
            genome.body,
            rng,
            config,
        ),
        front_leg=_mutate_morphology_chromosome(
            genome.front_leg,
            rng,
            config,
        ),
        rear_leg=_mutate_morphology_chromosome(
            genome.rear_leg,
            rng,
            config,
        ),
        joints=_mutate_morphology_chromosome(
            genome.joints,
            rng,
            config,
        ),
        brain=_mutate_brain(
            genome.brain,
            rng,
            config,
        ),
    )


# ============================================================
# Reproduction
# ============================================================

def reproduce(
    parent_a: Genome,
    parent_b: Genome,
    rng: np.random.Generator | None = None,
    mutation_config: MutationConfig = DEFAULT_MUTATION_CONFIG,
) -> Genome:
    """
    Create one offspring:

        Parent A ─┐
                  ├── crossover
        Parent B ─┘
                        │
                        ▼
                    child genome
                        │
                        ▼
                     mutation
                        │
                        ▼
                    offspring
    """

    rng = _get_rng(rng)

    child = crossover(
        parent_a,
        parent_b,
        rng=rng,
    )

    child = mutate(
        child,
        rng=rng,
        config=mutation_config,
    )

    return child