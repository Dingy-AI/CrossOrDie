from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


# ============================================================
# Gene definitions
# ============================================================

class BodyGene(IntEnum):
    LENGTH = 0
    HEIGHT = 1
    DENSITY = 2
    BALANCE = 3
    FRICTION = 4


class LegGene(IntEnum):
    ATTACHMENT = 0
    UPPER_LENGTH = 1
    LOWER_LENGTH = 2
    THICKNESS = 3
    FOOT_LENGTH = 4
    HIP_STRENGTH = 5
    KNEE_STRENGTH = 6


class JointGene(IntEnum):
    FRONT_HIP_RANGE = 0
    FRONT_KNEE_RANGE = 1
    REAR_HIP_RANGE = 2
    REAR_KNEE_RANGE = 3


# Derive counts directly from the gene definitions so they
# cannot accidentally get out of sync later.
BODY_GENE_COUNT = len(BodyGene)
FRONT_LEG_GENE_COUNT = len(LegGene)
REAR_LEG_GENE_COUNT = len(LegGene)
JOINT_GENE_COUNT = len(JointGene)

TOTAL_MORPHOLOGY_GENES = (
    BODY_GENE_COUNT
    + FRONT_LEG_GENE_COUNT
    + REAR_LEG_GENE_COUNT
    + JOINT_GENE_COUNT
)


# ============================================================
# Helpers
# ============================================================

def _random_genes(count: int) -> np.ndarray:
    """
    Generate a chromosome with normalized genes in [0, 1].

    Used only for direct Genome() construction.
    For deterministic generation, prefer Genome.random(rng).
    """
    return np.random.random(count).astype(np.float32)


def _validate_chromosome(
    chromosome: np.ndarray,
    expected_size: int,
    name: str,
) -> np.ndarray:
    """
    Convert chromosome to float32 and validate its contents.

    All morphology genes are normalized to [0, 1].
    """
    chromosome = np.asarray(
        chromosome,
        dtype=np.float32,
    )

    if chromosome.shape != (expected_size,):
        raise ValueError(
            f"{name} chromosome must have shape "
            f"({expected_size},), got {chromosome.shape}"
        )

    if not np.all(np.isfinite(chromosome)):
        raise ValueError(
            f"{name} chromosome contains non-finite values."
        )

    if np.any(chromosome < 0.0) or np.any(chromosome > 1.0):
        raise ValueError(
            f"{name} chromosome genes must be within [0, 1]."
        )

    return chromosome


# ============================================================
# Genome
# ============================================================

@dataclass(slots=True)
class Genome:
    """
    Genetic representation of a CrossOrDie creature.

    Morphology genes are normalized floats in [0, 1].

    Physical values such as leg length, body mass, motor
    strength, etc. are derived from these genes by the
    creature/body builder.

    The genome does NOT directly store physical units.
    """

    body: np.ndarray = field(
        default_factory=lambda: _random_genes(
            BODY_GENE_COUNT
        )
    )

    front_leg: np.ndarray = field(
        default_factory=lambda: _random_genes(
            FRONT_LEG_GENE_COUNT
        )
    )

    rear_leg: np.ndarray = field(
        default_factory=lambda: _random_genes(
            REAR_LEG_GENE_COUNT
        )
    )

    joints: np.ndarray = field(
        default_factory=lambda: _random_genes(
            JOINT_GENE_COUNT
        )
    )

    # Neural controller weights will be added once the
    # controller architecture is defined.
    #
    # The brain will eventually be stored as one flattened
    # vector so crossover/mutation can operate on it easily.
    brain: np.ndarray | None = None

    def __post_init__(self):
        """
        Validate all chromosomes whenever a Genome is created.
        """

        self.body = _validate_chromosome(
            self.body,
            BODY_GENE_COUNT,
            "body",
        )

        self.front_leg = _validate_chromosome(
            self.front_leg,
            FRONT_LEG_GENE_COUNT,
            "front_leg",
        )

        self.rear_leg = _validate_chromosome(
            self.rear_leg,
            REAR_LEG_GENE_COUNT,
            "rear_leg",
        )

        self.joints = _validate_chromosome(
            self.joints,
            JOINT_GENE_COUNT,
            "joints",
        )

        if self.brain is not None:
            self.brain = np.asarray(
                self.brain,
                dtype=np.float32,
            )

            if self.brain.ndim != 1:
                raise ValueError(
                    "brain chromosome must be a flat 1D array."
                )

            if not np.all(np.isfinite(self.brain)):
                raise ValueError(
                    "brain chromosome contains non-finite values."
                )

    # ========================================================
    # Construction
    # ========================================================

    @classmethod
    def random(
        cls,
        rng: np.random.Generator | None = None,
    ) -> "Genome":
        """
        Create a completely random morphology.

        Passing a Generator allows deterministic/reproducible
        populations.
        """

        if rng is None:
            rng = np.random.default_rng()

        return cls(
            body=rng.random(
                BODY_GENE_COUNT,
                dtype=np.float32,
            ),
            front_leg=rng.random(
                FRONT_LEG_GENE_COUNT,
                dtype=np.float32,
            ),
            rear_leg=rng.random(
                REAR_LEG_GENE_COUNT,
                dtype=np.float32,
            ),
            joints=rng.random(
                JOINT_GENE_COUNT,
                dtype=np.float32,
            ),
        )

    @classmethod
    def from_flat_morphology(
        cls,
        genes: np.ndarray,
        brain: np.ndarray | None = None,
    ) -> "Genome":
        """
        Reconstruct a Genome from its flattened morphology.

        Useful later for:
            - crossover
            - mutation
            - saving/loading populations
            - analysis
        """

        genes = np.asarray(
            genes,
            dtype=np.float32,
        )

        if genes.shape != (TOTAL_MORPHOLOGY_GENES,):
            raise ValueError(
                "Flat morphology must contain exactly "
                f"{TOTAL_MORPHOLOGY_GENES} genes, "
                f"got {genes.shape}."
            )

        index = 0

        body = genes[
            index:index + BODY_GENE_COUNT
        ].copy()

        index += BODY_GENE_COUNT

        front_leg = genes[
            index:index + FRONT_LEG_GENE_COUNT
        ].copy()

        index += FRONT_LEG_GENE_COUNT

        rear_leg = genes[
            index:index + REAR_LEG_GENE_COUNT
        ].copy()

        index += REAR_LEG_GENE_COUNT

        joints = genes[
            index:index + JOINT_GENE_COUNT
        ].copy()

        return cls(
            body=body,
            front_leg=front_leg,
            rear_leg=rear_leg,
            joints=joints,
            brain=None if brain is None else brain.copy(),
        )

    # ========================================================
    # Access
    # ========================================================

    def copy(self) -> "Genome":
        """
        Deep copy the genome.
        """

        return Genome(
            body=self.body.copy(),
            front_leg=self.front_leg.copy(),
            rear_leg=self.rear_leg.copy(),
            joints=self.joints.copy(),
            brain=(
                None
                if self.brain is None
                else self.brain.copy()
            ),
        )

    @property
    def chromosomes(self) -> dict[str, np.ndarray]:
        """
        Named morphology chromosomes.

        Primarily useful for mutation, crossover and debugging.
        """

        return {
            "body": self.body,
            "front_leg": self.front_leg,
            "rear_leg": self.rear_leg,
            "joints": self.joints,
        }

    def flatten_morphology(self) -> np.ndarray:
        """
        Return all morphology genes as one 1D array.

        Ordering:
            body
            front leg
            rear leg
            joints
        """

        return np.concatenate(
            [
                self.body,
                self.front_leg,
                self.rear_leg,
                self.joints,
            ]
        ).astype(
            np.float32,
            copy=False,
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    @property
    def morphology_gene_count(self) -> int:
        return TOTAL_MORPHOLOGY_GENES

    @property
    def total_gene_count(self) -> int:
        brain_count = (
            0
            if self.brain is None
            else self.brain.size
        )

        return (
            TOTAL_MORPHOLOGY_GENES
            + brain_count
        )