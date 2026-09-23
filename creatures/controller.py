import math

import numpy as np

from creatures.body import MOTOR_COUNT
from creatures.observation import OBSERVATION_SIZE
from evolution.genome import Genome


# ============================================================
# Neural architecture
# ============================================================

INPUT_SIZE = OBSERVATION_SIZE
HIDDEN_SIZE = 8
OUTPUT_SIZE = MOTOR_COUNT


# ------------------------------------------------------------
# Parameter counts
# ------------------------------------------------------------

INPUT_HIDDEN_WEIGHT_COUNT = (
    INPUT_SIZE * HIDDEN_SIZE
)

HIDDEN_BIAS_COUNT = HIDDEN_SIZE

HIDDEN_OUTPUT_WEIGHT_COUNT = (
    HIDDEN_SIZE * OUTPUT_SIZE
)

OUTPUT_BIAS_COUNT = OUTPUT_SIZE


BRAIN_GENE_COUNT = (
    INPUT_HIDDEN_WEIGHT_COUNT
    + HIDDEN_BIAS_COUNT
    + HIDDEN_OUTPUT_WEIGHT_COUNT
    + OUTPUT_BIAS_COUNT
)


# ============================================================
# Parameter layout
# ============================================================
#
# Flattened brain chromosome:
#
# [
#     W1,
#     b1,
#     W2,
#     b2,
# ]
#
# W1 shape:
#     (8, 17)
#
# b1 shape:
#     (8,)
#
# W2 shape:
#     (4, 8)
#
# b2 shape:
#     (4,)
#


W1_START = 0
W1_END = (
    W1_START
    + INPUT_HIDDEN_WEIGHT_COUNT
)

B1_START = W1_END
B1_END = (
    B1_START
    + HIDDEN_BIAS_COUNT
)

W2_START = B1_END
W2_END = (
    W2_START
    + HIDDEN_OUTPUT_WEIGHT_COUNT
)

B2_START = W2_END
B2_END = (
    B2_START
    + OUTPUT_BIAS_COUNT
)


assert B2_END == BRAIN_GENE_COUNT


# ============================================================
# Brain initialization
# ============================================================

def random_brain(
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a random neural-controller chromosome.

    Xavier-style initialization is used for the network
    weights so the initial network is not immediately
    saturated.

    Biases begin near zero.

    Returns
    -------
    np.ndarray
        Flat float32 vector with BRAIN_GENE_COUNT parameters.
    """

    if rng is None:
        rng = np.random.default_rng()

    # --------------------------------------------------------
    # Input -> hidden
    # --------------------------------------------------------

    w1_limit = math.sqrt(
        6.0
        / (
            INPUT_SIZE
            + HIDDEN_SIZE
        )
    )

    w1 = rng.uniform(
        -w1_limit,
        w1_limit,
        size=(
            HIDDEN_SIZE,
            INPUT_SIZE,
        ),
    ).astype(
        np.float32
    )

    # Small random biases rather than exact zero so initial
    # individuals are not excessively symmetric.
    b1 = rng.uniform(
        -0.05,
        0.05,
        size=(
            HIDDEN_SIZE,
        ),
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Hidden -> output
    # --------------------------------------------------------

    w2_limit = math.sqrt(
        6.0
        / (
            HIDDEN_SIZE
            + OUTPUT_SIZE
        )
    )

    w2 = rng.uniform(
        -w2_limit,
        w2_limit,
        size=(
            OUTPUT_SIZE,
            HIDDEN_SIZE,
        ),
    ).astype(
        np.float32
    )

    b2 = rng.uniform(
        -0.05,
        0.05,
        size=(
            OUTPUT_SIZE,
        ),
    ).astype(
        np.float32
    )

    brain = np.concatenate(
        [
            w1.flatten(),
            b1,
            w2.flatten(),
            b2,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    return brain


def with_random_brain(
    genome: Genome,
    rng: np.random.Generator | None = None,
    overwrite: bool = False,
) -> Genome:
    """
    Return a copy of a Genome with a neural controller.

    The supplied genome itself is never modified.

    If the genome already contains a brain, it is preserved
    unless overwrite=True.
    """

    result = genome.copy()

    if (
        result.brain is None
        or overwrite
    ):
        result.brain = random_brain(
            rng
        )

    return result


# ============================================================
# Neural controller
# ============================================================

class NeuralController:
    """
    Small feed-forward neural controller.

    Architecture:

        17 observations
                ↓
        8 tanh hidden neurons
                ↓
        4 tanh motor outputs

    Motor outputs are always in [-1, 1].

    Output order matches CreatureBody:

        0 = front hip
        1 = front knee
        2 = rear hip
        3 = rear knee

    The controller is stateless.

    Its behavior is entirely determined by:
        - current observation
        - brain chromosome
    """

    def __init__(
        self,
        brain: np.ndarray,
    ):
        brain = np.asarray(
            brain,
            dtype=np.float32,
        )

        if brain.shape != (
            BRAIN_GENE_COUNT,
        ):
            raise ValueError(
                "Brain chromosome must have shape "
                f"({BRAIN_GENE_COUNT},), "
                f"got {brain.shape}."
            )

        if not np.all(
            np.isfinite(brain)
        ):
            raise ValueError(
                "Brain chromosome contains "
                "non-finite values."
            )

        # Store our own copy so external mutation of the
        # genome cannot silently alter an active controller.
        self.brain = brain.copy()

        self._decode_parameters()

    # ========================================================
    # Construction
    # ========================================================

    @classmethod
    def from_genome(
        cls,
        genome: Genome,
    ) -> "NeuralController":
        """
        Construct a controller from a Genome.
        """

        if genome.brain is None:
            raise ValueError(
                "Genome does not contain a brain chromosome. "
                "Use with_random_brain() when creating the "
                "initial population."
            )

        return cls(
            genome.brain
        )

    # ========================================================
    # Parameter decoding
    # ========================================================

    def _decode_parameters(self):
        """
        Convert the flat chromosome into matrix views used
        during inference.
        """

        self.w1 = (
            self.brain[
                W1_START:W1_END
            ]
            .reshape(
                HIDDEN_SIZE,
                INPUT_SIZE,
            )
        )

        self.b1 = self.brain[
            B1_START:B1_END
        ]

        self.w2 = (
            self.brain[
                W2_START:W2_END
            ]
            .reshape(
                OUTPUT_SIZE,
                HIDDEN_SIZE,
            )
        )

        self.b2 = self.brain[
            B2_START:B2_END
        ]

    # ========================================================
    # Inference
    # ========================================================

    def forward(
        self,
        observation,
    ) -> np.ndarray:
        """
        Convert one observation vector into four motor
        commands.

        Parameters
        ----------
        observation
            Array-like with shape (17,).

        Returns
        -------
        np.ndarray
            Float32 vector with shape (4,), bounded to [-1, 1].
        """

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (
            INPUT_SIZE,
        ):
            raise ValueError(
                "Observation must have shape "
                f"({INPUT_SIZE},), "
                f"got {observation.shape}."
            )

        if not np.all(
            np.isfinite(observation)
        ):
            raise ValueError(
                "Observation contains "
                "non-finite values."
            )

        # Observations should already be normalized by
        # observation.py, but clipping here gives us another
        # layer of protection.
        observation = np.clip(
            observation,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Hidden layer
        # ----------------------------------------------------

        hidden = np.tanh(
            self.w1 @ observation
            + self.b1
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        output = np.tanh(
            self.w2 @ hidden
            + self.b2
        )

        return output.astype(
            np.float32,
            copy=False,
        )

    def __call__(
        self,
        observation,
    ) -> np.ndarray:
        """
        Allows:

            action = controller(observation)

        instead of:

            action = controller.forward(observation)
        """

        return self.forward(
            observation
        )