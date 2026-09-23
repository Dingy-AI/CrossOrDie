from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from evolution.genetics import MutationConfig

from evolution.population import (
    Population,
    PopulationConfig,
)


from evolution.evaluator import (
    EvaluationConfig,
    FitnessResult,
)
from evolution.genome import Genome
from evolution.population import (
    GenerationReport,
    GenerationStats,
)


# ============================================================
# Checkpoint format
# ============================================================

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """
    Data loaded from a saved CrossOrDie checkpoint.
    """

    genome: Genome

    format_version: int

    generation: int | None

    fitness_result: dict | None

    generation_stats: dict | None

    evaluation_config: dict | None


# ============================================================
# Save
# ============================================================

def save_genome_checkpoint(
    path: str | Path,
    genome: Genome,
    *,
    generation: int | None = None,
    fitness_result: FitnessResult | None = None,
    generation_stats: GenerationStats | None = None,
    evaluation_config: EvaluationConfig | None = None,
):
    """
    Save one genome and its associated metadata.

    The checkpoint stores:

        morphology chromosomes
        brain chromosome
        generation
        fitness result
        generation statistics
        evaluation configuration
        checkpoint format version

    Files use NumPy's compressed NPZ format.
    """

    path = Path(path)

    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "format_version":
            CHECKPOINT_FORMAT_VERSION,

        "generation":
            generation,

        "fitness_result":
            (
                None
                if fitness_result is None
                else asdict(fitness_result)
            ),

        "generation_stats":
            (
                None
                if generation_stats is None
                else asdict(generation_stats)
            ),

        "evaluation_config":
            (
                None
                if evaluation_config is None
                else asdict(evaluation_config)
            ),
    }

    metadata_json = json.dumps(
        metadata
    )

    # Brain is optional at the Genome level, although trained
    # champions should always have one.
    if genome.brain is None:

        has_brain = False

        brain = np.empty(
            0,
            dtype=np.float32,
        )

    else:

        has_brain = True

        brain = genome.brain.astype(
            np.float32,
            copy=True,
        )

    # --------------------------------------------------------
    # Atomic-ish save
    # --------------------------------------------------------
    #
    # Write to a temporary file first, then replace the final
    # path. This reduces the chance of leaving behind a broken
    # checkpoint if writing is interrupted.

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open("wb") as file:

        np.savez_compressed(
            file,

            body=genome.body,

            front_leg=genome.front_leg,

            rear_leg=genome.rear_leg,

            joints=genome.joints,

            brain=brain,

            has_brain=np.array(
                has_brain,
                dtype=np.bool_,
            ),

            metadata=np.array(
                metadata_json
            ),
        )

    temp_path.replace(
        path
    )


# ============================================================
# Save generation champion
# ============================================================

def save_generation_champion(
    path: str | Path,
    report: GenerationReport,
    evaluation_config: EvaluationConfig,
):
    """
    Convenience wrapper for saving the champion contained in a
    GenerationReport.
    """

    save_genome_checkpoint(
        path=path,

        genome=report.best_genome,

        generation=(
            report.stats.generation
        ),

        fitness_result=(
            report.best_result
        ),

        generation_stats=(
            report.stats
        ),

        evaluation_config=(
            evaluation_config
        ),
    )


# ============================================================
# Load
# ============================================================

def load_genome_checkpoint(
    path: str | Path,
) -> LoadedCheckpoint:
    """
    Load a saved CrossOrDie genome checkpoint.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "body",
            "front_leg",
            "rear_leg",
            "joints",
            "brain",
            "has_brain",
            "metadata",
        }

        missing = (
            required
            - set(data.files)
        )

        if missing:

            raise ValueError(
                "Checkpoint is missing fields: "
                f"{sorted(missing)}"
            )

        metadata = json.loads(
            str(
                data["metadata"].item()
            )
        )

        format_version = metadata.get(
            "format_version"
        )

        if format_version != (
            CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported checkpoint format version: "
                f"{format_version}. "
                f"Expected "
                f"{CHECKPOINT_FORMAT_VERSION}."
            )

        has_brain = bool(
            data["has_brain"].item()
        )

        brain = (
            data["brain"].copy()
            if has_brain
            else None
        )

        genome = Genome(
            body=data["body"].copy(),

            front_leg=(
                data["front_leg"].copy()
            ),

            rear_leg=(
                data["rear_leg"].copy()
            ),

            joints=(
                data["joints"].copy()
            ),

            brain=brain,
        )

    return LoadedCheckpoint(
        genome=genome,

        format_version=(
            format_version
        ),

        generation=metadata.get(
            "generation"
        ),

        fitness_result=metadata.get(
            "fitness_result"
        ),

        generation_stats=metadata.get(
            "generation_stats"
        ),

        evaluation_config=metadata.get(
            "evaluation_config"
        ),
    )


# ============================================================
# Evaluation config reconstruction
# ============================================================

def evaluation_config_from_checkpoint(
    checkpoint: LoadedCheckpoint,
) -> EvaluationConfig:
    """
    Reconstruct the exact EvaluationConfig used when the
    checkpoint was created.
    """

    if checkpoint.evaluation_config is None:

        raise ValueError(
            "Checkpoint does not contain an "
            "evaluation configuration."
        )

    return EvaluationConfig(
        **checkpoint.evaluation_config
    )

# ============================================================
# Population checkpoint format
# ============================================================

POPULATION_CHECKPOINT_FORMAT_VERSION = 1


def _json_default(value):
    """
    Convert NumPy values into JSON-compatible values.

    Mainly used for saving NumPy RNG state.
    """

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


# ============================================================
# Save full population
# ============================================================

def save_population_checkpoint(
    path: str | Path,
    population: Population,
):
    """
    Save an entire evolutionary population.

    This preserves:

        every genome
        morphology
        brain weights
        current generation
        population configuration
        evaluation configuration
        mutation configuration
        RNG state

    Loading this file allows evolution to continue from the
    exact saved population rather than restarting from one
    champion.
    """

    path = Path(path)

    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    genomes = population.genomes

    if not genomes:
        raise ValueError(
            "Cannot save an empty population."
        )

    # --------------------------------------------------------
    # Stack chromosomes
    # --------------------------------------------------------

    body = np.stack(
        [
            genome.body
            for genome in genomes
        ]
    ).astype(
        np.float32
    )

    front_leg = np.stack(
        [
            genome.front_leg
            for genome in genomes
        ]
    ).astype(
        np.float32
    )

    rear_leg = np.stack(
        [
            genome.rear_leg
            for genome in genomes
        ]
    ).astype(
        np.float32
    )

    joints = np.stack(
        [
            genome.joints
            for genome in genomes
        ]
    ).astype(
        np.float32
    )

    # A Population requires every genome to have a brain.
    for index, genome in enumerate(genomes):

        if genome.brain is None:
            raise ValueError(
                f"Genome {index} does not contain "
                "a brain chromosome."
            )

    brain = np.stack(
        [
            genome.brain
            for genome in genomes
        ]
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "checkpoint_type":
            "population",

        "format_version":
            POPULATION_CHECKPOINT_FORMAT_VERSION,

        # IMPORTANT:
        #
        # population.generation is the generation represented
        # by population.genomes.
        "generation":
            population.generation,

        "population_config":
            asdict(
                population.population_config
            ),

        "evaluation_config":
            asdict(
                population.evaluation_config
            ),

        "mutation_config":
            asdict(
                population.mutation_config
            ),

        # This lets us resume random selection/crossover from
        # the same RNG state.
        "rng_state":
            population.rng.bit_generator.state,
    }

    metadata_json = json.dumps(
        metadata,
        default=_json_default,
    )

    # --------------------------------------------------------
    # Safer temporary write
    # --------------------------------------------------------

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open("wb") as file:

        np.savez_compressed(
            file,

            body=body,
            front_leg=front_leg,
            rear_leg=rear_leg,
            joints=joints,
            brain=brain,

            metadata=np.array(
                metadata_json
            ),
        )

    temp_path.replace(
        path
    )


# ============================================================
# Load full population
# ============================================================

def load_population_checkpoint(
    path: str | Path,
) -> Population:
    """
    Restore a complete Population from disk.

    Evolution can continue directly from the loaded object.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Population checkpoint does not exist: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "body",
            "front_leg",
            "rear_leg",
            "joints",
            "brain",
            "metadata",
        }

        missing = (
            required
            - set(data.files)
        )

        if missing:
            raise ValueError(
                "Population checkpoint is missing fields: "
                f"{sorted(missing)}"
            )

        metadata = json.loads(
            str(
                data["metadata"].item()
            )
        )

        if metadata.get(
            "checkpoint_type"
        ) != "population":

            raise ValueError(
                "File is not a population checkpoint."
            )

        version = metadata.get(
            "format_version"
        )

        if version != (
            POPULATION_CHECKPOINT_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported population checkpoint "
                f"version: {version}. "
                "Expected "
                f"{POPULATION_CHECKPOINT_FORMAT_VERSION}."
            )

        body = data[
            "body"
        ].copy()

        front_leg = data[
            "front_leg"
        ].copy()

        rear_leg = data[
            "rear_leg"
        ].copy()

        joints = data[
            "joints"
        ].copy()

        brain = data[
            "brain"
        ].copy()

    # --------------------------------------------------------
    # Validate population dimensions
    # --------------------------------------------------------

    population_size = body.shape[0]

    if not (
        front_leg.shape[0]
        == population_size
        == rear_leg.shape[0]
        == joints.shape[0]
        == brain.shape[0]
    ):
        raise ValueError(
            "Population chromosome arrays have "
            "different population sizes."
        )

    # --------------------------------------------------------
    # Reconstruct genomes
    # --------------------------------------------------------

    genomes = []

    for index in range(
        population_size
    ):

        genomes.append(
            Genome(
                body=body[index],
                front_leg=front_leg[index],
                rear_leg=rear_leg[index],
                joints=joints[index],
                brain=brain[index],
            )
        )

    # --------------------------------------------------------
    # Reconstruct configuration
    # --------------------------------------------------------

    population_config = PopulationConfig(
        **metadata[
            "population_config"
        ]
    )

    evaluation_config = EvaluationConfig(
        **metadata[
            "evaluation_config"
        ]
    )

    mutation_config = MutationConfig(
        **metadata[
            "mutation_config"
        ]
    )

    if (
        population_config.population_size
        != population_size
    ):
        raise ValueError(
            "Saved population size does not match "
            "PopulationConfig."
        )

    # --------------------------------------------------------
    # Restore RNG
    # --------------------------------------------------------

    rng = np.random.default_rng()

    rng_state = metadata[
        "rng_state"
    ]

    expected_bit_generator = (
        rng.bit_generator.state[
            "bit_generator"
        ]
    )

    saved_bit_generator = (
        rng_state[
            "bit_generator"
        ]
    )

    if (
        saved_bit_generator
        != expected_bit_generator
    ):
        raise ValueError(
            "Saved RNG uses "
            f"{saved_bit_generator}, but this version "
            f"expects {expected_bit_generator}."
        )

    rng.bit_generator.state = (
        rng_state
    )

    # --------------------------------------------------------
    # Restore Population
    # --------------------------------------------------------

    return Population(
        genomes=genomes,
        rng=rng,

        population_config=(
            population_config
        ),

        evaluation_config=(
            evaluation_config
        ),

        mutation_config=(
            mutation_config
        ),

        generation=int(
            metadata[
                "generation"
            ]
        ),
    )