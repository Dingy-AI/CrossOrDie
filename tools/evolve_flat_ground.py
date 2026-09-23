import argparse
import sys
from pathlib import Path


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports
# ============================================================

from evolution.checkpoint import (
    load_genome_checkpoint,
    load_population_checkpoint,
    save_generation_champion,
    save_population_checkpoint,
)

from evolution.evaluator import (
    EvaluationConfig,
)

from evolution.population import (
    Population,
    PopulationConfig,
)


# ============================================================
# Default configuration
# ============================================================

DEFAULT_GENERATIONS = 100

SAVE_EVERY = 5


DEFAULT_POPULATION_CONFIG = PopulationConfig(
    population_size=100,
    elite_count=5,
    parent_fraction=0.30,
    tournament_size=3,
)


DEFAULT_EVALUATION_CONFIG = EvaluationConfig(
    episode_seconds=10.0,
)


CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "flat_ground"
)


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evolve CrossOrDie creatures on flat ground."
        )
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Population checkpoint to resume from. "
            "If omitted, a new random population is created."
        ),
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help=(
            "Number of additional generations to run. "
            f"Default: {DEFAULT_GENERATIONS}"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for a NEW population. "
            "Ignored when --resume is used."
        ),
    )

    return parser.parse_args()


# ============================================================
# Population creation / loading
# ============================================================

def create_population(
    args,
) -> Population:

    # --------------------------------------------------------
    # Resume existing evolution
    # --------------------------------------------------------

    if args.resume is not None:

        population = (
            load_population_checkpoint(
                args.resume
            )
        )

        print()
        print(
            "Resuming population:"
        )

        print(
            f"  checkpoint: "
            f"{args.resume}"
        )

        print(
            f"  generation: "
            f"{population.generation}"
        )

        print(
            f"  population: "
            f"{len(population.genomes)}"
        )

        print(
            f"  episode:    "
            f"{population.evaluation_config.episode_seconds:.1f}s"
        )

        print()

        return population

    # --------------------------------------------------------
    # New population
    # --------------------------------------------------------

    population = Population.random(
        population_config=(
            DEFAULT_POPULATION_CONFIG
        ),
        evaluation_config=(
            DEFAULT_EVALUATION_CONFIG
        ),
        seed=args.seed,
    )

    print()
    print(
        "Creating new population:"
    )

    print(
        f"  seed:       {args.seed}"
    )

    print(
        f"  generation: {population.generation}"
    )

    print(
        f"  population: {len(population.genomes)}"
    )

    print()

    return population


# ============================================================
# Best-ever initialization
# ============================================================

def get_existing_best_fitness(
    resume: bool,
) -> float:
    """
    When resuming, preserve the historical best-ever value.

    For a fresh run we deliberately start from -inf so the
    new run establishes its own best value.
    """

    if not resume:
        return float(
            "-inf"
        )

    best_path = (
        CHECKPOINT_DIR
        / "best_ever.npz"
    )

    if not best_path.exists():
        return float(
            "-inf"
        )

    try:

        checkpoint = (
            load_genome_checkpoint(
                best_path
            )
        )

        if (
            checkpoint.fitness_result
            is None
        ):
            return float(
                "-inf"
            )

        return float(
            checkpoint.fitness_result[
                "fitness"
            ]
        )

    except Exception as error:

        print(
            "Warning: could not read existing "
            "best-ever checkpoint:"
        )

        print(
            f"  {error}"
        )

        return float(
            "-inf"
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.generations < 1:

        raise ValueError(
            "--generations must be at least 1."
        )

    # --------------------------------------------------------
    # Load / create population
    # --------------------------------------------------------

    population = create_population(
        args
    )

    # IMPORTANT:
    #
    # If we're resuming, this comes from the saved population.
    # We do NOT replace it with DEFAULT_EVALUATION_CONFIG.
    evaluation_config = (
        population.evaluation_config
    )

    # --------------------------------------------------------
    # Best-ever state
    # --------------------------------------------------------

    best_ever_fitness = (
        get_existing_best_fitness(
            resume=(
                args.resume is not None
            )
        )
    )

    if (
        best_ever_fitness
        != float("-inf")
    ):

        print(
            "Historical best fitness: "
            f"{best_ever_fitness:.2f}"
        )

        print()

    # ========================================================
    # Logging / checkpoint callback
    # ========================================================

    def on_generation(report):

        nonlocal best_ever_fitness

        stats = report.stats

        print(
            f"Generation {stats.generation:4d} | "
            f"best={stats.best_fitness:8.2f} | "
            f"mean={stats.mean_fitness:8.2f} | "
            f"median={stats.median_fitness:8.2f} | "
            f"std={stats.fitness_std:8.2f} | "
            f"goals={stats.goal_count}"
        )

        # ----------------------------------------------------
        # Generation champion
        # ----------------------------------------------------

        if (
            stats.generation
            % SAVE_EVERY
            == 0
        ):

            champion_path = (
                CHECKPOINT_DIR
                / (
                    f"gen_"
                    f"{stats.generation:04d}"
                    f"_champion.npz"
                )
            )

            save_generation_champion(
                champion_path,
                report,
                evaluation_config,
            )

            print(
                "  saved champion: "
                f"{champion_path.name}"
            )

        # ----------------------------------------------------
        # Historical population checkpoint
        # ----------------------------------------------------
        #
        # report describes generation N.
        #
        # By the time this callback runs,
        # Population.step() has already bred generation N + 1.
        #
        # Therefore population.generation refers to the
        # CURRENT genomes stored in population.genomes.
        # ----------------------------------------------------

        if (
            population.generation
            % SAVE_EVERY
            == 0
        ):

            population_path = (
                CHECKPOINT_DIR
                / (
                    "population_gen_"
                    f"{population.generation:04d}"
                    ".npz"
                )
            )

            save_population_checkpoint(
                population_path,
                population,
            )

            print(
                "  saved population: "
                f"{population_path.name}"
            )

        # ----------------------------------------------------
        # Latest population
        # ----------------------------------------------------

        save_population_checkpoint(
            CHECKPOINT_DIR
            / "latest_population.npz",
            population,
        )

        # ----------------------------------------------------
        # Best-ever champion
        # ----------------------------------------------------

        if (
            stats.best_fitness
            > best_ever_fitness
        ):

            best_ever_fitness = (
                stats.best_fitness
            )

            save_generation_champion(
                CHECKPOINT_DIR
                / "best_ever.npz",
                report,
                evaluation_config,
            )

            print(
                "  new best-ever Cowie "
                f"({stats.best_fitness:.2f})"
            )

    # ========================================================
    # Run evolution
    # ========================================================

    start_generation = (
        population.generation
    )

    target_generation = (
        start_generation
        + args.generations
    )

    print(
        f"Running generations "
        f"{start_generation} "
        f"through "
        f"{target_generation - 1}"
    )

    print()

    population.run(
        generations=args.generations,
        callback=on_generation,
    )

    # --------------------------------------------------------
    # Explicit final save
    # --------------------------------------------------------
    #
    # latest_population is already saved by the callback, but
    # this makes the end-of-run guarantee obvious.

    save_population_checkpoint(
        CHECKPOINT_DIR
        / "latest_population.npz",
        population,
    )

    final_population_path = (
        CHECKPOINT_DIR
        / (
            "population_gen_"
            f"{population.generation:04d}"
            ".npz"
        )
    )

    save_population_checkpoint(
        final_population_path,
        population,
    )

    print()
    print(
        "Evolution complete."
    )

    print(
        f"Current generation: "
        f"{population.generation}"
    )

    print(
        "Final population saved:"
    )

    print(
        f"  {final_population_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()