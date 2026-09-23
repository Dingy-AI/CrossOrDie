from dataclasses import dataclass
from typing import Callable

import numpy as np

from creatures.controller import with_random_brain

from evolution.evaluator import (
    EvaluationConfig,
    FitnessResult,
    DEFAULT_EVALUATION_CONFIG,
    evaluate_genome,
)

from evolution.genetics import (
    MutationConfig,
    DEFAULT_MUTATION_CONFIG,
    reproduce,
)

from evolution.genome import Genome


# ============================================================
# Population configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class PopulationConfig:
    """
    Controls the evolutionary population.

    Initial defaults:

        population size     = 100
        elites              = 5
        parent pool         = top 30%
        tournament size     = 3
    """

    population_size: int = 100

    # Best N individuals survive completely unchanged.
    elite_count: int = 5

    # Fraction of the population allowed to reproduce.
    parent_fraction: float = 0.30

    # Number of candidates sampled when selecting each parent.
    tournament_size: int = 3

    def __post_init__(self):

        if self.population_size < 2:
            raise ValueError(
                "population_size must be at least 2."
            )

        if not (
            0 <= self.elite_count
            < self.population_size
        ):
            raise ValueError(
                "elite_count must satisfy "
                "0 <= elite_count < population_size."
            )

        if not (
            0.0
            < self.parent_fraction
            <= 1.0
        ):
            raise ValueError(
                "parent_fraction must be within (0, 1]."
            )

        if self.tournament_size < 1:
            raise ValueError(
                "tournament_size must be at least 1."
            )


DEFAULT_POPULATION_CONFIG = (
    PopulationConfig()
)


# ============================================================
# Evaluated individual
# ============================================================

@dataclass(frozen=True, slots=True)
class EvaluatedIndividual:
    """
    A genome paired with the result of its evaluation.
    """

    genome: Genome
    result: FitnessResult

    @property
    def fitness(self) -> float:
        return self.result.fitness


# ============================================================
# Generation statistics
# ============================================================

@dataclass(frozen=True, slots=True)
class GenerationStats:

    generation: int

    population_size: int

    best_fitness: float
    mean_fitness: float
    median_fitness: float
    worst_fitness: float
    fitness_std: float

    goal_count: int


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """
    Summary returned after completing one generation.

    The champion genome is copied so it remains safe even after
    the population continues evolving.
    """

    stats: GenerationStats

    best_genome: Genome
    best_result: FitnessResult


# ============================================================
# Population
# ============================================================

class Population:
    """
    CrossOrDie evolutionary population.

    Lifecycle:

        population
            ↓
        evaluate
            ↓
        rank
            ↓
        preserve elites
            ↓
        tournament-select parents
            ↓
        crossover
            ↓
        mutation
            ↓
        next generation
    """

    def __init__(
        self,
        genomes: list[Genome],
        *,
        rng: np.random.Generator,
        population_config: PopulationConfig = (
            DEFAULT_POPULATION_CONFIG
        ),
        evaluation_config: EvaluationConfig = (
            DEFAULT_EVALUATION_CONFIG
        ),
        mutation_config: MutationConfig = (
            DEFAULT_MUTATION_CONFIG
        ),
        generation: int = 0,
    ):

        if len(genomes) != (
            population_config.population_size
        ):
            raise ValueError(
                "Genome count must equal population_size. "
                f"Expected "
                f"{population_config.population_size}, "
                f"got {len(genomes)}."
            )

        for i, genome in enumerate(genomes):

            if genome.brain is None:
                raise ValueError(
                    f"Genome {i} does not contain "
                    "a brain chromosome."
                )

        if generation < 0:
            raise ValueError(
                "generation cannot be negative."
            )

        self.genomes = genomes

        self.rng = rng

        self.population_config = (
            population_config
        )

        self.evaluation_config = (
            evaluation_config
        )

        self.mutation_config = (
            mutation_config
        )

        self.generation = generation

    # ========================================================
    # Initial population
    # ========================================================

    @classmethod
    def random(
        cls,
        *,
        population_config: PopulationConfig = (
            DEFAULT_POPULATION_CONFIG
        ),
        evaluation_config: EvaluationConfig = (
            DEFAULT_EVALUATION_CONFIG
        ),
        mutation_config: MutationConfig = (
            DEFAULT_MUTATION_CONFIG
        ),
        seed: int | None = None,
    ) -> "Population":
        """
        Create a completely random Generation 0 population.

        Each individual receives:

            23 random morphology genes
            180 random brain genes
        """

        rng = np.random.default_rng(
            seed
        )

        genomes = []

        for _ in range(
            population_config.population_size
        ):

            genome = Genome.random(
                rng
            )

            genome = with_random_brain(
                genome,
                rng=rng,
            )

            genomes.append(
                genome
            )

        return cls(
            genomes=genomes,
            rng=rng,
            population_config=population_config,
            evaluation_config=evaluation_config,
            mutation_config=mutation_config,
            generation=0,
        )

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
    ) -> list[EvaluatedIndividual]:
        """
        Evaluate every genome in the current population.

        Evaluation is currently sequential.

        Later this is one of the first places we can add
        multiprocessing.
        """

        evaluated = []

        for genome in self.genomes:

            result = evaluate_genome(
                genome,
                config=self.evaluation_config,
            )

            evaluated.append(
                EvaluatedIndividual(
                    genome=genome,
                    result=result,
                )
            )

        return evaluated

    # ========================================================
    # Ranking
    # ========================================================

    @staticmethod
    def rank(
        evaluated: list[EvaluatedIndividual],
    ) -> list[EvaluatedIndividual]:
        """
        Highest fitness first.
        """

        return sorted(
            evaluated,
            key=lambda individual: (
                individual.fitness
            ),
            reverse=True,
        )

    # ========================================================
    # Parent pool
    # ========================================================

    def _parent_pool_size(
        self,
    ) -> int:

        population_size = (
            self.population_config.population_size
        )

        fraction = (
            self.population_config.parent_fraction
        )

        size = int(
            np.ceil(
                population_size
                * fraction
            )
        )

        # We always want at least two possible parents.
        return min(
            population_size,
            max(
                2,
                size,
            ),
        )

    # ========================================================
    # Tournament selection
    # ========================================================

    def _select_parent(
        self,
        parent_pool: list[EvaluatedIndividual],
        exclude_index: int | None = None,
    ) -> tuple[int, Genome]:
        """
        Select one parent using tournament selection.

        Example with tournament_size=3:

            randomly sample:
                Cowie #12 fitness 54
                Cowie #03 fitness 71
                Cowie #20 fitness 48

            winner:
                Cowie #03
        """

        available_indices = [
            i
            for i in range(
                len(parent_pool)
            )
            if i != exclude_index
        ]

        if not available_indices:
            raise ValueError(
                "No parents available for selection."
            )

        tournament_size = min(
            self.population_config.tournament_size,
            len(available_indices),
        )

        sampled_indices = (
            self.rng.choice(
                available_indices,
                size=tournament_size,
                replace=False,
            )
        )

        best_index = max(
            sampled_indices,
            key=lambda index: (
                parent_pool[
                    int(index)
                ].fitness
            ),
        )

        best_index = int(
            best_index
        )

        return (
            best_index,
            parent_pool[
                best_index
            ].genome,
        )

    # ========================================================
    # Breeding
    # ========================================================

    def breed_next_generation(
        self,
        evaluated: list[EvaluatedIndividual],
    ) -> list[Genome]:
        """
        Produce the next generation from an evaluated current
        generation.

        Elites are copied unchanged.

        Remaining slots are filled with children from
        tournament-selected parents.
        """

        if len(evaluated) != (
            self.population_config.population_size
        ):
            raise ValueError(
                "Evaluated population size does not match "
                "configured population size."
            )

        ranked = self.rank(
            evaluated
        )

        next_generation = []

        # ----------------------------------------------------
        # Elitism
        # ----------------------------------------------------

        elite_count = (
            self.population_config.elite_count
        )

        for individual in ranked[
            :elite_count
        ]:

            next_generation.append(
                individual.genome.copy()
            )

        # ----------------------------------------------------
        # Parent pool
        # ----------------------------------------------------

        parent_pool_size = (
            self._parent_pool_size()
        )

        parent_pool = ranked[
            :parent_pool_size
        ]

        # ----------------------------------------------------
        # Children
        # ----------------------------------------------------

        while len(
            next_generation
        ) < (
            self.population_config.population_size
        ):

            parent_a_index, parent_a = (
                self._select_parent(
                    parent_pool
                )
            )

            _, parent_b = (
                self._select_parent(
                    parent_pool,
                    exclude_index=parent_a_index,
                )
            )

            child = reproduce(
                parent_a,
                parent_b,
                rng=self.rng,
                mutation_config=(
                    self.mutation_config
                ),
            )

            next_generation.append(
                child
            )

        return next_generation

    # ========================================================
    # Statistics
    # ========================================================

    def _build_report(
        self,
        evaluated: list[EvaluatedIndividual],
    ) -> GenerationReport:

        ranked = self.rank(
            evaluated
        )

        fitnesses = np.asarray(
            [
                individual.fitness
                for individual in evaluated
            ],
            dtype=np.float64,
        )

        best = ranked[0]

        stats = GenerationStats(
            generation=self.generation,

            population_size=len(
                evaluated
            ),

            best_fitness=float(
                fitnesses.max()
            ),

            mean_fitness=float(
                fitnesses.mean()
            ),

            median_fitness=float(
                np.median(
                    fitnesses
                )
            ),

            worst_fitness=float(
                fitnesses.min()
            ),

            fitness_std=float(
                fitnesses.std()
            ),

            goal_count=sum(
                individual.result.reached_goal
                for individual in evaluated
            ),
        )

        return GenerationReport(
            stats=stats,
            best_genome=(
                best.genome.copy()
            ),
            best_result=(
                best.result
            ),
        )

    # ========================================================
    # One generation
    # ========================================================

    def step(
        self,
    ) -> GenerationReport:
        """
        Run one complete generation:

            evaluate generation N
                ↓
            collect statistics
                ↓
            breed generation N + 1
                ↓
            increment generation counter

        The returned report describes generation N.
        """

        evaluated = self.evaluate()

        report = self._build_report(
            evaluated
        )

        self.genomes = (
            self.breed_next_generation(
                evaluated
            )
        )

        self.generation += 1

        return report

    # ========================================================
    # Multiple generations
    # ========================================================

    def run(
        self,
        generations: int,
        callback: Callable[
            [GenerationReport],
            None,
        ]
        | None = None,
    ) -> list[GenerationReport]:
        """
        Run multiple generations.

        callback, when supplied, is called once after every
        completed generation.
        """

        if generations < 1:
            raise ValueError(
                "generations must be at least 1."
            )

        reports = []

        for _ in range(
            generations
        ):

            report = self.step()

            reports.append(
                report
            )

            if callback is not None:
                callback(
                    report
                )

        return reports