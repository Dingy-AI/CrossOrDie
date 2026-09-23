import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from creatures.controller import with_random_brain
from evolution.evaluator import evaluate_genome
from evolution.genome import Genome


rng = np.random.default_rng(42)


for i in range(10):

    genome = Genome.random(
        rng
    )

    genome = with_random_brain(
        genome,
        rng,
    )

    result = evaluate_genome(
        genome
    )

    print(
        f"Cowie {i:02d} | "
        f"fitness={result.fitness:8.2f} | "
        f"max_x={result.max_x:8.2f} | "
        f"final_x={result.final_x:8.2f} | "
        f"time={result.simulated_seconds:5.2f}s | "
        f"{result.termination_reason}"
    )