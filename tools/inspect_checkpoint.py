import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from evolution.checkpoint import (
    load_genome_checkpoint,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "checkpoint",
    type=Path,
)

args = parser.parse_args()


checkpoint = load_genome_checkpoint(
    args.checkpoint
)


print(
    f"Format:      "
    f"{checkpoint.format_version}"
)

print(
    f"Generation:  "
    f"{checkpoint.generation}"
)


if checkpoint.fitness_result:

    result = (
        checkpoint.fitness_result
    )

    print(
        f"Fitness:     "
        f"{result['fitness']:.2f}"
    )

    print(
        f"Max progress:"
        f" {result['max_progress']:.2f}"
    )

    print(
        f"Final X:     "
        f"{result['final_x']:.2f}"
    )

    print(
        f"Termination: "
        f"{result['termination_reason']}"
    )


print()
print(
    "Morphology genes:"
)

print(
    checkpoint.genome.flatten_morphology()
)

print()
print(
    "Brain genes:",
    checkpoint.genome.brain.size
    if checkpoint.genome.brain is not None
    else 0,
)