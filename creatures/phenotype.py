from dataclasses import dataclass
import math

from evolution.genome import (
    Genome,
    BodyGene,
    LegGene,
    JointGene,
)


# ============================================================
# Physical ranges
# ============================================================
#
# These are starting values, not sacred constants.
# We will tune them once we see creatures interacting with
# Pymunk.
#
# Pymunk uses arbitrary units, so for now we are treating
# world units approximately like pixels.


BODY_LENGTH_RANGE = (90.0, 180.0)
BODY_HEIGHT_RANGE = (35.0, 70.0)

# Relative density rather than Pymunk's literal density.
# body.py will convert this into actual mass.
BODY_DENSITY_RANGE = (0.65, 1.60)

# Center-of-mass shift as a fraction of torso length.
BODY_BALANCE_RANGE = (-0.20, 0.20)

BODY_FRICTION_RANGE = (0.60, 1.40)


UPPER_LEG_LENGTH_RANGE = (35.0, 95.0)
LOWER_LEG_LENGTH_RANGE = (30.0, 90.0)
LEG_THICKNESS_RANGE = (8.0, 20.0)
FOOT_LENGTH_RANGE = (15.0, 50.0)

# Starting motor force ranges.
# These will likely need tuning once physics is active.
HIP_STRENGTH_RANGE = (15_000.0, 75_000.0)
KNEE_STRENGTH_RANGE = (10_000.0, 60_000.0)


# Attachment positions are fractions of torso length measured
# from the torso center.
#
# Negative = rear
# Positive = front
FRONT_ATTACHMENT_RANGE = (0.10, 0.45)
REAR_ATTACHMENT_RANGE = (-0.45, -0.10)


# Joint angular ranges in degrees.
FRONT_HIP_RANGE = (25.0, 120.0)
FRONT_KNEE_RANGE = (20.0, 135.0)

REAR_HIP_RANGE = (25.0, 120.0)
REAR_KNEE_RANGE = (20.0, 135.0)


# ============================================================
# Phenotype data structures
# ============================================================

@dataclass(frozen=True, slots=True)
class BodyPhenotype:
    length: float
    height: float

    density_scale: float

    # Horizontal center-of-mass shift in world units.
    balance_offset: float

    friction: float


@dataclass(frozen=True, slots=True)
class LegPhenotype:
    # Horizontal offset from torso center in world units.
    attachment_offset: float

    upper_length: float
    lower_length: float

    thickness: float
    foot_length: float

    hip_strength: float
    knee_strength: float


@dataclass(frozen=True, slots=True)
class JointPhenotype:
    # Stored in radians because that is what Pymunk uses.
    front_hip_range: float
    front_knee_range: float

    rear_hip_range: float
    rear_knee_range: float


@dataclass(frozen=True, slots=True)
class Phenotype:
    body: BodyPhenotype

    front_leg: LegPhenotype
    rear_leg: LegPhenotype

    joints: JointPhenotype


# ============================================================
# Gene decoding
# ============================================================

def lerp(
    minimum: float,
    maximum: float,
    gene: float,
) -> float:
    """
    Convert a normalized [0, 1] gene into a physical value.
    """

    return minimum + (maximum - minimum) * float(gene)


def _decode_leg(
    genes,
    body_length: float,
    attachment_range: tuple[float, float],
) -> LegPhenotype:

    attachment_fraction = lerp(
        attachment_range[0],
        attachment_range[1],
        genes[LegGene.ATTACHMENT],
    )

    attachment_offset = (
        attachment_fraction * body_length
    )

    return LegPhenotype(
        attachment_offset=attachment_offset,

        upper_length=lerp(
            *UPPER_LEG_LENGTH_RANGE,
            genes[LegGene.UPPER_LENGTH],
        ),

        lower_length=lerp(
            *LOWER_LEG_LENGTH_RANGE,
            genes[LegGene.LOWER_LENGTH],
        ),

        thickness=lerp(
            *LEG_THICKNESS_RANGE,
            genes[LegGene.THICKNESS],
        ),

        foot_length=lerp(
            *FOOT_LENGTH_RANGE,
            genes[LegGene.FOOT_LENGTH],
        ),

        hip_strength=lerp(
            *HIP_STRENGTH_RANGE,
            genes[LegGene.HIP_STRENGTH],
        ),

        knee_strength=lerp(
            *KNEE_STRENGTH_RANGE,
            genes[LegGene.KNEE_STRENGTH],
        ),
    )


def decode_genome(
    genome: Genome,
) -> Phenotype:
    """
    Convert a normalized Genome into physical creature traits.

    Genome:
        abstract hereditary information

    Phenotype:
        actual physical properties used by the simulator
    """

    # --------------------------------------------------------
    # Torso
    # --------------------------------------------------------

    body_length = lerp(
        *BODY_LENGTH_RANGE,
        genome.body[BodyGene.LENGTH],
    )

    body_height = lerp(
        *BODY_HEIGHT_RANGE,
        genome.body[BodyGene.HEIGHT],
    )

    density_scale = lerp(
        *BODY_DENSITY_RANGE,
        genome.body[BodyGene.DENSITY],
    )

    balance_fraction = lerp(
        *BODY_BALANCE_RANGE,
        genome.body[BodyGene.BALANCE],
    )

    balance_offset = (
        balance_fraction
        * body_length
    )

    friction = lerp(
        *BODY_FRICTION_RANGE,
        genome.body[BodyGene.FRICTION],
    )

    body = BodyPhenotype(
        length=body_length,
        height=body_height,
        density_scale=density_scale,
        balance_offset=balance_offset,
        friction=friction,
    )

    # --------------------------------------------------------
    # Legs
    # --------------------------------------------------------

    front_leg = _decode_leg(
        genome.front_leg,
        body_length,
        FRONT_ATTACHMENT_RANGE,
    )

    rear_leg = _decode_leg(
        genome.rear_leg,
        body_length,
        REAR_ATTACHMENT_RANGE,
    )

    # --------------------------------------------------------
    # Joint limits
    # --------------------------------------------------------

    joints = JointPhenotype(

        front_hip_range=math.radians(
            lerp(
                *FRONT_HIP_RANGE,
                genome.joints[
                    JointGene.FRONT_HIP_RANGE
                ],
            )
        ),

        front_knee_range=math.radians(
            lerp(
                *FRONT_KNEE_RANGE,
                genome.joints[
                    JointGene.FRONT_KNEE_RANGE
                ],
            )
        ),

        rear_hip_range=math.radians(
            lerp(
                *REAR_HIP_RANGE,
                genome.joints[
                    JointGene.REAR_HIP_RANGE
                ],
            )
        ),

        rear_knee_range=math.radians(
            lerp(
                *REAR_KNEE_RANGE,
                genome.joints[
                    JointGene.REAR_KNEE_RANGE
                ],
            )
        ),
    )

    return Phenotype(
        body=body,
        front_leg=front_leg,
        rear_leg=rear_leg,
        joints=joints,
    )