import math
from dataclasses import dataclass

import numpy as np
import pymunk

from creatures.body import (
    CreatureBody,
    JointAssembly,
)


# ============================================================
# Observation layout
# ============================================================

TORSO_ANGLE_SIN = 0
TORSO_ANGLE_COS = 1
TORSO_ANGULAR_VELOCITY = 2

TORSO_VELOCITY_X = 3
TORSO_VELOCITY_Y = 4

VERTICAL_DISPLACEMENT = 5
GOAL_DISTANCE = 6

FRONT_HIP_ANGLE = 7
FRONT_HIP_VELOCITY = 8

FRONT_KNEE_ANGLE = 9
FRONT_KNEE_VELOCITY = 10

REAR_HIP_ANGLE = 11
REAR_HIP_VELOCITY = 12

REAR_KNEE_ANGLE = 13
REAR_KNEE_VELOCITY = 14

FRONT_FOOT_CONTACT = 15
REAR_FOOT_CONTACT = 16


OBSERVATION_SIZE = 17


# ============================================================
# Normalization configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class ObservationConfig:

    # Pixel-ish world units / second.
    velocity_scale: float = 600.0

    # Radians / second.
    angular_velocity_scale: float = 10.0

    # How far vertically the torso needs to move before the
    # observation saturates.
    vertical_displacement_scale: float = 500.0


DEFAULT_OBSERVATION_CONFIG = ObservationConfig()


# ============================================================
# Helpers
# ============================================================

def _clip(value: float) -> float:
    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


def _wrap_angle(angle: float) -> float:
    """
    Wrap an angle into [-pi, pi].
    """

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def _joint_state(
    joint: JointAssembly,
    angular_range: float,
    angular_velocity_scale: float,
) -> tuple[float, float]:
    """
    Return normalized relative joint angle and angular velocity.
    """

    body_a = joint.pivot.a
    body_b = joint.pivot.b

    relative_angle = _wrap_angle(
        body_b.angle
        - body_a.angle
    )

    # phenotype joint range represents the TOTAL allowable
    # angular span.
    half_range = angular_range / 2.0

    if half_range <= 0.0:
        normalized_angle = 0.0
    else:
        normalized_angle = _clip(
            relative_angle
            / half_range
        )

    relative_angular_velocity = (
        body_b.angular_velocity
        - body_a.angular_velocity
    )

    normalized_velocity = _clip(
        relative_angular_velocity
        / angular_velocity_scale
    )

    return (
        normalized_angle,
        normalized_velocity,
    )


def _shape_has_external_contact(
    space: pymunk.Space,
    shape: pymunk.Shape,
    creature: CreatureBody,
) -> bool:
    """
    Return True when a creature shape overlaps/touches something
    that is not another shape belonging to the same creature.

    This lets the feet act as simple contact sensors.
    """

    creature_shapes = set(
        creature.shapes
    )

    query_results = space.shape_query(
        shape
    )

    for result in query_results:

        other_shape = result.shape

        if other_shape is None:
            continue

        if other_shape not in creature_shapes:
            return True

    return False


# ============================================================
# Observation generation
# ============================================================

def build_observation(
    creature: CreatureBody,
    spawn_position: tuple[float, float],
    goal_x: float,
    config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
) -> np.ndarray:
    """
    Build the neural-network observation vector for a creature.

    All continuous values are normalized to approximately
    [-1, 1].

    Foot contacts are binary:
        0 = not touching terrain
        1 = touching terrain
    """

    torso = creature.torso_body

    observation = np.zeros(
        OBSERVATION_SIZE,
        dtype=np.float32,
    )

    # ========================================================
    # Torso orientation
    # ========================================================

    observation[
        TORSO_ANGLE_SIN
    ] = math.sin(
        torso.angle
    )

    observation[
        TORSO_ANGLE_COS
    ] = math.cos(
        torso.angle
    )

    observation[
        TORSO_ANGULAR_VELOCITY
    ] = _clip(
        torso.angular_velocity
        / config.angular_velocity_scale
    )

    # ========================================================
    # Torso velocity
    # ========================================================

    observation[
        TORSO_VELOCITY_X
    ] = _clip(
        torso.velocity.x
        / config.velocity_scale
    )

    observation[
        TORSO_VELOCITY_Y
    ] = _clip(
        torso.velocity.y
        / config.velocity_scale
    )

    # ========================================================
    # Position relative to episode
    # ========================================================

    spawn_x = float(
        spawn_position[0]
    )

    spawn_y = float(
        spawn_position[1]
    )

    course_length = (
        float(goal_x)
        - spawn_x
    )

    if course_length <= 0.0:
        raise ValueError(
            "goal_x must be greater than spawn_position.x"
        )

    # Positive = creature has fallen below its starting level.
    observation[
        VERTICAL_DISPLACEMENT
    ] = _clip(
        (
            torso.position.y
            - spawn_y
        )
        / config.vertical_displacement_scale
    )

    # 1.0 = at starting X position
    # 0.0 = at goal
    # negative = past the goal
    observation[
        GOAL_DISTANCE
    ] = _clip(
        (
            goal_x
            - torso.position.x
        )
        / course_length
    )

    # ========================================================
    # Joint states
    # ========================================================

    phenotype = creature.phenotype

    front_hip = _joint_state(
        creature.front_leg.hip,
        phenotype.joints.front_hip_range,
        config.angular_velocity_scale,
    )

    front_knee = _joint_state(
        creature.front_leg.knee,
        phenotype.joints.front_knee_range,
        config.angular_velocity_scale,
    )

    rear_hip = _joint_state(
        creature.rear_leg.hip,
        phenotype.joints.rear_hip_range,
        config.angular_velocity_scale,
    )

    rear_knee = _joint_state(
        creature.rear_leg.knee,
        phenotype.joints.rear_knee_range,
        config.angular_velocity_scale,
    )

    observation[
        FRONT_HIP_ANGLE
    ] = front_hip[0]

    observation[
        FRONT_HIP_VELOCITY
    ] = front_hip[1]

    observation[
        FRONT_KNEE_ANGLE
    ] = front_knee[0]

    observation[
        FRONT_KNEE_VELOCITY
    ] = front_knee[1]

    observation[
        REAR_HIP_ANGLE
    ] = rear_hip[0]

    observation[
        REAR_HIP_VELOCITY
    ] = rear_hip[1]

    observation[
        REAR_KNEE_ANGLE
    ] = rear_knee[0]

    observation[
        REAR_KNEE_VELOCITY
    ] = rear_knee[1]

    # ========================================================
    # Foot contact sensors
    # ========================================================

    observation[
        FRONT_FOOT_CONTACT
    ] = float(
        _shape_has_external_contact(
            creature.space,
            creature.front_foot_shape,
            creature,
        )
    )

    observation[
        REAR_FOOT_CONTACT
    ] = float(
        _shape_has_external_contact(
            creature.space,
            creature.rear_foot_shape,
            creature,
        )
    )

    return observation