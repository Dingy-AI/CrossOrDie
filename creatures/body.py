from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pymunk

from creatures.phenotype import (
    Phenotype,
    LegPhenotype,
)


# ============================================================
# Physics constants
# ============================================================

# Converts our pixel-ish morphology dimensions into reasonable
# rigid-body masses.
MASS_PER_AREA = 0.001

MIN_BODY_MASS = 0.25

TORSO_BEVEL_RADIUS = 5.0

BODY_ELASTICITY = 0.05

# Neural controllers will eventually output values in [-1, 1].
# Those values become target relative angular velocities.
DEFAULT_MAX_MOTOR_RATE = 6.0  # radians / second


# ============================================================
# Motor ordering
# ============================================================

class MotorIndex(IntEnum):
    FRONT_HIP = 0
    FRONT_KNEE = 1
    REAR_HIP = 2
    REAR_KNEE = 3


MOTOR_COUNT = len(MotorIndex)


# ============================================================
# Component containers
# ============================================================

@dataclass(slots=True)
class JointAssembly:
    pivot: pymunk.PivotJoint
    limit: pymunk.RotaryLimitJoint
    motor: pymunk.SimpleMotor


@dataclass(slots=True)
class LegAssembly:
    upper_body: pymunk.Body
    lower_body: pymunk.Body

    upper_shape: pymunk.Segment
    lower_shape: pymunk.Segment
    foot_shape: pymunk.Segment

    hip: JointAssembly
    knee: JointAssembly


# ============================================================
# Creature body
# ============================================================

class CreatureBody:
    """
    Physical representation of one CrossOrDie creature.

    A Genome is decoded into a Phenotype before reaching this
    class.

    CreatureBody is responsible only for constructing and
    controlling the Pymunk rigid-body system.

    Topology:

        torso

        front upper leg
        front lower leg + foot

        rear upper leg
        rear lower leg + foot

        4 actuated joints:
            front hip
            front knee
            rear hip
            rear knee
    """

    def __init__(
        self,
        space: pymunk.Space,
        phenotype: Phenotype,
        spawn_position: tuple[float, float],
        collision_group: int = 1,
        max_motor_rate: float = DEFAULT_MAX_MOTOR_RATE,
    ):
        if collision_group == 0:
            raise ValueError(
                "collision_group must be non-zero so creature "
                "parts can ignore self-collisions."
            )

        if max_motor_rate <= 0.0:
            raise ValueError(
                "max_motor_rate must be greater than zero."
            )

        self.space = space
        self.phenotype = phenotype

        self.spawn_position = (
            float(spawn_position[0]),
            float(spawn_position[1]),
        )

        self.collision_group = collision_group
        self.max_motor_rate = max_motor_rate

        self._shape_filter = pymunk.ShapeFilter(
            group=collision_group
        )

        # ----------------------------------------------------
        # Construct torso
        # ----------------------------------------------------

        (
            self.torso_body,
            self.torso_shape,
        ) = self._create_torso()

        # ----------------------------------------------------
        # Construct legs
        # ----------------------------------------------------

        self.front_leg = self._create_leg(
            phenotype.front_leg,
            phenotype.joints.front_hip_range,
            phenotype.joints.front_knee_range,
        )

        self.rear_leg = self._create_leg(
            phenotype.rear_leg,
            phenotype.joints.rear_hip_range,
            phenotype.joints.rear_knee_range,
        )

    # ========================================================
    # Construction
    # ========================================================

    def _create_torso(
        self,
    ) -> tuple[pymunk.Body, pymunk.Poly]:

        body_pheno = self.phenotype.body

        length = body_pheno.length
        height = body_pheno.height

        # ----------------------------------------------------
        # Mass
        # ----------------------------------------------------

        area = length * height

        mass = max(
            MIN_BODY_MASS,
            area
            * MASS_PER_AREA
            * body_pheno.density_scale,
        )

        # ----------------------------------------------------
        # Balance
        # ----------------------------------------------------
        #
        # The body's origin represents its center of mass.
        #
        # Instead of moving Pymunk's center_of_gravity after
        # construction, we shift the torso geometry relative
        # to the body origin.
        #
        # balance_offset > 0:
        #     center of mass moves toward the creature's front
        #
        # Therefore the visible geometry is shifted backward
        # relative to the body's origin.

        geometry_center_x = -body_pheno.balance_offset

        half_length = length / 2.0
        half_height = height / 2.0

        vertices = [
            (
                geometry_center_x - half_length,
                -half_height,
            ),
            (
                geometry_center_x + half_length,
                -half_height,
            ),
            (
                geometry_center_x + half_length,
                half_height,
            ),
            (
                geometry_center_x - half_length,
                half_height,
            ),
        ]

        moment = pymunk.moment_for_poly(
            mass,
            vertices,
            radius=TORSO_BEVEL_RADIUS,
        )

        body = pymunk.Body(
            mass,
            moment,
        )

        body.position = self.spawn_position

        shape = pymunk.Poly(
            body,
            vertices,
            radius=TORSO_BEVEL_RADIUS,
        )

        self._configure_shape(
            shape,
            friction=body_pheno.friction,
        )

        self.space.add(
            body,
            shape,
        )

        return body, shape

    def _create_leg(
        self,
        leg: LegPhenotype,
        hip_range: float,
        knee_range: float,
    ) -> LegAssembly:

        torso = self.phenotype.body

        # ----------------------------------------------------
        # Hip location
        # ----------------------------------------------------
        #
        # torso_body.position is the center of mass.
        #
        # Torso geometry is offset by -balance_offset.
        # Attachment positions are relative to the geometric
        # center of the torso.

        torso_geometry_center_x = (
            self.torso_body.position.x
            - torso.balance_offset
        )

        hip_x = (
            torso_geometry_center_x
            + leg.attachment_offset
        )

        hip_y = (
            self.torso_body.position.y
            + torso.height / 2.0
        )

        hip_position = (
            hip_x,
            hip_y,
        )

        # ----------------------------------------------------
        # Upper leg
        # ----------------------------------------------------

        upper_center = (
            hip_x,
            hip_y + leg.upper_length / 2.0,
        )

        (
            upper_body,
            upper_shape,
        ) = self._create_limb_segment(
            position=upper_center,
            length=leg.upper_length,
            thickness=leg.thickness,
        )

        # ----------------------------------------------------
        # Knee
        # ----------------------------------------------------

        knee_position = (
            hip_x,
            hip_y + leg.upper_length,
        )

        # ----------------------------------------------------
        # Lower leg + foot
        # ----------------------------------------------------

        lower_center = (
            hip_x,
            knee_position[1]
            + leg.lower_length / 2.0,
        )

        (
            lower_body,
            lower_shape,
            foot_shape,
        ) = self._create_lower_leg(
            position=lower_center,
            leg=leg,
        )

        # ----------------------------------------------------
        # Constraints
        # ----------------------------------------------------

        hip = self._create_joint(
            body_a=self.torso_body,
            body_b=upper_body,
            pivot_position=hip_position,
            angular_range=hip_range,
            max_force=leg.hip_strength,
        )

        knee = self._create_joint(
            body_a=upper_body,
            body_b=lower_body,
            pivot_position=knee_position,
            angular_range=knee_range,
            max_force=leg.knee_strength,
        )

        return LegAssembly(
            upper_body=upper_body,
            lower_body=lower_body,

            upper_shape=upper_shape,
            lower_shape=lower_shape,
            foot_shape=foot_shape,

            hip=hip,
            knee=knee,
        )

    # ========================================================
    # Body-part construction
    # ========================================================

    def _create_limb_segment(
        self,
        position: tuple[float, float],
        length: float,
        thickness: float,
    ) -> tuple[
        pymunk.Body,
        pymunk.Segment,
    ]:

        radius = thickness / 2.0

        area = length * thickness

        mass = max(
            MIN_BODY_MASS,
            area
            * MASS_PER_AREA
            * self.phenotype.body.density_scale,
        )

        start = (
            0.0,
            -length / 2.0,
        )

        end = (
            0.0,
            length / 2.0,
        )

        moment = pymunk.moment_for_segment(
            mass,
            start,
            end,
            radius,
        )

        body = pymunk.Body(
            mass,
            moment,
        )

        body.position = position

        shape = pymunk.Segment(
            body,
            start,
            end,
            radius,
        )

        self._configure_shape(
            shape,
            friction=self.phenotype.body.friction,
        )

        self.space.add(
            body,
            shape,
        )

        return body, shape

    def _create_lower_leg(
        self,
        position: tuple[float, float],
        leg: LegPhenotype,
    ) -> tuple[
        pymunk.Body,
        pymunk.Segment,
        pymunk.Segment,
    ]:

        radius = leg.thickness / 2.0

        # ----------------------------------------------------
        # Lower leg geometry
        # ----------------------------------------------------

        leg_start = (
            0.0,
            -leg.lower_length / 2.0,
        )

        leg_end = (
            0.0,
            leg.lower_length / 2.0,
        )

        # ----------------------------------------------------
        # Foot geometry
        # ----------------------------------------------------
        #
        # Foot is part of the same rigid body as the lower leg.
        # This means we do NOT introduce another joint or
        # another neural-network output.

        foot_y = leg.lower_length / 2.0

        foot_start = (
            -leg.foot_length / 2.0,
            foot_y,
        )

        foot_end = (
            leg.foot_length / 2.0,
            foot_y,
        )

        # ----------------------------------------------------
        # Mass
        # ----------------------------------------------------

        lower_area = (
            leg.lower_length
            * leg.thickness
        )

        foot_area = (
            leg.foot_length
            * leg.thickness
        )

        density_scale = (
            self.phenotype.body.density_scale
        )

        lower_mass = max(
            MIN_BODY_MASS,
            lower_area
            * MASS_PER_AREA
            * density_scale,
        )

        foot_mass = max(
            MIN_BODY_MASS / 2.0,
            foot_area
            * MASS_PER_AREA
            * density_scale,
        )

        total_mass = (
            lower_mass
            + foot_mass
        )

        # Let each component contribute to rotational inertia.
        moment = (
            pymunk.moment_for_segment(
                lower_mass,
                leg_start,
                leg_end,
                radius,
            )
            +
            pymunk.moment_for_segment(
                foot_mass,
                foot_start,
                foot_end,
                radius,
            )
        )

        body = pymunk.Body(
            total_mass,
            moment,
        )

        body.position = position

        lower_shape = pymunk.Segment(
            body,
            leg_start,
            leg_end,
            radius,
        )

        foot_shape = pymunk.Segment(
            body,
            foot_start,
            foot_end,
            radius,
        )

        self._configure_shape(
            lower_shape,
            friction=self.phenotype.body.friction,
        )

        self._configure_shape(
            foot_shape,
            friction=self.phenotype.body.friction,
        )

        self.space.add(
            body,
            lower_shape,
            foot_shape,
        )

        return (
            body,
            lower_shape,
            foot_shape,
        )

    # ========================================================
    # Joint construction
    # ========================================================

    def _create_joint(
        self,
        body_a: pymunk.Body,
        body_b: pymunk.Body,
        pivot_position: tuple[float, float],
        angular_range: float,
        max_force: float,
    ) -> JointAssembly:

        # Physical hinge.
        pivot = pymunk.PivotJoint(
            body_a,
            body_b,
            pivot_position,
        )

        # No collision between directly connected bodies.
        pivot.collide_bodies = False

        # We interpret the genetic joint range as the TOTAL
        # allowable angular span.
        half_range = angular_range / 2.0

        limit = pymunk.RotaryLimitJoint(
            body_a,
            body_b,
            -half_range,
            half_range,
        )

        limit.collide_bodies = False

        # Motor starts stationary.
        motor = pymunk.SimpleMotor(
            body_a,
            body_b,
            0.0,
        )

        # Very important:
        # SimpleMotor without max_force behaves essentially
        # like it has enormous/infinite motor authority.
        motor.max_force = max_force
        motor.collide_bodies = False

        self.space.add(
            pivot,
            limit,
            motor,
        )

        return JointAssembly(
            pivot=pivot,
            limit=limit,
            motor=motor,
        )

    # ========================================================
    # Shape configuration
    # ========================================================

    def _configure_shape(
        self,
        shape: pymunk.Shape,
        friction: float,
    ):
        shape.friction = friction
        shape.elasticity = BODY_ELASTICITY

        # All shapes belonging to this creature use the same
        # non-zero group and therefore do not collide with one
        # another.
        shape.filter = self._shape_filter

    # ========================================================
    # Motor control
    # ========================================================

    def apply_motor_commands(
        self,
        commands,
    ):
        """
        Apply normalized motor commands.

        Expected order:

            0 front hip
            1 front knee
            2 rear hip
            3 rear knee

        Each command is clipped to [-1, 1] and converted into
        a target relative angular velocity.
        """

        commands = np.asarray(
            commands,
            dtype=np.float32,
        )

        if commands.shape != (MOTOR_COUNT,):
            raise ValueError(
                f"Motor commands must have shape "
                f"({MOTOR_COUNT},), got {commands.shape}."
            )

        if not np.all(np.isfinite(commands)):
            raise ValueError(
                "Motor commands must contain only finite values."
            )

        commands = np.clip(
            commands,
            -1.0,
            1.0,
        )

        motors = self.motors

        for motor, command in zip(
            motors,
            commands,
        ):
            motor.rate = (
                float(command)
                * self.max_motor_rate
            )

    def stop_motors(self):
        for motor in self.motors:
            motor.rate = 0.0

    # ========================================================
    # Convenient accessors
    # ========================================================

    @property
    def motors(
        self,
    ) -> tuple[
        pymunk.SimpleMotor,
        pymunk.SimpleMotor,
        pymunk.SimpleMotor,
        pymunk.SimpleMotor,
    ]:
        return (
            self.front_leg.hip.motor,
            self.front_leg.knee.motor,
            self.rear_leg.hip.motor,
            self.rear_leg.knee.motor,
        )

    @property
    def bodies(
        self,
    ) -> tuple[pymunk.Body, ...]:
        return (
            self.torso_body,

            self.front_leg.upper_body,
            self.front_leg.lower_body,

            self.rear_leg.upper_body,
            self.rear_leg.lower_body,
        )

    @property
    def shapes(
        self,
    ) -> tuple[pymunk.Shape, ...]:
        return (
            self.torso_shape,

            self.front_leg.upper_shape,
            self.front_leg.lower_shape,
            self.front_leg.foot_shape,

            self.rear_leg.upper_shape,
            self.rear_leg.lower_shape,
            self.rear_leg.foot_shape,
        )

    @property
    def joints(
        self,
    ) -> tuple[JointAssembly, ...]:
        return (
            self.front_leg.hip,
            self.front_leg.knee,
            self.rear_leg.hip,
            self.rear_leg.knee,
        )

    @property
    def constraints(
        self,
    ) -> tuple[pymunk.Constraint, ...]:

        result = []

        for joint in self.joints:
            result.extend(
                [
                    joint.pivot,
                    joint.limit,
                    joint.motor,
                ]
            )

        return tuple(result)

    @property
    def front_foot_shape(
        self,
    ) -> pymunk.Segment:
        return self.front_leg.foot_shape

    @property
    def rear_foot_shape(
        self,
    ) -> pymunk.Segment:
        return self.rear_leg.foot_shape

    @property
    def position(
        self,
    ) -> pymunk.Vec2d:
        """
        Creature position is currently defined by torso center
        of mass.
        """

        return self.torso_body.position

    # ========================================================
    # Cleanup
    # ========================================================

    def remove_from_space(self):
        """
        Remove the entire creature from its Pymunk Space.
        """

        self.stop_motors()

        self.space.remove(
            *self.constraints,
            *self.shapes,
            *self.bodies,
        )