from dataclasses import dataclass
import math

import pymunk


# ============================================================
# Collision types
# ============================================================

COLLISION_TERRAIN = 1
COLLISION_BRIDGE = 2


# ============================================================
# World configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class WorldConfig:
    """
    Geometry and physical properties of the CrossOrDie world.

    Coordinate system:

        +X = right
        +Y = down
    """

    # --------------------------------------------------------
    # Important episode positions
    # --------------------------------------------------------

    spawn_x: float = 300.0
    spawn_y: float = 420.0

    goal_x: float = 1840.0

    # Creature is considered dead below this point.
    death_y: float = 950.0

    # --------------------------------------------------------
    # Bridge
    # --------------------------------------------------------

    bridge_start_x: float = 760.0
    bridge_end_x: float = 1460.0

    bridge_start_y: float = 670.0
    bridge_end_y: float = 670.0

    bridge_segments: int = 18

    # Positive Y points downward, so positive sag makes the
    # bridge droop.
    bridge_sag: float = 55.0

    bridge_radius: float = 7.0
    bridge_friction: float = 1.25

    # --------------------------------------------------------
    # Terrain
    # --------------------------------------------------------

    terrain_radius: float = 8.0
    terrain_friction: float = 1.20

    elasticity: float = 0.03

    def __post_init__(self):

        if self.goal_x <= self.spawn_x:
            raise ValueError(
                "goal_x must be greater than spawn_x."
            )

        if self.bridge_end_x <= self.bridge_start_x:
            raise ValueError(
                "bridge_end_x must be greater than "
                "bridge_start_x."
            )

        if self.bridge_segments < 2:
            raise ValueError(
                "bridge_segments must be at least 2."
            )

        if self.death_y <= max(
            self.bridge_start_y,
            self.bridge_end_y,
        ):
            raise ValueError(
                "death_y must be below the bridge."
            )


DEFAULT_WORLD_CONFIG = WorldConfig()


# ============================================================
# CrossOrDie world
# ============================================================

class CrossOrDieWorld:
    """
    Static Version-1 CrossOrDie environment.

    Contains:

        irregular starting plateau
        sloped canyon entrance
        sagging segmented bridge
        irregular destination plateau

    The bridge is currently STATIC.

    Rendering is deliberately not handled here.
    This class only owns simulation geometry.
    """

    def __init__(
        self,
        space: pymunk.Space,
        config: WorldConfig = DEFAULT_WORLD_CONFIG,
    ):
        self.space = space
        self.config = config

        self.terrain_shapes: list[pymunk.Segment] = []
        self.bridge_shapes: list[pymunk.Segment] = []

        # ----------------------------------------------------
        # Generate terrain geometry
        # ----------------------------------------------------

        self.left_surface = (
            self._create_left_surface()
        )

        self.right_surface = (
            self._create_right_surface()
        )

        self.bridge_points = (
            self._create_bridge_points()
        )

        # ----------------------------------------------------
        # Build Pymunk collision geometry
        # ----------------------------------------------------

        self._build_terrain()
        self._build_bridge()

    # ========================================================
    # Important positions
    # ========================================================

    @property
    def spawn_position(
        self,
    ) -> tuple[float, float]:

        return (
            self.config.spawn_x,
            self.config.spawn_y,
        )

    @property
    def goal_x(self) -> float:
        return self.config.goal_x

    @property
    def death_y(self) -> float:
        return self.config.death_y

    @property
    def course_length(self) -> float:

        return (
            self.config.goal_x
            - self.config.spawn_x
        )

    # ========================================================
    # Terrain definition
    # ========================================================

    def _create_left_surface(
        self,
    ) -> list[tuple[float, float]]:
        """
        Irregular starting terrain.

        Ends exactly where the bridge begins.
        """

        c = self.config

        return [
            (-800.0, 648.0),
            (-400.0, 650.0),
            (0.0, 646.0),
            (130.0, 638.0),
            (260.0, 644.0),
            (390.0, 634.0),
            (510.0, 641.0),
            (610.0, 646.0),
            (680.0, 654.0),
            (
                c.bridge_start_x,
                c.bridge_start_y,
            ),
        ]

    def _create_right_surface(
        self,
    ) -> list[tuple[float, float]]:
        """
        Irregular destination terrain.

        Begins exactly where the bridge ends.
        """

        c = self.config

        return [
            (
                c.bridge_end_x,
                c.bridge_end_y,
            ),
            (1530.0, 655.0),
            (1610.0, 644.0),
            (1710.0, 637.0),
            (1840.0, 642.0),
            (1990.0, 635.0),
            (2180.0, 643.0),
            (2450.0, 637.0),
            (2800.0, 640.0),
        ]

    # ========================================================
    # Bridge generation
    # ========================================================

    def _create_bridge_points(
        self,
    ) -> list[tuple[float, float]]:
        """
        Generate a smooth sagging bridge.

        Sag curve:

            4 * t * (1 - t)

        equals:

            0 at either endpoint
            1 at the center
        """

        c = self.config

        points = []

        for index in range(
            c.bridge_segments + 1
        ):

            t = (
                index
                / c.bridge_segments
            )

            x = (
                c.bridge_start_x
                + (
                    c.bridge_end_x
                    - c.bridge_start_x
                )
                * t
            )

            base_y = (
                c.bridge_start_y
                + (
                    c.bridge_end_y
                    - c.bridge_start_y
                )
                * t
            )

            sag_factor = (
                4.0
                * t
                * (1.0 - t)
            )

            y = (
                base_y
                + c.bridge_sag
                * sag_factor
            )

            points.append(
                (
                    x,
                    y,
                )
            )

        return points

    # ========================================================
    # Pymunk construction
    # ========================================================

    def _build_terrain(self):

        c = self.config

        self.terrain_shapes.extend(
            self._create_segment_chain(
                points=self.left_surface,
                radius=c.terrain_radius,
                friction=c.terrain_friction,
                elasticity=c.elasticity,
                collision_type=COLLISION_TERRAIN,
            )
        )

        self.terrain_shapes.extend(
            self._create_segment_chain(
                points=self.right_surface,
                radius=c.terrain_radius,
                friction=c.terrain_friction,
                elasticity=c.elasticity,
                collision_type=COLLISION_TERRAIN,
            )
        )

    def _build_bridge(self):

        c = self.config

        self.bridge_shapes.extend(
            self._create_segment_chain(
                points=self.bridge_points,
                radius=c.bridge_radius,
                friction=c.bridge_friction,
                elasticity=c.elasticity,
                collision_type=COLLISION_BRIDGE,
            )
        )

    def _create_segment_chain(
        self,
        *,
        points: list[tuple[float, float]],
        radius: float,
        friction: float,
        elasticity: float,
        collision_type: int,
    ) -> list[pymunk.Segment]:

        shapes = []

        for start, end in zip(
            points[:-1],
            points[1:],
        ):

            shape = pymunk.Segment(
                self.space.static_body,
                start,
                end,
                radius,
            )

            shape.friction = friction
            shape.elasticity = elasticity

            shape.collision_type = (
                collision_type
            )

            self.space.add(
                shape
            )

            shapes.append(
                shape
            )

        return shapes

    # ========================================================
    # Episode state
    # ========================================================

    def progress(
        self,
        position: pymunk.Vec2d,
    ) -> float:
        """
        Forward distance from starting position.
        """

        return float(
            position.x
            - self.config.spawn_x
        )

    def reached_goal(
        self,
        position: pymunk.Vec2d,
    ) -> bool:

        return (
            position.x
            >= self.config.goal_x
        )

    def fell_into_canyon(
        self,
        position: pymunk.Vec2d,
    ) -> bool:

        return (
            position.y
            >= self.config.death_y
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    @property
    def shape_count(self) -> int:

        return (
            len(self.terrain_shapes)
            + len(self.bridge_shapes)
        )

    @property
    def bridge_lowest_y(self) -> float:
        """
        Largest Y value because +Y points downward.
        """

        return max(
            point[1]
            for point in self.bridge_points
        )

    @property
    def bridge_length(self) -> float:
        """
        Actual curved bridge length rather than straight-line
        horizontal distance.
        """

        total = 0.0

        for start, end in zip(
            self.bridge_points[:-1],
            self.bridge_points[1:],
        ):

            dx = end[0] - start[0]
            dy = end[1] - start[1]

            total += math.hypot(
                dx,
                dy,
            )

        return total