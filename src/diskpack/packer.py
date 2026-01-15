import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Tuple, Dict, Set
from enum import Enum

# Type aliases
Polygon = np.ndarray
Point = np.ndarray
GridKey = Tuple[int, int]
Circle = Tuple[float, float, float]

# Threshold for switching between vectorized and spatial index approaches
VECTORIZED_THRESHOLD = 750


class PackingMode(Enum):
    """Available packing strategies."""
    RANDOM = "random"           # Random sampling (original)
    HEX_GRID = "hex_grid"       # Hexagonal grid (fixed radius only)
    FRONT = "front"             # Front-based packing (highest density)


@dataclass
class PackingConfig:
    """Configuration parameters for the circle packing algorithm."""
    padding: float = 1.5
    min_radius: float = 1.0
    grid_resolution_divisor: float = 25
    max_failed_attempts: int = 200
    mega_circle_threshold: float = 0.5
    ray_cast_epsilon: float = 1e-10
    sample_batch_size: int = 50
    fixed_radius: Optional[float] = None
    use_hex_grid: bool = True
    use_front_packing: bool = False  # New option for front-based packing
    verbose: bool = False


@dataclass
class PackingProgress:
    """Tracks the current state of the packing algorithm."""
    circles_placed: int = 0
    failed_attempts: int = 0
    max_failed_attempts: int = 200

    @property
    def progress_ratio(self) -> float:
        """How close we are to stopping (0.0 = just started, 1.0 = about to stop)."""
        return self.failed_attempts / self.max_failed_attempts

    def __str__(self) -> str:
        return f"Placed: {self.circles_placed} | Failed attempts: {self.failed_attempts}/{self.max_failed_attempts} ({self.progress_ratio:.0%})"


class PolygonGeometry:
    """Handles geometric calculations for polygon boundaries."""

    def __init__(self, polygons: List[Polygon], epsilon: float = 1e-10):
        self.polygons = [np.array(p, dtype=float) for p in polygons]
        self.epsilon = epsilon
        self._compute_bounds()
        self._precompute_edges()

    def _compute_bounds(self) -> None:
        all_vertices = np.vstack(self.polygons)
        self.min_coords = np.min(all_vertices, axis=0)
        self.max_coords = np.max(all_vertices, axis=0)

    def _precompute_edges(self) -> None:
        """Precompute edge data for vectorized distance calculations."""
        all_p1 = []
        all_p2 = []
        for poly in self.polygons:
            n = len(poly)
            for i in range(n):
                all_p1.append(poly[i])
                all_p2.append(poly[(i + 1) % n])

        self.edge_starts = np.array(all_p1)
        self.edge_ends = np.array(all_p2)
        self.edge_vecs = self.edge_ends - self.edge_starts
        self.edge_lengths_sq = np.sum(self.edge_vecs ** 2, axis=1)
        self.edge_lengths = np.sqrt(self.edge_lengths_sq)
        
        # Compute inward normals for each edge
        self.edge_normals = np.zeros_like(self.edge_vecs)
        for i, (start, vec) in enumerate(zip(self.edge_starts, self.edge_vecs)):
            # Perpendicular (rotate 90 degrees)
            normal = np.array([-vec[1], vec[0]])
            if self.edge_lengths[i] > 0:
                normal = normal / self.edge_lengths[i]
            # Check if normal points inward (toward polygon center)
            midpoint = start + vec / 2
            test_point = midpoint + normal * 0.001
            if not self._point_in_polygon_single(test_point):
                normal = -normal
            self.edge_normals[i] = normal

    def _point_in_polygon_single(self, point: Point) -> bool:
        """Check if a single point is inside the polygon."""
        x, y = point[0], point[1]
        inside = False
        for poly in self.polygons:
            n = len(poly)
            for i in range(n):
                p1, p2 = poly[i], poly[(i + 1) % n]
                if ((p1[1] > y) != (p2[1] > y)) and \
                   (x < (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1] + self.epsilon) + p1[0]):
                    inside = not inside
        return inside

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Even-Odd Rule for interior detection, supports holes."""
        x, y = points[:, 0], points[:, 1]
        inside = np.zeros(len(points), dtype=bool)

        for poly in self.polygons:
            n = len(poly)
            for i in range(n):
                p1, p2 = poly[i], poly[(i + 1) % n]
                crosses_edge = (p1[1] > y) != (p2[1] > y)
                dy = p2[1] - p1[1] + self.epsilon
                x_intercept = (p2[0] - p1[0]) * (y - p1[1]) / dy + p1[0]
                inside ^= crosses_edge & (x < x_intercept)

        return inside

    def distance_to_boundary(self, point: Point) -> float:
        """Vectorized distance to nearest polygon edge."""
        to_point = point - self.edge_starts
        dots = np.sum(to_point * self.edge_vecs, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.clip(dots / self.edge_lengths_sq, 0, 1)
            t = np.where(self.edge_lengths_sq == 0, 0, t)

        projections = self.edge_starts + t[:, np.newaxis] * self.edge_vecs
        distances = np.linalg.norm(point - projections, axis=1)

        return float(np.min(distances))

    def distances_to_boundary_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorized distance calculation for multiple points at once."""
        to_point = points[:, np.newaxis, :] - self.edge_starts[np.newaxis, :, :]
        dots = np.sum(to_point * self.edge_vecs[np.newaxis, :, :], axis=2)

        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.clip(dots / self.edge_lengths_sq[np.newaxis, :], 0, 1)
            t = np.where(self.edge_lengths_sq[np.newaxis, :] == 0, 0, t)

        projections = (
            self.edge_starts[np.newaxis, :, :] +
            t[:, :, np.newaxis] * self.edge_vecs[np.newaxis, :, :]
        )
        distances = np.linalg.norm(points[:, np.newaxis, :] - projections, axis=2)

        return np.min(distances, axis=1)
    
    def closest_point_on_edge(self, point: Point, edge_idx: int) -> Tuple[Point, float]:
        """Find closest point on a specific edge and return it with distance."""
        start = self.edge_starts[edge_idx]
        vec = self.edge_vecs[edge_idx]
        length_sq = self.edge_lengths_sq[edge_idx]
        
        if length_sq == 0:
            return start, np.linalg.norm(point - start)
        
        t = np.clip(np.dot(point - start, vec) / length_sq, 0, 1)
        closest = start + t * vec
        return closest, np.linalg.norm(point - closest)


@dataclass
class SpatialIndex:
    """Grid-based spatial index for efficient collision detection."""
    cell_size: float
    origin: np.ndarray
    mega_threshold: float
    grid: Dict[GridKey, List[int]] = field(default_factory=dict)
    mega_circles: List[int] = field(default_factory=list)

    _centers: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    _radii: np.ndarray = field(default_factory=lambda: np.empty(0))

    def add_circle(self, index: int, center: Point, radius: float) -> None:
        self._centers = np.vstack([self._centers, center]) if len(self._centers) > 0 else center.reshape(1, 2)
        self._radii = np.append(self._radii, radius)

        if radius > self.cell_size * self.mega_threshold:
            self.mega_circles.append(index)
        else:
            key = self._get_cell_key(center)
            self.grid.setdefault(key, []).append(index)

    def get_nearby_indices(self, point: Point) -> Iterator[int]:
        yield from self.mega_circles
        center_key = self._get_cell_key(point)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_key = (center_key[0] + dx, center_key[1] + dy)
                if neighbor_key in self.grid:
                    yield from self.grid[neighbor_key]

    def distance_to_circles(self, point: Point) -> float:
        """Get minimum distance from point to any existing circle's edge."""
        if len(self._centers) == 0:
            return float('inf')

        indices = list(self.get_nearby_indices(point))
        if not indices:
            return float('inf')

        centers = self._centers[indices]
        radii = self._radii[indices]

        distances = np.linalg.norm(centers - point, axis=1) - radii
        return float(np.min(distances))

    def _get_cell_key(self, point: Point) -> GridKey:
        cell_coords = ((point - self.origin) // self.cell_size).astype(int)
        return (int(cell_coords[0]), int(cell_coords[1]))


@dataclass(order=True)
class FrontCandidate:
    """A candidate position for placing a circle in front-based packing."""
    priority: float  # Negative radius for max-heap behavior with min-heap
    center: Point = field(compare=False)
    radius: float = field(compare=False)
    source_type: str = field(compare=False)  # 'edge', 'circle-circle', 'circle-edge'
    source_ids: Tuple = field(compare=False)  # IDs of circles/edges that generated this candidate


class CirclePacker:
    """Packs circles within polygon boundaries using various strategies."""

    def __init__(self, polygons: List[Polygon], config: Optional[PackingConfig] = None):
        self.config = config or PackingConfig()
        self.geometry = PolygonGeometry(polygons, self.config.ray_cast_epsilon)
        self.centers: List[Point] = []
        self.radii: List[float] = []
        self.progress = PackingProgress(max_failed_attempts=self.config.max_failed_attempts)

        extent = max(self.geometry.max_coords - self.geometry.min_coords)
        cell_size = extent / self.config.grid_resolution_divisor
        self.spatial_index = SpatialIndex(
            cell_size=cell_size,
            origin=self.geometry.min_coords,
            mega_threshold=self.config.mega_circle_threshold
        )

        # Cache for numpy arrays
        self._centers_arr: Optional[np.ndarray] = None
        self._radii_arr: Optional[np.ndarray] = None
        self._cache_valid = False

    def _invalidate_cache(self) -> None:
        self._cache_valid = False

    def _get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._cache_valid or self._centers_arr is None:
            if len(self.centers) > 0:
                self._centers_arr = np.array(self.centers)
                self._radii_arr = np.array(self.radii)
            else:
                self._centers_arr = np.empty((0, 2))
                self._radii_arr = np.empty(0)
            self._cache_valid = True
        return self._centers_arr, self._radii_arr

    def _sample_candidate_points(self, count: int) -> np.ndarray:
        points = np.random.uniform(
            self.geometry.min_coords,
            self.geometry.max_coords,
            size=(count, 2)
        )
        return points[self.geometry.contains_points(points)]

    def _compute_max_radius(self, point: Point) -> float:
        max_radius = self.geometry.distance_to_boundary(point)
        circle_dist = self.spatial_index.distance_to_circles(point)
        max_radius = min(max_radius, circle_dist)
        return max_radius - self.config.padding

    def _compute_max_radii_batch(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([])

        max_radii = self.geometry.distances_to_boundary_batch(points)

        if len(self.centers) == 0:
            return max_radii - self.config.padding

        centers_arr, radii_arr = self._get_arrays()

        if len(self.centers) < VECTORIZED_THRESHOLD:
            dists = np.linalg.norm(
                points[:, np.newaxis, :] - centers_arr[np.newaxis, :, :],
                axis=2
            ) - radii_arr
            min_circle_dists = np.min(dists, axis=1)
            max_radii = np.minimum(max_radii, min_circle_dists)
        else:
            for i, point in enumerate(points):
                indices = list(self.spatial_index.get_nearby_indices(point))
                if indices:
                    nearby_centers = centers_arr[indices]
                    nearby_radii = radii_arr[indices]
                    distances = np.linalg.norm(nearby_centers - point, axis=1) - nearby_radii
                    max_radii[i] = min(max_radii[i], np.min(distances))

        return max_radii - self.config.padding

    def _find_best_placement(self, candidates: np.ndarray) -> Optional[Tuple[Point, float]]:
        if len(candidates) == 0:
            return None

        radii = self._compute_max_radii_batch(candidates)
        fixed = self.config.fixed_radius

        if fixed is not None:
            valid_mask = radii >= fixed
            if not np.any(valid_mask):
                return None
            valid_indices = np.where(valid_mask)[0]
            best_idx = valid_indices[0]
            return candidates[best_idx], fixed
        else:
            best_idx = np.argmax(radii)
            best_radius = radii[best_idx]

            if best_radius >= self.config.min_radius:
                return candidates[best_idx], best_radius
            return None

    def _place_circle(self, center: Point, radius: float) -> None:
        idx = len(self.centers)
        self.centers.append(center)
        self.radii.append(radius)
        self.spatial_index.add_circle(idx, center, radius)
        self._invalidate_cache()

    def _is_valid_placement(self, center: Point, radius: float) -> bool:
        """Check if a circle placement is valid (inside polygon, no overlaps)."""
        # Check if center is inside polygon
        if not self.geometry._point_in_polygon_single(center):
            return False
        
        # Check boundary distance
        boundary_dist = self.geometry.distance_to_boundary(center)
        if boundary_dist < radius + self.config.padding - 1e-9:
            return False
        
        # Check circle overlaps
        if len(self.centers) > 0:
            centers_arr, radii_arr = self._get_arrays()
            distances = np.linalg.norm(centers_arr - center, axis=1)
            min_allowed = radii_arr + radius + self.config.padding
            if np.any(distances < min_allowed - 1e-9):
                return False
        
        return True

    # =========================================================================
    # Hex Grid Packing
    # =========================================================================

    def _generate_hex_grid(self, radius: float) -> np.ndarray:
        spacing = (radius + self.config.padding) * 2
        dy = spacing * np.sqrt(3) / 2

        min_x, min_y = self.geometry.min_coords
        max_x, max_y = self.geometry.max_coords

        min_x -= spacing
        min_y -= spacing
        max_x += spacing
        max_y += spacing

        points = []
        row = 0
        y = min_y

        while y <= max_y:
            x_offset = (spacing / 2) if row % 2 else 0
            x = min_x + x_offset

            while x <= max_x:
                points.append([x, y])
                x += spacing

            y += dy
            row += 1

        return np.array(points) if points else np.empty((0, 2))

    def _pack_hex_grid(self) -> List[Circle]:
        radius = self.config.fixed_radius
        circles = []

        grid_points = self._generate_hex_grid(radius)

        if len(grid_points) == 0:
            return circles

        inside_mask = self.geometry.contains_points(grid_points)
        interior_points = grid_points[inside_mask]

        min_clearance = radius + self.config.padding
        boundary_distances = self.geometry.distances_to_boundary_batch(interior_points)
        valid_mask = boundary_distances >= min_clearance

        valid_points = interior_points[valid_mask]

        if self.config.verbose:
            print(f"Hex grid: {len(grid_points)} total -> {len(interior_points)} inside -> {len(valid_points)} valid")

        for point in valid_points:
            self._place_circle(point, radius)
            circles.append((float(point[0]), float(point[1]), float(radius)))

        if self.config.verbose:
            print(f"Done! Placed {len(circles)} circles")

        return circles

    # =========================================================================
    # Front-Based Packing
    # =========================================================================

    def _find_tangent_circle_two_circles(
        self, c1: Point, r1: float, c2: Point, r2: float, r: float
    ) -> List[Point]:
        """
        Find positions where a circle of radius r is tangent to two existing circles.
        Returns 0, 1, or 2 valid positions.
        """
        d = np.linalg.norm(c2 - c1)
        
        # Distance from each center to the new circle's center
        d1 = r1 + r + self.config.padding
        d2 = r2 + r + self.config.padding
        
        # Check if solution exists (triangle inequality)
        if d > d1 + d2 or d < abs(d1 - d2) or d < 1e-10:
            return []
        
        # Solve using law of cosines
        # d1^2 = d^2 + d2^2 - 2*d*d2*cos(angle at c2)
        # Actually easier: find intersection of two circles centered at c1, c2
        
        # Using the formula for circle-circle intersection
        a = (d1**2 - d2**2 + d**2) / (2 * d)
        h_sq = d1**2 - a**2
        
        if h_sq < 0:
            return []
        
        h = np.sqrt(h_sq)
        
        # Unit vector from c1 to c2
        u = (c2 - c1) / d
        # Perpendicular
        v = np.array([-u[1], u[0]])
        
        # Midpoint along c1-c2 axis
        p = c1 + a * u
        
        # Two solutions
        solutions = []
        if h < 1e-10:
            solutions.append(p)
        else:
            solutions.append(p + h * v)
            solutions.append(p - h * v)
        
        return solutions

    def _find_tangent_circle_edge(
        self, edge_idx: int, r: float
    ) -> List[Tuple[Point, float]]:
        """
        Find positions along an edge where circles of radius r can be placed.
        Returns list of (center, t) where t is position along edge [0, 1].
        """
        start = self.geometry.edge_starts[edge_idx]
        vec = self.geometry.edge_vecs[edge_idx]
        normal = self.geometry.edge_normals[edge_idx]
        length = self.geometry.edge_lengths[edge_idx]
        
        if length < 1e-10:
            return []
        
        # Circle center is offset from edge by radius + padding
        offset = r + self.config.padding
        
        # Generate positions along the edge
        positions = []
        spacing = (r + self.config.padding) * 2
        
        t = offset / length  # Start offset from edge start
        while t < 1 - offset / length:
            point_on_edge = start + t * vec
            center = point_on_edge + offset * normal
            positions.append((center, t))
            t += spacing / length
        
        return positions

    def _find_tangent_circle_circle_and_edge(
        self, circle_idx: int, edge_idx: int, r: float
    ) -> List[Point]:
        """
        Find positions where a circle of radius r is tangent to both
        an existing circle and a polygon edge.
        """
        c = self.centers[circle_idx]
        rc = self.radii[circle_idx]
        
        start = self.geometry.edge_starts[edge_idx]
        vec = self.geometry.edge_vecs[edge_idx]
        normal = self.geometry.edge_normals[edge_idx]
        length = self.geometry.edge_lengths[edge_idx]
        
        if length < 1e-10:
            return []
        
        # New circle must be:
        # 1. At distance rc + r + padding from circle center
        # 2. At distance r + padding from edge
        
        edge_offset = r + self.config.padding
        circle_dist = rc + r + self.config.padding
        
        # The center lies on a line parallel to the edge, offset by edge_offset
        # And on a circle around c with radius circle_dist
        
        # Line: point = start + t * vec + edge_offset * normal
        # Circle: |point - c| = circle_dist
        
        # Substitute: |start + t * vec + edge_offset * normal - c| = circle_dist
        # Let p0 = start + edge_offset * normal - c
        # |p0 + t * vec| = circle_dist
        # |p0|^2 + 2*t*(p0 . vec) + t^2*|vec|^2 = circle_dist^2
        
        p0 = start + edge_offset * normal - c
        a = np.dot(vec, vec)  # |vec|^2
        b = 2 * np.dot(p0, vec)
        c_coef = np.dot(p0, p0) - circle_dist**2
        
        discriminant = b**2 - 4 * a * c_coef
        
        if discriminant < 0:
            return []
        
        solutions = []
        sqrt_disc = np.sqrt(discriminant)
        
        for t in [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]:
            if 0 <= t <= 1:
                center = start + t * vec + edge_offset * normal
                solutions.append(center)
        
        return solutions

    def _get_max_radius_at_point(self, center: Point) -> float:
        """Get the maximum radius that can fit at a given center point."""
        # Distance to boundary
        max_r = self.geometry.distance_to_boundary(center)
        
        # Distance to existing circles
        if len(self.centers) > 0:
            centers_arr, radii_arr = self._get_arrays()
            distances = np.linalg.norm(centers_arr - center, axis=1) - radii_arr
            max_r = min(max_r, np.min(distances))
        
        return max_r - self.config.padding

    def _pack_front(self) -> List[Circle]:
        """
        Pack circles using front-based algorithm.
        Achieves higher density by systematically filling from edges inward.
        """
        circles = []
        min_r = self.config.min_radius
        fixed_r = self.config.fixed_radius
        
        # Priority queue: (negative_radius, center, radius, source_info)
        # Using negative radius for max-heap behavior
        candidates: List[Tuple[float, int, Point, float]] = []
        candidate_id = 0
        
        # Track which circle pairs we've already processed
        processed_pairs: Set[Tuple[int, int]] = set()
        processed_circle_edge: Set[Tuple[int, int]] = set()
        
        def add_candidate(center: Point, radius: float):
            nonlocal candidate_id
            if radius >= min_r:
                heapq.heappush(candidates, (-radius, candidate_id, center, radius))
                candidate_id += 1

        # Phase 1: Seed candidates along all edges
        if self.config.verbose:
            print("Phase 1: Seeding edge candidates...")
        
        for edge_idx in range(len(self.geometry.edge_starts)):
            if fixed_r is not None:
                # Fixed radius mode: place along edges
                edge_positions = self._find_tangent_circle_edge(edge_idx, fixed_r)
                for center, t in edge_positions:
                    if self._is_valid_placement(center, fixed_r):
                        add_candidate(center, fixed_r)
            else:
                # Variable radius: sample points along edge and compute max radius
                edge_positions = self._find_tangent_circle_edge(edge_idx, min_r)
                for center, t in edge_positions:
                    if self.geometry._point_in_polygon_single(center):
                        max_r = self._get_max_radius_at_point(center)
                        if max_r >= min_r:
                            add_candidate(center, max_r)
        
        if self.config.verbose:
            print(f"  Initial candidates: {len(candidates)}")
        
        # Phase 2: Main loop - place circles and generate new candidates
        if self.config.verbose:
            print("Phase 2: Placing circles...")
        
        iterations = 0
        max_iterations = 100000  # Safety limit
        
        while candidates and iterations < max_iterations:
            iterations += 1
            
            # Pop best candidate
            neg_radius, _, center, radius = heapq.heappop(candidates)
            
            # Recompute max radius (things may have changed)
            if fixed_r is not None:
                actual_radius = fixed_r
                if not self._is_valid_placement(center, actual_radius):
                    continue
            else:
                actual_radius = self._get_max_radius_at_point(center)
                if actual_radius < min_r:
                    continue
                if not self._is_valid_placement(center, actual_radius):
                    continue
            
            # Place the circle
            self._place_circle(center, actual_radius)
            circles.append((float(center[0]), float(center[1]), float(actual_radius)))
            
            if self.config.verbose and len(circles) % 50 == 0:
                print(f"  Placed {len(circles)} circles, {len(candidates)} candidates remaining")
            
            new_circle_idx = len(self.centers) - 1
            
            # Generate new candidates from circle-circle tangencies
            for other_idx in range(new_circle_idx):
                pair = (min(other_idx, new_circle_idx), max(other_idx, new_circle_idx))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                other_center = self.centers[other_idx]
                other_radius = self.radii[other_idx]
                
                # Try to find tangent circles
                if fixed_r is not None:
                    tangent_centers = self._find_tangent_circle_two_circles(
                        center, actual_radius, other_center, other_radius, fixed_r
                    )
                    for tc in tangent_centers:
                        if self._is_valid_placement(tc, fixed_r):
                            add_candidate(tc, fixed_r)
                else:
                    # For variable radius, try a few different radii
                    for test_r in [min_r, min_r * 2, min_r * 4]:
                        tangent_centers = self._find_tangent_circle_two_circles(
                            center, actual_radius, other_center, other_radius, test_r
                        )
                        for tc in tangent_centers:
                            if self.geometry._point_in_polygon_single(tc):
                                max_r = self._get_max_radius_at_point(tc)
                                if max_r >= min_r:
                                    add_candidate(tc, max_r)
            
            # Generate new candidates from circle-edge tangencies
            for edge_idx in range(len(self.geometry.edge_starts)):
                ce_pair = (new_circle_idx, edge_idx)
                if ce_pair in processed_circle_edge:
                    continue
                processed_circle_edge.add(ce_pair)
                
                if fixed_r is not None:
                    tangent_centers = self._find_tangent_circle_circle_and_edge(
                        new_circle_idx, edge_idx, fixed_r
                    )
                    for tc in tangent_centers:
                        if self._is_valid_placement(tc, fixed_r):
                            add_candidate(tc, fixed_r)
                else:
                    for test_r in [min_r, min_r * 2]:
                        tangent_centers = self._find_tangent_circle_circle_and_edge(
                            new_circle_idx, edge_idx, test_r
                        )
                        for tc in tangent_centers:
                            if self.geometry._point_in_polygon_single(tc):
                                max_r = self._get_max_radius_at_point(tc)
                                if max_r >= min_r:
                                    add_candidate(tc, max_r)
        
        if self.config.verbose:
            print(f"Done! Placed {len(circles)} circles in {iterations} iterations")
        
        return circles

    # =========================================================================
    # Random Sampling Packing
    # =========================================================================

    def _pack_random(self) -> Iterator[Circle]:
        self.progress = PackingProgress(max_failed_attempts=self.config.max_failed_attempts)

        while self.progress.failed_attempts < self.config.max_failed_attempts:
            candidates = self._sample_candidate_points(self.config.sample_batch_size)
            result = self._find_best_placement(candidates)

            if result is not None:
                center, radius = result
                self._place_circle(center, radius)
                self.progress.circles_placed += 1
                self.progress.failed_attempts = 0

                if self.config.verbose and self.progress.circles_placed % 25 == 0:
                    print(self.progress)

                yield (float(center[0]), float(center[1]), float(radius))
            else:
                self.progress.failed_attempts += 1

                if self.config.verbose and self.progress.failed_attempts % 50 == 0:
                    print(self.progress)

        if self.config.verbose:
            print(f"Done! {self.progress}")

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def generate(self) -> Iterator[Circle]:
        """
        Generate circles until no more can be placed.

        Strategy selection:
        1. If use_front_packing=True: use front-based algorithm (highest density)
        2. Else if fixed_radius and use_hex_grid=True: use hex grid (fastest for fixed)
        3. Else: use random sampling

        Yields:
            Tuples of (x, y, radius) for each placed circle.
        """
        # Front-based packing (highest density)
        if self.config.use_front_packing:
            yield from self._pack_front()
            return
        
        # Hex grid for fixed radius (unless disabled)
        if self.config.fixed_radius is not None and self.config.use_hex_grid:
            yield from self._pack_hex_grid()
            return

        # Random sampling (original method)
        yield from self._pack_random()

    def pack(self) -> List[Circle]:
        """Pack circles and return them as a list."""
        return list(self.generate())
