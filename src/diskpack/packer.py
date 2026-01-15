import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Tuple, Dict

# Type aliases
Polygon = np.ndarray
Point = np.ndarray
GridKey = Tuple[int, int]
Circle = Tuple[float, float, float]


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
        """
        Vectorized distance calculation for multiple points at once.
        """
        n_points = len(points)
        n_edges = len(self.edge_starts)

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


class CirclePacker:
    """Packs circles within polygon boundaries using random sampling."""

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

    def _sample_candidate_points(self, count: int) -> np.ndarray:
        points = np.random.uniform(
            self.geometry.min_coords,
            self.geometry.max_coords,
            size=(count, 2)
        )
        return points[self.geometry.contains_points(points)]

    def _compute_max_radius(self, point: Point) -> float:
        """Compute max radius for a single point."""
        max_radius = self.geometry.distance_to_boundary(point)
        circle_dist = self.spatial_index.distance_to_circles(point)
        max_radius = min(max_radius, circle_dist)
        return max_radius - self.config.padding

    def _compute_max_radii_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorized max radius computation for multiple points."""
        if len(points) == 0:
            return np.array([])

        max_radii = self.geometry.distances_to_boundary_batch(points)

        if len(self.centers) > 0:
            centers_arr = np.array(self.centers)
            radii_arr = np.array(self.radii)

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

    def _generate_hex_grid(self, radius: float) -> np.ndarray:
        """
        Generate a hexagonal grid of points within the bounding box.
        Hex grid is the optimal packing arrangement for equal circles.
        """
        spacing = (radius + self.config.padding) * 2
        dy = spacing * np.sqrt(3) / 2

        min_x, min_y = self.geometry.min_coords
        max_x, max_y = self.geometry.max_coords

        # Add margin to ensure coverage
        min_x -= spacing
        min_y -= spacing
        max_x += spacing
        max_y += spacing

        points = []
        row = 0
        y = min_y

        while y <= max_y:
            # Offset every other row by half spacing
            x_offset = (spacing / 2) if row % 2 else 0
            x = min_x + x_offset

            while x <= max_x:
                points.append([x, y])
                x += spacing

            y += dy
            row += 1

        return np.array(points) if points else np.empty((0, 2))

    def _pack_hex_grid(self) -> List[Circle]:
        """
        Pack circles using hexagonal grid placement.
        Much faster and denser than random sampling for fixed radius.
        """
        radius = self.config.fixed_radius
        circles = []

        # Generate hex grid
        grid_points = self._generate_hex_grid(radius)

        if len(grid_points) == 0:
            return circles

        # Filter to points inside polygon
        inside_mask = self.geometry.contains_points(grid_points)
        interior_points = grid_points[inside_mask]

        # Filter to points with enough clearance from boundary
        min_clearance = radius + self.config.padding
        boundary_distances = self.geometry.distances_to_boundary_batch(interior_points)
        valid_mask = boundary_distances >= min_clearance

        valid_points = interior_points[valid_mask]

        if self.config.verbose:
            print(f"Hex grid: {len(grid_points)} total -> {len(interior_points)} inside -> {len(valid_points)} valid")

        # All valid points become circles (no collision check needed - hex grid guarantees no overlap)
        for point in valid_points:
            self._place_circle(point, radius)
            circles.append((float(point[0]), float(point[1]), float(radius)))

        if self.config.verbose:
            print(f"Done! Placed {len(circles)} circles")

        return circles

    def _pack_random(self) -> Iterator[Circle]:
        """
        Pack circles using random sampling.
        Used for variable radius mode or when organic placement is desired.
        """
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

    def generate(self) -> Iterator[Circle]:
        """
        Generate circles until no more can be placed.

        For fixed_radius mode with use_hex_grid=True (default), uses optimized hex grid.
        For fixed_radius mode with use_hex_grid=False, uses random sampling for organic look.
        For variable radius mode, uses random sampling with best-fit selection.

        Yields:
            Tuples of (x, y, radius) for each placed circle.
        """
        # Use hex grid for fixed radius (unless disabled)
        if self.config.fixed_radius is not None and self.config.use_hex_grid:
            yield from self._pack_hex_grid()
            return

        # Random sampling mode (variable radius OR fixed radius with organic placement)
        yield from self._pack_random()

    def pack(self) -> List[Circle]:
        """Pack circles and return them as a list."""
        return list(self.generate())
