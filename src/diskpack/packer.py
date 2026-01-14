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

    def _compute_bounds(self) -> None:
        all_vertices = np.vstack(self.polygons)
        self.min_coords = np.min(all_vertices, axis=0)
        self.max_coords = np.max(all_vertices, axis=0)

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
        min_distance = float('inf')
        for poly in self.polygons:
            for i in range(len(poly)):
                p1, p2 = poly[i], poly[(i + 1) % len(poly)]
                min_distance = min(min_distance, self._point_to_segment_distance(point, p1, p2))
        return min_distance

    @staticmethod
    def _point_to_segment_distance(point: Point, seg_start: Point, seg_end: Point) -> float:
        segment_vec = seg_end - seg_start
        segment_length_sq = np.sum(segment_vec ** 2)
        if segment_length_sq == 0:
            return np.linalg.norm(point - seg_start)
        t = np.clip(np.dot(point - seg_start, segment_vec) / segment_length_sq, 0, 1)
        projection = seg_start + t * segment_vec
        return np.linalg.norm(point - projection)


@dataclass
class SpatialIndex:
    """Grid-based spatial index for efficient collision detection."""
    cell_size: float
    origin: np.ndarray
    mega_threshold: float
    grid: Dict[GridKey, List[int]] = field(default_factory=dict)
    mega_circles: List[int] = field(default_factory=list)

    def add_circle(self, index: int, center: Point, radius: float) -> None:
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
        max_radius = self.geometry.distance_to_boundary(point)
        for idx in self.spatial_index.get_nearby_indices(point):
            distance_to_circle = np.linalg.norm(self.centers[idx] - point) - self.radii[idx]
            max_radius = min(max_radius, distance_to_circle)
        return max_radius - self.config.padding

    def _find_best_placement(self, candidates: np.ndarray, fixed_radius: Optional[float]) -> Optional[Tuple[Point, float]]:
        best_point, best_radius = None, 0
        for point in candidates:
            radius = self._compute_max_radius(point)
            if fixed_radius is not None:
                radius = fixed_radius if radius >= fixed_radius else -1
            if radius > best_radius:
                best_point, best_radius = point, radius

        if best_point is not None and best_radius >= self.config.min_radius:
            return best_point, best_radius
        return None

    def _place_circle(self, center: Point, radius: float) -> None:
        idx = len(self.centers)
        self.centers.append(center)
        self.radii.append(radius)
        self.spatial_index.add_circle(idx, center, radius)

    def generate(self, fixed_radius: Optional[float] = None, verbose: bool = False) -> Iterator[Circle]:
        """
        Generate circles until no more can be placed.
        
        Args:
            fixed_radius: If provided, all circles will have this exact radius.
            verbose: If True, print progress updates periodically.
        
        Yields:
            Tuples of (x, y, radius) for each placed circle.
        """
        self.progress = PackingProgress(max_failed_attempts=self.config.max_failed_attempts)
        
        while self.progress.failed_attempts < self.config.max_failed_attempts:
            candidates = self._sample_candidate_points(self.config.sample_batch_size)
            result = self._find_best_placement(candidates, fixed_radius)
            
            if result is not None:
                center, radius = result
                self._place_circle(center, radius)
                self.progress.circles_placed += 1
                self.progress.failed_attempts = 0
                
                if verbose and self.progress.circles_placed % 25 == 0:
                    print(self.progress)
                
                yield (float(center[0]), float(center[1]), float(radius))
            else:
                self.progress.failed_attempts += 1
                
                if verbose and self.progress.failed_attempts % 50 == 0:
                    print(self.progress)

        if verbose:
            print(f"Done! {self.progress}")

    def pack(self, fixed_radius: Optional[float] = None, verbose: bool = False) -> List[Circle]:
        """Pack circles and return them as a list."""
        return list(self.generate(fixed_radius, verbose=verbose))
