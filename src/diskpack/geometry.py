"""
Geometry utilities for circle packing.

Contains:
- PolygonGeometry: boundary calculations, point-in-polygon tests
- SpatialIndex: grid-based spatial indexing for collision detection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Iterator, Tuple, Dict

# Type aliases
Polygon = np.ndarray
Point = np.ndarray
GridKey = Tuple[int, int]

# Threshold for switching between vectorized and spatial index approaches
VECTORIZED_THRESHOLD = 750


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
        self.extent = max(self.max_coords - self.min_coords)

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
        
        # Compute inward normals
        self.edge_normals = np.zeros_like(self.edge_vecs)
        for i, (start, vec) in enumerate(zip(self.edge_starts, self.edge_vecs)):
            normal = np.array([-vec[1], vec[0]])
            if self.edge_lengths[i] > 0:
                normal = normal / self.edge_lengths[i]
            midpoint = start + vec / 2
            test_point = midpoint + normal * 0.001
            if not self.contains_point(test_point):
                normal = -normal
            self.edge_normals[i] = normal

    def contains_point(self, point: Point) -> bool:
        """Check if a single point is inside the polygon (even-odd rule)."""
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
        """Vectorized even-odd rule for multiple points."""
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
        """Distance from a point to the nearest polygon edge."""
        to_point = point - self.edge_starts
        dots = np.sum(to_point * self.edge_vecs, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.clip(dots / self.edge_lengths_sq, 0, 1)
            t = np.where(self.edge_lengths_sq == 0, 0, t)

        projections = self.edge_starts + t[:, np.newaxis] * self.edge_vecs
        distances = np.linalg.norm(point - projections, axis=1)

        return float(np.min(distances))

    def distances_to_boundary_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorized distance calculation for multiple points."""
        if len(points) == 0:
            return np.array([])
            
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
        """Add a circle to the spatial index."""
        self._centers = np.vstack([self._centers, center]) if len(self._centers) > 0 else center.reshape(1, 2)
        self._radii = np.append(self._radii, radius)

        if radius > self.cell_size * self.mega_threshold:
            self.mega_circles.append(index)
        else:
            key = self._get_cell_key(center)
            self.grid.setdefault(key, []).append(index)

    def get_nearby_indices(self, point: Point) -> Iterator[int]:
        """Yield indices of circles that might be near a point."""
        yield from self.mega_circles
        center_key = self._get_cell_key(point)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_key = (center_key[0] + dx, center_key[1] + dy)
                if neighbor_key in self.grid:
                    yield from self.grid[neighbor_key]

    def get_circles_in_region(self, min_pt: Point, max_pt: Point) -> List[int]:
        """Get all circle indices that might intersect a rectangular region."""
        indices = set(self.mega_circles)
        
        min_key = self._get_cell_key(min_pt)
        max_key = self._get_cell_key(max_pt)
        
        for gx in range(min_key[0] - 1, max_key[0] + 2):
            for gy in range(min_key[1] - 1, max_key[1] + 2):
                if (gx, gy) in self.grid:
                    indices.update(self.grid[(gx, gy)])
        
        return list(indices)

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
        """Convert a point to its grid cell coordinates."""
        cell_coords = ((point - self.origin) // self.cell_size).astype(int)
        return (int(cell_coords[0]), int(cell_coords[1]))
