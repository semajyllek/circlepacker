"""
Main circle packing implementation.

Contains CirclePacker class with multiple packing strategies:
- Random sampling (original)
- Hex grid (optimal for fixed radius)
- Front-based (fills corners well)
- Hybrid (state-of-the-art, combines all approaches)
"""

import numpy as np
import heapq
from typing import List, Optional, Iterator, Tuple, Set

from .config import PackingConfig, PackingProgress, Circle, Point, Polygon
from .geometry import PolygonGeometry, SpatialIndex, VECTORIZED_THRESHOLD


class CirclePacker:
    """
    State-of-the-art circle packer with multiple strategies.
    
    Usage:
        # Basic usage
        packer = CirclePacker([polygon_vertices])
        circles = packer.pack()
        
        # With configuration
        config = PackingConfig(use_hybrid_packing=True, verbose=True)
        packer = CirclePacker([polygon_vertices], config)
        circles = packer.pack()
        
        # Fixed radius
        config = PackingConfig(fixed_radius=5.0)
        packer = CirclePacker([polygon_vertices], config)
        circles = packer.pack()
    """

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

        self._centers_arr: Optional[np.ndarray] = None
        self._radii_arr: Optional[np.ndarray] = None
        self._cache_valid = False
        
        self._max_possible_radius = self._estimate_max_radius()

    def _estimate_max_radius(self) -> float:
        """Estimate the maximum radius that could fit in this polygon."""
        n_samples = 100
        points = np.random.uniform(
            self.geometry.min_coords,
            self.geometry.max_coords,
            size=(n_samples, 2)
        )
        inside = self.geometry.contains_points(points)
        if not np.any(inside):
            return self.geometry.extent / 4
        
        interior_points = points[inside]
        distances = self.geometry.distances_to_boundary_batch(interior_points)
        return float(np.max(distances))

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

    def _get_max_radius_at_point(self, center: Point) -> float:
        max_r = self.geometry.distance_to_boundary(center)
        
        if len(self.centers) > 0:
            centers_arr, radii_arr = self._get_arrays()
            
            if len(self.centers) < VECTORIZED_THRESHOLD:
                distances = np.linalg.norm(centers_arr - center, axis=1) - radii_arr
                max_r = min(max_r, np.min(distances))
            else:
                indices = list(self.spatial_index.get_nearby_indices(center))
                if indices:
                    distances = np.linalg.norm(centers_arr[indices] - center, axis=1) - radii_arr[indices]
                    max_r = min(max_r, np.min(distances))
        
        return max_r - self.config.padding

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

    def _is_valid_placement(self, center: Point, radius: float) -> bool:
        if not self.geometry.contains_point(center):
            return False
        
        boundary_dist = self.geometry.distance_to_boundary(center)
        if boundary_dist < radius + self.config.padding - 1e-9:
            return False
        
        if len(self.centers) > 0:
            centers_arr, radii_arr = self._get_arrays()
            distances = np.linalg.norm(centers_arr - center, axis=1)
            min_allowed = radii_arr + radius + self.config.padding
            if np.any(distances < min_allowed - 1e-9):
                return False
        
        return True

    def _place_circle(self, center: Point, radius: float) -> None:
        idx = len(self.centers)
        self.centers.append(center)
        self.radii.append(radius)
        self.spatial_index.add_circle(idx, center, radius)
        self._invalidate_cache()

    # =========================================================================
    # Tangent Circle Geometry (for front-based packing)
    # =========================================================================

    def _find_tangent_circle_two_circles(
        self, c1: Point, r1: float, c2: Point, r2: float, r: float
    ) -> List[Point]:
        """Find positions where a circle of radius r is tangent to two circles."""
        d = np.linalg.norm(c2 - c1)
        d1 = r1 + r + self.config.padding
        d2 = r2 + r + self.config.padding
        
        if d > d1 + d2 or d < abs(d1 - d2) or d < 1e-10:
            return []
        
        a = (d1**2 - d2**2 + d**2) / (2 * d)
        h_sq = d1**2 - a**2
        
        if h_sq < 0:
            return []
        
        h = np.sqrt(h_sq)
        u = (c2 - c1) / d
        v = np.array([-u[1], u[0]])
        p = c1 + a * u
        
        solutions = []
        if h < 1e-10:
            solutions.append(p)
        else:
            solutions.append(p + h * v)
            solutions.append(p - h * v)
        
        return solutions

    def _find_tangent_circle_edge(self, edge_idx: int, r: float) -> List[Tuple[Point, float]]:
        """Find positions along an edge where circles of radius r can be placed."""
        start = self.geometry.edge_starts[edge_idx]
        vec = self.geometry.edge_vecs[edge_idx]
        normal = self.geometry.edge_normals[edge_idx]
        length = self.geometry.edge_lengths[edge_idx]
        
        if length < 1e-10:
            return []
        
        offset = r + self.config.padding
        positions = []
        spacing = (r + self.config.padding) * 2
        
        t = offset / length
        while t < 1 - offset / length:
            point_on_edge = start + t * vec
            center = point_on_edge + offset * normal
            positions.append((center, t))
            t += spacing / length
        
        return positions

    def _find_tangent_circle_circle_and_edge(
        self, circle_idx: int, edge_idx: int, r: float
    ) -> List[Point]:
        """Find positions where a circle is tangent to both a circle and an edge."""
        c = self.centers[circle_idx]
        rc = self.radii[circle_idx]
        
        start = self.geometry.edge_starts[edge_idx]
        vec = self.geometry.edge_vecs[edge_idx]
        normal = self.geometry.edge_normals[edge_idx]
        length = self.geometry.edge_lengths[edge_idx]
        
        if length < 1e-10:
            return []
        
        edge_offset = r + self.config.padding
        circle_dist = rc + r + self.config.padding
        
        p0 = start + edge_offset * normal - c
        a = np.dot(vec, vec)
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

    # =========================================================================
    # Hex Grid Packing
    # =========================================================================

    def _generate_hex_grid(self, radius: float, min_pt: Point = None, max_pt: Point = None) -> np.ndarray:
        """Generate a hexagonal grid of points."""
        spacing = (radius + self.config.padding) * 2
        dy = spacing * np.sqrt(3) / 2

        if min_pt is None:
            min_pt = self.geometry.min_coords
        if max_pt is None:
            max_pt = self.geometry.max_coords

        min_x, min_y = min_pt[0] - spacing, min_pt[1] - spacing
        max_x, max_y = max_pt[0] + spacing, max_pt[1] + spacing

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

    def _pack_hex_grid(self, radius: float = None, min_pt: Point = None, max_pt: Point = None) -> List[Circle]:
        """Pack circles using hexagonal grid placement."""
        if radius is None:
            radius = self.config.fixed_radius
        circles = []

        grid_points = self._generate_hex_grid(radius, min_pt, max_pt)

        if len(grid_points) == 0:
            return circles

        inside_mask = self.geometry.contains_points(grid_points)
        interior_points = grid_points[inside_mask]

        if len(interior_points) == 0:
            return circles

        min_clearance = radius + self.config.padding
        boundary_distances = self.geometry.distances_to_boundary_batch(interior_points)
        valid_mask = boundary_distances >= min_clearance

        valid_points = interior_points[valid_mask]

        # Check against existing circles
        final_points = []
        for point in valid_points:
            if self._is_valid_placement(point, radius):
                final_points.append(point)

        if self.config.verbose:
            print(f"Hex grid: {len(grid_points)} total -> {len(interior_points)} inside -> {len(final_points)} valid")

        for point in final_points:
            self._place_circle(point, radius)
            circles.append((float(point[0]), float(point[1]), float(radius)))

        return circles

    # =========================================================================
    # Random Sampling Packing
    # =========================================================================

    def _pack_random(self, min_radius: float = None, max_attempts: int = None) -> List[Circle]:
        """Pack circles using random sampling."""
        if min_radius is None:
            min_radius = self.config.min_radius
        if max_attempts is None:
            max_attempts = self.config.max_failed_attempts
            
        circles = []
        consecutive_fails = 0

        while consecutive_fails < max_attempts:
            candidates = self._sample_candidate_points(self.config.sample_batch_size)
            
            if len(candidates) == 0:
                consecutive_fails += 1
                continue
                
            radii = self._compute_max_radii_batch(candidates)
            
            if self.config.fixed_radius is not None:
                valid_mask = radii >= self.config.fixed_radius
                if not np.any(valid_mask):
                    consecutive_fails += 1
                    continue
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[0]
                best_radius = self.config.fixed_radius
            else:
                best_idx = np.argmax(radii)
                best_radius = radii[best_idx]
            
            if best_radius >= min_radius:
                center = candidates[best_idx]
                self._place_circle(center, best_radius)
                circles.append((float(center[0]), float(center[1]), float(best_radius)))
                consecutive_fails = 0
                
                if self.config.verbose and len(circles) % 25 == 0:
                    self.progress.circles_placed = len(self.centers)
                    print(self.progress)
            else:
                consecutive_fails += 1

        return circles

    # =========================================================================
    # Front-Based Packing
    # =========================================================================

    def _pack_front(self, min_radius_threshold: float = 0.0) -> List[Circle]:
        """
        Front-based packing with optional minimum radius threshold.
        Only places circles >= min_radius_threshold * max_possible_radius.
        """
        circles = []
        min_r = self.config.min_radius
        fixed_r = self.config.fixed_radius
        
        effective_min = max(min_r, min_radius_threshold * self._max_possible_radius)
        
        candidates: List[Tuple[float, int, Point, float]] = []
        candidate_id = 0
        
        processed_pairs: Set[Tuple[int, int]] = set()
        processed_circle_edge: Set[Tuple[int, int]] = set()
        
        def add_candidate(center: Point, radius: float):
            nonlocal candidate_id
            if radius >= effective_min:
                heapq.heappush(candidates, (-radius, candidate_id, center, radius))
                candidate_id += 1

        # Seed candidates along edges
        for edge_idx in range(len(self.geometry.edge_starts)):
            if fixed_r is not None:
                edge_positions = self._find_tangent_circle_edge(edge_idx, fixed_r)
                for center, t in edge_positions:
                    if self._is_valid_placement(center, fixed_r):
                        add_candidate(center, fixed_r)
            else:
                for test_r in [effective_min, effective_min * 2, effective_min * 4, self._max_possible_radius * 0.5]:
                    if test_r < effective_min:
                        continue
                    edge_positions = self._find_tangent_circle_edge(edge_idx, test_r)
                    for center, t in edge_positions:
                        if self.geometry.contains_point(center):
                            max_r = self._get_max_radius_at_point(center)
                            if max_r >= effective_min:
                                add_candidate(center, max_r)

        iterations = 0
        max_iterations = 100000
        
        while candidates and iterations < max_iterations:
            iterations += 1
            
            neg_radius, _, center, radius = heapq.heappop(candidates)
            
            if fixed_r is not None:
                actual_radius = fixed_r
                if not self._is_valid_placement(center, actual_radius):
                    continue
            else:
                actual_radius = self._get_max_radius_at_point(center)
                if actual_radius < effective_min:
                    continue
                if not self._is_valid_placement(center, actual_radius):
                    continue
            
            self._place_circle(center, actual_radius)
            circles.append((float(center[0]), float(center[1]), float(actual_radius)))
            
            new_circle_idx = len(self.centers) - 1
            
            # Generate circle-circle candidates
            for other_idx in range(new_circle_idx):
                pair = (min(other_idx, new_circle_idx), max(other_idx, new_circle_idx))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                other_center = self.centers[other_idx]
                other_radius = self.radii[other_idx]
                
                if fixed_r is not None:
                    tangent_centers = self._find_tangent_circle_two_circles(
                        center, actual_radius, other_center, other_radius, fixed_r
                    )
                    for tc in tangent_centers:
                        if self._is_valid_placement(tc, fixed_r):
                            add_candidate(tc, fixed_r)
                else:
                    for test_r in [effective_min, effective_min * 2, effective_min * 4]:
                        tangent_centers = self._find_tangent_circle_two_circles(
                            center, actual_radius, other_center, other_radius, test_r
                        )
                        for tc in tangent_centers:
                            if self.geometry.contains_point(tc):
                                max_r = self._get_max_radius_at_point(tc)
                                if max_r >= effective_min:
                                    add_candidate(tc, max_r)
            
            # Generate circle-edge candidates
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
                    for test_r in [effective_min, effective_min * 2]:
                        tangent_centers = self._find_tangent_circle_circle_and_edge(
                            new_circle_idx, edge_idx, test_r
                        )
                        for tc in tangent_centers:
                            if self.geometry.contains_point(tc):
                                max_r = self._get_max_radius_at_point(tc)
                                if max_r >= effective_min:
                                    add_candidate(tc, max_r)

        return circles

    # =========================================================================
    # Hybrid Packing (State-of-the-Art)
    # =========================================================================

    def _find_gaps(self, min_gap_size: float) -> List[Tuple[Point, float]]:
        """Find gaps in the current packing where more circles could fit."""
        gaps = []
        
        grid_spacing = min_gap_size
        x_range = np.arange(
            self.geometry.min_coords[0] + grid_spacing,
            self.geometry.max_coords[0],
            grid_spacing
        )
        y_range = np.arange(
            self.geometry.min_coords[1] + grid_spacing,
            self.geometry.max_coords[1],
            grid_spacing
        )
        
        for x in x_range:
            for y in y_range:
                point = np.array([x, y])
                if not self.geometry.contains_point(point):
                    continue
                
                max_r = self._get_max_radius_at_point(point)
                if max_r >= min_gap_size:
                    gaps.append((point, max_r))
        
        return gaps

    def _pack_hybrid(self) -> List[Circle]:
        """
        State-of-the-art hybrid packing algorithm.
        
        Phase 1: Large circles using front-based packing
        Phase 2: Medium circles using front-based packing
        Phase 3: Small circles using micro hex grids in gaps
        Phase 4: Cleanup using targeted random sampling
        """
        all_circles = []
        
        if self.config.fixed_radius is not None:
            # Fixed radius mode
            if self.config.verbose:
                print("=" * 60)
                print("HYBRID MODE (Fixed Radius)")
                print("=" * 60)
            
            self.progress.phase = "Front"
            if self.config.verbose:
                print(f"\nPhase 1: Front-based placement (r={self.config.fixed_radius})")
            
            circles = self._pack_front()
            all_circles.extend(circles)
            
            if self.config.verbose:
                print(f"  Placed {len(circles)} circles")
            
            # Fill remaining gaps with hex grid
            self.progress.phase = "Hex Fill"
            if self.config.verbose:
                print(f"\nPhase 2: Hex grid gap filling")
            
            before = len(all_circles)
            circles = self._pack_hex_grid(self.config.fixed_radius)
            all_circles.extend(circles)
            
            if self.config.verbose:
                print(f"  Added {len(all_circles) - before} circles")
            
        else:
            # Variable radius mode
            if self.config.verbose:
                print("=" * 60)
                print("HYBRID MODE (Variable Radius)")
                print("=" * 60)
                print(f"Estimated max radius: {self._max_possible_radius:.1f}")
            
            # Phase 1: Large circles
            self.progress.phase = "Large"
            large_threshold = self.config.hybrid_large_threshold
            if self.config.verbose:
                print(f"\nPhase 1: Large circles (>= {large_threshold*100:.0f}% of max)")
            
            circles = self._pack_front(min_radius_threshold=large_threshold)
            all_circles.extend(circles)
            
            if self.config.verbose:
                print(f"  Placed {len(circles)} large circles")
            
            # Phase 2: Medium circles
            self.progress.phase = "Medium"
            medium_threshold = self.config.hybrid_medium_threshold
            if self.config.verbose:
                print(f"\nPhase 2: Medium circles (>= {medium_threshold*100:.0f}% of max)")
            
            before = len(all_circles)
            circles = self._pack_front(min_radius_threshold=medium_threshold)
            all_circles.extend(circles)
            
            if self.config.verbose:
                print(f"  Placed {len(all_circles) - before} medium circles")
            
            # Phase 3: Micro hex grid fill
            self.progress.phase = "Micro-fill"
            min_gap = self.config.hybrid_micro_grid_min_gap
            if self.config.verbose:
                print(f"\nPhase 3: Micro hex grid fill (gaps >= {min_gap})")
            
            gaps = self._find_gaps(min_gap)
            before = len(all_circles)
            
            for gap_center, gap_radius in gaps:
                micro_radius = max(self.config.min_radius, gap_radius * 0.4)
                min_pt = gap_center - gap_radius
                max_pt = gap_center + gap_radius
                circles = self._pack_hex_grid(micro_radius, min_pt, max_pt)
                all_circles.extend(circles)
            
            if self.config.verbose:
                print(f"  Filled {len(gaps)} gaps, added {len(all_circles) - before} circles")
            
            # Phase 4: Random cleanup
            self.progress.phase = "Cleanup"
            if self.config.verbose:
                print(f"\nPhase 4: Random sampling cleanup")
            
            before = len(all_circles)
            circles = self._pack_random(
                min_radius=self.config.min_radius,
                max_attempts=self.config.max_failed_attempts // 2
            )
            all_circles.extend(circles)
            
            if self.config.verbose:
                print(f"  Added {len(all_circles) - before} cleanup circles")
        
        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"TOTAL: {len(all_circles)} circles")
            print("=" * 60)
        
        return all_circles

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def generate(self) -> Iterator[Circle]:
        """
        Generate circles using the configured strategy.

        Strategy priority:
        1. use_hybrid_packing=True: State-of-the-art multi-phase algorithm
        2. use_front_packing=True: Front-based algorithm
        3. fixed_radius + use_hex_grid=True: Hex grid (fastest for fixed)
        4. Default: Random sampling

        Yields:
            Tuples of (x, y, radius) for each placed circle.
        """
        if self.config.use_hybrid_packing:
            yield from self._pack_hybrid()
            return
        
        if self.config.use_front_packing:
            yield from self._pack_front()
            return
        
        if self.config.fixed_radius is not None and self.config.use_hex_grid:
            yield from self._pack_hex_grid()
            return

        yield from self._pack_random()

    def pack(self) -> List[Circle]:
        """Pack circles and return them as a list."""
        return list(self.generate())
