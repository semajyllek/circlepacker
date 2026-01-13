import unittest
import numpy as np
from circlepacker.packer import CirclePacker

class TestPackerLogic(unittest.TestCase):
    def setUp(self):
        """Initialize a standard 100x100 square for testing."""
        self.vertices = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.packer = CirclePacker(self.vertices, padding=0.1)

    # --- Test Geometric Utilities ---

    def test_wall_distance(self):
        """Verify the boundary distance calculation is accurate."""
        # A point at (5, 50) should be exactly 5 units from the left wall
        dist = self.packer._get_wall_dist(np.array([5.0, 50.0]))
        self.assertAlmostEqual(dist, 5.0)

    # --- Test Spatial Hashing & Mega-Circles ---

    def test_mega_circle_routing(self):
        """Ensure large circles are correctly routed to the global list."""
        # cell_size is 4 in a 100x100 grid (100/25). r > 2 is 'Mega'.
        self.packer.centers.append(np.array([50.0, 50.0]))
        self.packer.radii.append(10.0) 
        self.packer._add_to_lookup(0)
        
        self.assertIn(0, self.packer.mega_circles)
        self.assertEqual(len(self.packer.grid), 0)

    def test_grid_neighbor_lookup(self):
        """Ensure small circles are routed to and found in the local grid."""
        # Add a small circle (r=0.5)
        self.packer.centers.append(np.array([10.0, 10.0]))
        self.packer.radii.append(0.5)
        self.packer._add_to_lookup(0)
        
        # Checking neighbors for a point at (11, 10)
        # It should detect circle 0 in the neighborhood
        min_dist = self.packer._check_grid_neighbors(np.array([11.0, 10.0]), 100.0)
        # Dist center-to-center is 1.0, minus radius 0.5 = 0.5
        self.assertAlmostEqual(min_dist, 0.5)

    # --- Test Safety & Overlap Prevention ---

    def test_no_overlap_guarantee(self):
        """Verify the integrated get_max_safe_radius prevents overlaps."""
        # Add a large circle
        self.packer.centers.append(np.array([50.0, 50.0]))
        self.packer.radii.append(20.0)
        self.packer._add_to_lookup(0)
        
        # Test a point at (50, 75). Distance to circle edge is 5.0.
        safe_r = self.packer.get_max_safe_radius(np.array([50.0, 75.0]))
        
        # Radius must be <= 5.0 - padding
        self.assertLessEqual(safe_r, 5.0 - self.packer.padding)

    # --- Test Sampling & Performance ---

    def test_vectorized_sampling(self):
        """Ensure the batch sampler handles polygon boundaries correctly."""
        best_pt, best_r = self.packer._find_best_candidate(num_samples=50)
        
        if best_pt is not None:
            # The best point MUST be inside the square
            self.assertTrue(0 <= best_pt[0] <= 100)
            self.assertTrue(0 <= best_pt[1] <= 100)
            self.assertGreater(best_r, 0)

    def test_generator_convergence(self):
        """Verify the generator stops when patience is exhausted."""
        # Use a tiny area and high min_radius to force early saturation
        small_packer = CirclePacker([(0,0), (10,0), (10,10), (0,10)])
        count = 0
        for _ in small_packer.generate(attempts=100, min_radius=8.0, patience=2):
            count += 1
        
        # Should stop very quickly as a radius of 8 won't fit twice
        self.assertLess(count, 5)

if __name__ == '__main__':
    unittest.main()
