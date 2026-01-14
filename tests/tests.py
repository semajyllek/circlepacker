import unittest
import numpy as np
from diskpack.packer import CirclePacker, PackingConfig, PolygonGeometry



class TestCirclePacker(unittest.TestCase):
    def setUp(self):
        """Set up a simple 10x10 square for testing."""
        self.square_verts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        self.square = [self.square_verts]
        # Use small padding/min_radius to allow for dense filling
        self.config = PackingConfig(padding=0.1, min_radius=0.5, patience_before_stop=100)

    def _calculate_poly_area(self, segments: list) -> float:
        """Calculates area of polygons (including holes) using the Shoelace Formula."""
        total_area = 0
        for poly in segments:
            x = poly[:, 0]
            y = poly[:, 1]
            # Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            # If it's a hole (clockwise), it naturally subtracts if we don't use abs here,
            # but since our parser handles holes via Even-Odd, we treat all as positive 
            # and subtract holes manually if needed. For this test, we assume shells.
            total_area += area
        return total_area

    def test_filling_density(self):
        """Verify that the packer fills a minimum percentage of the polygon area."""
        packer = CirclePacker(self.square, self.config)
        circles = list(packer.generate())
        
        # Calculate areas
        poly_area = self._calculate_poly_area(self.square)
        circle_area = sum(np.pi * (r**2) for _, _, r in circles)
        
        fill_percentage = (circle_area / poly_area) * 100
        
        print(f"\nFill Analysis: {len(circles)} circles placed.")
        print(f"Total Area: {poly_area:.2f} | Circle Area: {circle_area:.2f}")
        print(f"Packing Density: {fill_percentage:.2f}%")
        
        # Assert a minimum density threshold. 
        # For a random packer, 30-50% is a reasonable 'success' floor depending on radius.
        self.assertGreater(fill_percentage, 25.0, f"Packing density too low: {fill_percentage:.2f}%")

    def test_geometry_containment(self):
        """Verify the Even-Odd rule correctly identifies the interior."""
        geo = PolygonGeometry(self.square)
        test_pts = np.array([
            [5, 5],   # Center
            [15, 5],  # Far outside
            [-1, -1]  # Outside
        ])
        results = geo.contains_points(test_pts)
        self.assertTrue(results[0])
        self.assertFalse(results[1])
        self.assertFalse(results[2])

    def test_no_overlap_integrity(self):
        """Mathematically verify no circles overlap including padding."""
        packer = CirclePacker(self.square, self.config)
        circles = list(packer.generate())
        
        for i, (x1, y1, r1) in enumerate(circles):
            for j, (x2, y2, r2) in enumerate(circles):
                if i == j: continue
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Distance must be >= sum of radii + padding
                min_sep = r1 + r2 + self.config.padding
                self.assertGreaterEqual(dist, min_sep - 1e-9)

if __name__ == '__main__':
    unittest.main()
