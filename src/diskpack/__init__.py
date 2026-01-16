"""
diskpack - State-of-the-art circle packing for arbitrary polygons.

Usage:
    from diskpack import CirclePacker, PackingConfig
    
    # Basic usage
    packer = CirclePacker([polygon_vertices])
    circles = packer.pack()
    
    # With configuration
    config = PackingConfig(use_hybrid_packing=True, verbose=True)
    packer = CirclePacker([polygon_vertices], config)
    circles = packer.pack()
    
    # Fixed radius (uses optimal hex grid)
    config = PackingConfig(fixed_radius=5.0)
    packer = CirclePacker([polygon_vertices], config)
    circles = packer.pack()

Packing modes:
    - Random sampling: Original greedy algorithm
    - Hex grid: Optimal for fixed radius, blazing fast
    - Front-based: Fills corners well, more circles
    - Hybrid: State-of-the-art, combines all approaches for best density
"""

from .config import PackingConfig, PackingProgress, PackingMode, Circle, Point, Polygon
from .packer import CirclePacker
from .geometry import PolygonGeometry, SpatialIndex

__all__ = [
    "CirclePacker",
    "PackingConfig",
    "PackingProgress",
    "PackingMode",
    "PolygonGeometry",
    "SpatialIndex",
    "Circle",
    "Point", 
    "Polygon",
]

__version__ = "0.3.0"
