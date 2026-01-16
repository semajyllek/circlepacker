"""
Configuration and type definitions for circle packing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

# Type aliases
Polygon = np.ndarray
Point = np.ndarray
GridKey = Tuple[int, int]
Circle = Tuple[float, float, float]  # (x, y, radius)


class PackingMode(Enum):
    """Available packing strategies."""
    RANDOM = "random"
    HEX_GRID = "hex_grid"
    FRONT = "front"
    HYBRID = "hybrid"


@dataclass
class PackingConfig:
    """
    Configuration parameters for the circle packing algorithm.
    
    Basic parameters:
        padding: Minimum gap between circles and between circles and edges
        min_radius: Smallest circle that will be placed
        fixed_radius: If set, all circles will have this exact radius
        
    Algorithm selection:
        use_hex_grid: Use hexagonal grid for fixed radius (fastest)
        use_front_packing: Use front-based algorithm
        use_hybrid_packing: Use state-of-the-art multi-phase algorithm
        
    Performance tuning:
        max_failed_attempts: Stop after this many consecutive failures
        sample_batch_size: Points sampled per iteration (random mode)
        grid_resolution_divisor: Controls spatial index granularity
        
    Hybrid mode parameters:
        hybrid_large_threshold: Phase 1 minimum (fraction of max radius)
        hybrid_medium_threshold: Phase 2 minimum (fraction of max radius)  
        hybrid_micro_grid_min_gap: Minimum gap size for micro hex fill
    """
    # Basic parameters
    padding: float = 1.5
    min_radius: float = 1.0
    fixed_radius: Optional[float] = None
    
    # Algorithm selection
    use_hex_grid: bool = True
    use_front_packing: bool = False
    use_hybrid_packing: bool = False
    
    # Performance tuning
    max_failed_attempts: int = 200
    sample_batch_size: int = 50
    grid_resolution_divisor: float = 25
    mega_circle_threshold: float = 0.5
    ray_cast_epsilon: float = 1e-10
    
    # Hybrid mode parameters
    hybrid_large_threshold: float = 0.5
    hybrid_medium_threshold: float = 0.25
    hybrid_micro_grid_min_gap: float = 5.0
    
    # Output
    verbose: bool = False


@dataclass
class PackingProgress:
    """Tracks the current state of the packing algorithm."""
    circles_placed: int = 0
    failed_attempts: int = 0
    max_failed_attempts: int = 200
    phase: str = ""

    @property
    def progress_ratio(self) -> float:
        """How close to stopping (0.0 = just started, 1.0 = done)."""
        return self.failed_attempts / self.max_failed_attempts if self.max_failed_attempts > 0 else 0

    def __str__(self) -> str:
        phase_str = f"[{self.phase}] " if self.phase else ""
        return f"{phase_str}Placed: {self.circles_placed} | Failed: {self.failed_attempts}/{self.max_failed_attempts} ({self.progress_ratio:.0%})"
