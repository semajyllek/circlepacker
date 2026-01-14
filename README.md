# diskpack

A Python library for packing circles within arbitrary polygon boundaries.

![Circle packing example](https://via.placeholder.com/600x400?text=Circle+Packing+Example)

## Installation

```bash
pip install git+https://github.com/semajyllek/diskpack.git
```

Or clone and install locally:

```bash
git clone https://github.com/semajyllek/diskpack.git
cd diskpack
pip install -e .
```

## Quick Start

```python
import numpy as np
from diskpack import CirclePacker, PackingConfig

# Define a polygon (list of [x, y] vertices)
square = np.array([
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100]
])

# Pack circles
packer = CirclePacker([square])
circles = packer.pack()

# Each circle is (x, y, radius)
for x, y, r in circles[:5]:
    print(f"Circle at ({x:.1f}, {y:.1f}) with radius {r:.1f}")
```

## How It Works

The algorithm uses a **greedy random sampling** approach:

1. **Sample** random points within the polygon's bounding box
2. **Filter** to keep only points inside the polygon (using the even-odd rule)
3. **Compute** the maximum radius at each point without overlapping boundaries or existing circles
4. **Place** the largest valid circle from each batch
5. **Repeat** until no more circles can be placed

### Key Techniques

**Even-Odd Rule for Point-in-Polygon**

To determine if a point is inside a polygon (including polygons with holes), we cast a ray from the point and count edge crossings. An odd count means inside, even means outside.

**Spatial Indexing**

Checking every existing circle for collisions is O(n) per placement. We use a grid-based spatial index to only check nearby circles, bringing average-case down to O(1).

Large circles that span multiple grid cells are stored separately as "mega circles" and always checked—this prevents missed collisions at cell boundaries.

**Distance Calculations**

The maximum radius at any point is the minimum of:
- Distance to the nearest polygon edge
- Distance to the nearest existing circle's boundary

We subtract a configurable padding to prevent circles from touching.

## Configuration

Customize the packing behavior with `PackingConfig`:

```python
from diskpack import PackingConfig

config = PackingConfig(
    padding=1.5,                  # Gap between circles and boundaries
    min_radius=1.0,               # Smallest circle to place
    max_failed_attempts=200,      # Stop after this many failed attempts
    sample_batch_size=50,         # Points sampled per iteration
    grid_resolution_divisor=25,   # Controls spatial index cell size
    mega_circle_threshold=0.5,    # Radius/cell_size ratio for "mega" circles
)

packer = CirclePacker([polygon], config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `padding` | 1.5 | Minimum gap between circles and between circles and edges |
| `min_radius` | 1.0 | Circles smaller than this won't be placed |
| `max_failed_attempts` | 200 | Algorithm stops after this many consecutive failed placements |
| `sample_batch_size` | 50 | Number of random points tested per iteration |
| `grid_resolution_divisor` | 25 | Higher = smaller grid cells = more memory, faster lookups |
| `mega_circle_threshold` | 0.5 | Circles with radius > cell_size × this value are tracked globally |

## Progress Tracking

Monitor the packing process with `verbose=True`:

```python
packer = CirclePacker([polygon], config)
circles = packer.pack(verbose=True)
```

Output:
```
Placed: 25 | Failed attempts: 0/200 (0%)
Placed: 50 | Failed attempts: 0/200 (0%)
Placed: 75 | Failed attempts: 0/200 (0%)
Placed: 75 | Failed attempts: 50/200 (25%)
...
Done! Placed: 82 | Failed attempts: 200/200 (100%)
```

You can also check progress after packing:

```python
print(packer.progress)
# Placed: 82 | Failed attempts: 200/200 (100%)
```

## Fixed-Radius Mode

Pack circles of uniform size:

```python
circles = packer.pack(fixed_radius=5.0)
```

## Complex Polygons

The library handles concave polygons and multiple polygon boundaries:

```python
# Star shape
angles = np.linspace(0, 2 * np.pi, 11)[:-1]
star = []
for i, a in enumerate(angles):
    r = 100 if i % 2 == 0 else 40
    star.append([r * np.cos(a), r * np.sin(a)])

packer = CirclePacker([np.array(star)])
circles = packer.pack()
```

```python
# Multiple boundaries (e.g., polygon with a hole)
outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
inner = np.array([[40, 40], [60, 40], [60, 60], [40, 60]])  # hole

packer = CirclePacker([outer, inner])
```

## Visualization

```python
import matplotlib.pyplot as plt

def visualize(polygons, circles):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw polygon boundaries
    for poly in polygons:
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=2)
    
    # Draw circles
    for x, y, r in circles:
        ax.add_patch(plt.Circle((x, y), r, fill=True, alpha=0.7))
    
    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.show()

visualize([square], circles)
```

## Generator Mode

For large packings or progress tracking, use the generator:

```python
packer = CirclePacker([polygon])

for i, (x, y, r) in enumerate(packer.generate()):
    print(f"Placed circle {i}: radius {r:.2f}")
    
    if i >= 100:  # Stop early
        break
```

## Performance Tips

- **Increase `sample_batch_size`** for denser packings (more candidates per iteration)
- **Decrease `max_failed_attempts`** if you're okay with slightly less dense results
- **Increase `grid_resolution_divisor`** for polygons with many small circles
- **Use `fixed_radius`** when you know the circle size—avoids max-radius computation

## Requirements

- Python 3.8+
- NumPy ≥ 1.20

## License

MIT
