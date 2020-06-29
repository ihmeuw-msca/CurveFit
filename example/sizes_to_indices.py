#! /usr/bin/env python3
"""
{begin_markdown sizes_to_indices_xam}
{spell_markdown utils}

# Example and Test of sizes_to_indices

## Function Documentation
[size_to_indices](sizes_to_indices.md)

## Example Source Code
```python"""
import numpy

from curvefit.core.utils import sizes_to_indices

sizes = [2, 4, 3]
indices = sizes_to_indices(sizes)
assert all(indices[0] == numpy.array([0, 1]))
assert all(indices[1] == numpy.array([2, 3, 4, 5]))
assert all(indices[2] == numpy.array([6, 7, 8]))

print('sizes_to_indices.py: OK')

"""```
{end_markdown sizes_to_indices_xam}
"""