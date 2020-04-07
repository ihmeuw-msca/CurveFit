'''[begin_markdown sizes_to_indices_xam]

# Example and Test of sizes_to_indices

```python'''
import sys
import sandbox
sandbox.path()
import curvefit
#
sizes   = [ 2, 4, 3 ]
indices = curvefit.core.utils.sizes_to_indices(sizes)
assert indices[0] == range(0, 2)
assert indices[1] == range(2, 2+4)
assert indices[2] == range(2+4, 2+4+3)
print('sizes_to_indices.py: OK')
sys.exit(0)
'''```
[end_markdown sizes_to_indices_xam]'''
