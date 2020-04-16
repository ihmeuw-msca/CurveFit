#! /usr/bin/env python3
'''{begin_markdown unzip_x_xam}
{spell_markdown params ndim}

# Example and Test of unzip_x

## Function Documentation
[unzip_x](unzip_x.md)

## Example Source Code
```python'''
import sys
import numpy
import sandbox
sandbox.path()
import curvefit
#
num_groups = 2
num_fe     = 3
x          = numpy.array( range( (num_groups + 1) * num_fe ) )
fe, re     = curvefit.core.effects2params.unzip_x(x, num_groups, num_fe)
assert fe.ndim == 1
assert re.ndim == 2
assert fe.shape[0] == num_fe
assert re.shape[0] == num_groups
assert re.shape[1] == num_fe
assert all( fe == x[0 : num_fe] )
for i in range(num_groups) :
    assert all( re[i,:] == x[(i+1)*num_fe : (i+2)*num_fe] )
print('unzip_x.py: OK')
sys.exit(0)
'''```
{end_markdown unzip_x_xam}'''
