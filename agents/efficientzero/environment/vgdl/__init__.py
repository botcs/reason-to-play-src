"""VGDL package compatibility helpers.

This file aliases legacy absolute imports (e.g., `import ontology`) to the
package modules (e.g., `vgdl.ontology`) so older code paths continue to work
when importing `vgdl` as a package.
"""

import sys as _sys

from . import ontology as _ontology
from . import tools as _tools
from . import colors as _colors
from . import core as _core
from . import util as _util

_sys.modules.setdefault('ontology', _ontology)
_sys.modules.setdefault('tools', _tools)
_sys.modules.setdefault('colors', _colors)
_sys.modules.setdefault('core', _core)
_sys.modules.setdefault('util', _util)
