import os
import sys

# Make `src/` importable as top-level modules (scene, motion, depth, estimator, ...)
_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
