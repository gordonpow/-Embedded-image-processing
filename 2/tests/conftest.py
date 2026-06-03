import os
import sys

# Make `src/` importable as top-level modules (scene, motion, depth, estimator, ...)
_ROOT = os.path.dirname(os.path.dirname(__file__))
for _sub in ("src", "benchmarks"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
