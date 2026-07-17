"""Enable `python -m cane_eval` as a PATH-independent entry point.

Equivalent to the `cane-eval` console script, but works even when the
script directory (e.g. Windows Scripts/) is not on PATH.
"""

from cane_eval.cli import main

if __name__ == "__main__":
    main()
