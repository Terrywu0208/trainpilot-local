"""Re-export shim: the validator tests live in the
``tests/test_validators/`` package (so the suite can be run as
``pytest tests/test_validators``); this module re-exports them so they
are also collected from ``tests/test_validators_*.py``."""
from test_validators.test_validators import *  # noqa: F401,F403
