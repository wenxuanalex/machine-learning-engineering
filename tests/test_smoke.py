def test_smoke_imports():
    """Minimal smoke test so CI has a real signal.

    We keep this intentionally lightweight: just prove core packages import.
    """

    # Base analytics stack
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401

    assert np.__version__
    assert pd.__version__
