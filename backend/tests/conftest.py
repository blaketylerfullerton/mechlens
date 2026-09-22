import os
import sys
from pathlib import Path

import pytest

# `app` is a package; tests import it as one (`from app.schema import ...`),
# so `backend/` is what has to be on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _restore_environ():
    """`main()` writes MECHLENS_API_TOKEN/MECHLENS_DATA_DIR straight into the
    environment, and monkeypatch cannot undo a variable that was absent when
    the test started. A leaked token switches BearerAuthMiddleware on for every
    later `create_app()`, so every unauthenticated request 401s."""
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)
