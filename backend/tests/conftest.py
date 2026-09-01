import sys
from pathlib import Path

# `app` is a package; tests import it as one (`from app.schema import ...`),
# so `backend/` is what has to be on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
