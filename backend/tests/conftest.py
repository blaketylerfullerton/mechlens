import sys
from pathlib import Path

# The app modules import each other flat (`from schema import ...`), so tests
# need backend/app on the path the same way running a script from there does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
