import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Pick up an in-tree build of the C++ module if present (cmake -B build).
for d in (ROOT / "build", ROOT / "cpp" / "build"):
    if d.is_dir():
        sys.path.insert(0, str(d))
