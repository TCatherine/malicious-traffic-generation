import sys
from pathlib import Path

PATHS = [item for item in Path(__file__).parent.iterdir() if item.is_dir()]
for i, path in enumerate(PATHS):
    sys.path.insert(i, str(path))
