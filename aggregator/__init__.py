import sys
from pathlib import Path

PATHS = [item for item in Path(__file__).parent.iterdir() if item.is_dir()]
for path in enumerate(PATHS):
    sys.path.append(str(path))

from .main import run
