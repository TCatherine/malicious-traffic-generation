import sys

from pathlib import Path
from .main import run

PATH = Path(__file__).parent
sys.path.append(str(PATH))
