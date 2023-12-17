import sys
from pathlib import Path

PATH = Path(__file__).parent
sys.path.append(str(PATH))
sys.path.append(str(PATH / 'data'))
sys.path.append(str(PATH / 'models'))
sys.path.append(str(PATH / 'models/basic_gan'))
sys.path.append(str(PATH / 'models/basic_vae'))

from .main import run
