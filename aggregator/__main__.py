import sys
from pathlib import Path

PATH = Path(__file__).parent
sys.path.append(str(PATH))
sys.path.append(str(PATH / 'parser'))
sys.path.append(str(PATH / 'models/basic_gan'))
