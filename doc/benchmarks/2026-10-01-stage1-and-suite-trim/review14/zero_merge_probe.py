import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'bin'))
from Layers import Error
Error().merge(object(),weight=0.)
