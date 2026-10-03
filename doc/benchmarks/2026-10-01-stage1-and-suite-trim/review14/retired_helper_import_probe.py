from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[4]
os.environ['BASICMODEL_DEVICE']='cpu'
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from test_input_word_cursor import _build_nongrammar_model
_build_nongrammar_model()
