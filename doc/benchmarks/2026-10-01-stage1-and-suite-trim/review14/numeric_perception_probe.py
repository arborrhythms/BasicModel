from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[4]
os.environ.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false')
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from test_input_word_cursor import test_nongrammar_config_disables_cursor_and_next_word_is_none
test_nongrammar_config_disables_cursor_and_next_word_is_none()
print('PASS: the numeric default builds and keeps its word cursor disabled')
