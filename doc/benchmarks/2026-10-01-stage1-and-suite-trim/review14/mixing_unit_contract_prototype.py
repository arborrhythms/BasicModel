from pathlib import Path
import os,sys,json
from types import MethodType
ROOT=Path(__file__).resolve().parents[4]
os.environ.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false')
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch,util
from test_mm_xor import _fresh_model
util.TheCompileBackend='none'
def loop(cond,body,args):
    while bool(cond(*args)):args=body(*args)
    return args
torch.while_loop=loop
model,_,_=_fresh_model(str(ROOT/'data/XOR_grammar.xml'))
model.perceptualSpace._serial_word_capacity = int(model.perceptualSpace.outputShape[0])
model.perceptualSpace._embed_ladder = model.perceptualSpace._embed_ladder_word_major
raw = model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
with torch.no_grad(): model._lex_embed_stem(raw)
isp = model.inputSpace
expected = [[b'hello', b' ', b'world'], [b'hello', b' ', b'there'],
            [b'loving', b' ', b'world'], [b'loving', b' ', b'there']]
assert isp._word_active_mask.sum(-1).tolist() == [3] * 4, isp._word_active_mask
for b, words in enumerate(expected):
    actual = [bytes(isp._ar_target_word_bytes[b,w][isp._ar_target_word_mask[b,w]].tolist()) for w in range(3)]
    assert actual == words, (actual, words)
assert isp._ar_grammar_leaf_mask[:, :3].tolist() == [[True,False,True]] * 4
assert bool((isp._ar_grammar_object_rows[:, [0,2]] >= 0).all())
assert model._word_symbol_concept_ids() is None
model.End()
print('PASS: three complete units, exact bytes including separator; only two object grammar leaves')
