"""Read-only baseline inspection of grammar leaf staging; no training/seed."""
import json
import warnings
import torch
from test_mm_xor import _fresh_model
import util

util.TheCompileBackend = 'none'
model, _, _ = _fresh_model('data/XOR_grammar.xml')
model.eval()
try:
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter('ignore')
        raw = model.inputSpace.prepInput(['hello world', 'hello there', 'loving world', 'loving there'])
        model(raw)
    owner = model._concept_owner()
    print('OWNER', owner.nWhat, owner.nDim, tuple(owner.similarity_codebook.getW().shape))
    print('TERMINAL', model.conceptualSpace.nWhat, model.conceptualSpace.nDim)
    print('ROOT', model._stm_single_S)
    for word in ('hello', 'world', 'there', 'loving'):
        wid = owner.definitions.word(form=word)
        obj = owner.definitions.deref(wid)
        print('WORD', word, 'word', wid, 'object', obj, 'row', owner._csw_row_of(obj))
    for name in ('_word_active_mask', '_ar_word_concept_rows', '_ar_word_object_rows', '_ar_word_object_atoms'):
        print(name, getattr(model.inputSpace, name))
    print('PS', {k: v for k, v in model.perceptualSpace._forward_input.items()
                 if k in ('tokens', 'indices', 'native_indices', 'part_spans', 'native_part_spans', 'word_texts')})
    print('STAGED_EXTENTS', model._reading_word_extents)
finally:
    model.End()
    model.symbolSpace.soft_reset()
