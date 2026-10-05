import torch

def test_native_admission_snapshot():
    from test_mm_xor import _fresh_model, _PROJECT
    from pathlib import Path
    m=_fresh_model(str(Path(_PROJECT)/'data/MM_xor.xml'))[0]
    with torch.no_grad():m(m.inputSpace.prepInput(['hi no','hi go','we no','we go']))
    fi=m.perceptualSpace._forward_input
    for n in ('word_texts','tokens','part_spans','native_part_spans','native_indices'):
        print(n,fi.get(n))
    for n in ('_attention_forms','_attention_spans','_attention_words','_reading_word_extents'):
        print(n,getattr(m,n,None))
    print('offsets',getattr(m.inputSpace,'_ar_word_part_offsets',None))
    print('raw',m._staged_concepts_in)
    print('defs',m._concept_owner().definitions._by_form)
