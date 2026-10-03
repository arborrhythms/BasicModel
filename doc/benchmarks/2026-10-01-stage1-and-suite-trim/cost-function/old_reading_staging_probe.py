"""Read-only staged-shape audit after mandatory-reconstruction failures; no optimizer."""
import os,sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from test_mm_xor import _fresh_model
cases=[]
for name in ['LM_5M','MM_decode','MM_bpe','MM_grammar','XOR_grammar','POS_smoke']:
    m,_,_=_fresh_model(str(ROOT/'data'/f'{name}.xml'))
    language=m.symbolSpace.languageLayer
    d=dict(config=name,serial=m.serial,operation_layer=type(getattr(language,'operation_layer',None)).__name__,concept_binding=m.concept_binding,serial_object_meta=m.serial_object_meta,protocol=m.sentence_protocol,synthesis=m.perceptualSpace.synthesis_mode,analysis=m.wholeSpace.analysis_mode)
    def check():
        import torch
        for k in ('_word_active_mask','_packed_sentence_ids','_ar_grammar_object_rows','_ar_concept_lookup_rows'):
            v=getattr(m.inputSpace,k,None);d[k]=dict(shape=list(v.shape),nonnegative=int((v>=0).sum())) if torch.is_tensor(v) else str(v)
        d['percept_keys']=list((getattr(m.perceptualSpace,'_forward_input',None) or {}).keys())
    m._validate_reconstruction_bank=check
    try:m._lex_embed_stem(m.inputSpace.prepInput(['hello world','hello there','loving world','loving there']))
    except Exception as e:d['error']=repr(e)
    print(json.dumps(d),flush=True);cases.append(d);m.End();m.symbolSpace.soft_reset()
(HERE/'old-reading-staging-audit.json').write_text(json.dumps(cases,indent=2)+'\n')
