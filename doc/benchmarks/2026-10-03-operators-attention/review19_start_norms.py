"""One XOR forward to the first reconstruction boundary; stop before training."""
import json,math,os,sys
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test'),str(H)]
import torch
from Models import BasicModel,BaseModel,ModelFactory
from review17_run_audit import geometry,json_value
from util import TheXMLConfig

class ObservedStart(Exception):pass

def forbidden_step(*args,**kwargs):
    raise AssertionError('The initial-norm probe must stop before any backward or optimizer step')

def observe(model,record):
    model._review17_margin=float(TheXMLConfig.space('ConceptualSpace','latticeMargin',0.))
    with torch.no_grad():
        start=geometry(model,record)
        forms=[]
        for stage,book in enumerate(start['codes']):
            names={r['row']:r['word'] for r in book['support']}
            for row,v in zip(book['rows'],book['forms']['values']):
                forms.append(dict(stage=stage,row=row,word=names[row],l2=float(v.norm()),max_abs=float(v.abs().max())))
        roots=[dict(sentence=text,l2=float(v.norm()),max_abs=float(v.abs().max()))
               for text,v in zip(start['root_inputs'],start['roots']['values'])]
    result=dict(kind='One first-forward norm probe, not a gate',training_runs=0,optimizer_steps=0,
        backward_calls=0,seed_override=None,source_of_norms='First ordinary greedy reconstruction trial; before any trial cost or owner update',
        forms=forms,roots=roots,geometry=start)
    (H/'review19-start-norms.json').write_text(json.dumps(result,indent=2,default=json_value)+'\n')
    raise ObservedStart()

with patch.object(BasicModel,'_reconstruct_trial',observe),patch.object(BaseModel,'_backward_training_loss',forbidden_step),patch.object(BasicModel,'_sentence_train_step',forbidden_step):
    try:ModelFactory.run('data/XOR_grammar.xml')
    except ObservedStart:print('Captured first-forward norms; zero backward calls and optimizer steps.')
    else:raise AssertionError('The first reconstruction boundary was not reached')
