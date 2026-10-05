"""Descriptive timing and fixed-probe fields, with no learning or new seed."""
import json, os, time
from pathlib import Path
import torch


def test_open_cost_and_fixed_probe_levels(monkeypatch):
 from test_mm_xor import _fresh_model
 from CategoricalDiscrimination import FIXED_PROBES, categorical_discrimination
 import ModelAttention
 model,_,_=_fresh_model('data/XOR_grammar.xml'); model.eval()
 stage_times=[]; open_times=[]; forward_times=[]
 original=ModelAttention.stage_input; field=ModelAttention.read_code_field
 def stage(*args,**kwargs):
  start=time.perf_counter();value=original(*args,**kwargs);stage_times.append(time.perf_counter()-start);return value
 def opening(*args,**kwargs):
  start=time.perf_counter();value=field(*args,**kwargs);open_times.append(time.perf_counter()-start);return value
 monkeypatch.setattr(ModelAttention,'stage_input',stage)
 monkeypatch.setattr(ModelAttention,'read_code_field',opening)
 with torch.no_grad():
  for _ in range(6):
   value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
   start=time.perf_counter();model.forward(value);forward_times.append(time.perf_counter()-start)
 report=dict(timing=dict(config='data/XOR_grammar.xml',batch=4,words=8,warmup=1,measurements=5,
  seed=None,forward_seconds=forward_times[1:],stage_input_seconds=stage_times[1:],
  field_open_seconds=open_times[1:],before_boundary='_sentence_prelude',
  after_boundaries=['ModelAttention.stage_input','ModelAttention.read_code_field'],
  comparable_whole_forward=True,comparable_open_boundaries=False))
 # Use the unchanged language pilot's dictionary capacity for all fixed probes.
 # First exposure admits observations; the measured pass reads that same bank.
 model.End();del model
 import gc;gc.collect()
 model,_,_=_fresh_model('data/BasicModel.xml');model.eval()
 results={}
 with torch.no_grad():
  for probe in FIXED_PROBES.values():
   for text in probe['texts']:
    model.forward(model.inputSpace.prepInput([text]));model.End()
  for name,probe in FIXED_PROBES.items():
   readings={'word':[],'sentence':[]};counts={level:dict(both=0,observed=0,neither=0,brackets=0) for level in readings}
   missing=[]
   for text in probe['texts']:
    model.forward(model.inputSpace.prepInput([text]))
    words=model._word_expectation_input[0]
    live=model._attention_forms[-1]
    readings['word'].append(((words*live[...,None]).sum(1)/live.sum(1).clamp_min(1)[:,None])[0].cpu())
    slots=model._tensor_final_end_slots
    assert torch.is_tensor(slots), 'fixed probe produced no sentence-level reading'
    readings['sentence'].append(slots[0].flatten().cpu())
    poles=model._attention_native_poles
    for level,values,valid in (
     ('word',poles,live),('sentence',poles.amax(1,keepdim=True),live.any(1,keepdim=True))):
     positive=values[...,0]>0;negative=values[...,1]>0
     counts[level]['both']+=int((positive&negative&valid).sum())
     counts[level]['observed']+=int(((positive|negative)&valid).sum())
     counts[level]['neither']+=int((~(positive|negative)&valid).sum())
     counts[level]['brackets']+=int(valid.sum())
    if not live.any():missing.append(text)
    model.End()
   results[name]={level:dict(**counts[level],
      both_rate=counts[level]['both']/max(1,counts[level]['brackets']),
      categorical_discrimination=categorical_discrimination(values,probe['labels']))
      for level,values in readings.items()}
   results[name]['unread_inputs']=missing
 report['fixed_probes']=dict(config='data/BasicModel.xml',seed=None,training_updates=0,
  phase='second exposure after native admission; no gradient updates',
  pole_definition='native word-object poles; sentence is their pooled open bracket',
  byte='disabled, not measured',row='disabled, not measured',sets=results)
 Path(os.environ['ATTENTION_PROFILE_OUTPUT']).write_text(json.dumps(report,indent=2)+'\n')
