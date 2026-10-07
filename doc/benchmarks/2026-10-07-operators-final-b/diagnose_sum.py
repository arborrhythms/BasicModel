"""One requested all-on replay of the measured sum-08, on its frozen source."""
from pathlib import Path
import hashlib,json,os,random,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FROZEN=HERE/'diagnosis/frozen'
PRIOR=HERE.parent/'2026-10-07-operators-final'
os.chdir(FROZEN)
sys.path[:0]=[str(FROZEN/'bin'),str(FROZEN/'test'),str(PRIOR),str(HERE.parent/'2026-10-03-operators-attention')]
import torch,numpy as np
from unittest.mock import patch
from Models import BasicModel,ModelFactory
from AnswerComparison import AnswerComparison
from Spaces import OutputSpace
from SentenceUnderstanding import SentenceRecordReader
import test_explicit_dimensions as gates
from operators_run_audit import observe_run,json_value
from operators_gate_observer import capture_final_derivations

entry=torch.load(PRIOR/'measurements/sum-08/unseeded-entry.pt',weights_only=False,map_location='cpu')
random.setstate(entry['python']);np.random.set_state(entry['numpy']);torch.set_rng_state(entry['torch'])
epoch=0;comparison=False;current=None;rows=[];handles=[];hooked=set();gradients={}
run_epoch,head,judge,readout,record_forward,run=BasicModel.runEpoch,BasicModel._forward_head,AnswerComparison.forward,OutputSpace._apply_readout,SentenceRecordReader.forward,ModelFactory.run
def epochs(model,*args,**kwargs):
    global epoch
    if kwargs.get('optimizer') is not None:epoch+=1
    return run_epoch(model,*args,**kwargs)
def judges(*args,**kwargs):
    global comparison
    old=comparison;comparison=True
    try:return judge(*args,**kwargs)
    finally:comparison=old
def heads(model,*args,**kwargs):
    global current
    record=kwargs.get('understanding');old=current
    if not comparison and record is not None:
        width=model._concept_owner().similarity_codebook.mereology.percept_event_width
        current=dict(epoch=epoch,training=bool(model._sentence_training),trial=getattr(model,'_sentence_trial',None),
            root_form_norm=record.root[:,:width].detach().norm(dim=-1).tolist(),
            root_meaning_norm=record.root[:,width:].detach().norm(dim=-1).tolist(),
            features_norm=record.reader_features().norm(dim=-1).tolist())
        rows.append(current)
        for name,p in model.named_parameters():
            if (name.startswith('inputSpace.outputSpace.') or name.startswith('answer_record_reader.')) and p.requires_grad and id(p) not in hooked:
                hooked.add(id(p))
                def observe(g,name=name):
                    gradients.setdefault(str(epoch),{}).setdefault(name,[]).append(dict(norm=float(g.detach().norm()),maximum=float(g.detach().abs().max())))
                    return g
                handles.append(p.register_hook(observe))
    try:
        result=head(model,*args,**kwargs)
        if current is not old and current is not None:
            current['output']=model.normalizer.denormalize(result.materialize(),which='output').detach().reshape(-1).tolist()
        return result
    finally:current=old
def readouts(space,value):
    if current is not None and not comparison:
        current['pre_activation']=value.detach().reshape(-1).tolist()
        current['bias']=None if space._readout_bias is None else space._readout_bias.detach().reshape(-1).tolist()
    return readout(space,value)
def records(reader,record):
    result=record_forward(reader,record)
    if current is not None and not comparison:
        current['record_contribution']=result.detach().reshape(-1).tolist()
    return result

with patch.object(BasicModel,'runEpoch',epochs),patch.object(BasicModel,'_forward_head',heads),patch.object(AnswerComparison,'forward',judges),patch.object(OutputSpace,'_apply_readout',readouts),patch.object(SentenceRecordReader,'forward',records),patch.object(ModelFactory,'run',lambda ignored:run(str(HERE/'diagnosis/sum-control.xml'))),capture_final_derivations(),observe_run() as audit:
    model=gates._run_xor_grammar_in_process()
for handle in handles:handle.remove()
data=model.inputSpace.data
prediction=torch.stack(data.reconstructed_output).reshape(-1).detach().cpu()
target=torch.stack(data.test_output).reshape(-1).to(prediction)
original=json.loads((PRIOR/'measurements/sum-08/measurement.json').read_text())
result=dict(mse=float((prediction-target).square().mean()),predictions=prediction.tolist(),expected_mse=original['mse'],
    reproduces=prediction.tolist()==original['answers'],epochs=epoch,per_read=rows,gradients=gradients,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source='frozen measured combined source',seed=None)
(HERE/'diagnosis/sum08.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'diagnosis/run-audit.json').write_text(json.dumps(audit,indent=2,default=json_value)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('per_read','gradients')}),flush=True)
assert result['reproduces'], 'the requested measured all-on replay did not reproduce'
