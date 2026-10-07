"""Replay one untrained frozen round-2b fixture under old/new pole handoffs."""
import ast, hashlib, json, os, sys, zipfile
from pathlib import Path
from unittest.mock import patch
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
from Models import BasicModel
from Language import OperationSelectionLayer
from test_mm_xor import _fresh_model
from ModelAttention import read_poles


def main():
    archive = zipfile.ZipFile(HERE/'before/source.zip')
    text = archive.read('bin/Models.py').decode()
    tree = ast.parse(text)
    old = next(node for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='BasicModel'
               for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_attention_sentence_payload')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[old],type_ignores=[]),'frozen-round2-handoff','exec'),namespace)
    old_payload = namespace[old.name]
    source = dict(before=ast.get_source_segment(text,old),
                  after=ast.get_source_segment((ROOT/'bin/Models.py').read_text(),
                      next(n for c in ast.parse((ROOT/'bin/Models.py').read_text()).body
                           if isinstance(c,ast.ClassDef) and c.name=='BasicModel'
                           for n in c.body if isinstance(n,ast.FunctionDef) and n.name==old.name)))
    (HERE/'handoff-bodies.json').write_text(json.dumps(source,indent=2)+'\n')
    state_path=HERE/'disjunction-initial-rng.pt'
    state = torch.load(state_path,weights_only=True) if state_path.exists() else torch.get_rng_state()
    if not state_path.exists(): torch.save(state,state_path)
    result = {}
    original_forward, original_attend = OperationSelectionLayer.forward,OperationSelectionLayer.attend
    original_commit = BasicModel._commit_sentence
    def disjunction(module,x,**kw):
        stop=(x.shape[1]-1)*module.r_reduce+x.shape[1]*module.r_apply
        depth=kw.get('depth',torch.full((len(x),),x.shape[1]))
        kw['replay_action']=torch.where(depth>1,1,stop)
        return original_forward(module,x,**kw)
    def narrowing(module,keys,legal,space,**kw):
        available=legal.flatten(1)
        action=available.long().argmax(-1)
        for op in (0,2,1,4,3,5):
            slots=legal[:,:,op]
            action=torch.where(slots.any(-1),slots.long().argmax(-1)*legal.shape[-1]+op,action)
        kw['replay_action']=action
        return original_attend(module,keys,legal,space,**kw)
    for label in ('round2b','round2c','landing_handoff'):
        torch.set_rng_state(state)
        model,_,data=_fresh_model('data/XOR_grammar.xml')
        captured={}
        def commit(m,state,sid,active,*args):
            texts,unavailable=m.reconstruct_grammar_sentence(state,sid,active)
            from Language import LanguageSpace
            bank=m._sentence_primed_bank
            root=state[1][9][:,sid]
            op=next(getattr(op,'gl',op) for op in m._stm_reducer().ops
                    if getattr(getattr(op,'gl',op),'rule_name','')=='disjunction')
            left,right,valid=LanguageSpace._bounded_binary_reconstruction(op,root,
                torch.zeros_like(root),torch.zeros(len(root),dtype=torch.bool),
                torch.zeros(len(root),dtype=torch.bool),bank.codes,bank.valid,16)
            relative=(op.compose(left,right)-root).square().sum(-1)/root.square().sum(-1).clamp_min(1e-30)
            captured.update(texts=texts,unavailable=unavailable.tolist(),
                argmin_relative_residual=relative.tolist(),
                costs=m._last_sentence_credit['components'][:,0,0].tolist(),
                actions=m._attention_words.actions.tolist(),
                poles=m._attention_poles.tolist(),
                activations=state[1][0].tolist())
            return original_commit(m,state,sid,active,*args)
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(OperationSelectionLayer,'forward',disjunction))
            stack.enter_context(patch.object(OperationSelectionLayer,'attend',narrowing))
            stack.enter_context(patch.object(BasicModel,'_commit_sentence',commit))
            if label=='round2b':
                from Interpret import InterpretLayer
                import Models
                def saved_method(path, cls, method, namespace):
                    source=archive.read(path).decode()
                    tree=ast.parse(source)
                    node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
                              for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
                    env=dict(namespace)
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen-round2b/'+path,'exec'),env)
                    return env[method]
                stack.enter_context(patch.object(InterpretLayer,'forward',saved_method('bin/Interpret.py','InterpretLayer','forward',dict(torch=torch))))
                stack.enter_context(patch.object(BasicModel,'_pushed_word_slab',saved_method('bin/Models.py','BasicModel','_pushed_word_slab',vars(Models))))
            elif label=='landing_handoff':
                stack.enter_context(patch.object(BasicModel,'_attention_sentence_payload',lambda m,p,i:p))
                stack.enter_context(patch('ModelAttention.reference_evidence',lambda m,a,c:None))
            raw,target=next(iter(data.data_loader(split='test',num_streams=4)))
            batch=model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)
            captured['inputs']=[model._bytes_to_text(x).rstrip(chr(0)) for x in raw]
            with torch.no_grad():
                model.runBatch(train=False,optimizer=None,batchSize=4,split='test',batch_override=batch)
        captured['multisets']=sum(sorted(a.split())==sorted(b.split()) for a,b in zip(captured['inputs'],captured['texts']))
        result[label]=captured
        model.End();model.symbolSpace.soft_reset()
    result['fixture']=dict(configuration='data/XOR_grammar.xml',training_steps=0,
        random_initialization='one captured RNG state, reused; no seed selection',
        forced='disjunction; narrowing priority not, and, or, descend, gloss, divide',
        frozen_manifest_sha256=hashlib.sha256((HERE/'before/source.json').read_bytes()).hexdigest())
    assert result['round2c']['multisets']==4
    assert result['round2c']['argmin_relative_residual']==[0.]*4
    assert result['round2c']['costs']==result['landing_handoff']['costs']
    (HERE/'disjunction-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(costs=v['costs'],argmin_relative_residual=v['argmin_relative_residual'],multisets=v['multisets'],texts=v['texts'])
                      for k,v in result.items() if k!='fixture'}))


if __name__=='__main__': main()
