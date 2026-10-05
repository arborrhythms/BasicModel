from pathlib import Path
import torch

def test_capacity_before(monkeypatch,tmp_path):
    from Interpret import InterpretLayer
    from recon_bench import run_config
    lookup=InterpretLayer.lookup_word
    def read(self,parts,wholes,**kwargs):
        try:return lookup(self,parts,wholes,**kwargs)
        except RuntimeError:
            index=self.owner.definitions
            word=index.word(form=kwargs.get('form'))
            print('FAILED_WORD',parts,wholes,kwargs,'old',None if word is None else index.description(word),flush=True)
            print('CAPS',self.owner._order_caps(),'rows',self.owner._csw_rows,flush=True)
            raise
    monkeypatch.setattr(InterpretLayer,'lookup_word',read)
    run_config('data/MM_xor_fixture.xml',epochs=1,seed=0,out_dir=str(tmp_path))

def test_mixed_gradient_before(monkeypatch):
    import test_output_path_supervised as tests
    original=tests._answer_training_probe
    def run(*args):
        result=original(*args)
        print('GRADS',[None if p is None else float(p.norm()) for p in result['total_grads']],flush=True)
        print('PARAMS',result['active_index'],result['changed'],[tuple(p.shape) for p in result['params']],flush=True)
        return result
    monkeypatch.setattr(tests,'_answer_training_probe',run)
    tests.test_mixed_supplied_numeric_and_automatic_text_trains_only_supplied_row(None)
