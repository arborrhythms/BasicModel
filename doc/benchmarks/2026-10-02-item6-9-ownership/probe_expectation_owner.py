import sys,os,json
from pathlib import Path
root=Path(__file__).resolve().parents[3];sys.path[:0]=[str(root/'bin'),str(root/'test')]
os.environ['RUN_SLOW']='1'
import Models,pytest
original=Models.BasicModel._sentence_train_step
def inspect(self,loss):
 o=self._sentence_optimizer;predictor=list(self.symbolSpace.discourse._inter_predictor.parameters()); registered={id(p) for g in o.param_groups for p in g['params']}
 groups=self.objective_parameter_groups(o)
 print('EXPECTATION_OWNER',json.dumps(dict(sid=self._open_sentence_slot,trial=self._sentence_trial,predictors=len(predictor),in_optimizer=sum(id(p) in registered for p in predictor),owner_parameters=len(groups['expectation']),terms=self._sentence_cost_registry.breakdown())),flush=True)
 return original(self,loss)
Models.BasicModel._sentence_train_step=inspect
raise SystemExit(pytest.main(['-q','-s','test/test_expectation_defaults.py::test_enable_after_disabled_construction_joins_optimizer_and_off_keeps_input_learning']))
