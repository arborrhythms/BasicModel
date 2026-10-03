"""Save every forced-unary choice before repairing the zero-root failure."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from test_subspace_what_stm_contract import _xor_model,_shared_compose
m=_xor_model.__wrapped__();chooser=m.languageSpace._tree_layer(2).chooser;score=chooser.score_unary
chooser.score_unary=lambda *a,**k:(lambda pair:(pair[0],torch.full_like(pair[1],1e6)))(score(*a,**k))
state,_=_shared_compose(m);state[0][:, 1, 1::2].neg_();rounds=2*state[0].shape[1];events=[]
try:
 for step in range(rounds):
  choice=m.languageSpace.choose_operation(state,torch.tensor([True]),slots=1,sample=False,rounds_left=rounds-step, replay_action=(torch.tensor([0]) if step == rounds - 1 else None))
  before=state[0].detach().tolist()
  state=m.conceptualSpace.apply_language_choice(state,choice)
  events.append(dict(step=step,depth=state[1].tolist(),before=before,after=state[0].detach().tolist(),
   choice={n:v.detach().tolist() if torch.is_tensor(v) else str(v) for n,v in choice._asdict().items()}))
 Path(__file__).with_suffix('.json').write_text(json.dumps(dict(events=events),indent=2)+'\n')
 assert bool((state[0][:,0].abs().sum(-1)>0).all())
finally:m.End()
