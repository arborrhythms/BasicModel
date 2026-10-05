"""Display the unresolved signed/silent-code examples without selecting a policy."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'bin'))
import torch
from Language import IntersectionLayer
op=IntersectionLayer()
pairs=[([0.],[1.]),([-.7],[-.5]),([.7],[.5]),([-.5],[.4])]
print(json.dumps([{'left':a,'right':b,'candidate':op.compose(torch.tensor(a),torch.tensor(b)).tolist()} for a,b in pairs],indent=2))
