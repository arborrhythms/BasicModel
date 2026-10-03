from pathlib import Path
import json, torch
ROOT=Path(__file__).resolve().parents[3]
rows=[]
for p in list((ROOT/'data').glob('*.ckpt'))+list((ROOT/'test/fixtures').glob('*.pt')):
    state=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    meta={k:type(v).__name__ for k,v in state.items()}
    saved=state.get('optimizer') or state.get('optimizer_state') or state.get('optimizer_state_dict')
    def description(value):
        if not isinstance(value,dict):return type(value).__name__
        return {k:(len(v) if isinstance(v,(dict,list)) else type(v).__name__) for k,v in value.items()}
    rows.append(dict(path=str(p.relative_to(ROOT)),keys=meta,optimizer=description(saved)))
print(json.dumps(rows,indent=2))
(Path(__file__).parent/'checkpoint-inventory.json').write_text(json.dumps(rows,indent=2))
