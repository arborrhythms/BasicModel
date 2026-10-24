"""Read-only evidence after the captured output-walk failure's native call."""
import json,os
from pathlib import Path
import pytest,torch

@pytest.fixture(autouse=True)
def probe(monkeypatch):
    import Generative,Models
    walk=Models.BasicModel._output_generate_walk
    def finish(self,*args,**kwargs):
        result=walk(self,*args,**kwargs)
        event=args[0];depth=event.ne(0).any(-1).sum(-1);parent=event[torch.arange(len(event)),depth-1]
        bank=kwargs['basis'];language=self.languageSpace
        with torch.no_grad():
            _,_,legal=Generative.inverse_menu(language,parent,depth>0,
                basis=bank,basis_valid=kwargs['basis_valid'])
        record=dict(same_codes=torch.equal(bank[:,0],bank[:,1]),legal=legal.tolist(),
            names=[*language._generate_binary_names,*language._generate_unary_names,'STOP'],
            parent=parent.tolist(),emitted=result[1].tolist(),pending=result[2].tolist())
        Path(os.environ['OUTPUT_PROBE']).write_text(json.dumps(record,indent=2)+'\n')
        return result
    monkeypatch.setattr(Models.BasicModel,'_output_generate_walk',finish)
