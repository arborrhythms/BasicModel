"""Observe admission's unresolved typed-reference fallback."""
import json
import os
from pathlib import Path
import pytest
import torch
from arrival_capture import plain

@pytest.fixture(autouse=True)
def observe_lexical(request,monkeypatch):
    from Language import LanguageSpace
    from Interpret import InterpretLayer
    # Read the class owning the capture method from the runtime module.
    import Language
    cls = next(c for c in vars(Language).values() if isinstance(c,type)
               and 'resolve_lexical_references' in c.__dict__)
    resolve, interpret = cls.resolve_lexical_references, InterpretLayer.forward
    events=[]
    def write():
        (Path(os.environ['CONTRACT_CAPTURE']).parent/'lexical-probe.json').write_text(json.dumps(plain(events),indent=2)+'\n')
    def forward(self,word,*args,**kwargs):
        value=interpret(self,word,*args,**kwargs)
        if not torch.is_tensor(word):
            events.append(dict(call='interpret',word=word,requested=kwargs.get('order'),result=value,
                field_pending=word in self._field_pending,objects=self.owner.definitions.objects(word)))
            write()
        return value
    def resolved(self,owner,word_rows,concept_ids,actions,**kwargs):
        value=resolve(self,owner,word_rows,concept_ids,actions,**kwargs)
        events.append(dict(call='references',admit=kwargs.get('admit'),source=concept_ids,result=value,
            readings=[dict(surface=owner.word_surface_for_row(int(row)),
                concepts=[(cid,owner._concept_source_order(cid)) for cid in owner.word_concepts(owner.word_surface_for_row(int(row)))])
                for row in word_rows]))
        write()
        return value
    monkeypatch.setattr(InterpretLayer,'forward',forward)
    monkeypatch.setattr(cls,'resolve_lexical_references',resolved)
