from pathlib import Path
import inspect, json
import pytest
import ReferenceParticipants
from ClauseJournal import finish_clause
from test_clause_acceptance import SentenceFixture
from test_identity_threading import recorded_identity_reading

source=inspect.getsource(ReferenceParticipants.participant_terms)
start=source.index('        if lower or name == \'bind\':\n')
end=source.index('    return eligible, inspected',start)
replacement='''        if lower or name == 'bind':
            positions = {(head or len(children))-1: True}
        elif name in ('equal', 'part', 'whole'):
            positions = {i: True for i in range(len(children))}
        elif name == 'lift' and len(children) == 2:
            positions = {i: i+1 != head for i in range(len(children))}
        elif noun and head and not clause:
            positions = {head-1: True}
        elif name == 'verb' and len(children) == 2:
            positions = {i: i+1 != head for i in range(len(children))}
        elif head:
            positions = {head-1: noun}
        else:
            positions = {i: noun for i in range(len(children))}
        # Inspect every subtree for its own declared positions. A non-head
        # operand is not itself a noun, but can contain a prior determination.
        pending.extend((child, positions.get(i, False))
                       for i, child in reversed(tuple(enumerate(children))))
'''
proposed=source[:start]+replacement+source[end:]
namespace={}; exec(proposed,namespace)
result={}
with pytest.MonkeyPatch.context() as patch:
 f=SentenceFixture(patch)
 entry,_,handle=recorded_identity_reading(f,('bind',('lower',('lower_unary','subject'),'tail')))
 try:
  clause=finish_clause(f.language,entry,registry=f.registry)
  f.cs.components.reference_close.preview(clause,())
 except RuntimeError as e:
  result['before']=str(e)
 patch.setattr(ReferenceParticipants,'participant_terms',namespace['participant_terms'])
 clause=finish_clause(f.language,entry,registry=f.registry)
 plan=f.cs.components.reference_close.preview(clause,())
 result['after']=[dict(mode=item['mode'],selected=item['selected'],handle=item['node'].identity['handle']) for item in plan['items']]
 assert result['before']=='a compose bind names an unwritten introduction'
 assert len(plan['items'])==2 and all(item['selected']==('new',0) for item in plan['items'])
root=Path('doc/benchmarks/2026-10-23-identity-threading/final/source-06')
(root/'participant-walk-proposal.py').write_text(proposed)
(root/'participant-proposed-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
