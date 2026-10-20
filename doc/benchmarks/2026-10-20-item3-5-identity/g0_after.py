"""G0: production native cube and actual compose carrier, after migration."""
from pathlib import Path
import json
import sys
import tempfile

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
# The receipt is under doc/benchmarks/<round>; source is never imported from /tmp.
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from identity_experiment import make_model

DEST=Path(__file__).resolve().parent
TREES={
    'cat':'cat','dog':'dog','the cat':('lower','the','cat'),
    'the cat again':('lower','the','cat'),'a cat':('lower','a','cat'),
    'the dog':('lower','the','dog'),
    'the black cat':('lower','the',('intersection','black','cat')),
    'the small cat':('lower','the',('intersection','small','cat')),
    'black cat':('intersection','black','cat'),
}


def measure(points):
    all_points=torch.stack(tuple(points.values()))
    def pair(left,right):
        q,c=points[left],points[right]
        meet=torch.minimum(q,c)
        return dict(request=left,candidate=right,
            equal=bool(torch.equal(q,c)),distance=float((q-c).norm()),
            meet_equals_request=bool(torch.allclose(meet,q,atol=1e-6,rtol=0)),
            meet_equals_candidate=bool(torch.allclose(meet,c,atol=1e-6,rtol=0)),
            candidate_covered=float(meet.sum()/c.sum()) if bool(c.sum()!=0) else None,
            request_covered=float(meet.sum()/q.sum()) if bool(q.sum()!=0) else None)
    pairs=[pair(a,b) for a,b in (
        ('the cat','the cat again'),('the cat','a cat'),('the cat','the dog'),
        ('the cat','the black cat'),('the cat','the small cat'),
        ('the black cat','the small cat'),('black cat','the black cat'))]
    return dict(minimum=float(all_points.min()),maximum=float(all_points.max()),
        negative_coordinates=int((all_points<0).sum()),width=all_points.shape[-1],
        points={name:value.tolist() for name,value in points.items()},questions=pairs)


def run():
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        model=make_model(Path(directory),patch)
        model.eval()
        try:
            with torch.no_grad():
                model._start_spaces_for_forward()
                raw=model.inputSpace.prepInput(['cat dog the a black small'])
                model._lex_embed_stem(raw)
                owner=model._concept_owner()
                bank=owner.similarity_codebook
                words={word:owner.interpret.word_for_form(word) for word in ('cat','dog','the','a','black','small')}
                cubes={word:bank.lookup_rows(owner._csw_row_of(identity)).detach().clone()
                       for word,identity in words.items()}
                assert all(bool(((point>=0)&(point<=1)).all()) for point in cubes.values())
                language=model.languageSpace
                operations={rule.method_name:index for index,rule in enumerate(language._compose_binary_rules)}
                def points(leaves):
                    def fold(tree):
                        if isinstance(tree,str):return leaves[tree]
                        name,left,right=tree
                        a,b=fold(left),fold(right)
                        return language.forward_binary_step(a[None],b[None],
                            torch.tensor([operations[name]],device=a.device),
                            torch.tensor([True],device=a.device))[0].detach()
                    return {name:fold(tree) for name,tree in TREES.items()}
                report=dict(source_commit=__import__('subprocess').check_output(
                    ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    training_steps=0,seed_selection='none; fixed native identity codes',
                    native_ids=words,operators=operations,
                    native_cube=measure(points(cubes)),
                    compose_carrier=measure(points({word:owner.interpret.activate(point, point.new_tensor(1.), evidence=point.new_tensor([1.,0.]))
                        for word,point in cubes.items()})))
                (DEST/'G0-after.json').write_text(json.dumps(report,indent=2)+'\n')
                text='# G0: centroid and containment on production codes\n\n'
                text+='No training or seed search. Native bank codes and the actual binding carrier are measured separately. Distances are diagnostics only.\n'
                for mode in ('native_cube','compose_carrier'):
                    result=report[mode]
                    text+=f'\n## {mode}\n\nRange {result["minimum"]:.8g} to {result["maximum"]:.8g}; {result["negative_coordinates"]} negative coordinates.\n'
                    for i,q in enumerate(result['questions']):
                        text+=f'\n### Question {i+1}: {q["request"]} / {q["candidate"]}\n\n| Measurement | Value |\n|---|---:|\n'
                        for key,value in q.items():
                            if key not in ('request','candidate'):text+=f'| {key} | {value} |\n'
                (DEST/'G0-after.md').write_text(text)
                print(json.dumps({key:{k:v for k,v in report[key].items() if k!='points'}
                                  for key in ('native_cube','compose_carrier')},indent=2))
        finally:
            model.End();model.symbolSpace.soft_reset()

if __name__=='__main__':run()
