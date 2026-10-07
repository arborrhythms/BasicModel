"""Static path audit and frozen-prefix certificate before any standing training."""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]

def main():
    import torch
    from Interpret import activate_code
    from MeaningCodes import compose
    from tetralemma_audit import audit
    from test_operators_final_b import forced_prefix_certificate
    from test_tetralemma_leaf import CORNERS
    path = audit()
    assert not path['violations'], path['violations']
    (HERE/'tetralemma-path-audit.json').write_text(json.dumps(path, indent=2)+'\n')
    code = torch.tensor([[1., .5, .25]])
    atom = torch.cat((torch.tensor([[.6,.8]]), code, code*0), -1)
    leaves = [activate_code(atom, torch.tensor([.4]), 2, evidence=torch.tensor([corner]))
              for corner in CORNERS]
    report = dict(corners=[dict(pair=corner, leaf=leaf[0].tolist()) for corner,leaf in zip(CORNERS,leaves)],
        connectives={name:[dict(left=CORNERS[i],right=CORNERS[j],meaning=compose(name,a[:,2:],b[:,2:])[0].tolist())
            for i,a in enumerate(leaves) for j,b in enumerate(leaves)]
            for name in ('conjunction','disjunction','sum')},
        required_evidence=dict(inputs=[[1.,0.],[0.,1.]], stored_pair=[1.,1.],
            test='test/test_tetralemma_leaf.py::test_required_evidence_read_of_true_and_false_is_both'),
        path_sites=len(path['sites']), direct_accesses=len(path['direct_accesses']),
        inverse='one entry per word, form-only shortlist and pair choice; recover both lanes independently',
        tests='test/test_tetralemma_leaf.py: actual walk handoff, interpretation, pushed/reference slabs and stored row; all corner pairs, graded inverse and mutation-tested AST guard')
    (HERE/'tetralemma-certificate.json').write_text(json.dumps(report,indent=2)+'\n')
    prefix=forced_prefix_certificate()
    assert prefix['snapshot_counts']==[2,2]
    assert all(r['multisets']==4 and r['residual']==[0.]*4 and r['R']==[0.]*4 for r in prefix['reads'])
    (HERE/'repair-certificates.json').write_text(json.dumps(prefix,indent=2)+'\n')
    print(json.dumps(dict(path_sites=len(path['sites']), violations=0, prefix=prefix)))

if __name__=='__main__': main()
