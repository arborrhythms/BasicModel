"""Static declaration/consumer inventory; no model or training initialization."""
import ast, json, sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'bin'))
from Language import Grammar, GRAMMAR_LAYER_CLASSES

def main():
    grammar=Grammar();grammar.load_from_grammar_file('complete.grammar')
    properties=('clause_form','head_role','relation_kind','scope_transparent','polarity_effect',
        'meaning_mode','same_reference_idempotent','predicate_identity','field_eligible','order_delta',
        'case_head_role','determiner_mode','substrate','mandatory','footprint_reads','footprint_writes')
    rules=[dict(canonical=r.canonical,**{p:getattr(r,p) for p in properties}) for r in grammar.rules+grammar.thought_rules+grammar.ps_rules]
    names=('op_name','rule_name','method_name')
    comparisons=[];consumers=[]
    consumer_names={'lookup_rows','binding','occurrence_terms','diffuse_memberships','centroid_forms',
        'activate_code','compose_blocks','occurrence_memberships','stamp_input_when','set_time'}
    for path in sorted((ROOT/'bin').glob('*.py')):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Compare):
                text=ast.unparse(node)
                if any(n in text for n in names) and any(isinstance(n,ast.Constant) and isinstance(n.value,str) for n in ast.walk(node)):
                    comparisons.append(dict(file=str(path.relative_to(ROOT)),line=node.lineno,expression=text))
            if isinstance(node,ast.Call):
                name=getattr(node.func,'attr',getattr(node.func,'id',None))
                if name in consumer_names:consumers.append(dict(file=str(path.relative_to(ROOT)),line=node.lineno,consumer=name))
    semantic_name_tests=[r for r in comparisons if any(f"'{op}'" in r['expression'] or f'"{op}"' in r['expression'] for op in GRAMMAR_LAYER_CLASSES)]
    # Semantic spelling is resolved at grammar registration, never by a
    # runtime branch on op_name/rule_name/method_name.
    assert not semantic_name_tests, semantic_name_tests
    result=dict(rules=rules,declared_rules=len(rules),runtime_semantic_name_tests=semantic_name_tests,
        nonsemantic_name_comparisons=comparisons,consumers=consumers,consumer_counts=dict(Counter(r['consumer'] for r in consumers)),
        when_inventory=json.loads((ROOT/'test/fixtures/when-readers-round4a0.json').read_text()),
        attention='unchanged 6.8 where-mask walk; no relevance weight or soft filter',
        meanings='content-key codes -> detached occurrence means -> independent bilattice -> kept sentence row',
        priming='combined concept/membership degree budget -> detached codebook retrieval prior',
        form='native lower bound -> adjacent ceiling -> containment projection -> fixed binding projection; index uses c == 1')
    (HERE/'consumer-census.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rules=len(rules),consumers=len(consumers),semantic_name_tests=len(semantic_name_tests))),flush=True)

if __name__=='__main__':main()
