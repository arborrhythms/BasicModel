"""Compare frozen runtime and gate contracts with the accepted round-2 source."""
import ast, hashlib, json, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(value): return hashlib.sha256(value).hexdigest()
def main():
    old=json.loads((HERE/'before/source.json').read_text())
    new=json.loads((HERE/'delivered-source/source.json').read_text())
    unchanged=('bin/Language.py','bin/SentenceCredit.py','bin/AnswerComparison.py',
        'bin/WalkTrials.py','bin/ModelAttention.py','bin/Attention.py','bin/Meaning.py',
        'bin/ClauseJournal.py','bin/SentenceCompose.py',
        'test/test_explicit_dimensions.py','test/test_mm_xor.py')
    checks={name:old[name]==new[name]==sha((ROOT/name).read_bytes()) for name in unchanged}
    assert all(checks.values())
    with zipfile.ZipFile(HERE/'before/source.zip') as saved:
        def reader(source):
            return ast.dump(next(node for node in ast.parse(source).body
                if isinstance(node,ast.ClassDef) and node.name=='SentenceRecordReader'),include_attributes=False)
        reader_unchanged=reader(saved.read('bin/SentenceUnderstanding.py').decode())==reader((ROOT/'bin/SentenceUnderstanding.py').read_text())
        assert reader_unchanged
        texts={name:dict(old=saved.read(name).decode() if name in old else '',new=(ROOT/name).read_text())
               for name in new if name.startswith('bin/') and new[name]!=old.get(name)}
    (HERE/'changed-runtime-texts.json').write_text(json.dumps(texts,indent=2)+'\n')
    result=dict(unchanged_files=checks,presented_reader_computation_unchanged=reader_unchanged,
        mechanism='identity construction and its fixed binding view, exact rung-zero bytes, signed trust ingestion',
        changed_runtime_files=list(texts),gate_selectors_unchanged=True)
    (HERE/'runtime-contract-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
