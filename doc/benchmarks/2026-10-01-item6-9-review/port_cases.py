"""Complete case bodies and their changed shared fixture, for all seventeen ports."""
import hashlib
import json
from pathlib import Path
import sys
from audit import bodies

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
selectors = json.loads((HERE/'port-selectors.json').read_text())


def cases():
    result = []
    for node in selectors:
        file, *symbols = node.split('::')
        name = '.'.join(symbol.split('[',1)[0] for symbol in symbols)
        source = ROOT/file
        result.append(dict(nodeid=node, file=file, symbol=name,
            body=bodies(source)[name], file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            shared_fixture=('test/reading_fixtures.py::commit_reading'
                if 'test_closing_discards_operations' not in node and 'test_missing_absorbed_operand' not in node else None)))
    return result


if sys.argv[1]=='before':
    path=HERE/'ports/old-cases.json'
    assert not path.exists()
    path.write_text(json.dumps(cases(),indent=2)+'\n')
elif sys.argv[1]=='after':
    before=json.loads((HERE/'ports/old-cases.json').read_text())
    after=cases()
    changes=json.loads((HERE/'ports/bodies.json').read_text())
    helper=next(change for change in changes if change['symbol']=='commit_reading')
    entries=[]
    for old,new in zip(before,after):
        assert old['nodeid']==new['nodeid']
        entries.append(dict(nodeid=new['nodeid'],file=new['file'],symbol=new['symbol'],
            old_body=old['body'],new_body=new['body'],
            body_unchanged=old['body']==new['body'],shared_fixture=new['shared_fixture'],
            fixture_old_body=helper['old_body'] if new['shared_fixture'] else None,
            fixture_new_body=helper['new_body'] if new['shared_fixture'] else None,
            reason=('Fixture gains the two step-7 fields; all case assertions are unchanged.'
                    if new['shared_fixture'] else
                    'The direct sentence-state builder gains the two step-7 fields and checks their disposal.'
                    if 'test_closing_discards_operations' in new['nodeid'] else
                    'The decided inverse returns a hard least-residual candidate pair; boundedness and the known right operand are preserved.')))
    (HERE/'ports/case-bodies.json').write_text(json.dumps(entries,indent=2)+'\n')
    lines=['# Seventeen ports', '', 'Every case’s complete old and new body is in `case-bodies.json`. '
           'For the fifteen unchanged case bodies, the complete old/new shared fixture is included too.', '',
           '| Case | Case body unchanged | Reason |', '|---|---|---|']
    lines += ['| '+entry['nodeid']+' | '+str(entry['body_unchanged'])+' | '+entry['reason']+' |' for entry in entries]
    (HERE/'ports/README.md').write_text('\n'.join(lines)+'\n')
else:
    raise ValueError(sys.argv[1])
