"""The explicitly requested read-only Claude review, after all scoped gates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
source = json.loads((HERE/'frozen-source.json').read_text())['source']
summary = json.loads((HERE/'gate-summary.json').read_text())
assert summary['source'] == source and summary['all_requested_gates_pass']
assert not subprocess.check_output(['git','diff',source,'--name-only','--','bin','test','data'],
                                   cwd=ROOT,text=True).strip()
command = ['/Users/arogers/.local/bin/claude', '--print', '--model', 'opus',
    '--safe-mode', '--restricted', '--strict-mcp-config', '--tools', 'Read,Grep,Glob',
    '--allowedTools', 'Read,Grep,Glob', '--permission-mode', 'dontAsk',
    '--permission-prompts', 'none', '--output-format', 'stream-json', '--verbose']
request = (HERE/'claude-review-request.md').read_bytes()
record = dict(source=source,command=command,request_sha256=hashlib.sha256(request).hexdigest(),
              started=datetime.now(timezone.utc).isoformat())
path = HERE/'claude-review-run.json'
assert not path.exists()
path.write_text(json.dumps(record,indent=2)+'\n')
with (HERE/'claude-review-stream.jsonl').open('xb') as output, (HERE/'claude-review-stderr.log').open('xb') as error:
    result = subprocess.run(command,cwd=ROOT,input=request,stdout=output,stderr=error)
record.update(exit=result.returncode,finished=datetime.now(timezone.utc).isoformat())
for line in (HERE/'claude-review-stream.jsonl').read_text().splitlines():
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        continue
    if item.get('type') == 'system' and item.get('model'):
        record['model'] = item['model']
    if item.get('type') == 'result':
        record.update(is_error=item.get('is_error'),subtype=item.get('subtype'))
        if item.get('result'):
            (HERE/'claude-review.md').write_text(item['result']+'\n')
            record['report'] = 'claude-review.md'
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
