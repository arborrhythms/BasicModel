"""Read-only Claude review of the dated 2b spec questions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
base = '0b030e3ebdeca87f9ebf80739237437c3c75ea8f'
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not (HERE/'review-input.json').exists()
(HERE/'review-input.json').write_text(json.dumps(dict(review_input=source, production_base=base, scope='spec questions only; no implementation or gates'), indent=2)+'\n')
assert not subprocess.check_output(['git', 'diff', base, '--name-only', '--', 'bin', 'test', 'data'],
                                   cwd=ROOT, text=True).strip()
command = ['/Users/arogers/.local/bin/claude', '--print', '--model', 'opus',
    '--safe-mode', '--restricted', '--strict-mcp-config', '--tools', 'Read,Grep,Glob',
    '--allowedTools', 'Read,Grep,Glob', '--permission-mode', 'dontAsk',
    '--permission-prompts', 'none', '--output-format', 'stream-json', '--verbose']
request = (HERE/'claude-review-request.md').read_bytes()
run = dict(source=source, command=command, request_sha256=hashlib.sha256(request).hexdigest(),
           started=datetime.now(timezone.utc).isoformat())
path = HERE/'claude-review-run.json'
assert not path.exists()
path.write_text(json.dumps(run, indent=2)+'\n')
with (HERE/'claude-review-stream.jsonl').open('wb') as output, (HERE/'claude-review-stderr.log').open('wb') as error:
    result = subprocess.run(command, cwd=ROOT, input=request, stdout=output, stderr=error)
run.update(exit=result.returncode, finished=datetime.now(timezone.utc).isoformat())
for line in (HERE/'claude-review-stream.jsonl').read_text().splitlines():
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        continue
    if item.get('type') == 'system' and item.get('model'):
        run['model'] = item['model']
    if item.get('type') == 'result':
        run['is_error'] = item.get('is_error')
        run['subtype'] = item.get('subtype')
        if item.get('result'):
            (HERE/'claude-review.md').write_text(item['result']+'\n')
            run['report'] = 'claude-review.md'
path.write_text(json.dumps(run, indent=2)+'\n')
print(json.dumps(run, indent=2))
raise SystemExit(result.returncode)
