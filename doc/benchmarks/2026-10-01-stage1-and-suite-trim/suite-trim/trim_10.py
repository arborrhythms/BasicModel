"""Install the bounded weekly target and its coverage-age warning."""
import ast,json,shutil
from port_ledger import HERE,ROOT

paths=['test/slow_tests.py','test/bounded_tests.py','Makefile']
before={name:(ROOT/name).read_text() for name in paths}
shutil.copyfile(HERE/'slow_tests.draft.py',ROOT/'test/slow_tests.py')
shutil.copyfile(HERE/'slow_weekly.draft.py',ROOT/'test/slow_weekly.py')
p=ROOT/'test/bounded_tests.py';source=p.read_text()
position=source.index('\ndef worker_environment(root):')
helper='''
def slow_coverage_warning(root, now=None):
    """A full sweep reports missing, stale or incomplete weekly coverage."""
    from datetime import datetime, timezone
    path = Path(root) / 'tmp/slow-tests/latest.json'
    if not path.exists():
        return 'No weekly slow-test record; run make test_slow_weekly.'
    try:
        record = json.loads(path.read_text())
        date = datetime.fromisoformat(record['date'])
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        current = datetime.now(timezone.utc) if now is None else now
        age = (current - date).total_seconds() / 86400
    except (OSError, ValueError, TypeError, KeyError):
        return 'The weekly slow-test record is unreadable; run make test_slow_weekly.'
    if age > 7:
        return f'Latest weekly slow-test record is {age:.1f} days old (over seven days).'
    if not record.get('complete') or not record.get('source_matched'):
        return 'Latest weekly slow-test run did not complete on matching source.'
    if record.get('exit_code'):
        return 'Latest weekly slow-test run has failures; see tmp/slow-tests/latest.json.'
    return None

'''
source=source[:position]+ '\n'+helper+source[position:]
needle='''    def save():
        result["elapsed_seconds"]'''
assert needle in source
source=source.replace(needle,'''    if requires_suite_lock(selectors, keyword, marker):
        warning = slow_coverage_warning(root)
        result["warnings"] = [] if warning is None else [warning]
        if warning is not None:
            print("[slow coverage warning] " + warning, flush=True)

    def save():
        result["elapsed_seconds"]''',1)
needle='''        "<p>Collection, worker logs, process limits and complete coverage are recorded in result.json.</p>"'''
assert needle in source
source=source.replace(needle,needle+'''\n        + "".join(f"<p><strong>Warning:</strong> {html.escape(warning)}</p>"
                  for warning in result.get("warnings", []))
        +''',1)
# An explicit + above separates the expression from the remaining adjacent strings.
ast.parse(source);p.write_text(source)
p=ROOT/'Makefile';source=p.read_text();needle='\npreflight : $(VENV_STAMP)'
assert needle in source
source=source.replace(needle,'''
# Existing environment only; the item 6.9 baseline holds the venv rebuild.
# 8 GiB ordinary workers; 24 GiB only for the native production stage-1 arms.
.PHONY: test_slow_weekly
test_slow_weekly:
\tPYTHONPATH=bin:test $(VENV_PYTHON) test/slow_weekly.py

preflight : $(VENV_STAMP)''',1);p.write_text(source)
(HERE/'weekly-target-source-changes.json').write_text(json.dumps([
 {'path':name,'old':old,'new':(ROOT/name).read_text()} for name,old in before.items()
]+[{'path':'test/slow_weekly.py','old':None,'new':(ROOT/'test/slow_weekly.py').read_text()}],indent=2)+'\n')
print('Installed weekly target: 8 GiB ordinary / 24 GiB native, no environment rebuild.')
