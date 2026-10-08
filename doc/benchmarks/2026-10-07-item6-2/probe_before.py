from pathlib import Path
p=Path('bin/Queries.py').read_text()
for name in ('ask','query','isTrue','exist','isEqual','isImplied','implies','gain'):
    assert f"ThoughtExecutorDescriptor('{name}'" in p, f"missing thought face: {name}"
