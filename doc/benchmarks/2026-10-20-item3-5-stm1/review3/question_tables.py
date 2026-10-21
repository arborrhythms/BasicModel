"""Render the native question records; no result selection or regrading."""
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
output = ['# Review 3: decomposition questions on `47652a9fe`', '',
    'Original parameter snapshots, seeds 4199722147 / 2428440506, budgets',
    '512 / 768, forced scopes and R + A keep law are unchanged. Both arms',
    'keep prime on and φ=.5; gains are all zero or all .5. Every companion',
    'question is reported below. These are mechanism certificates.', '']
all_questions = []
for arm in ('zero', 'enabled'):
    for length in (1, 2):
        directory = (HERE/f'decomposition-{arm}-29-reports'/
            f'test_forced_native_decomposition_through_training_driver[{length}]'/
            f'decomposition-{length}')
        path = directory/'questions.jsonl'
        questions = list(map(json.loads, path.read_text().splitlines()))
        for question in questions:
            all_questions.append(dict(arm=arm, **question))
            output += [f'## {arm}, length {length}, document {question["document"]}', '',
                '| Measurement | Result |', '|---|---|']
            for key in ('answer', 'bound_correct', 'expected_chain_length',
                        'inference_count', 'chain_correct', 'work', 'open',
                        'committed_binding', 'operations'):
                output.append(f'| {key.replace("_", " ")} | `{json.dumps(question[key])}` |')
            output += ['', f'[Native question records]({path.relative_to(HERE)}); '
                f'[bounded run](decomposition-{arm}-29/report.html).', '']
output.insert(2, f'{sum(bool(q["chain_correct"]) for q in all_questions)}/{len(all_questions)} '
    'companion questions have the correct proof chain.\n')
(HERE/'decomposition-29-questions.md').write_text('\n'.join(output)+'\n')
(HERE/'decomposition-29-questions.json').write_text(json.dumps(all_questions, indent=2)+'\n')
