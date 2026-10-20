"""Affected mechanism and lexical boundaries, without a landing sweep."""
import os
from pathlib import Path
import subprocess
import sys

receipt = Path(__file__).resolve().parent
root = receipt.parents[2]
name = 'candidate-scoped'
if (receipt/name).exists():
    raise FileExistsError(receipt/name)
files = (
    'bind_thought', 'clause_acceptance', 'definition_integrity', 'definition_rows',
    'definition_sparsity', 'definition_taxonomy', 'generation_lesson',
    'grammar_lesson_context', 'grammar_word_learning', 'grammatical_query_vps',
    'independent_components', 'item6_2_thinking', 'math_chain_repair2',
    'math_chain_section14', 'operator_catalogue', 'operators_final_b',
    'reference_close', 'reference_mentions', 'serial_reverse_errors',
    'thought_operation_catalog', 'tied_operator_reconstruction', 'walk_trials',
    'when_grammar_rules', 'word_interpretation',
)
env = dict(os.environ, RUN_SLOW='1', BASICMODEL_DEVICE='cpu',
    ITEM35_BIND_RECEIPT=str(receipt/'generated-candidate'))
command = [sys.executable, 'test/test_report.py',
    *(f'test/test_{name}.py' for name in files),
    '--workers', '3', '--memory-gib', '24', '--timeout', '900',
    '--suite-timeout', '1800', '--batch-size', '16', '--max-files', '1',
    '--run-dir', str(receipt/name)]
with (receipt/(name+'.log')).open('w') as stream:
    result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                            stderr=subprocess.STDOUT)
print(name, result.returncode, flush=True)
sys.exit(result.returncode)
