"""Twenty fresh processes per concrete test-29 and test-31 case, unseeded."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
selectors = ['test/test_item7_word_admission.py::test_29_shared_whole_does_not_evidence_an_absent_word']
for configuration in ('MM_xor.xml', 'MM_grammar.xml', 'XOR_grammar.xml'):
    selectors.append('test/test_item7_word_admission.py::test_29_each_reading_publishes_only_its_words[' + configuration + ']')
    selectors.append('test/test_item7_definitions.py::test_31_read_words_have_definition_rows_without_changing_the_configuration[' + configuration + ']')
raise SystemExit(subprocess.call([sys.executable, str(HERE / 'run_checks.py'),
                                sys.argv[1], str(ROOT), *(selectors * 20)]))
