"""One bounded sweep; candidate commits replace source archives and checks.

Keep the existing process, memory, selection and timeout controls. Disable
only the supervisor's source snapshots and manifest, per Alec's October 11
measurement instruction. Fresh pytest workers retain their ordinary behavior.
Arguments are those of test/bounded_tests.py.
"""
from unittest.mock import patch

import bounded_tests


if __name__ == '__main__':
    write = bounded_tests.write_json
    def report(path, value):
        if path.name != 'source-manifest.json':
            write(path, value)
    with patch.object(bounded_tests, 'source_snapshot', lambda root: None), \
         patch.object(bounded_tests, 'documentation_snapshot', lambda root: None), \
         patch.object(bounded_tests, 'write_json', report):
        code, _ = bounded_tests.main()
    raise SystemExit(code)
