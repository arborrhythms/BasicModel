"""Verify the closing source and retained evidence without changing old receipts."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVIEW = HERE.parent / '2026-10-08-math-chain-repair-2'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot
from verification import validate

spec = importlib.util.spec_from_file_location('retained_closing_checks', REVIEW / 'finalize_closing.py')
retained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)


def main():
    source = source_snapshot(ROOT)
    checked = validate(source)
    checks = dict(
        closing_helpers=retained.verify(ROOT, REVIEW / 'closing-source-4/helpers.json'),
        closing_artifacts=retained.verify(REVIEW, REVIEW / 'closing-artifacts-sha256.json'),
        stopped_files=retained.verify(REVIEW, REVIEW / 'stopped-by-decision/retained-files-sha256.json'),
        stop_receipt=retained.verify(REVIEW / 'stopped-by-decision', REVIEW / 'stopped-by-decision/receipt-sha256.json'),
        old_campaign_helpers=retained.verify(ROOT, REVIEW / 'measured-source/measurement-helpers.json'),
        standing_campaign_helpers=retained.verify(ROOT, HERE / 'measured-source/measurement-helpers.json'),
        prior_receipts=[retained.verify(HERE.parent / name, HERE.parent / name / 'receipt-sha256.json')
            for name in ('2026-10-07-math-chain', '2026-10-08-math-chain-repair')])
    contracts = retained.read(REVIEW / 'frozen-contracts.json')
    assert all(name == 'bin/BindingAnswers.py' or retained.digest(ROOT / name) == value
               for name, value in contracts.items())
    binding = retained.read(REVIEW / 'binding-answers-verifier.json')
    assert retained.digest(ROOT / 'bin/BindingAnswers.py') == binding['current_sha256']
    assert binding['functions_byte_identical']['matches']
    corrections = retained.read(REVIEW / 'protocol-corrections-14-13.json')
    assert retained.digest(REVIEW / 'protocol.json') == corrections['original_protocol_sha256']
    assert len(corrections['corrections']) == 4
    summary = retained.read(HERE / 'standing-summary.json')
    assert summary['condition_satisfied'] and summary['attempts'] == 30
    xml = ET.parse(HERE / 'landing-doc-links.xml')
    suites = xml.getroot().findall('testsuite')
    assert sum(int(x.attrib['tests']) for x in suites) == 348
    assert all(int(x.attrib['errors']) == int(x.attrib['failures']) == 0 for x in suites)
    result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        exact_closing_source=checked, retained_evidence=checks,
        frozen_contracts_unchanged=True, binding_matches_unchanged=True,
        original_protocol_unchanged=True, four_corrections_recorded=True,
        document_links=dict(passed=348,failed=0,result='landing-doc-links.xml'),
        standing_condition_satisfied=True, learning_claim=False)
    with (HERE / 'landing-integrity.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
