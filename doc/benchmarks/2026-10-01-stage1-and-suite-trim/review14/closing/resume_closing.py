"""Finish untouched cases and summarize; never restart closing measurements."""
import json
from datetime import datetime, timezone
from pathlib import Path

import resume_sweep
import summarize_sweep

HERE = Path(__file__).resolve().parent


def status(stage, **fields):
    (HERE / 'status.json').write_text(json.dumps(dict(
        stage=stage, updated=datetime.now(timezone.utc).isoformat(),
        continuation='first-attempt accounting repaired', **fields), indent=2) + '\n')


if __name__ == '__main__':
    status('full_sweep')
    try:
        receipt = resume_sweep.run()
        if not receipt['complete']:
            status('full_sweep', complete=False, remaining=len(receipt['unattempted']))
            raise SystemExit(1)
        assert receipt['source_matched']
        status('sweep_summary')
        summary = summarize_sweep.summarize()
        assert summary['complete'] and summary['source_matched']
        status('complete', sweep=summary)
    except Exception as exc:
        status('error', error=repr(exc))
        raise
