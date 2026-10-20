"""Receipt instrumentation only; leave the closing transaction unchanged."""
import json
import os
from pathlib import Path


def pytest_configure(config):
    from PartClosing import PartClosing
    preflight = PartClosing.preflight

    def record(self):
        try:
            return preflight(self)
        except RuntimeError:
            data = dict(next_id=self.alloc.next_id, planned_end=self._next,
                capacity=self.owner.nVectors, pools=self.owner._order_caps(),
                rows=list(self.alloc.layer()._tensor_rows.items()),
                plan=[dict(symbol=r['symbol'], seat=r['seat'], edge=r['edge'],
                    pair=r['pair'], types=r['types'], order=r['value'].order,
                    reference=r['value'].reference, word=r['value'].word_id,
                    operation=r['value'].operation,
                    children=len(r['value'].children)) for r in self._plan])
            with Path(os.environ['CLOSING_CAPACITY_LOG']).open('a') as out:
                out.write(json.dumps(data) + '\n')
            raise
    PartClosing.preflight = record
