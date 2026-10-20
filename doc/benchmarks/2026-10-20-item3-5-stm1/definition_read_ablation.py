"""Diagnostic only: restore the candidate's untyped structural edge read."""
def pytest_configure(config):
    from Layers import ConceptAllocator

    def candidate_records(self, cid):
        raw = self.store_of(cid).constituents(cid)
        return list(dict.fromkeys((*raw,
            *(('part', ('sym', part)) for part in self.parts.parts(cid)),
            *(('whole', ('sym', whole)) for whole in self.parts.wholes(cid)))))

    ConceptAllocator.records = candidate_records
