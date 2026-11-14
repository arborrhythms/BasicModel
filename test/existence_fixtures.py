"""Read one explicit claim without inventing a conceptual field for a vector."""
from Queries import ThoughtLTMCapability
from QueryWork import QueryWorkBudget

def read_claim(store, row):
    return ThoughtLTMCapability(store=lambda: store, equal=lambda a,b: 1., tau_id=.9).end_evidence(
        store.occurrence_of(row), max_records=1, work=QueryWorkBudget(1))
