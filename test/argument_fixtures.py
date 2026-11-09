"""Native word geometry for isolated formal-argument mechanism certificates."""
import torch
from PartsOfSpeech import PartsOfSpeech
from definition_fixtures import agreeing_views
from test_review13_subspaces import fixture


def argument_inventory(arguments, count=4):
    owner, bank, _, _ = fixture(meaning_pairs=max(4, count), capacity=max(32, len(arguments)+count+8))
    parts = PartsOfSpeech(arguments, owner)
    # The fixture owns its concept space for as long as its grammar catalogue.
    object.__setattr__(parts, '_fixture_owner', owner)
    words = tuple(owner.interpret.lookup_word([4+i], (), form=f'form-{i}') for i in range(count))
    for i, word in enumerate(words):
        row = owner._csw_row_of(word)
        bank.mereology.placement[row] = torch.eye(bank.mereology.meaning_pairs)[i]
        bank.mereology.placement_witnesses[row] = 1
    agreeing_views(owner, *words)
    return owner, parts, words


def view_counts(parts):
    bank = parts.owner.similarity_codebook
    return {name: int(bank.trust.views[parts.owner._csw_row_of(cid)])
            for name, cid in parts.concepts.items()}
