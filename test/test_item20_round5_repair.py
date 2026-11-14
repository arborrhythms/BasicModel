"""Declared sentence banks and one-stream short-term addressing."""
from Memory import STM0
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from test_item20_round5_memory import receipt


@pytest.fixture(scope='module')
def default_model():
    from data import TheData
    from Models import BaseModel
    TheData.load('xor')
    model, _ = BaseModel.from_config('data/model.xml', data=TheData)
    yield model
    model.End()


def test_default_pinned_banks_use_the_declared_word_bound(default_model):
    strata = default_model.conceptualSpace.stm.strata
    recent = default_model.conceptualSpace.stm.capacity
    assert default_model.serial_word_capacity == strata.words == 256
    assert recent == 8
    assert len(strata.banks) == 3 * default_model._concept_owner().nStreams
    assert all(bank.capacity == 256 + 3*256 + 2*recent == 1040 for bank in strata.banks)
    assert all(not hasattr(bank, 'stream')
               for bank in (*strata.banks, *strata.recent_stores))
    declared = sum(value.numel()*value.element_size() for value in strata.buffers())
    # Ordered-pair stores are linked rather than registered child modules.
    linked = sum(value.numel()*value.element_size()
                 for bank in (*strata.banks, *strata.recent_stores)
                 for value in bank.parts.buffers())
    assert declared + linked < .25 * 2**30
    receipt('bank-bound', dict(words=strata.words, streams=strata.streams,
        banks=len(strata.banks), rows_per_bank=strata.banks[0].capacity,
        stream_columns=0, declared_bytes=declared, declared_gib=declared/2**30,
        linked_pair_bytes=linked, all_buffer_bytes=declared+linked))


def test_pinned_capture_takes_256_words_and_rejects_257(default_model):
    strata = default_model.conceptualSpace.stm.strata
    bank = strata.banks[0]
    shape = tuple(bank.slots.shape)
    def program(words):
        return SimpleNamespace(leaves=bank.slots.new_zeros(words, shape[-1]),
            actions=torch.tensor([(0, -1, word) for word in range(words)]),
            concept_ids=torch.full((words,), -1, dtype=torch.long))
    accepted = strata.capture(program(256), default_model.languageSpace, 0)
    assert len(accepted.tree.store) == len(accepted.tree.roots) == 256
    with pytest.raises(OverflowError, match='257/256 words'):
        strata.capture(program(257), default_model.languageSpace, 0)
    assert all(tuple(store.slots.shape) == shape for store in strata.banks)
    strata.clear()
    receipt('capture-bound', dict(accepted_words=256, rejected_words=257,
        fixed_shape=list(shape), resized=False))


def test_construction_requires_the_word_bound_in_config_or_defaults(tmp_path, monkeypatch):
    import Models
    from data import TheData
    tree = ET.parse('data/model.xml')
    architecture = tree.getroot().find('architecture')
    architecture.remove(architecture.find('serialWordCapacity'))
    own, defaults = tmp_path/'own.xml', tmp_path/'defaults.xml'
    tree.write(own); tree.write(defaults)
    initialize = Models.init_config
    monkeypatch.setattr(Models, 'init_config', lambda path=None, defaults_path=None:
        initialize(path=path, defaults_path=str(defaults)))
    TheData.load('xor')
    with pytest.raises(ValueError, match='<serialWordCapacity>'):
        Models.BaseModel.from_config(str(own), data=TheData)
    receipt('required-sentence-bound', dict(own_declared=False,
        defaults_declared=False, raised=True, element='serialWordCapacity'))
