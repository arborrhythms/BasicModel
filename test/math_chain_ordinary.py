"""Real document presentation and the immutable measurement observer."""
from pathlib import Path
import importlib.util
import json
import re
import torch
from math_chain_corpus import flatten
from test_math_chain import ROOT, build_model


def observer_class():
    path = ROOT/'doc/benchmarks/2026-10-07-math-chain/math_observer.py'
    spec = importlib.util.spec_from_file_location('frozen_math_observer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Observer


def episode_observer():
    path = ROOT/'doc/benchmarks/2026-10-08-math-chain-repair-2/episode_state.py'
    spec = importlib.util.spec_from_file_location('episode_footprint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ordinary_model(folder, budget=32):
    text = (ROOT/'data/MM_math_chain.xml').read_text()
    text = re.sub(r'<ltmCapacity>\d+</ltmCapacity>', '<ltmCapacity>8192</ltmCapacity>', text)
    text = re.sub(r'<attentionBudget>\d+</attentionBudget>', f'<attentionBudget>{budget}</attentionBudget>', text)
    config = folder/'ordinary.xml'
    config.write_text(text)
    return build_model(config)


def stage(model, documents):
    texts, targets, addresses = flatten(documents, supplied=True)
    data = model.inputSpace.data
    data.train_input, data.train_output = texts, [torch.zeros(1) for _ in texts]
    data.text_answers['train'] = targets
    data.source_addresses['train'] = [dict(address, split='train') for address in addresses]
    data.math_chain_documents = {'train': documents}


def train_documents(model, documents, folder, *, epochs=1, after=None, episode=None):
    from MathChainTraining import present
    from ThoughtReferences import open_slots, bindings
    stage(model, documents)
    observations = []
    optimizer = model.getOptimizer(lr=.001)
    with observer_class()(folder) as observer:
        if episode is not None:
            from Models import BasicModel
            from unittest.mock import patch
            original = BasicModel.run_selected_thought
            def observe_episode(model, *args, **kwargs):
                return episode(original, model, *args, **kwargs)
            observer.stack.enter_context(patch.object(BasicModel, 'run_selected_thought', observe_episode))
        def record(model, split, rows, result):
            observer.after_batch(model, split, rows, result)
            fields = model._sentence_fields[0]
            episodes = dict(model._last_closing_thoughts)
            store = model.symbolSpace.ltm_store
            draw = model._last_sentence_credit['departure']
            departures = draw['compose_round'].tolist()
            for row, source in enumerate(rows):
                field = fields[row]
                index = store.index_of_row(field.row_id)
                assert index is not None, 'ordinary closing was not committed'
                value = store.meaning_of(index)
                observations.append(dict(epoch=observer.context['epoch'], source=source,
                    row_id=field.row_id, address=store.occurrence_of(index),
                    open=open_slots(value), forward=bindings(value).get('_forward_references', ()),
                    episode=row in episodes, departure=departures[row],
                    query_open=() if field.query is None else open_slots(field.query)))
            if after is not None:
                after(model, split, rows, result, observations, observer)
        for epoch in range(1, epochs+1):
            observer.context = dict(epoch=epoch, phase='ordinary_certificate')
            present(model, split='train', optimizer=optimizer, batch_size=8, after_batch=record)
    (folder/'ordinary.json').write_text(json.dumps(observations, indent=2)+'\n')
    return observations
