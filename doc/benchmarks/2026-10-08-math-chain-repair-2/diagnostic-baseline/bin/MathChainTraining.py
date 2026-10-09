"""Document-preserving presentation of the ordinary opaque-word corpus.

This driver never reads answers, operand values or worked-step metadata when
forming a batch. Equal-length documents share a stream group; every source
sentence is consumed once, with its existing address and loss-side label.
"""
from collections import OrderedDict

import torch


def document_batches(addresses, batch_size):
    if batch_size < 1:
        raise ValueError('positive batch size required')
    documents = OrderedDict()
    for index, address in enumerate(addresses):
        documents.setdefault(address['document'], []).append(index)
    lengths = OrderedDict()
    for indices in documents.values():
        lengths.setdefault(len(indices), []).append(indices)
    for size, documents in lengths.items():
        for start in range(0, len(documents), batch_size):
            group = documents[start:start + batch_size]
            for sentence in range(size):
                yield [indices[sentence] for indices in group], sentence == size - 1


def present(model, *, split, optimizer=None, batch_size=8, after_batch=None):
    """Use the normal sentence owner, chooser, credit and reset boundaries."""
    data = model.inputSpace.data
    inputs, outputs = getattr(data, split + '_input'), getattr(data, split + '_output')
    training = optimizer is not None
    model.train(training)
    model.symbolSpace.expectation.reset()
    memory = model._what_memory()
    if memory is not None:
        memory.reset()
    total, batches = 0, 0
    with torch.set_grad_enabled(training):
        for rows, end in document_batches(data.source_addresses[split], batch_size):
            texts = [inputs[index] for index in rows]
            x = model.inputSpace.prepInput(texts)
            y = model.outputSpace.prepOutput([outputs[index] for index in rows])
            if model.teacher is not None:
                model.teacher.stage_batch_sources(split, rows)
            ends = [end] * len(rows)
            model._thought_document_ends = tuple(ends)
            result, _ = model.runBatch(train=training, batchNum=total,
                batchSize=len(rows), split=split, optimizer=optimizer,
                batch_override=(x, y), source_rows=rows)
            if after_batch is not None:
                after_batch(model, split, rows, result)
            model._thought_document_ends = ()
            del result
            model.flush_word_buffers()
            model.dispatch_per_row_reset(ends)
            model.dispatch_soft_reset()
            model.post_tick_compact()
            total += len(rows)
            batches += 1
    return dict(sentences=total, batches=batches)
