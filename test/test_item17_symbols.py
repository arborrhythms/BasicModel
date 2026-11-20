"""Item 1.7, round 4: symbols and the orders above.

Certificate 51 of the sigma-pi specification, §21: the Sigma takes the for
channel and the against channel as two inputs, each with weights of its own
(rule 59). The certificates of the round's other steps join this file.
"""
import json
from pathlib import Path

import pytest
import torch

from OrderZero import OrderZeroSigma, combine, read_word


def _read_lesson_at_start(sentences):
    """The lesson's model after ``sentences`` sentences read with a learning
    rate of zero: every column of the order-zero Sigma is as its writers
    made it, at its start. Returns the model, the patch to undo and every
    percept field the order-zero Sigma read."""
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    from Spaces import ConceptualSpace
    patch = MonkeyPatch()
    use_eager_reading(patch)
    fields = []
    original = ConceptualSpace.cs_read_percept_field

    def kept(self, perception, **kwargs):
        fields.append(perception)
        return original(self, perception, **kwargs)
    patch.setattr(ConceptualSpace, 'cs_read_percept_field', kept)
    import Language
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    model._tensor_peer_while_eager = True
    optimizer = model.getOptimizer(lr=0.)
    curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
    texts, targets = stage_documents(TheData, curriculum['stages'][0])
    admit_training_forms(model, texts, targets)
    count = 0

    class Complete(Exception):
        pass

    def batch(*args, **kwargs):
        nonlocal count
        count += 1
        if count == sentences:
            raise Complete
    try:
        present(model, split='train', optimizer=optimizer, batch_size=1, after_batch=batch)
    except Complete:
        pass
    return model, patch, fields


@pytest.fixture(scope='module')
def lesson_at_start():
    model, patch, fields = _read_lesson_at_start(16)
    yield model._concept_owner(), fields
    model.End()
    patch.undo()


def _one_input(sigma, combined, columns):
    """Rule 3's read: one input of for less against, one signed weight."""
    weight = sigma.weight.detach()[:, torch.as_tensor(columns)]
    return torch.tanh((combined[..., 0, :] - combined[..., 1, :]) @ weight)


# Certificate 51: two channels at the Sigma (rule 59).

def test_51a_at_its_start_every_concept_reads_what_one_input_of_for_less_against_read(lesson_at_start):
    """(a) The lesson's sixteen sentences read with no learning: every column
    the writers made has its against weights at its for weights negated, and
    every concept reads on the two channels what it read on one input of for
    less against, on the field and at the word step."""
    owner, fields = lesson_at_start
    sigma, basic = owner.order_zero, owner.basic_percepts
    members = owner._concept_allocator.order_zero_members
    columns = sorted({row for row, _, _ in members.started()})
    assert columns
    assert torch.equal(sigma.against.detach(), -sigma.weight.detach())
    # The field: every reading's combined vector in the sixteen sentences,
    # read by every concept started from a nominated row.
    against_lane = readings = 0
    largest = 0.
    for perception in fields:
        if not len(perception.keys):
            continue
        combined = owner.order_zero_input(perception).detach()
        two = sigma.concepts(combined, columns).detach()
        one = _one_input(sigma, combined, columns)
        torch.testing.assert_close(two, one, atol=1e-6, rtol=0)
        readings += int(combined[..., 0, :].shape[:-1].numel())
        against_lane += int((combined[..., 1, :sigma.content_width] > 0).any(-1).sum())
        largest = max(largest, float((two - one).abs().max()))
    assert readings and against_lane
    # The word step: each nominated row's word, its run of parts and its
    # wholes present, read by its two columns; on one input it read its
    # basic percept's read pair, for less against, by its for weights.
    rows = [row for row in range(len(basic.index)) if bool(basic.promoted[row])]
    concepts = torch.tensor([owner.concept_of_basic_percept(row) for row in rows])
    reading = owner.order_zero_reading(concepts, torch.tensor(rows)).detach()
    evidence = torch.ones(len(rows), 1, 2)
    mask = torch.ones(len(rows), 1, 2, dtype=torch.bool)
    place = torch.zeros(len(rows), 1, sigma.nInput - sigma.content_width)
    words = read_word(evidence, mask, reading[:, None], place, content_width=sigma.content_width)
    pairs = basic.read_codes(torch.tensor(rows)).reshape(len(rows), 2, -1).to(sigma.weight)
    weight = sigma.weight.detach()[:sigma.content_width, concepts].t()
    expected = torch.tanh(((pairs[:, 0] - pairs[:, 1]) * weight).sum(-1))[:, None]
    torch.testing.assert_close(words, expected, atol=1e-6, rtol=0)
    against_read = int(pairs[:, 1].any(-1).sum())
    print(dict(concepts=len(columns), fields=len(fields), readings=readings,
               readings_with_an_against_lane=against_lane, words=len(rows),
               words_whose_pair_has_an_against_side=against_read, largest_difference=largest))


def test_51b_the_gradient_reaches_the_for_and_the_against_weights_apart():
    """(b) Each channel's weights take the gradient of their own input."""
    width, content = 10, 8
    sigma = OrderZeroSigma(width, 3, content_width=content)
    codes = torch.rand(3, 2, content)
    for column in range(3):
        sigma.start_column(column, codes[column, 0], against=codes[column, 1])
    presence = torch.tensor([[.9, .1], [.2, .7], [.5, .5]])[:, None]
    x = combine(presence, codes, where=torch.rand(1, (width - content) // 2),
                when=torch.rand(1, (width - content) // 2))
    assert bool(x[..., 0, :].any()) and bool(x[..., 1, :].any())
    y = sigma.concepts(x, [0, 1, 2])
    target = torch.tensor([[1., -1., 0.]])
    ((y - target) ** 2).sum().backward()
    g = (2 * (y - target) * (1 - y ** 2)).detach()
    torch.testing.assert_close(sigma.weight.grad, x[0, 0].detach()[:, None] * g, atol=1e-6, rtol=0)
    torch.testing.assert_close(sigma.against.grad, x[0, 1].detach()[:, None] * g, atol=1e-6, rtol=0)
    assert not torch.allclose(sigma.weight.grad, -sigma.against.grad)


def test_51c_a_concept_of_absence_is_learned_on_the_against_channel_alone():
    """(c) Learning, printed: a concept started from a percept, taught to be
    present where the percept is witnessed against and to say nothing where
    it is witnessed for. The Sigma starts from the percept's own column, so
    nothing depends on a draw."""
    content = 24
    code = torch.zeros(1, content)
    code[0, [2, 9, 17]] = 1.
    pairs = torch.stack((code, torch.zeros_like(code)), -2)
    sigma = OrderZeroSigma(content, 1, content_width=content)
    sigma.start_column(0, code[0])
    witnessed_for = combine(torch.tensor([[[1., 0.]]]), pairs)[0]
    witnessed_against = combine(torch.tensor([[[0., 1.]]]), pairs)[0]
    inputs = torch.stack((witnessed_for, witnessed_against))
    target = torch.tensor([[0.], [1.]])
    unit = code[0] / code[0].norm()
    before = dict(reads=sigma.concepts(inputs, [0]).detach().flatten().tolist(),
                  for_weight=float(sigma.weight[:, 0] @ unit), against_weight=float(sigma.against[:, 0] @ unit))
    optimizer = torch.optim.Adam(sigma.parameters(), lr=.05)
    for _ in range(500):
        optimizer.zero_grad()
        ((sigma.concepts(inputs, [0]) - target) ** 2).mean().backward()
        optimizer.step()
    reads = sigma.concepts(inputs, [0]).detach().flatten()
    after = dict(reads=reads.tolist(), for_weight=float(sigma.weight[:, 0] @ unit),
                 against_weight=float(sigma.against[:, 0] @ unit))
    print(dict(before=before, after=after))
    assert abs(float(reads[0])) < .05 and float(reads[1]) > .9
    assert abs(after['for_weight']) < .05 * after['against_weight']


def test_a_sigma_saved_on_one_input_loads_its_against_weights_as_its_for_weights_negated():
    sigma = OrderZeroSigma(6, 2, content_width=4)
    with torch.no_grad():
        sigma.weight.uniform_(-1, 1)
    saved = {'weight': sigma.weight.detach().clone()}
    restored = OrderZeroSigma(6, 2, content_width=4)
    restored.load_state_dict(saved, strict=True)
    assert torch.equal(restored.against.detach(), -saved['weight'])
