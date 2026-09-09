"""The math dataset (mathematical thinking spec section 4; plan Phase 1):
generator invariants, structure splits, position independence, and
``Data.what()`` returning the one-hot answer."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from exact import MathProblemGenerator, split_by_structure  # noqa: E402
from What import What  # noqa: E402


@pytest.mark.parametrize("stage", [1, 2])
def test_generator_invariants(stage):
    R = 24
    gen = MathProblemGenerator(seed=1, range=R, depths=(1, 2, 3),
                               distractors=(0, 1, 2), stage=stage)
    for p in gen.problems(30):
        names = set().union(*(e.vars for e in p.equations))
        # every constant / value in range; the answer is the solution's query
        assert 0 <= p.answer < R
        assert all(0 <= v < R for v in p.solution.values())
        assert p.solution[p.query] == p.answer
        assert p.query in p.chain and p.chain <= names
        # distractors never touch the chain
        for e in p.equations:
            if e.solved_var is not None and e.solved_var not in p.chain:
                assert not (e.vars & p.chain), (e.render(), p.chain)
        # unique solution (brute force on small problems)
        if len(names) <= 3:
            sols = MathProblemGenerator.brute_force(p)
            assert len(sols) == 1 and sols[0][p.query] == p.answer
        if stage == 1:
            assert p.depth == len(p.order) - 1
        surface = p.surface()
        assert surface.endswith(f"what is {p.query} ?") and " ; " in surface


def test_split_by_structure_is_disjoint_and_holds_out_deep_problems():
    gen = MathProblemGenerator(seed=2, range=64, depths=(1, 2, 3), stage=1)
    deep = MathProblemGenerator(seed=3, range=64, depths=(4, 5, 6), stage=1)
    problems = gen.problems(200) + deep.problems(40)
    splits = split_by_structure(problems, train_depths=(1, 2, 3))
    assert all(splits[k] for k in ("train", "validation", "test"))
    structures = {k: {p.structure for p in v} for k, v in splits.items()}
    assert not (structures["train"] & structures["test"])
    assert not (structures["train"] & structures["validation"])
    assert all(p.depth <= 3 for p in splits["train"] + splits["validation"])
    assert any(p.depth >= 4 for p in splits["test"])
    assert all(p.depth <= 3 or p in splits["test"] for p in problems)


def test_positions_are_independent_of_answers_and_premise_order_varies():
    from data import Data
    a = Data(); a.load("math", dat={"mathProblems": 64, "mathRange": 32, "mathSeed": 4})
    b = Data(); b.load("math", dat={"mathProblems": 64, "mathRange": 32, "mathSeed": 5})
    answers_a = [int(torch.as_tensor(t).argmax()) for t in a.train_output]
    assert answers_a != sorted(answers_a)                  # not ordered by answer
    # Two seeds give different presentations; the same seed is reproducible.
    c = Data(); c.load("math", dat={"mathProblems": 64, "mathRange": 32, "mathSeed": 4})
    assert a.train_input == c.train_input and a.train_input != b.train_input
    # Premise order was shuffled: the query's defining premise is not always last.
    positions = []
    for text in a.train_input:
        clauses = [s.strip() for s in text.split(";")][:-1]
        q = text.split("what is")[1].split("?")[0].strip()
        positions.append([i for i, s in enumerate(clauses) if s.startswith(q + " =")][0]
                         == len(clauses) - 1)
    assert not all(positions)


def test_data_what_supervised_returns_the_onehot_answer():
    from data import Data
    data = Data()
    data.load("math", dat={"mathProblems": 32, "mathRange": 16, "mathDepths": "1-2",
                           "mathTestDepths": "3", "mathDistractors": 1})
    assert data.has_supervised_outputs
    assert data.math_range == 16
    n = data.what_extent("train")
    assert n == len(data.math_problems["train"]) > 0
    for row in range(min(n, 8)):
        problem = data.math_problems["train"][row]
        assert data.train_input[row] == problem.surface()
        desired = data.what(What.supervised(row))
        assert desired.available and desired.provenance == "data"
        target = torch.as_tensor(desired.what)
        assert target.shape == (16,) and int(target.argmax()) == problem.answer
        assert float(target.sum()) == 1.0
        # present questions still return the surface; addresses are per row
        assert data.what(What.present(row)).what == problem.surface()
        assert data.source_address("train", row)["row"] == row
    # depth-3 problems were held out of train
    assert all(p.depth <= 2 for p in data.math_problems["train"])
    assert any(p.depth == 3 for p in data.math_problems["test"])


def test_xor_dataset_is_unchanged():
    from data import Data
    data = Data(); data.load("xor")
    assert data.train_input[:2] == ["hello world", "hello there"]
    assert not hasattr(data, "math_problems")
