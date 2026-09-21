"""Measured generativity; failure rates are evidence, not hidden assertions."""
import json

import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from MemoryIndex import unfold_idea
from test_prepared_answer_boundary import _native_answer_model


def test_current_grammar_recovery_by_depth_training_and_chain_length(tmp_path):
    """Keep old ideas fixed while the current shared operator/MLP are trained.

    No original tree is supplied to the unfold. The observations include null
    results; this is not a claim of learned generativity or held-out utility.
    """
    torch.manual_seed(946)
    model = _native_answer_model(tmp_path, True)
    language = model.languageSpace
    try:
        op_index = list(language._generate_binary_names).index('lower')
        op = language._generate_binary_ops[op_index]
        from Queries import _basis
        basis = _basis(model.conceptualSpace).detach().clone()
        leaves = basis[:5]
        assert len(leaves) == 5 and bool((torch.pdist(leaves) > 0).all())
        records = []
        for length in (1, 2, 3, 5):
            value = leaves[length - 1:length]
            operations = (('code', length - 1),)
            for code in reversed(range(length - 1)):
                value = op.compose(leaves[code:code + 1], value)
                operations = (('binary', op_index), ('code', code), *operations)
            store = TernaryTruthStore(leaves.shape[-1], capacity=1)
            row = store.append_meaning(ConceptualMeaning.from_description(value[0]),
                                      leaf_codes=(tuple(range(length)), (), ()))
            records.append((length, store, row, operations))
        parameters = tuple(language.generate_policy.parameters()) + tuple(op.parameters())
        optimizer = torch.optim.Adam(parameters, lr=.001)
        reports = []
        for update in range(9):
            if update in (0, 1, 8):
                for length, store, row, expected in records:
                    recovered = unfold_idea(language, basis, store.slots[row, 0], 32)
                    exact_codes = recovered['codes'] == store.leaf_terms(row, 0)
                    reports.append(dict(depth=length - 1, chain_length=length, updates=update,
                        code_recovery=float(exact_codes and recovered['complete']),
                        derivation_recovery=float(recovered['complete'] and recovered['operations'] == expected),
                        completed=recovered['complete'], work=recovered['spent']))
            if update == 8:
                break
            optimizer.zero_grad()
            # The ordinary generate MLP is taught code-stop versus a real
            # lower split. Shared numerical credit is tied inverse error.
            left, right = leaves[:-1], leaves[1:]
            parent = op.compose(left, right)
            children = op.generate(parent)
            inputs = torch.cat((leaves, parent.detach()))
            stop = language.generate_policy_logits(inputs).shape[-1] - 1
            targets = torch.tensor([stop] * len(leaves) + [op_index] * len(parent))
            cost = torch.nn.functional.cross_entropy(language.generate_policy_logits(inputs), targets)
            cost = cost + sum((child - target).square().mean()
                              for child, target in zip(children, (left, right)))
            cost.backward()
            optimizer.step()
        print('[mind-generativity] ' + json.dumps(reports, sort_keys=True))
        assert len(reports) == 12 and all(row['work'] <= 32 for row in reports)
        assert all(0 <= row['derivation_recovery'] <= row['code_recovery'] <= 1 for row in reports)
        # Composed ideas stay off-codebook. Quantization is idempotent on a code.
        for code, leaf in enumerate(leaves):
            assert int((basis - leaf).square().sum(-1).argmin()) == code
        composed = op.compose(leaves[0:1], leaves[1:2]).detach()
        assert not bool(torch.isclose(basis, composed, atol=1e-6).all(-1).any())
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
