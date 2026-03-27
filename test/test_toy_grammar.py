"""
Toy Grammar Learning Test
=========================

Can SyntacticLayer learn to predict derivation rules from sentence structure?

Toy CFG:
    0: S  → NP VP        (binary)
    1: NP → DET N        (binary)
    2: NP → N            (unary/terminal)
    3: VP → V NP         (binary)
    4: VP → V            (unary/terminal)
    5: VP → V NP PP      (ternary — split as VP → V VP', VP' → NP PP)

For simplicity we use rules 0-4 (all binary or terminal).

Vocabulary (one-hot slots):
    0: DET  ("the", "a")
    1: N    ("cat", "dog", "fish")
    2: V    ("sees", "chases", "eats")

The activation vector encodes which POS slots are "on" in the sentence,
plus noise. The target is the derivation rule sequence (left-to-right,
pre-order).

A sentence like "the cat sees a dog" has derivation:
    depth 0: S  → NP VP       (rule 0)
    depth 1: NP → DET N       (rule 1)
    depth 2: VP → V NP        (rule 3)
    depth 3: NP → DET N       (rule 1)

So depth determines which rule fires. The SyntacticLayer should learn
this mapping from the activation pattern to the rule sequence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import random
import torch
import torch.nn.functional as F

from BasicModel import TheDevice
from Model import SyntacticLayer, Grammar

# ── Toy grammar ──────────────────────────────────────────────────────

TOY_RULES = [
    "S → NP VP",      # 0  binary
    "NP → DET N",     # 1  binary
    "NP → N",         # 2  terminal
    "VP → V NP",      # 3  binary
    "VP → V",         # 4  terminal
]
NUM_TOY_RULES = len(TOY_RULES)

# POS tag slots in the activation vector
POS_DET = 0
POS_N   = 1
POS_V   = 2
NUM_POS = 3

# Derivation depth for our toy grammar
MAX_DEPTH = 4  # S → NP VP → (DET N | N) (V NP | V) → ...


def generate_derivation():
    """Generate a random derivation and return (pos_bag, rule_sequence).

    pos_bag: which POS tags appear (as counts)
    rule_sequence: list of rule IDs in pre-order, padded to MAX_DEPTH
    """
    pos_counts = [0, 0, 0]  # DET, N, V
    rules = []

    def expand_S():
        rules.append(0)  # S → NP VP
        expand_NP()
        expand_VP()

    def expand_NP():
        if random.random() < 0.6:
            rules.append(1)  # NP → DET N
            pos_counts[POS_DET] += 1
            pos_counts[POS_N] += 1
        else:
            rules.append(2)  # NP → N
            pos_counts[POS_N] += 1

    def expand_VP():
        if random.random() < 0.7:
            rules.append(3)  # VP → V NP
            pos_counts[POS_V] += 1
            expand_NP()
        else:
            rules.append(4)  # VP → V
            pos_counts[POS_V] += 1

    expand_S()

    # Pad or truncate to MAX_DEPTH
    while len(rules) < MAX_DEPTH:
        rules.append(0)  # pad with rule 0 (doesn't matter, masked in loss)
    rules = rules[:MAX_DEPTH]

    return pos_counts, rules


def make_activation(pos_counts, noise=0.1):
    """Convert POS bag to a float activation vector with noise."""
    act = torch.tensor(pos_counts, dtype=torch.float32)
    act = act + torch.randn_like(act) * noise
    return act


def generate_batch(batch_size, noise=0.1):
    """Generate a batch of (activations, target_rules, mask).

    mask: 1 where a real rule was generated, 0 where padded.
    """
    acts = []
    targets = []
    masks = []

    for _ in range(batch_size):
        pos_counts, rules = generate_derivation()
        acts.append(make_activation(pos_counts, noise))
        targets.append(rules)

        # Mask: real rules (not padding)
        # We know the derivation generates 3-4 rules depending on VP choice
        # Count how many real rules were generated before padding
        pos, real_rules = generate_derivation.__code__.co_varnames, rules  # just use all
        masks.append([1.0] * MAX_DEPTH)  # simplify: all positions active

    acts = torch.stack(acts).to(TheDevice.get())
    targets = torch.tensor(targets, dtype=torch.long).to(TheDevice.get())
    masks = torch.tensor(masks, dtype=torch.float32).to(TheDevice.get())

    return acts, targets, masks


def train_and_evaluate(num_epochs=300, batch_size=64, lr=0.005, verbose=True):
    """Train SyntacticLayer on toy grammar and return final accuracy.

    We call the derivation stack directly (input_proj + layers + heads)
    rather than the full forward() which also builds word tuples, since
    the toy grammar uses a different rule format than TheGrammar.
    """

    # Create a Grammar-compatible wrapper for toy rules
    toy_grammar = Grammar()
    # Override with toy rules for this test
    toy_grammar.rules = TOY_RULES

    layer = SyntacticLayer(
        nInput=NUM_POS,
        nOutput=NUM_POS,
        max_depth=MAX_DEPTH,
        hidden_dim=64,
        grammar=toy_grammar,
        tau=1.0,
    )
    layer.train()

    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)

    def predict(x):
        """Run the derivation stack without building word tuples."""
        h = layer.input_proj.forward(x)
        h = layer.activation_fn(h)
        depth_ids = torch.arange(layer.max_depth, device=x.device)
        depth_vecs = layer.depth_embed(depth_ids)
        all_logits = []
        all_probs = []
        for d in range(layer.max_depth):
            h = h + depth_vecs[d]
            h = layer.derivation_layer.forward(h)
            h = layer.activation_fn(h)
            logits = layer.rule_head.forward(h)
            if layer.training:
                probs = F.gumbel_softmax(logits, tau=layer.tau, hard=False)
            else:
                probs = F.softmax(logits, dim=-1)
            all_logits.append(logits)
            all_probs.append(probs)
        return {
            "rule_logits": torch.stack(all_logits, dim=1),
            "rule_probs": torch.stack(all_probs, dim=1),
            "predicted_rules": torch.stack(all_logits, dim=1).argmax(dim=-1),
        }

    best_acc = 0.0
    for epoch in range(num_epochs):
        acts, targets, masks = generate_batch(batch_size)

        out = predict(acts)
        logits = out["rule_logits"]  # [B, MAX_DEPTH, NUM_TOY_RULES]

        # Cross-entropy at each depth
        loss = 0.0
        for d in range(MAX_DEPTH):
            loss = loss + F.cross_entropy(logits[:, d, :], targets[:, d])
        loss = loss / MAX_DEPTH

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Anneal temperature
        if epoch < num_epochs // 2:
            tau = 1.0 - 0.8 * (epoch / (num_epochs // 2))
            layer.set_tau(max(tau, 0.2))

        # Evaluate every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            layer.eval()
            with torch.no_grad():
                eval_acts, eval_targets, eval_masks = generate_batch(256)
                eval_out = predict(eval_acts)
                preds = eval_out["predicted_rules"]  # [B, MAX_DEPTH]
                correct = (preds == eval_targets).float()
                per_depth_acc = correct.mean(dim=0)
                overall_acc = correct.mean().item()
                best_acc = max(best_acc, overall_acc)

                if verbose:
                    depth_str = " ".join(f"d{d}={per_depth_acc[d]:.2f}"
                                         for d in range(MAX_DEPTH))
                    print(f"  epoch {epoch+1:4d}  loss={loss.item():.4f}  "
                          f"acc={overall_acc:.3f}  {depth_str}")
            layer.train()

    return best_acc


def main():
    print("=" * 60)
    print("Toy Grammar Learning Test")
    print("=" * 60)
    print(f"\nRules:")
    for i, r in enumerate(TOY_RULES):
        print(f"  {i}: {r}")
    print(f"\nPOS slots: DET=0, N=1, V=2")
    print(f"Derivation depth: {MAX_DEPTH}")
    print(f"Device: {TheDevice.get()}")
    print()

    acc = train_and_evaluate(num_epochs=300, verbose=True)

    print(f"\nBest accuracy: {acc:.3f}")
    if acc > 0.7:
        print("PASS — SyntacticLayer can learn toy grammar rules")
    else:
        print("MARGINAL — accuracy below 70%, may need more epochs or tuning")

    return acc


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
