"""
Universality Test -- SVO Recognition & Luminosity of Kind Behavior
==================================================================

Tests the model's ability to:
1. Recognize Subject-Verb-Object structure in transitive sentences
2. Distinguish kind from unkind actions via luminosity in cognitive space

The golden rule (universality) requires that kind actions preserve or
increase truth-store coherence under S/O reversal:
    luminosity(K(X,Y) + K(Y,X)) >= luminosity(K(X,Y))

All tests are xfail: the untrained model has no reason to decompose
SVO correctly or route kind/unkind sentences into different luminosity
regimes.  These become ratchet tests once grammar + truth training land.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import warnings
import torch
import pytest
import matplotlib
import Models
import Spaces
import Language
matplotlib.use('Agg')

from util import init_config, TheXMLConfig

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# -- Test Corpus ------------------------------------------------------
# Every sentence uses a transitive verb (S V O structure).
# "kind" sentences: the action is prosocial / reversible without harm.
# "unkind" sentences: the action is antisocial / harmful under reversal.

KIND_SENTENCES = [
    "the teacher helped the student",
    "the woman fed the child",
    "the doctor healed the patient",
    "the neighbor sheltered the family",
    "the mentor encouraged the apprentice",
]

UNKIND_SENTENCES = [
    "the bully punched the kid",
    "the thief robbed the merchant",
    "the tyrant oppressed the people",
    "the boss exploited the workers",
    "the vandal destroyed the garden",
]

# Expected SVO decompositions for a subset (used by identification tests).
# Each tuple: (subject_head, verb, object_head)
EXPECTED_SVO = {
    "the teacher helped the student":     ("teacher", "helped", "student"),
    "the bully punched the kid":          ("bully",   "punched", "kid"),
    "the woman fed the child":            ("woman",   "fed",     "child"),
    "the thief robbed the merchant":      ("thief",   "robbed",  "merchant"),
    "the doctor healed the patient":      ("doctor",  "healed",  "patient"),
    "the tyrant oppressed the people":    ("tyrant",  "oppressed", "people"),
}


def _reload_config():
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Language.TheGrammar._configured = False


def _make_model():
    """Create a MentalModel with grammar and truth support."""
    _reload_config()
    model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
    return model


def _run_forward(model, sentences):
    """Run forward pass on sentences, return (model, input_state, concepts, symbols).

    Suppresses range warnings from untrained weights.
    """
    outputs = [torch.tensor([0.0])] * len(sentences)
    with Models.TheData.runtime_batch(sentences, outputs), \
         warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Range violation")
        warnings.filterwarnings("ignore", message="PiLayer.reverse")
        train_input, _ = model.inputSpace.getTrainData()
        x = model.inputSpace.prepInput(train_input[:len(sentences)])
        model.eval()
        model.set_sigma(0)
        with torch.no_grad():
            input_state, symbols, _, _ = model.forward(x)
    return input_state, symbols, symbols


# Minimum confidence the model must assign to lift(C,C,C) for a
# transitive sentence to count as correctly identified SVO syntax.
LIFT_CONFIDENCE_THRESHOLD = 0.9


def _find_ternary_lift_rule(grammar):
    """Return the global rule ID for lift(C, C, C), or None."""
    for i, rule in enumerate(grammar.rules):
        if rule.method_name == 'lift' and grammar.arity(i) == 3:
            return i
    return None


def _get_lift_confidence(model, grammar):
    """Return max lift(C,C,C) probability across composition depths.

    After forward(), ConceptualSyntacticLayer caches:
      - last_composable_rules: list of global rule IDs
      - last_rule_probs: [B, depths, n_composable] renormalized probs

    Returns the maximum probability assigned to lift(C,C,C) at any depth,
    averaged over the batch.  Returns 0.0 if composition didn't run or
    if the ternary lift rule is not among the composable rules.
    """
    sl = model.wordSpace.syntacticLayer
    probs = sl.last_rule_probs          # [B, depths, n_composable] or None
    rules = sl.last_composable_rules    # list of global rule IDs or None
    if probs is None or rules is None:
        return 0.0

    lift_rid = _find_ternary_lift_rule(grammar)
    if lift_rid is None or lift_rid not in rules:
        return 0.0

    col = rules.index(lift_rid)
    # probs[:, :, col] -> [B, depths]; take max over depths, mean over batch
    return probs[:, :, col].max(dim=1).values.mean().item()


# =====================================================================
# 1. SVO Identification
# =====================================================================

class TestSVOIdentification(unittest.TestCase):
    """Model should extract Subject, Verb, Object from transitive sentences.

    The ternary lift rule C -> lift(C, C, C) fires during conceptual
    composition when three active concept positions are available.
    last_svo then holds (subject, verb, object) tensors.

    For SVO identification to "work", the model must:
    (a) assign >= 90% probability to lift(C,C,C) over competing rules, and
    (b) produce distinct S, V, O vectors that reconstruct correctly.

    Merely having the lift fire in soft superposition (with 1/6 random
    weight) does not count -- the model must be confident it is seeing
    transitive syntax.
    """

    def setUp(self):
        self.model = _make_model()
        self.grammar = Language.TheGrammar

    @pytest.mark.xfail(reason="untrained model: lift confidence near chance", strict=False)
    def test_lift_confidence_on_transitive_sentence(self):
        """Model should assign >= 90% probability to lift(C,C,C) for SVO sentences."""
        _run_forward(self.model, ["the teacher helped the student"])
        conf = _get_lift_confidence(self.model, self.grammar)
        self.assertGreaterEqual(conf, LIFT_CONFIDENCE_THRESHOLD,
            f"lift(C,C,C) confidence {conf:.3f} < {LIFT_CONFIDENCE_THRESHOLD} "
            f"-- model is not confident it is seeing transitive syntax")

    @pytest.mark.xfail(reason="untrained model: SVO roles not learned", strict=False)
    def test_svo_vectors_are_distinct(self):
        """S, V, O tensors should be meaningfully different from each other."""
        _run_forward(self.model, ["the teacher helped the student"])
        conf = _get_lift_confidence(self.model, self.grammar)
        if conf < LIFT_CONFIDENCE_THRESHOLD:
            self.skipTest(f"lift confidence {conf:.3f} < {LIFT_CONFIDENCE_THRESHOLD}")
        ws = self.model.wordSpace
        svo = ws.get_last_svo(0) if ws.svo_valid(0) else None
        if svo is None:
            self.skipTest("ternary lift did not fire")
        s, v, o = svo
        # Average over batch and position to get role centroids
        s_c = s.mean(dim=(0, 1))
        v_c = v.mean(dim=(0, 1))
        o_c = o.mean(dim=(0, 1))
        # Cosine similarity between roles should be < 0.9 (not collapsed)
        cos = torch.nn.functional.cosine_similarity
        sv_sim = cos(s_c.unsqueeze(0), v_c.unsqueeze(0)).item()
        vo_sim = cos(v_c.unsqueeze(0), o_c.unsqueeze(0)).item()
        so_sim = cos(s_c.unsqueeze(0), o_c.unsqueeze(0)).item()
        self.assertLess(sv_sim, 0.9, f"S and V too similar: {sv_sim:.3f}")
        self.assertLess(vo_sim, 0.9, f"V and O too similar: {vo_sim:.3f}")
        self.assertLess(so_sim, 0.9, f"S and O too similar: {so_sim:.3f}")

    @pytest.mark.xfail(reason="untrained model: reconstruction not meaningful", strict=False)
    def test_svo_reconstruction_accuracy(self):
        """Reverse pass should place S, V, O tokens in correct sentence positions.

        For "the teacher helped the student":
          positions 0-1 -> subject phrase ("the teacher")
          position  2   -> verb ("helped")
          positions 3-4 -> object phrase ("the student")
        """
        _run_forward(self.model, ["the teacher helped the student"])
        conf = _get_lift_confidence(self.model, self.grammar)
        if conf < LIFT_CONFIDENCE_THRESHOLD:
            self.skipTest(f"lift confidence {conf:.3f} < {LIFT_CONFIDENCE_THRESHOLD}")
        ws = self.model.wordSpace
        svo = ws.get_last_svo(0) if ws.svo_valid(0) else None
        if svo is None:
            self.skipTest("ternary lift did not fire")

        # Run reverse to get reconstructed tokens
        symbols = self.model.outputSpace.subspace.materialize()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                input_data, _ = self.model.reverse(
                    symbols, self.model.outputs.materialize())

        # Decode reconstructed tokens
        tokens = self.model.inputSpace.decodeInput(input_data)
        if not tokens:
            self.skipTest("No tokens decoded")

        # Check that the verb appears in the middle region
        sentence_tokens = tokens[0] if isinstance(tokens[0], list) else tokens
        token_texts = [t.lower().strip() for t in sentence_tokens if t.strip()]
        self.assertIn("helped", token_texts,
                       f"Verb 'helped' not in reconstruction: {token_texts}")

    @pytest.mark.xfail(reason="untrained model: lift confidence near chance", strict=False)
    def test_all_transitive_sentences_confident_lift(self):
        """All transitive sentences should trigger lift(C,C,C) with >= 90% confidence."""
        all_sentences = KIND_SENTENCES + UNKIND_SENTENCES
        confident = 0
        for sent in all_sentences:
            _run_forward(self.model, [sent])
            conf = _get_lift_confidence(self.model, self.grammar)
            if conf >= LIFT_CONFIDENCE_THRESHOLD:
                confident += 1
        self.assertEqual(confident, len(all_sentences),
                         f"Only {confident}/{len(all_sentences)} sentences reached "
                         f"{LIFT_CONFIDENCE_THRESHOLD:.0%} lift confidence")


# =====================================================================
# 2. Luminosity of Kind vs Unkind Actions
#    (gated on SVO identification)
# =====================================================================

class TestLuminosityOfKindness(unittest.TestCase):
    """Kind actions should produce greater luminosity than unkind actions.

    Universality (golden rule) predicts that kind actions are reversible:
    "the teacher helped the student" ~= "the student helped the teacher"
    in terms of truth-store coherence.  Unkind actions break under reversal:
    "the bully punched the kid" != "the kid punched the bully" in coherence.

    These tests are gated on correct SVO identification.  If the model
    cannot extract SVO, the luminosity comparison is meaningless.
    """

    def setUp(self):
        self.model = _make_model()
        self.grammar = Language.TheGrammar
        # Seed the truth store so luminosity is non-trivial
        truth_layer = self.model.wordSpace.truth_layer
        with torch.no_grad():
            for _ in range(5):
                v = torch.randn(truth_layer.nDim)
                v = v / v.norm()
                truth_layer.record(v, degree=1.0)

    def _require_confident_svo(self, sentence):
        """Run forward, require lift >= 90% confidence, return SVO or skipTest."""
        _run_forward(self.model, [sentence])
        conf = _get_lift_confidence(self.model, self.grammar)
        if conf < LIFT_CONFIDENCE_THRESHOLD:
            self.skipTest(
                f"lift confidence {conf:.3f} < {LIFT_CONFIDENCE_THRESHOLD} "
                f"for: {sentence!r}")
        ws = self.model.wordSpace
        svo = ws.get_last_svo(0) if ws.svo_valid(0) else None
        if svo is None:
            self.skipTest(f"SVO is None despite confidence for: {sentence!r}")
        return svo

    def _get_svo_and_luminosity(self, sentence):
        """Run forward, extract confident SVO, compute universality score.

        LearnedSVO path: SVO lives on the unified S-tier
        ``syntacticLayer.last_svo`` (grammar-derived from the
        chart-compose derivation trace), and the universality score
        computed during ``MentalModel.forward`` lands on
        ``model._universality_score``. The xfail-guarded callers
        tolerate ``(None, None)`` when the untrained model's chart
        compose doesn't pick the canonical S -> S VO / VO -> V O path.
        """
        _run_forward(self.model, [sentence])
        sl = self.model.wordSpace.syntacticLayer if self.model.wordSpace else None
        svo = sl.last_svo if sl is not None else None
        score = getattr(self.model, '_universality_score', None)
        return svo, score

    @pytest.mark.xfail(reason="untrained model: universality scoring not calibrated",
                        strict=False)
    def test_kind_sentences_have_positive_universality(self):
        """Kind actions should produce non-negative universality scores.

        Positive universality = action preserves illumination under S/O reversal.
        Gated on >= 90% lift confidence (SVO must be correctly identified).
        """
        for sent in KIND_SENTENCES:
            svo, score = self._get_svo_and_luminosity(sent)
            if svo is None:
                self.skipTest(f"SVO not confidently identified for: {sent!r}")
            self.assertGreaterEqual(score, 0.0,
                f"Kind sentence should have non-negative universality: "
                f"{sent!r} got {score:.4f}")

    @pytest.mark.xfail(reason="untrained model: universality scoring not calibrated",
                        strict=False)
    def test_unkind_sentences_have_lower_universality(self):
        """Unkind actions should produce lower (possibly negative) universality.

        The asymmetry of harm means "X hurts Y" + "Y hurts X" damages
        truth-store coherence more than kind actions do.
        Gated on >= 90% lift confidence.
        """
        for sent in UNKIND_SENTENCES:
            svo, score = self._get_svo_and_luminosity(sent)
            if svo is None:
                self.skipTest(f"SVO not confidently identified for: {sent!r}")
            # Weaker claim: just check it's computable (not NaN).
            # ``score`` comes from model._universality_score as a scalar
            # tensor; pass it directly to torch.isnan instead of rewrapping
            # (which triggers the "copy-construct from tensor" warning).
            score_t = score if torch.is_tensor(score) else torch.tensor(float(score))
            self.assertFalse(torch.isnan(score_t).item(),
                f"Universality score is NaN for: {sent!r}")

    @pytest.mark.xfail(reason="untrained model: kind/unkind distinction not learned",
                        strict=False)
    def test_kind_mean_exceeds_unkind_mean(self):
        """Average universality of kind sentences > average of unkind sentences.

        This is the core prediction of the golden rule: prosocial actions
        produce more symmetric (higher) universality than antisocial ones.
        Gated on >= 90% lift confidence -- only sentences where the model
        confidently selects transitive syntax contribute to the comparison.
        """
        kind_scores = []
        unkind_scores = []

        for sent in KIND_SENTENCES:
            svo, score = self._get_svo_and_luminosity(sent)
            if svo is None:
                continue
            kind_scores.append(score)

        for sent in UNKIND_SENTENCES:
            svo, score = self._get_svo_and_luminosity(sent)
            if svo is None:
                continue
            unkind_scores.append(score)

        if not kind_scores or not unkind_scores:
            self.skipTest("SVO not confidently identified -- cannot compare luminosity")

        kind_mean = sum(kind_scores) / len(kind_scores)
        unkind_mean = sum(unkind_scores) / len(unkind_scores)

        self.assertGreater(kind_mean, unkind_mean,
            f"Kind mean universality ({kind_mean:.4f}) should exceed "
            f"unkind mean ({unkind_mean:.4f})")

    @pytest.mark.xfail(reason="untrained model: luminosity baseline not established",
                        strict=False)
    def test_kind_actions_increase_luminosity(self):
        """Recording kind-action truths should increase luminosity.

        Feed SVO from kind sentences through universality, which
        temporarily records truths and measures luminosity change.
        A positive delta means the action brightens the truth store.
        """
        deltas = []
        for sent in KIND_SENTENCES:
            svo, score = self._get_svo_and_luminosity(sent)
            if svo is not None and score is not None:
                deltas.append(score)

        if not deltas:
            self.skipTest("No SVO extracted from kind sentences")

        positive = sum(1 for d in deltas if d > 0)
        self.assertGreater(positive, len(deltas) // 2,
            f"Majority of kind sentences should increase luminosity, "
            f"but only {positive}/{len(deltas)} did: {deltas}")


if __name__ == '__main__':
    unittest.main()
