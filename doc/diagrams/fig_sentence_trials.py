#!/usr/bin/env python3
"""One sentence, two readings: the trial comparison that keeps a reading and credits the chooser.

Source: doc/GradientFlow.md ("Current gradient boundaries", "The training step").
Embedded in doc/GradientFlow.md. Run this file, or gen_diagrams.py to write every figure.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_diagrams import *  # noqa: F401,F403  (palette and drawing primitives)

FILENAME = "sentence_trials.svg"


def make():
    W, H = 1260, 700
    b = []
    b.append(card(30, 110, 230, 132, "Sentence, perceived once", [
        "every word perceived and", "symbolized once, after the",
        "previous sentence's commit;", "the primed bank staged"], BLUE_L, BLUE))
    b.append(card(300, 88, 390, 78, "Greedy reading", [
        "the chooser's best option at every round"], "#fff", BLUE_DK))
    b.append(card(300, 184, 390, 104, "Explore reading: one departure", [
        "draw a walk, an eligible round, an alternative;",
        "greedy everywhere else; the credit is corrected",
        "by K · R_walk · W"], "#fff", ORG))
    b.append(txt(495, 308, "both composed and costed at identical parameters before either trains",
                 fs=12, fill=GRY, italic=True))
    b.append(ortho([(260, 176), (280, 176), (280, 127), (300, 127)], BLUE_DK, "arrd"))
    b.append(ortho([(260, 176), (280, 176), (280, 236), (300, 236)], ORG, "arro"))

    b.append(card(730, 92, 160, 70, "R + A", ["detached cost"], GRY_L, BLUE_DK, center=True))
    b.append(card(730, 201, 160, 70, "R + A", ["detached cost"], GRY_L, ORG, center=True))
    b.append(ortho([(690, 127), (730, 127)], BLUE_DK, "arrd"))
    b.append(ortho([(690, 236), (730, 236)], ORG, "arro"))
    b.append(txt(716, 290, "R: reconstruction · A: answer error, when supplied", anchor="start", fs=11.5, fill=GRY, italic=True))

    b.append(diamond(990, 182, 140, 100, "keep?", "#fff", BLUE_DK, fs=15))
    b.append(ortho([(890, 127), (905, 127), (905, 182), (918, 182)], BLUE_DK, "arrd"))
    b.append(ortho([(890, 236), (905, 236), (905, 182), (918, 182)], ORG, "arro"))
    for k, row in enumerate(["explore is kept only if", "its R + A is strictly lower;", "a tie keeps greedy"]):
        b.append(txt(1072, 166 + k * 20, row, anchor="start", fs=12.5, fill=BLUE_DK))

    b.append(card(30, 342, 390, 112, "Expectation (E)", [
        "trains on both trials, each on its own local",
        "residual, with inputs and targets detached;",
        "it enters neither keeping nor credit"], TEAL_L, TEAL))
    b.append(card(450, 342, 390, 112, "Chooser credit", [
        "the score function on the detached advantage,",
        "greedy cost minus explore cost, at the",
        "departure the explore reading took"], ORG_L, ORG))
    b.append(card(870, 342, 360, 112, "Commit the kept reading", [
        "its end state is written at the kept close",
        "(MTM tokens, LTM0 rows), detached: no",
        "gradient crosses into the next sentence"], GRN_L, GRN_DK))
    b.append(ortho([(990, 232), (990, 324), (1050, 324), (1050, 342)], GRN_DK, "arrd"))
    b.append(ortho([(990, 324), (645, 324), (645, 342)], ORG, "arro"))

    b.append(txt(30, 494, "Parameter owners  (separateChooser = true, the default)", anchor="start",
                 fs=15, fill=BLUE_DK, bold=True))
    b.append(card(30, 506, 390, 136, "Reconstruction (R)", [
        "perception, the sole writer of native PS/WS",
        "prototypes and their evidence; the field",
        "dictionaries; compose operators and their",
        "tied inverses; the compose chooser; the",
        "shared generate decoder"], BLUE_L, BLUE))
    b.append(card(450, 506, 390, 136, "Expectation (E)", [
        "the shared symbol activation predictor",
        "(ActivationPredictor) on local residuals;",
        "every source and target detached; percept",
        "moments update from observations without",
        "gradients"], TEAL_L, TEAL))
    b.append(card(870, 506, 360, 136, "Answer (A)", [
        "answer readers, presented and comparison;",
        "learned retrieval for thought and generation;",
        "question conditioners; intuit's weights;",
        "the concluded understanding is detached, and",
        "supplied-answer gradients never reach R"], ORG_L, ORG))
    b.append(txt(W // 2, 666, "With separateChooser = false the shared understanding parameters receive "
                 "s·R + A + C, one s ~ U[1 − j, 1] per sentence.", fs=12, fill=GRY, italic=True))
    b.append(txt(W // 2, 688, "The backward arrangement is under the Stage 2 review "
                 "(GradientFlow.md, Read first).", fs=12, fill=GRY, italic=True))
    return figure(W, H, "One sentence, two readings: the trial comparison",
                  "GradientFlow.md: the training step and the current gradient boundaries", b, MARKERS)


if __name__ == "__main__":
    write(FILENAME, make())
