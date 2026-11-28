#!/usr/bin/env python3
"""Draw the supervised sentence path in SentenceLooks, RequestRead and WholePaths."""
from gen_diagrams import (
    BLUE, BLUE_DK, BLUE_L, GRN_DK, GRN_L, GRY, MARKERS, ORG, ORG_L,
    PURP, PURP_L, TEAL, TEAL_L, card, figure, ortho, txt, write,
)

FILENAME = "sentence_loop.svg"


def make():
    width, height = 1180, 682
    body = [
        card(28, 144, 224, 94, "Open attention · look zero", [
            "Locate percepts across the input.", "Read the opening conceptual field."], BLUE_L, BLUE),
        card(323, 144, 206, 94, "Request the next look", [
            "Read the opening, kept trace,", "place and whole STM stack."], BLUE_L, BLUE),
        card(601, 144, 256, 94, "Update the field at the fixation", [
            "Decay the trace. Apply expectation", "on ascent and attention on descent."], TEAL_L, TEAL),
        card(927, 144, 224, 94, "Compose in STM", [
            "Shift the word that arrived.", "Reduce the stack."], PURP_L, PURP),
        ortho([(252, 188), (323, 188)]),
        ortho([(529, 188), (601, 188)]),
        ortho([(857, 188), (927, 188)]),
        ortho([(1040, 144), (1040, 112), (426, 112), (426, 144)], PURP, "arrp"),
        txt(730, 103, "Next look reads the kept trace and the updated stack", fs=13, fill=PURP),
        card(28, 321, 224, 94, "Reverse the opening field", [
            "Decode once at observed places", "and reuse across candidate paths."], TEAL_L, TEAL),
        ortho([(140, 238), (140, 321)]),
        txt(150, 280, "shared opening", fs=12, anchor="start", fill=GRY),
        card(323, 321, 206, 94, "Publish understanding", [
            "Read the composed meaning", "and its conceptual field."], PURP_L, PURP),
        ortho([(426, 238), (426, 321)]),
        txt(436, 280, "reading ends", fs=12, anchor="start", fill=GRY),
        card(601, 321, 256, 94, "Closing thought", [
            "Read this path’s understanding", "and resolve its answer."], PURP_L, PURP),
        ortho([(529, 367), (601, 367)]),
        card(927, 321, 224, 94, "Generate the text answer", [
            "Write terminal occurrences and", "realize them through reverse."], ORG_L, ORG),
        ortho([(857, 367), (927, 367)]),
        card(28, 500, 224, 92, "Reconstruction cost · R", [
            "Score the opening inverse and", "the path’s reconstruction terms."], GRN_L, GRN_DK),
        ortho([(140, 415), (140, 500)]),
        ortho([(360, 415), (360, 475), (222, 475), (222, 500)]),
        card(927, 500, 224, 92, "Answer cost · A", [
            "Score the supervised output,", "including a target of silence."], GRN_L, GRN_DK),
        ortho([(1040, 415), (1040, 500)]),
        ortho([(426, 415), (426, 457), (970, 457), (970, 500)], ORG, "arro"),
        txt(697, 448, "Numeric output reads the composed state through an affine head", fs=13, fill=ORG),
        card(413, 500, 354, 92, "Keep one complete path per stream", [
            "Minimize R + A. The greedy path wins a tie.", "Commit the winning reading and effects."],
             GRN_L, GRN_DK, center=True),
        ortho([(252, 546), (413, 546)], GRN_DK, "arrg"),
        ortho([(927, 546), (767, 546)], GRN_DK, "arrg"),
        txt(width / 2, 633, "With the unit delay enabled, each look can read percepts and concepts "
            "observed one step earlier. The opening has no attention mask.", fs=13, fill=BLUE_DK),
        txt(width / 2, 656, "Without output supervision, R selects the path first. Only the kept path runs closing thought.",
            fs=13, fill=GRY, italic=True),
    ]
    drawing = figure(width, height, "The sentence loop of one path",
                     "Grammar reading with supervised output · attention, composition, thought and the per-stream choice",
                     body, MARKERS)
    return drawing.replace("<svg ", f'<svg viewBox="0 0 {width} {height}" ', 1)


if __name__ == "__main__":
    write(FILENAME, make())
