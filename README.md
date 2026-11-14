# BasicModel

## Basic Model of Cognition

*The basic model of cognition relies on conceptual hyperplanes and perceptual prototypes to synthesize and analyze the input space. It uses a high-dimensional embedding to characterize mental space and integrates symbolic computation. Its three major operations are intersection (analysis: the top-down division of the presented unity into symbolic generalities), union (synthesis: the bottom-up aggregation of atoms into perceptual particulars), and equality (where symbols are elements that map across perceptual and conceptual domains). See [Philosophy](doc/Philosophy.md) for the analytic/synthetic orientation.*

## Documentation

Start with Installation, Architecture and Training; Params is the reference for
the XML knobs, and Testing for how changes are tested.

| Document | Description |
|---|---|
| [Installation](doc/Installation.md) | Setup, Makefile targets, `train.py` options, environment variables |
| [Architecture](doc/Architecture.md) | Pipeline design, layer types, invertible LDU factorisation, whole paths |
| [Runtime architecture](doc/Componentization.md) | Live ownership map, non-modular mechanisms, and componentization sequence |
| [BasicModel](doc/BasicModel.md) | Cognitive science foundations |
| [Spaces](doc/Spaces.md) | Runtime spaces: Input, Part/Perceptual, Modal, Conceptual, Whole/Symbolic, Output, and SymbolSpace |
| [STM](doc/STM.md) | Short-term memory: the working set of ideas, serial and parallel sequencing, attention, end-states and their chain into long-term memory |
| [Language](doc/Language.md) | The grammar: its layers and operators, and how a sentence is composed and generated |
| [Orders](doc/Orders.md) | Sets, membership and the ladders of concept order |
| [Mereology](doc/Mereology.md) | Parthood as the fundamental operation and the relations defined from it, asserted and geometric parthood, synthesis and analysis of the percept lattice |
| [Lexicon](doc/Lexicon.md) | Word-vector store: surface forms, codebook binding, LBG splitting |
| [Logic](doc/Logic.md) | Subsymbolic and symbolic logic operations |
| [Extend](doc/Extend.md) | An operator's fourth use, and the field of understanding |
| [Reasoning](doc/Reasoning.md) | Direct and derived truth, the thought episode and its operators, and intuition, understanding and thought |
| [Expectation](doc/ExpectationRetention.md) | The activation predictor and expectation at every level |
| [What and spacetime design](doc/WhatSpacetimeDesign.md) | The question-answering design: the `what` question, queryable spacetime, one understanding with input reconstruction and output construction, the interaction record, and thinking |
| [Selected meaning](doc/SelectedMeaning.md) | Selected linguistic meaning and the one thought controller |
| [Accessible mind](doc/AccessibleMind.md) | Thought operators as effects on the subsystems of the accessible mind |
| [Thought-operator contracts](doc/QueryContracts.md) | Contracts of the thought operators and their grammar-owned catalogue |
| [Query phases](doc/QueryPhases.md) | When query execution is permitted while a sentence is read |
| [Query work](doc/QueryWork.md) | Shared work accounting for a selected thought |
| [Thought history](doc/ThoughtHistory.md) | The chronological history of ordinary thought |
| [Taxonomy queries](doc/TaxonomyQueries.md) | PartOf evidence read from the conceptual taxonomy |
| [Existence evidence](doc/ExistenceEvidence.md) | Existence evidence and grammatical descriptions |
| [Nested retention](doc/NestedRetention.md) | Nested occurrence retention in LTM |
| [SymbolFirewall](doc/SymbolFirewall.md) | Governing principle: the boundary between subsymbolic and symbolic computation |
| [Training](doc/Training.md) | Two-phase training, SBOW embeddings, masked prediction modes, lessons and corpora |
| [Gradient flow](doc/GradientFlow.md) | Which objective trains which parameters |
| [Ergodic](doc/Ergodic.md) | Gradient energy sensor, adaptive exploration, factor-level noise injection |
| [Testing](doc/Testing.md) | The bounded test runner, test tiers, and the rules for tests and measurements |
| [Params](doc/Params.md) | XML configuration reference and migration notes |
| [LLM equivalence](doc/LLMEquivalence.md) | The sense in which a sentence-level model is equivalent to an LLM |
| [Philosophy](doc/Philosophy.md) | Kant's analysis/synthesis, Ramsey's theoretical roles, Buddhist epistemology (pramana, tetralemma) |
| [MachineMinds](doc/MachineMinds.md) | What machine minds are, feel, and know |
| [Future work](doc/FutureWork.md) | Directions decided but not yet built |
| [Development history](doc/DevelopmentHistory.md) | Superseded mechanisms and measurements, kept as the record |

## Overview

BasicModel is a parameterized neural architecture that answers the question "what is the truest thing we can say of this experience?".


Model configurations are specified in XML. See [doc/Architecture.md](doc/Architecture.md) for the full mathematical treatment.

Compose fires one operation per round from a joint operation/location softmax.
In training, each sentence runs one greedy complete path (reading, composition,
reconstruction, thought and output) and `explorePaths` alternatives, each
departing once from it. Each stream keeps its strictly cheapest path on
reconstruction plus answer cost, with ties to the greedy path, and the sentence
takes one optimizer step. `architecture.composeTemperature` defaults to `0`
(argmax) and sets the temperature of the training draws; model credit remains
untempered. Evaluation runs the greedy path alone with deterministic logit
argmax. `architecture.reducePressure` defaults to `1` and initializes the
learned gain that raises binary logits as occupancy and deadline urgency grow.
Hard deadlines reserve a slot for the next word and finish each sentence within
its one-slot absolute or three-slot relative end state. See
[Architecture](doc/Architecture.md#whole-paths-and-concept-episodes-2026-11-11)
and [Language](doc/Language.md#one-operation-per-round).

## Files

| File | Description |
|------|-------------|
| [bin/train.py](bin/train.py) | Two-phase training orchestrator: embeddings (Phase 1) then model (Phase 2), local or remote via SSH |
| [bin/Models.py](bin/Models.py) | Main entry point: model factory, training loop, `--compare`, optional `--report` |
| [bin/Layers.py](bin/Layers.py) | Layer library: SigmaLayer, PiLayer, ErgodicLayer, LinearLayer, TruthLayer |
| [bin/Spaces.py](bin/Spaces.py) | Space classes: InputSpace, PartSpace, ModalSpace, ConceptualSpace, WholeSpace, OutputSpace |
| [bin/Language.py](bin/Language.py) | Grammar layers and SymbolSpace: parsing and composition |
| [bin/embed.py](bin/embed.py) | Word vector training: CBOW/SBOW with negative sampling, `WordVectors` (gensim-compatible `.kv`) |
| [bin/serve.py](bin/serve.py) | HTTP server: OpenAI-compatible `/chat/completions` endpoint for WikiOracle integration |
| [data/](data/) | XML model configurations |
| [doc/Architecture.md](doc/Architecture.md) | Algorithm details: Sigma/Pi layers, ergodic exploration, gradient energy sensor |
| [doc/Componentization.md](doc/Componentization.md) | Software ownership, consolidation targets, and extraction gates |
| [doc/Params.md](doc/Params.md) | Full XML parameter reference |
| [doc/Training.md](doc/Training.md) | Embedding pretraining, CBOW/SBOW, masked prediction, `<trainEmbedding>` modes |
| [doc/Installation.md](doc/Installation.md) | Setup, Makefile targets, train.py options, remote training |
| [test/](test/) | Unit tests |

## Quick Start

External MNIST CSVs and the first FineWeb shard are downloaded and verified by
`.venv/bin/python bin/prepare_data.py`, using the URLs, sizes and hashes in
`data/external-files.json`. They remain local and ignored. Tests that need an
absent file skip with this preparation command; tests do not download data.
Stored research papers and locally produced weights remain in the repository.

```bash
# Set up virtual environment
make install

# Run a single model
make xor            # data/MM_xor.xml
make mnist          # data/mnist.xml (prepare the external data above first)

# Compare two models side-by-side (XML1 and XML2 have no defaults)
make compare XML1=data/ergodic.xml XML2=data/ergodic-only.xml

# Run tests
make test

# Generate PDF documentation
make doc
```

## XML Configuration

Models are configured via XML files in `data/`. Training and data parameters live in nested sub-elements:

```xml
<model>
  <architecture>
    <ergodic>false</ergodic>
    <weightsPath>output/BasicModel.ckpt</weightsPath>

    <data>
      <dataset>xor</dataset>        <!-- xor | mnist | text | curriculum | inline | math | ... -->
      <dataType>numeric</dataType>  <!-- embedding | numeric -->
    </data>

    <training>
      <numTrials>1</numTrials>
      <numEpochs>20</numEpochs>
      <batchSize>10</batchSize>
      <learningRate>0.001</learningRate>
      <certainty>false</certainty>
      <autoload>true</autoload>
      <autosave>false</autosave>
    </training>
  </architecture>

  <InputSpace> ... </InputSpace>
  <PartSpace> ... </PartSpace>
  <ConceptualSpace> ... </ConceptualSpace>
  <WholeSpace> ... </WholeSpace>
  <OutputSpace> ... </OutputSpace>
</model>
```

See [doc/Params.md](doc/Params.md) for the full parameter reference.

## Output

Runs produce an HTML report only when `Models.py` is invoked with `--report`
(the `make compare` target does this). Reports are timestamped under `output/`
and may include:

- **Error per Epoch** -- training and test loss curves
- **Accuracy per Digit** -- per-class breakdown for digit/classification configs

In compare mode, overlay plots show combined loss and accuracy across models
when report generation is enabled.
