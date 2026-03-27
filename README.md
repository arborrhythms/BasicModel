# BasicModel

## Basic Model of Cognition

*The basic model of cognition relies on conceptual hyperplanes and perceptual prototypes to synthesize and analyze the input space. It uses a high-dimensional embedding to characterize mental space and integrates symbolic computation. Its three major operations are intersection (which forms percepts from concepts), union (a bidirectional mapping between concepts and percepts), and equality (where symbols are elements that map across perceptual and conceptual domains).*

## Documentation

| Document | Description |
|---|---|
| [Architecture](doc/Architecture.md) | Pipeline design, layer types, invertible LDU factorisation |
| [Spaces](doc/Spaces.md) | All six spaces: Input → Perceptual → Conceptual → Symbolic → Syntactic → Output |
| [Ergodic](doc/Ergodic.md) | Gradient energy sensor, adaptive exploration, factor-level noise injection |
| [Training](doc/Training.md) | Two-phase training, SBOW embeddings, masked prediction modes |
| [Params](doc/Params.md) | Complete XML configuration reference |
| [BasicModel](doc/BasicModel.md) | Cognitive science foundations |
| [Language](doc/Language.md) | Grammar, word encoding, symbolic and syntactic spaces |
| [Logic](doc/Logic.md) | Subsymbolic and symbolic logic operations |
| [MachineMinds](doc/MachineMinds.md) | What machine minds are, feel, and know |
| [Installation](doc/Installation.md) | Setup, Makefile targets, environment variables |

## Overview

BasicModel is a parameterized neural architecture with three independent levers:

- **Ergodic** — adaptive bias-variance control via a gradient energy sensor (not gradient descent)
- **Certainty** — per-neuron certainty tracking; neurons graduate from exploration to exploitation independently
- **Reversible** — bidirectional training: forward prediction + backward reconstruction in a single optimizer pass

Model configurations are specified in XML and can be compared side-by-side. See [doc/Architecture.md](doc/Architecture.md) for the full mathematical treatment.

## Files

| File | Description |
|------|-------------|
| [bin/BasicModel.py](bin/BasicModel.py) | Main entry point: model factory, training loop, comparison plots, HTML report |
| [bin/Model.py](bin/Model.py) | Layer library: SigmaLayer, PiLayer, ErgodicLayer, LinearLayer, spaces |
| [bin/embed.py](bin/embed.py) | Word vector training: CBOW/SBOW with negative sampling, `WordVectors` (gensim-compatible `.kv`) |
| [bin/SigmaPi.py](bin/SigmaPi.py) | Standalone demo of the SigmaPi network solving XOR |
| [data/](data/) | XML model configurations |
| [doc/Architecture.md](doc/Architecture.md) | Algorithm details: Sigma/Pi layers, ergodic exploration, gradient energy sensor |
| [doc/Params.md](doc/Params.md) | Full XML parameter reference |
| [doc/Training.md](doc/Training.md) | Embedding pretraining, CBOW/SBOW, masked prediction, `<trainEmbedding>` modes |
| [doc/Installation.md](doc/Installation.md) | Setup, Makefile targets, train.py options, remote training |
| [test/](test/) | Unit tests |

## Quick Start

```bash
# Set up virtual environment
make install

# Run a single model
make simple          # data/simple.xml
make ergodic         # data/ergodic.xml

# Compare two models side-by-side
make compare         # defaults: data/simple.xml vs data/ergodic-only.xml
make compare XML1=data/simple.xml XML2=data/ergodic.xml

# Run tests
make test

# Generate PDF documentation
make doc_pdf
```

## XML Configuration

Models are configured via XML files in `data/`. Training and data parameters live in nested sub-elements:

```xml
<model>
  <architecture>
    <modelType>simple</modelType>   <!-- simple | lm | embedding -->
    <ergodic>false</ergodic>
    <certainty>false</certainty>
    <reconstruct>NONE</reconstruct>  <!-- NONE | symbols | output | both -->
    <maskedPrediction>NONE</maskedPrediction>

    <data>
      <dataset>xor</dataset>        <!-- xor | mnist | text | ... -->
    </data>

    <training>
      <numTrials>1</numTrials>
      <numEpochs>20</numEpochs>
      <batchSize>10</batchSize>
      <learningRate>0.001</learningRate>
      <weightsPath>output/BasicModel.ckpt</weightsPath>
      <autoload>true</autoload>
      <autosave>false</autosave>
    </training>
  </architecture>

  <InputSpace> ... </InputSpace>
  <PerceptualSpace> ... </PerceptualSpace>
  <ConceptualSpace> ... </ConceptualSpace>
  <SymbolicSpace> ... </SymbolicSpace>
  <OutputSpace> ... </OutputSpace>
</model>
```

See [doc/Params.md](doc/Params.md) for the full parameter reference.

## Output

Each run produces an HTML report (timestamped in `output/`) containing:

- **Error per Epoch** — training and test loss curves
- **Accuracy per Digit** — per-class accuracy breakdown

In compare mode, additional overlay plots show combined loss and accuracy across models with color-coded legends.
