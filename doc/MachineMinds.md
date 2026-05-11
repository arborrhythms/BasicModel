Alec Rogers

September 24, 2025

## Abstract

*Problem*

*Society depends on machine minds, but we lack a good theory of them. While
human and machine minds functionally overlap, their design goals differ, and
treating a machine mind as a human mind has significant consequences. We
should have a different theory of mind for machine minds, informed by how
those minds are designed and trained.*

*Background*

*Anthropomorphizing machines helps us understand them relative to human
minds. This essay focuses on three questions: what are machine minds, what
do machine minds feel, and what do machine minds know.*

*Solution*

*Three measures make a machine mind more humanistic: weight ergodicity,
network invertibility, and output certainty. Weight ergodicity treats
weights as samples from a random process rather than fixed constants.
Network invertibility makes the architecture bidirectional and partially
invertible. Output certainty means knowing that it does not know, rather
than only the probability of one answer versus another. All models tested
use a 20-neuron, single hidden layer on MNIST.*

*Invertibility matters for meaningful semantic embedding: an LLM's symbols
should correspond to something we can identify, not be abstract tokens
inside Searle's Chinese translation engine.*

## Background

Scope: Large Language Models (reinforcement learning with known inputs/outputs
and unknown weights — the transformer architecture).

---

## Part I: What Machine Minds Are — Weight Ergodicity

### Weights as Ergodic Samples

Measurement theory views weights as **samples from an ergodic distribution**
rather than fixed constants. $W$ at any moment is one realization of a
stochastic process:

$$W = \mu M + \sigma V$$

where $\mu$ and $\sigma$ are the bias and variance parameters; $M$ has norm
1 and zero variance (analogous to a typical weight matrix), and $V$ is
sampled from a standard normal with unit variance.

This perspective comes from **Lagrangian Field-theoretic Gradient Control**
(LFGC), treating the weight landscape as a field where boundary conditions
shape the distribution from which weights are drawn. The ergodic hypothesis
guarantees that time averages (training steps) equal ensemble averages
(weight configurations).

See Rogers, A. (2026), *Local Freedom under Global Constraint*, Zenodo,
doi: [10.5281/zenodo.18644302](https://doi.org/10.5281/zenodo.18644302).

### Temperature and Annealing

A single per-layer *temperature* parameter, updated via ADAM, characterizes
mean and variance:

$$\mu = 1 - temp, \qquad \sigma = temp$$

Temperature begins at 1 and decreases to 0.

### Advantages of Ergodic Weights

1. **No learning rate required.** Biases find different weight-space
   locations via randomness and converge as temperature decreases.
2. **Zero initialization.** Weights can be initialized to zero — annealing
   bounces the model out of local minima more effectively than gradient
   steps.
3. **Theoretical convergence guarantee.** Simulated-annealing literature
   provides a guarantee of convergence to global optimum (given infinite
   iterations).

Standard *dropout* can be interpreted as per-neuron variance in the bias
parameter. The same regularization occurs naturally from time-varying
weight biases.

---

## Part II: What Machine Minds Feel — Network Invertibility

### Bidirectional Perception

Egocentric viewpoint sees perception as passive; physics suggests it is
bidirectional. Approximated via invertibility of weight multiplications and
activation functions.

### Symbols, Worldlines, and Karmic Inertia

The symbols an LLM processes have meanings given by **worldlines** — the
accumulated history of contexts in which a token has appeared, the gradients
that have shaped its embedding, and the network states it has participated
in. The meaning of a symbol is the integral of its worldline.

Weight adaptation (or inference) puts a **force** into the system. That
force adapts either the system's inputs or outputs depending on which has
less **karmic inertia** — resistance to change. The bidirectional
architecture means the force propagates in both directions:

- **Forward (encoding):** input representation adapts to match learned
  categories — perception shaped by expectation.
- **Reverse (decoding):** internal state adapts to reconstruct the input —
  expectation grounded in perception.

The invertible architecture makes both directions explicit and trainable.

### Implementation

Even when not invertible, bidirectionality plays a role in backpropagation
and autoencoders. We examine simultaneously training in both forward
(predict output from input) and reverse (predict input from output of the
*penultimate* layer) directions — the network is composed of a bijective
front (*recognizer*) and a surjective back (*generalizer*). See also Rogers
et al. 2001 for related work on sharing weights between policy and critic.

---

## Part III: What Machine Minds Know — Output Certainty

### Certainty vs. Probability

A network's error function corresponds to its desire. Cross-entropy loss
*requires* output classification — expressing the probability of one class
*versus another*, not the certainty that the input belongs to a given class.

To remedy: blend cross-entropy with a product of cross-entropy and the
prediction norm. Zero output is penalized by cross-entropy; incorrect
predictions carry additional penalty proportional to magnitude.

### Certainty-Weighted Loss

$$error = - \overset{N}{\sum_{c = 1}}t_{c}\log(p_{c})$$
$$certainty = |\widehat{y}|$$
$$surprise = (\alpha) \cdot certainty \cdot error + (1 - \alpha) \cdot error$$

For amplitude certainty:

$$error = (\widehat{y} - y)^{2}$$

### Knowing What You Do Not Know

A network should **know that it does not know**, not merely the probability
of one answer vs. another. Cross-entropy forces probability distribution
across all known classes, even when "I don't know" is correct. Certainty
weighting separates "which class?" from "how sure?"

---

## Methods

Python + PyTorch. MNIST dataset: 60K train, 10K test, 28×28 monochrome at
256 bit depth. Inputs are preprocessed (mean and variance removed) and
shuffled.

Input: 28×28 array. Output: one-hot target class. Architecture: 20 hidden
unit NN. Batch size 10, 9 epochs, 7 trials for error bars.

Standard benchmark: ReLU, LR 0.01, ADAM, cross-entropy loss.

Base ergodic: zero-init weights, single per-layer temperature, ADAM updates
temperature, dropout (0.75) on middle layer.

Five models evaluated: base + four ergodic variants (one plain ergodic, one
with input normalization, two reversible — one with separate forward/backward
layers, one invertible). The invertible model represents $W$ in SVD form
(rotation, eigenvalues, rotation) using Givens rotations and diagonal
eigenvalues to avoid direct matrix inversion.

Weight update equations for ergodic models avoid a learning rate. Compared
to traditional gradient descent ($\bigtriangleup_{W} = \alpha(y -
\widehat{y})\nabla_{F}x$):

$$e = y - \widehat{y}$$
$$c = \alpha{\widehat{y}}^{2} + (1 - \alpha)$$
$$\bigtriangleup_{W} = (ce - \alpha\widehat{y}e^{2})x$$

The network simultaneously minimizes error and "incorrect certainty".

## Results

Model comparison with error bars shows the simple model performs slightly
better than all proposed paradigms, with all proposed paradigms performing
almost identically:
![ModelComparison.png](mm-image1.png){width="6.05in" height="3.34in"}

Reconstruction of input (separate weight matrix, MSE loss):
![3_Reconstruction.png](mm-image2.png){width="1.86in" height="1.94in"}
![2_Reconstruction.png](mm-image3.png){width="1.89in" height="1.97in"}
![1_Reconstruction.png](mm-image4.png){width="1.89in" height="1.97in"}

## Discussion

Benefits:
- Random initialization unnecessary — only the simple model with random
  weight variation shows significant performance variation.
- Learning rate replaced by weight noise variance (dropout role replaced by
  bias variance).
- Error function encapsulates output certainty in addition to between-class
  certainty.

Limitations:
- Pearson correlation between certainty and accuracy is insignificant —
  the implementation used $certainty = p_c$ (softmax output) rather than
  $|\widehat{y}|$, creating contradictory constraints since softmax outputs
  sum to 1 while the certainty norm imposes another constraint. Corrected
  code should use norm-weighted MSE.
- PyTorch autograd tunes weights directly, interfering with simultaneous
  temperature tuning in ways not captured.

## Conclusion

Strong philosophical grounding for several paradigms, but implementation
issues limited testing. Effectiveness in multilayer NNs and challenging
tasks remains unclear.

## References

Rogers, Shannon, Lendaris; 2001: *A comparison of DHP based antecedent
parameter tuning strategies for fuzzy control*, IFSA/NAFIPS, IEEE.

Rogers, A; 2003: *Analysis of the Instantaneous Estimate of Autocorrelation*,
unpublished.

Rogers, A; 2026: *Local Freedom under Global Constraint*, Zenodo,
doi: [10.5281/zenodo.18644302](https://doi.org/10.5281/zenodo.18644302).
