import math, os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
try:
    from torchviz import make_dot
except ImportError:
    make_dot = None
from matplotlib import pyplot as plt
from datasets import load_dataset
from wordvectors import WordVectors
#from transformers import BertTokenizer
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pandas as pd
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from Model import Layer, PiLayer, SigmaLayer, ReversibleSigmaLayer, ReversiblePiLayer # Import custom layers from Model.py
from Model import VQLayer, NormLayer, LinearLayer, ReversibleLinearLayer, AttentionLayer
from Model import GammaMem, ColumnUsageTracker, LiftingLayer, SoftMap, CertaintyWeightedCrossEntropy, epsilon
import torch.optim as optim
from functools import partial

from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BASE_DIR)  # basicmodel/ root
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def output_path(filename):
    return os.path.join(ensure_output_dir(), filename)

def output_stem(stem):
    return os.path.join(ensure_output_dir(), stem)


class Report:
    """Collects timestamped SVG figures and XML configs, then writes an HTML report."""
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.figures = []       # list of (title, svg_path)
        self.xml_configs = []   # list of (name, xml_content)

    def save_figure(self, fig, title):
        """Save a matplotlib figure as a timestamped SVG and register it."""
        safe = title.replace(" ", "_").replace("/", "-")
        filename = f"{self.timestamp}_{safe}.svg"
        path = output_path(filename)
        fig.savefig(path, format='svg', bbox_inches='tight')
        self.figures.append((title, filename))
        return path

    def add_xml(self, config_path):
        """Register an XML config file for inclusion in the report."""
        name = os.path.basename(config_path)
        with open(config_path, 'r') as f:
            self.xml_configs.append((name, f.read()))

    def write_html(self):
        """Write the collected figures and configs into a single HTML file."""
        if not self.figures:
            return None
        html_path = output_path(f"{self.timestamp}_report.html")
        lines = [
            '<!DOCTYPE html>',
            '<html><head>',
            f'<title>BasicModel Report {self.timestamp}</title>',
            '<style>',
            '  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; }',
            '  h1 { border-bottom: 2px solid #333; padding-bottom: .3em; }',
            '  h2 { margin-top: 2em; color: #444; }',
            '  .figure { margin: 1.5em 0; }',
            '  .figure img { max-width: 100%; border: 1px solid #ddd; }',
            '  pre { background: #f5f5f5; padding: 1em; overflow-x: auto; border-radius: 4px; }',
            '  .meta { color: #888; font-size: 0.9em; }',
            '</style>',
            '</head><body>',
            f'<h1>BasicModel Report</h1>',
            f'<p class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
        ]
        # Figures
        lines.append('<h2>Figures</h2>')
        for title, svg_file in self.figures:
            lines.append(f'<div class="figure"><h3>{title}</h3>')
            lines.append(f'<img src="{svg_file}" alt="{title}"></div>')
        # XML configs
        if self.xml_configs:
            lines.append('<h2>Configurations</h2>')
            for name, content in self.xml_configs:
                escaped = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                lines.append(f'<h3>{name}</h3><pre>{escaped}</pre>')
        lines.append('</body></html>')
        with open(html_path, 'w') as f:
            f.write('\n'.join(lines))
        from urllib.parse import quote
        file_url = "file://" + quote(os.path.abspath(html_path))
        print(f"Report saved to {file_url}")
        return html_path
TheReport = Report()

class PositionalEncoding(nn.Module):
    nDim   = 2
    index  = [-4, -3]
    #index = [embeddingSize-4, embeddingSize-3]
    p      = 0
    maxP   = 0
    period = [65521, 65537]

    def __init__(self, maxP=0):
        print("Creating positional encoding ...")
        super(PositionalEncoding, self).__init__()
        self.p    = 0
        self.maxP = maxP
        self.div_term = 2*math.pi / maxP
    def forward(self, x):
        batch = x.shape[0]
        n     = x.shape[1] if len(x.shape) > 1 else 1
        embeddingSize = x.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        # write to the last elements of the array
        position = torch.arange(self.p, self.p+batch*n, dtype=torch.float32) * self.div_term
        p1 = torch.sin(position * self.div_term).unsqueeze(0).unsqueeze(0)
        p2 = torch.cos(position * self.div_term).unsqueeze(0).unsqueeze(0)
        pos = torch.concatenate((p1, p2), dim=2)
        # normalize the positional encodings so that they do not overwhelm the data
        # pos = pos/ torch.norm(pos, dim=-1, keepdim=True)
        y = x.clone()
        y[:, :, index] = pos.reshape(batch, n, self.nDim)
        self.p += batch
        assert self.p < self.maxP, "Overflow in object embedding"
        return y
    def reverse(self, y): # from a positional encoding, retrieve indices
        embeddingSize = y.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        pos = y[:,:, index]
        y[:, :, index] = 0
        return y, pos
    @staticmethod
    def test():
        x=  torch.zeros([2,4,100])
        pe= PositionalEncoding(100)
        y = pe.forward(x)
        z = pe.reverse(y)
        print(z)
class TemporalEncoding(nn.Module):
    nDim= 2
    index  = [-2, -1]
    period = [1193, 2000147]
    t      = 0 #nn.Parameter(torch.zeros(1))

    def __init__(self, maxT=0):
        super().__init__()
        self.t    = 0
        self.maxT = maxT
    def forward(self, x):
        batch = x.shape[0]
        n = x.shape[1] if len(x.shape) > 1 else 1
        embeddingSize = x.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        # write to the last elements of the array
        # add in proportion to the norm of the existing features
        t1 = ( 0.5*(1+torch.cos(math.pi + 2*math.pi * torch.tensor(range(self.t, self.t+batch))/self.period[0] )) ).unsqueeze(0).unsqueeze(0)
        t2 = ( 0.5*(1+torch.cos(math.pi + 2*math.pi * torch.tensor(range(self.t, self.t+batch))/self.period[0] )) ).unsqueeze(0).unsqueeze(0)
        time = torch.concatenate((t1, t2), dim=2)
        y = x.clone()
        y[:, :, index] = time.reshape(batch, 1, self.nDim)
        return y

    def increment(self, batch):
        self.t += batch

    def reverse(self, y): # from a positional encoding, retrieve indices
        batch = y.shape[0]
        embeddingSize = y.shape[-1]
        index = np.add([embeddingSize, embeddingSize], self.index)
        t =  y[:, :, index]
        y[:, :, index] = 0
        return y, t
    @staticmethod
    def test():
        x=  torch.zeros([2,4,10])
        te= TemporalEncoding(4)
        y = te.forward(x)
        z = te.reverse(y)
        print(z)
class ObjectEncoding(nn.Module):
    #nWhat        = the "what" is encoded as a dimensionality that varies by subspace
    nWhere       = PositionalEncoding.nDim
    nWhen        = TemporalEncoding.nDim

    inputDim     = 0
    perceptDim   = 0
    conceptDim   = 0
    symbolDim    = 0
    outputDim    = 0

    nInput    = 2 ** 3  # the size of the context window
    nPercepts = 2 ** 4
    nConcepts = 2 ** 4  #
    nSymbols  = 2 ** 3  # must be equal to nConcepts (currently)
    nOutput   = 1  # The output (prediction) size

    objectSize = nWhere + nWhen
    nObjects   = 100*(nInput + nPercepts + nConcepts + nSymbols + nOutput)
    what       = lambda x : True
    where      = PositionalEncoding(nObjects)
    when       = TemporalEncoding(nObjects)

    def setDimensions(self, inputDim, perceptDim, conceptDim, outputDim):
        assert inputDim == perceptDim, "The input and percept dimensions do not match" # they are both input to concepts
        TheObjectEncoding.setInputDim(inputDim)
        TheObjectEncoding.setPerceptDim(perceptDim)
        TheObjectEncoding.setConceptDim(conceptDim)
        TheObjectEncoding.setSymbolDim(1)
        TheObjectEncoding.setOutputDim(outputDim)
    def setInputDim(self, nDim):
        assert self.nObjects != 0, "nObjects was not set"
        self.inputDim = nDim
    def setPerceptDim(self, nDim):
        assert self.nObjects != 0, "nObjects was not set"
        self.perceptDim = nDim
    def setConceptDim(self, nDim):
        assert self.nObjects != 0, "nObjects was not set"
        self.conceptDim = nDim
    def setSymbolDim(self, nDim):
        assert self.nObjects != 0, "nObjects was not set"
        #assert (nDim==0), "Symbols are zero-dimensional"
        self.symbolDim = nDim
    def setOutputDim(self, nDim):
        assert self.nObjects != 0, "nObjects was not set"
        self.outputDim = nDim

    def getEmbeddingSize(self, nDim):
        return nDim + self.objectSize
    def getInputEmbedding(self):
        return self.getEmbeddingSize(self.inputDim)
    def getPerceptEmbedding(self):
        return self.getEmbeddingSize(self.perceptDim)
    def getConceptEmbedding(self):
        return self.getEmbeddingSize(self.conceptDim)
    def getSymbolEmbedding(self):
        return self.getEmbeddingSize(self.symbolDim)
    def getOutputEmbedding(self):
        return self.outputDim # the output is not embedded

    def pad(self, objects, where=True, when=True):
        size = 0
        size += self.nWhere if where else 0
        size += self.nWhen if when else 0
        objects = F.pad(objects, (0, size))
        return objects
    def slice(self, object, where=True, when=True):
        size = 0
        size += self.nWhere if where else 0
        size += self.nWhen if when else 0
        objects = object[0:-size]
        return objects

    def forward(self, objects, what=False, where=True, when=True, pad=False):
        if self.nObjects == 1: # no positional encoding if there is only one object
            return objects
        if pad:
            objects = self.pad(objects)
        if what:
            objects = self.what(objects)
        if where:
            objects = self.where(objects)
        if when:
            objects = self.when(objects)
        return objects
    def reverse(self, objects):
        objects, space = self.where.reverse(objects)
        objects, time  = self.when.reverse(objects)
        return objects, space, time
    #@staticmethod
    #def removeEncoding(x):
    #    e = x.shape[-1]
    #    e -= TheObjectEncoding.objectSize
    #    if len(x.shape) == 2:
    #       x = x[:, 0:e]
    #   else:
    #        x = x[:, :, 0:e]
    #    return x
TheObjectEncoding = ObjectEncoding()


class Data():
    train_input       = []
    train_output      = []
    validation_input  = []
    validation_output = []
    test_input        = []
    test_output       = []

    inputLength       = 128
    combinedTokens    = []

    def load(self, dataset):
        if dataset == "mnist":
            self.loadMNist()
        if dataset == "xor":
            self.loadXOR()
        if dataset == "tomatoes":
            self.loadTomatoes()
    def shuffle(self):
        rand_indx = torch.randperm(len(self.train_output))
        self.train_input = self.train_input[rand_indx][:]
        self.train_output = self.train_output[rand_indx][:]
    def loadMNist(self):
        df = pd.read_csv(os.path.join(DATA_DIR, 'mnist_train.csv'))
        train = df.values
        df = pd.read_csv(os.path.join(DATA_DIR, 'mnist_test.csv'))
        test = df.values
        self.train_input  = torch.tensor(train[:, 1:]/255.0, dtype=torch.float)
        mnistMean = torch.mean(self.train_input)
        self.train_input = self.train_input - mnistMean
        mnistSTD = torch.std(self.train_input)
        self.train_input = self.train_input / mnistSTD
        self.train_output = torch.zeros((train.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(train[:, 0]):
            self.train_output[i][ndx:ndx+1] = 1.0
        self.test_input  = torch.tensor(test[:, 1:]/255.0, dtype=torch.float)
        self.test_input  = (self.test_input - mnistMean) / mnistSTD
        self.test_output = torch.zeros((test.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(test[:, 0]):
            self.test_output[i][ndx:ndx+1] = 1.0
        self.validation_input  = torch.tensor(test[:, 1:]/255.0, dtype=torch.float)
        self.validation_output = torch.zeros((test.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(test[:, 0]):
            self.validation_output[i][ndx:ndx+1] = 1.0
        self.inputLength = 28 * 28
    def loadXOR(self):
        data = {
            "train": {
                "text": ["hello world", "hello there", "loving world", "loving there" ], # nPercepts = 3
                "label": [[0], [1], [1], [0]]
                #"label": [[0, 1], [1, 0], [1, 0], [0, 1]]
            },
            "validation": {
                "text": ["hello world", "hello there", "loving world", "loving there" ], # nPercepts = 3
                "label": [[0], [1], [1], [0]]
                #"label": [[0, 1], [1, 0], [1, 0], [0, 1]]
            },
            "test": {
                "text": ["hello world", "hello there", "loving world", "loving there" ], # nPercepts = 3
                "label": [[0], [1], [1], [0]]
                #"label": [[0, 1], [1, 0], [1, 0], [0, 1]]
            }
        }
        self.train_input      = data["train"]["text"]
        self.train_output      = data["train"]["label"]
        self.validation_input = data["validation"]["text"]
        self.validation_output = data["validation"]["label"]
        self.test_input       = data["test"]["text"]
        self.test_output       = data["test"]["label"]
        self.processLM(data)
    def loadTomatoes(self):
        cache_file = os.path.join(DATA_DIR, "rottenTomatoes.data")

        # Load or cache the pre-trained Word2Vec model
        if os.path.exists(cache_file):
            print("Loading cached data...")
            data = torch.load(cache_file, weights_only=False)
        else:
            print("Downloading data...")
            data = load_dataset("rotten_tomatoes")
            torch.save(data, cache_file)
        self.processLM(data)
    def processLM(self, data, permute=True):
        # tokenize the sentences
        train_tokens      = data["train"]["text"]
        train_labels      = data["train"]["label"]
        validation_tokens = data["validation"]["text"]
        validation_labels = data["validation"]["label"]
        test_tokens       = data["test"]["text"]
        test_labels       = data["test"]["label"]

        self.combinedTokens = train_tokens + validation_tokens + test_tokens
        self.combinedTokens = list(set(self.combinedTokens))

        for i in range(len(self.train_input)):
            self.train_input[i]       = self.stringTensor(train_tokens[i])
            self.validation_input[i]  = self.stringTensor(validation_tokens[i])
            self.test_input[i]        = self.stringTensor(test_tokens[i])
            self.train_output[i]      = torch.tensor(train_labels[i], dtype=torch.float)
            self.validation_output[i] = torch.tensor(validation_labels[i], dtype=torch.float)
            self.test_output[i]       = torch.tensor(test_labels[i], dtype=torch.float)

        if permute:
            rand_indx = torch.randperm(len(self.train_output))
            self.train_input  = [self.train_input[i] for i in rand_indx]
            self.train_output = [self.train_output[i] for i in rand_indx]

    def tokenize(self, TheLanguageModel):
        self.train_input      = TheLanguageModel.tokenize(self.train_input)
        self.validation_input = TheLanguageModel.tokenize(self.validation_input)
        self.test_input       = TheLanguageModel.tokenize(self.test_input)
    def data(self):
        data = {
            "train": {
                "text": self.train_input,
                "label": self.train_output
            },
            "validation": {
                "text":self.validation_input,
                "label": self.validation_output
            },
            "test": {
                "text": self.test_input,
                "label": self.test_output
            }
        }
    def getEmbeddingSize(self):
        return self.getInputSize(), self.getOutputSize()
    def getInputSize(self):
        inShape = len(self.train_input)
        inputEmbeddingSize  = self.train_input[0].shape[0]
        return inputEmbeddingSize
    def getOutputSize(self):
        outShape = len(self.train_output)
        outputEmbeddingSize = self.train_output[0].shape[0]
        return outputEmbeddingSize
    def stringTensor(self, string):
        ascii_values = [ord(char) for char in string]
        tensor = torch.tensor(ascii_values, dtype=torch.int8)
        #zero   = torch.tensor(0, dtype=torch.int8)
        tensor = F.pad(tensor, (0, self.inputLength - tensor.size(0)), 'constant', 0) #ord(" "))
        assert tensor.shape[0]==self.inputLength
        return tensor
TheData = Data()
class Message():
    def __call__(self, txt, newline="\n"):
        print(txt, end=newline)
message = Message()


# A set of prototype vectors for each space
class VectorSet(nn.Module):
    nInput           = 0  # the number of inputs passed to the forward method
    nVectors         = 0  # the number of outputs expected from the forward method
    nDim             = 0  # the latent dimensionality
    embeddingSize    = 0  # this is the dimensionality and the objectEncoding for each vector
    vectors          = nn.Parameter(torch.randn([nVectors, embeddingSize]))  # this is the actual matrix of vectors
    snapDistance     = 0.1  # vectors at or below this distance snap to their corresponding prototypes
    eta              = 0.9
    codebookAct      = GammaMem(nVectors)
    frozen           = []
    returnOnlyFrozen = False
    freezingTemp     = 0.25
    # linear = nn.Linear(10, 5)
    # tracker = ColumnUsageTracker(linear, freezeThreshold=0.01)

    def getSize(self):
        return self.nVectors

    def create(self, nInput, nVectors, nDim, customVQ=True, signed=False):
        self.nInput    = nInput
        self.nVectors  = nVectors
        self.nDim      = nDim
        self.nVectors  = nVectors
        self.customVQ  = customVQ
        self.signed    = signed
        if nDim != None:
            self.embeddingSize = TheObjectEncoding.getEmbeddingSize(nDim)
    def updateWeights(self, embed_sum, cluster_size):
        # Zero out gradients for frozen indices
        weights = torch.ones(self.vq.codebook_size)
        if len(self.frozen) > 0:
            weights[self.frozen] = 0
            if not self.customVQ:
                self.vectors.grad[self.frozen,:] = 0
        return weights
    def freeze(self, activations = None):
        if activations is not None:
            # set the activations that are already frozen
            #unfrozenAct = [a if a not in self.frozen else 0 for a in activations ]
            unfrozenAct = [activations[a].detach().numpy() if a not in self.frozen else 0 for a in range(self.codebookSize)]
            #unfrozenAct = activations[unfrozen]
            #if len(unfrozen) > 0:
            indexMax = np.argmax(unfrozenAct)
            if unfrozenAct[indexMax] > self.freezingTemp:
                if not indexMax in self.frozen:
                    self.frozen.append(indexMax)
                    message(f"Frozen activation at {indexMax}")
    def addVectors(self, nVec=1, decay = 0.9):
        self.codebookSize = nVec
        self.codebookAct  = GammaMem(nVec)
        self.frozen       = [] #np.zeros(nVec)
        if self.customVQ:
            self.vq = VectorQuantize(
                dim = self.embeddingSize,
                codebook_size = nVec,
                threshold_ema_dead_code = 1,
                #num_quantizers=1,         # Return the N nearest quantized vectors
                decay = decay,              # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 1.0,   # the weight on the commitment loss
                #sample_codebook_temp=0.0, #
                #use_cosine_sim=True,
                #learnable_codebook=True,
                #ema_update=False,
                rotation_trick = True      # Set False to use the STE gradient estimator or True to use the rotation trick.
            )
        else:
            # self.vq = VQLayer(
            #    nDim          = nDim,
            #    codebookSize  = nVectors,
            #    numQuantizers = nVectors)
            vec = torch.randn([nVec, self.embeddingSize])
            for i in range(0, nVec):
                vec[i, :] = F.normalize(TheObjectEncoding(vec[i, :].unsqueeze(0).unsqueeze(0)), p=2, dim=1)
            self.vectors = vec[:, :]
    def forward(self, input, t=0):
        # X should be of size batch x nInput x nDim
        # Pad X if necessary
        x     = input
        batch = input.shape[0]
        act   = torch.zeros([batch, self.codebookSize])
        if self.customVQ:
            if x.shape[-1] == self.nDim:
                x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize])], dim=2)
            y   = torch.reshape(x, [-1, self.embeddingSize])

            quantized, indices, commit_loss = self.vq(y, ema_update_weight=self.updateWeights)

            err = torch.norm(y-quantized, dim=1)
            err = torch.reshape(err, x.shape[0:2])
            quantized = torch.reshape(quantized, x.shape)
            # pick the nVector symbols with the smallest reconstruction error
            # Get the top nVectors smallest reconstruction errors
            values_smallest, indices_smallest = torch.topk(err, k=self.nVectors, dim=1, largest=False)
            for i in range(0, indices_smallest.shape[0]):
                for j in range(0, indices_smallest.shape[1]):
                    if err[i,j] <= self.snapDistance:
                        cosSim = self.unsignedAngle(x[i, j, :].clone(), quantized[i, indices_smallest[i, j], :].clone())
                        x[i, j, :] = quantized[i, indices_smallest[i, j], :]
                        # Is this filling the entire activation matrix properly (on the LHS)?
                        assert torch.all(indices_smallest < self.codebookSize), "activation dimension is not correct."
                        act[i, indices_smallest[i, j]] = cosSim + t * random.random()
                    else:
                        #message("codebook miss")
                        x[i, j, :] = quantized[i, indices_smallest[i, j], :]
                        #x[i, j, :] = torch.zeros( [1, 1, self.embeddingSize])
        else:
            dists = self.codebookDistance(x)
            x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize])], dim=2)
            # Project the set of input vectors onto the basis vectors (the vector set).
            # Then compute the column norm of the basis, which result in activations (neuron power).
            # The top of those activations become the "Conscious" set of the current space.
            for b in range(0, x.shape[0]):
                for v in range(0, x.shape[1]):
                    nearestDist, nearestIdx = torch.topk(dists[b,v,:], 1, dim =-1, largest=True)
                    err = nearestDist[0]
                    if err <= self.snapDistance:
                        #message("Using prototype vector")
                        x[b,v,:] = self.vectors[nearestIdx,:]
                        act[b, v] = nearestDist[v]
                    #else:
                    #    x[b,v,:] = x[b,v,:]
                    # Update the codebook
                    # Train the closest vector even if it is not used.
                    if self.training:
                        self.vectors[nearestIdx, :] = self.eta * (self.vectors[nearestIdx, :]) + (1-self.eta) *  x[b,v,:]
                        #self.vectors[nearestIdx, :] = F.normalize(self.vectors[nearestIdx, :], p=2, dim=1)
        for b in range(0, x.shape[0]):
            self.codebookAct.delta(act[b,:])
            if self.returnOnlyFrozen:
                unfrozen = [a for a in range(self.codebookSize) if a not in self.frozen]
                act[b, unfrozen] = 0
            self.freeze(activations = self.codebookAct.get())
        return x

    # The following routine needs also to check if the inner product is positive,
    # otherwise the intersection of the hyperplanes outside of the unit circle
    # may indicate that the two regions are disjoint rather than parts of one another.
    # Therefore, there are three possible results:
    # part(a,b), part(b,a), or disjoint(a,b)
    def conceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        # Normalize vectors A and B
        A_norm = A / A.norm()
        B_norm = B / B.norm()
        # Find orthogonal vector to both A and B (cross product for 3D, generalized for nD)
        cross_prod = torch.linalg.cross(A_norm, B_norm)
        orthogonal_vector = cross_prod / cross_prod.norm()
        # Calculate distance of intersection hyperplane from origin
        distance = orthogonal_vector.norm()
        # Normalize distance to get a measure between 0 and 1
        measure = torch.clamp(distance, 0, 1)
        return measure

        # Example usage
        A = torch.tensor([1.0, 2.0, 3.0])
        B = torch.tensor([3.0, 2.0, 1.0])
        measure = hyperplaneParthood(A, B)

    # The following should be replaced with a mereological framework
    # that operates on the voronoi cells of the LVQ as atoms.
    def perceptParthood(A: torch.Tensor, B: torch.Tensor) -> float:
        """
        Computes the directional parthood ratio: how much of A is contained within B.

        Both A and B must be in [0, 1]^n and define axis-aligned hyperrectangles with origin.

        Parameters:
            A (torch.Tensor): vector defining hyperrectangle A (outer corner)
            B (torch.Tensor): vector defining hyperrectangle B (outer corner)
            eps (float): small value to avoid division by zero

        Returns:
            float: parthood ratio in [0, 1]
        """
        A, B = A.clamp(0, 1), B.clamp(0, 1)
        ratio = torch.minimum(A / (B + epsilon), torch.ones_like(A))
        return torch.prod(ratio).item()

       # Example usage
        A = torch.tensor([1.0, 2.0, 3.0])
        B = torch.tensor([3.0, 2.0, 1.0])
        measure = hyperrectParthood(A, B)

    def learn(self, x, target_idx, lr=0.01):
        """
        Simple LVQ prototype update
        x: (batch, nDim) input vectors
        target_idx: (batch,) indices of the "correct" prototype
        lr: learning rate
        """
        x = F.normalize(x, p=2, dim=-1)
        selected_vectors = self.vectors[target_idx]  # (batch, nDim )

        # LVQ update: move prototypes toward or away from inputs
        # We will implement "attraction" only for now (classic LVQ1)
        delta = lr * (x - selected_vectors)
        self.vectors.data[target_idx] += delta

        # Optional normalization to keep vectors on the sphere
        self.normalize()
    # --- Vector Insertion ---
    def replace(self, new_vectors):
        #assert(self.nVectors == self.vectors.shape[0])
        if self.customVQ:
            self.vq.codebook = torch.stack(new_vectors, dim=0)
        else:
            #vec = torch.randn([nVec, self.embeddingSize])
            #for i in range(0, nVec):
            #    vec[i, :] = F.normalize(TheObjectEncoding(vec[i, :].unsqueeze(0).unsqueeze(0)), p=2, dim=1)
            self.vectors = new_vectors
    def insert(self, new_vectors):
        """
        Insert one or more new vectors
        new_vectors: (nNew, nDim  )
        """
        new_vectors = F.normalize(new_vectors, p=2, dim =-1)
        for i in range(0, new_vectors.shape[0]):
            new_vectors[i, :] = TheObjectEncoding(new_vectors[i, :])
        self.vectors = nn.Parameter(torch.cat([self.vectors.data, new_vectors], dim =0))
        self.nVectors = self.vectors.shape[0]
    # --- Vector Removal ---
    def remove(self, indices):
        """
        Remove vectors by index
        indices: list or tensor of indices to remove
        """
        mask = torch.ones(self.vectors.shape[0], dtype=torch.bool, device=self.vectors.device)
        mask[indices] = False
        self.vectors = nn.Parameter(self.vectors.data[mask])
        self.nVectors = self.vectors.shape[0]
    # --- Fuzzy / Mereology Methods ---
    def norm(self, x):
        return torch.norm(x, dim=-1)
    def normalize(self, x=None):
        if x is None:
            with torch.no_grad():
                if self.signed:
                    self.vectors.data = F.normalize(self.vectors.data, p=2, dim =-1)
                else:
                    self.vectors.data = torch.maximum(torch.minimum(self.vectors.data, 1), 0)
                    self.vectors.data = F.normalize(self.vectors.data, p=2, dim =-1)
        else:
            if self.signed:
                x = F.normalize(x, p=2, dim=-1)
            else:
                x = torch.maximum(torch.minimum(x, torch.tensor(1.0)), torch.tensor(0.0))
                x = F.normalize(x, p=2, dim=-1)
            return x
    def negate(self, x):
        return 1 - x
    def distance(self, x, y):
        N = self.codebookSize
        dist = (x.T @ y) / N
        return dist
    def codebookDistance(self, x):
        vec = self.vectors[:, 0:-TheObjectEncoding.objectSize]
        # dist = self.angle(x.unsqueeze(2), vec.unsqueeze(0).unsqueeze(0))  # (batch, nInput, nFeatures)
        dist = x @ vec.T / self.nDim
        return dist
    def unsignedAngle(self, x, y, dim=-1):
        #xShape = x.shape
        #yShape = y.shape
        #x = torch.reshape(x, (-1, xShape[-1]))
        #y = torch.reshape(y, (-1, yShape[-1]))
        cos_sim = F.cosine_similarity(x, y, dim=-1)
        #dot = torch.sum(x * y, dim=dim)
        #norm_x = torch.norm(x, p=2, dim=dim)
        #norm_y = torch.norm(y, p=2, dim=dim)
        #cos_sim =  dot / (norm_x * norm_y + epsilon)
        # scale 0-1
        return 0.5 * (1-cos_sim) # scale [0-1]
    def equal(self, x, y):
        return 1.0 - self.angle(x, y)
    def part(self, x, y):
        return 1.0 - self.angle(x, y)
    def whole(self, x, y):
        return 1.0 - self.angle(y, x)
    def boundary(self, x, y):
        return torch.abs(self.part(x, y) - self.whole(x, y))
    def overlap(self, x, y):
        return torch.min(self.part(x, y), self.whole(x, y))
    def union(self, x, y):
        return torch.max(x, y)
    def intersection(self, x, y):
        return torch.min(x, y)

# In PassThrough, the perceptual space comes directly from the input space: percepts are not used.
class PassThrough(VectorSet):
    def create(self, nInput, nVectors, nDim):
        super().create(nInput, nVectors, nDim)
    def forward(self, x, t=0):
        batch = x.shape[0]
        y = np.zeros([batch, self.nVectors, self.embeddingSize])
        embeddingZeroPad = TheObjectEncoding.objectSize
        for b in range(batch):
            for n in range(0, self.nInput+1):
                data = np.concatenate((x[b:b + 1, :], np.zeros([1, embeddingZeroPad])), axis=1)
                data = TheObjectEncoding.forward(data)
                y[b, n, 0:self.embeddingSize] = data
            data = np.zeros([1, self.embeddingSize])
            for n in range(self.nInput, self.nVectors):
                y[b, n, 0:self.embeddingSize] = data
        y =  torch.from_numpy(np.array(y, dtype=np.float32))
        return y
    def reverse(self, y, t):
        x = y
        return x
class UnquantizedVSet(VectorSet):
    """Pass-through VectorSet that skips quantization entirely."""
    def create(self, nInput, nVectors, nDim):
        super().create(nInput, nVectors, nDim)
    def forward(self, x, t=0):
        return x
    def reverse(self, y, t=0):
        return y

# An LVQ implementation with an inverse.
class ReversibleDictionary(VectorSet):
    talking = True
    def addVectors(self, nVec=1, LM=None):
        super().addVectors(nVec=nVec, decay = 1.0)
        self.LM = LM
        dict    = LM.getDictionary()
        self.replace(list(dict.values()))
        self.words = list(dict.keys())
    def lookup(self, x):
        y = x
        for b in range(len(x)):
            for i in range(len(x[b])):
                y[b][i] =self.words.index(x[b][i])
        return y
    def forward(self, input, t):
        batch = input.shape[0]
        input = input.squeeze()
        # percepts are Batch x nVectors x nFeatures
        tokenized = self.LM.tokenize(input)
        indices   = self.lookup(tokenized)
        words     = self.vq.codebook[indices, :]
        if words.shape[1] < self.nVectors:
            words = torch.concatenate( (words, torch.zeros([batch, self.nVectors-words.shape[1], self.embeddingSize])), dim=1)
        return words
    def flatten(self, words):
        s = ""
        for w in words:
            s += str(w) + " "
        s = s.rstrip(" ")
        return s
    def reverse(self, y, t):
        batch = y.shape[0]
        nVec  = y.shape[1]

        untokenizedWords = [["" for _ in range(nVec)] for _ in range(batch)]
        for b in range(batch):
            for v in range(nVec):
                vec = y[b,v,:].unsqueeze(0)
                quant, index, _ = self.vq(vec)
                word = self.words[index]
                untokenizedWords[b][v] = word #[0][0]
                if self.talking:
                    message(word, newline = " " if v!=nVec-1 else "\n")
        words = self.LM.untokenize(untokenizedWords)

        #p = []
        #for b in range(batch):
        #    s = []
        #    for v in range(nVec):
        #        vec = y[b,v,:].unsqueeze(0)
        #       quant, index, _ = self.vq( vec,  )
        #        s.append(self.words[index])
        #    p.append(self.flatten(s))
        #words = p # self.LM.untokenize(p)
        #if self.talking:
        #    message(words)
        return  words
# This class assumes that the perceptual space is a dictionary full of word embeddings.
class LanguageModel(VectorSet):
    maxTokens = 0
    talking   = True

    def tokenize(self, data):
        tokenized = []
        for b in range(len(data)):
            sentence = "".join(chr(i) for i in data[b].tolist())
            sentence = sentence.rstrip("\x00")
            t = sentence.split(" ")
            #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            #t = tokenizer.tokenize(sentence)
            # compute the maximum number of tokens in a sentence
            self.maxTokens = max(self.maxTokens, len(t))
            tokenized.append(t)
        # if self.maxTokens > self.nInput:
        #    assert("word was clipped")
        return tokenized  # tokenize the sentences, and compute the maximum number of tokens
    def tokenizeList(self, data):
        tokenized = []
        for b in range(len(data)):
            sentence = data[b]
            t = sentence.split(" ")
            for w in t:
                tokenized.append(w)
        vocab = list(set(tokenized))
        vocab.append(" ")
        vocab.append("\x00")
        return vocab
    def untokenize(self, tokenized):
        data = []
        speech = ""
        for b in range(len(tokenized)):
            sentence = ""
            for w in range(self.nVectors):
                sentence += tokenized[b][w]
                if w < self.nVectors - 1:
                    sentence += " "
            data.append(TheData.stringTensor(sentence))
            speech += sentence
        #if self.talking:
        #    message(speech)
        data = torch.stack(data)
        return data.unsqueeze(1)
    def getVectors(self):
        return self.wv.get_normed_vectors()
    def getSize(self):
        return len(self.wv)
    def getDictionary(self):
        dictionary = dict({})
        for key in self.wv.index_to_key:
            word = torch.reshape(torch.tensor(self.wv[key]), [1, 1, self.nDim])
            dictionary[key] = TheObjectEncoding.forward(word, pad=True).squeeze()
        return dictionary
    def create(self, untokenized, nInput=None, nVectors=None, nDim=None, pretrained=True):
        super().create(nInput, nVectors, nDim)
        tokenized = self.tokenizeList(untokenized)
        data_embeddings_dir = os.path.join(PROJECT_DIR, "data", "embeddings")
        output_embeddings_dir = os.path.join(PROJECT_DIR, "output", "embeddings")
        os.makedirs(output_embeddings_dir, exist_ok=True)
        if pretrained:
            self.model_path = os.path.join(output_embeddings_dir, "word2vec_custom_pretrained.pt")
            super().create(nInput, nVectors, 100)
            if os.path.exists(self.model_path):
                print(f"Loading {self.model_path}...")
                self.wv = WordVectors.load(self.model_path)
            else:
                print("Loading pretrained embeddings...")
                pretrained_file = os.path.join(data_embeddings_dir, "enwiki_20180420_100d.txt")
                self.wv = WordVectors.load_word2vec_format(pretrained_file)
            vec = self.getVectors()
            nDim = len(vec[0])
            nVectors = len(vec)
            super().create(nInput, nVectors, nDim)
        else:
            self.model_path = os.path.join(output_embeddings_dir, "word2vec_custom.pt")
            nDim = 20
            super().create(nInput, nVectors, nDim)
            if os.path.exists(self.model_path):
                print(f"Loading {self.model_path}...")
                self.wv = WordVectors.load(self.model_path)
            else:
                print("Building vocabulary embeddings...")
                self.wv = WordVectors.from_vocab(tokenized, vector_size=nDim)
            vec = self.getVectors()
            nDim = len(vec[0])
            nVectors = len(vec)
            super().create(nInput, nVectors, nDim)
        self.wv.save(self.model_path)
        print(f"Saved embeddings to {self.model_path}")
    def forward(self, input, t=0):
        input = input.squeeze()
        self.batch = len(input)
        tokenized = self.tokenize(input)

        embeddings = []
        embeddingZeroPad = TheObjectEncoding.objectSize
        for sentence in tokenized:
            sentence_embeddings = []
            nTokens = 0
            for token in sentence:
                if token in self.wv:
                    t = np.concatenate((self.wv[token], np.zeros([embeddingZeroPad])), axis=0)
                    t = t / np.linalg.norm(t)
                    sentence_embeddings.append(t)
                else:
                    t = np.concatenate((np.random.randn(self.nDim), np.zeros([embeddingZeroPad])), axis=0)
                    t = t / np.linalg.norm(t)
                    sentence_embeddings.append(t)
                    warnings.warn('unknown token "{0}"'.format(token))
                nTokens += 1
                if nTokens >= self.nVectors:
                    break
            while len(sentence_embeddings) < self.nVectors:
                sentence_embeddings.append(np.zeros(self.embeddingSize))
            embeddings.append(sentence_embeddings)

        e = torch.from_numpy(np.array(embeddings, dtype=np.float32))
        return e
    def reverse(self, y, t=0.0):
        similarWords = [["" for _ in range(self.nVectors)] for _ in range(self.batch)]
        for b in range(self.batch):
            for w in range(self.nVectors):
                embedding = TheObjectEncoding.slice(y[b, w])
                word, score = self.wv.most_similar(embedding.detach().numpy(), topn=1)[0]
                similarWords[b][w] = word
        similarWords = self.untokenize(similarWords)
        return similarWords


class Space(nn.Module):
    name         = ""
    # vectorSet class-level default removed — use instance nn.ModuleList instead
    activation   = None
    processSymbols = False

    def __init__(self, inputShape, outputShape, nVectors, nDim, useVQ=False, customVQ=True, nPrototypes=0, reversePass=False, processSymbols=False):
        super(Space, self).__init__()
        self.inputShape   = inputShape
        self.outputShape  = outputShape
        self.nVectors     = nVectors
        self.nDim         = nDim # the latent dimensionality
        self.embeddingSize = TheObjectEncoding.getEmbeddingSize(self.nDim)
        self.batch        = 0
        self.vectorSet    = nn.ModuleList()
        self.useVQ        = useVQ
        self.customVQ     = customVQ
        self.nPrototypes  = nPrototypes
        self.reversePass = reversePass
        self.processSymbols = processSymbols
        self.params = []
        self.layers = []

    def getEmbeddedIO(self):
        input  = TheObjectEncoding.getEmbeddingSize(self.inputShape[1])
        output = TheObjectEncoding.getEmbeddingSize(self.outputShape[1])
        return input, output
    def lookup(self, x):
        activation = x[0]
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([torch.zeros([1,1, TheObjectEncoding.conceptDim]), x[:,:,1:]], dim=2)
        output, index, _ = self.vectors().vq(x)
        #output[:,:,0:TheObjectEncoding.conceptDim] = output[:,:,0:TheObjectEncoding.conceptDim] * activation  # multiply the codebook vector by the activation
        return output
    def dereference(self, symbols):
        # we get [ batch x nConcepts x symbolEmbedding ],
        # and must compute [ batch x nConcepts x conceptEmbedding ]
        assert list(symbols.shape) == [self.batch, self.nVectors, TheObjectEncoding.getSymbolEmbedding()], "Incorrect input size for dereference"
        input,_ = self.getEmbeddedIO()
        objects = torch.zeros(self.batch, self.nVectors, self.embeddingSize)
        for b in range(self.batch):
            for s in range(self.nVectors):
                x = self.lookup(symbols[b,s,:])
                objects[b,s,:] = x
        assert list(objects.shape) == [self.batch, self.nVectors, self.embeddingSize], "Incorrect output size for dereference"
        return objects

    def stats(self, x):
        #codebookUse = self.vectors().codebookUse
        #message(f"{self.name} Codebook activation: { np.sum(self.vectors().codebookAct.get()) }")
        return
    def vectors(self):
        # this is done to store by reference, since lists are mutable entities (?)
        return self.vectorSet[0]
    def createVectorSet(self, quantized=True):
        if quantized:
            self.vectorSet.append(VectorSet())
            self.vectors().create(self.inputShape[0], self.nVectors, self.nDim, self.customVQ)
            self.vectors().addVectors(nVec=self.nPrototypes)
        else:
            vs = UnquantizedVSet()
            vs.create(self.inputShape[0], self.nVectors, self.nDim)
            self.vectorSet.append(vs)
    def forwardBegin(self, x, t=0.0, reshape=False):
        self.batch = x.shape[0]
        if reshape:
            x = self.flatten(x, True)
            # assert list(x.shape) == [self.batch, self.inputShape[0] *self.inputShape[1]]
        else:
            input, _ = self.getEmbeddedIO()
            assert list(x.shape) == [self.batch, self.inputShape[0], input]
        return x
    def forwardEnd(self, x, t=0.0, reshape=False):
        if reshape:
            x = self.reshape(x, True)
        else:
        #    if x.shape[-1] == self.nDim: # zero-pad vectors without a position.
        #        # x = TheObjectEncoding.forward(x)
        #        x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize])], dim=2)
            _, output = self.getEmbeddedIO()
            assert list(x.shape)==[self.batch, self.outputShape[0], output]
        return x
    def reverseBegin(self, y, t=0.0, reshape=False):
        self.batch = y.shape[0]
        if reshape:
            y = self.flatten(y, False)
        else:
            _, output = self.getEmbeddedIO()
            assert list(y.shape) == [self.batch, self.outputShape[0], output]
        return y
    def reverseEnd(self, y, t=0.0, reshape=False):
        if reshape:
            y = self.reshape(y, False)
        else:
            input, _ = self.getEmbeddedIO()
            assert list(y.shape) == [self.batch, self.inputShape[0], input]
        return y
    def flatten(self, x, forward=True):
        input, output = self.getEmbeddedIO()
        if forward:
            x = x.reshape(self.batch, self.inputShape[0] * input)
        else:
            x = x.reshape(self.batch, self.outputShape[0] * self.outputShape[1])
        return x
    def reshape(self, y, forward=True):
        input, output = self.getEmbeddedIO()
        if forward:
            y = y.reshape(self.batch, self.outputShape[0], self.outputShape[1])
        else:
            y = y.reshape(self.batch, self.inputShape[0], input)
        return y
    def getParameters(self):
        return self.params
    def paramUpdate(self):
        for l in self.layers:
            l.paramUpdate()
class InputSpace(Space):
    name = "Inputs"
    def __init__(self, inputShape, outputShape, nVectors, nDim=None, LM=None, tokenizedInput=False, useVQ=True):
        super(InputSpace, self).__init__(inputShape, outputShape, nVectors, nDim, useVQ=useVQ)
        if LM is not None:
            self.nDim = LM.nDim
            if True:
                self.vectorSet.append(ReversibleDictionary())
                self.vectors().create(self.inputShape[0], self.nVectors, self.nDim, customVQ=True)
                self.vectors().addVectors(nVec=LM.getSize(), LM=LM)
            else:
                self.vectorSet.append(LM)
                #self.vectors().create(self.inputShape[0], self.nVectors, self.nDim)
                #self.vectors().addVectors(nVec=LM.getSize(), LM=LM)
        else:
            self.createVectorSet(quantized=self.useVQ)
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.input          = torch.FloatTensor
        self.tokenizedInput = tokenizedInput
        fullSize  = outputShape[0]*outputShape[1]
        self.lift = LiftingLayer(fullSize, fullSize)
    # The world presenting itself
    def forward(self, input, t=0, mask=None):
        self.batch = input.shape[0]
        assert list(input.shape) == [self.batch, self.inputShape[0], self.inputShape[1]]
        self.input = self.vectors().forward(input, t)
        #self.input = self.lift.forward(self.input)
        _, output = self.getEmbeddedIO()
        assert list(self.input.shape) == [self.batch, self.outputShape[0], output]
        return self.input
    def reverse(self, y, t=0):
        y = self.reverseBegin(y, t)
        #y = self.lift.reverse(y)
        #x = x.unsqueeze(0).unsqueeze(0)
        #x = torch.cat([torch.zeros([1, 1, TheObjectEncoding.conceptDim]), x[:, :, 1:]], dim=2)
        #output[:, :, 0:TheObjectEncoding.conceptDim] = output[:, :, 0:TheObjectEncoding.conceptDim] * activation  # multiply the codebook vector by the activation
        self.input = self.vectors().reverse(y, t)
        #where, when = TheObjectEncoding.reverse(self.percepts)
        #self.input = self.reverseEnd(self.input, t)
        return self.input
class PerceptualSpace(Space):
    name = "Percepts"
    hasAttention = True

    def __init__(self, inputShape, outputShape, nVectors, nDim, useVQ=True, reversePass=False, nPrototypes=0, processSymbols=False):
        super(PerceptualSpace, self).__init__(inputShape, outputShape, nVectors, nDim, useVQ=useVQ, nPrototypes=nPrototypes, reversePass=reversePass, processSymbols=processSymbols)
        input, output = self.getEmbeddedIO()
        self.attention = AttentionLayer(self.nDim, self.nDim)
        if reversePass:
            if inputShape[0]*2 == nVectors:
                self.pi  = ReversiblePiLayer(input, self.nDim, permuteInput=False)  # Hidden layer using PiLayer
                self.forwardPi, self.reversePi = self.pi.forward, self.pi.reverse
            else:
                self.pi1      = PiLayer(input, self.nDim, permuteInput=False)  # Hidden layer using PiLayer
                self.pi2      = PiLayer(self.nDim, input, permuteInput=False)  # Hidden layer using PiLayer
                self.forwardPi, self.reversePi = self.pi1.forward, self.pi2.forward
        else:
            self.pi        = PiLayer(input, self.nDim, permuteInput=False)  # Hidden layer using PiLayer
            self.forwardPi = self.pi.forward
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.createVectorSet()
    def distance(self, x, y):
        # This is a product distance that looks roughly like a star.
        # It has an orthogonalizing effect on its inputs.
        # it is not immediately clear what certainty looks like in this domain,
        # but the suggestion is to use a tunable transfer function whose slope represents certainty.
        return torch.prod( [1-x, 1-y] )
    def certainty(self, x):
        pass
    # Perception
    def forward(self, x, t=0):
        x = self.forwardBegin(x, t)
        x = self.forwardPi(x, t)
        if self.hasAttention:
            x = self.attention.forward(x, t)
        if self.useVQ:
            x  = self.vectors().forward(x , t)
        if self.processSymbols:
            # Turn the percept dimensionality into the symbol dimensionality, but preserve the codebook's positional encoding
            encoding = x[:,:,-TheObjectEncoding.objectSize:]
            x = torch.norm( x[:,:,0:-TheObjectEncoding.objectSize], dim=2 ) / (2*self.nVectors)
            x = x.unsqueeze(-1)
            x = torch.concatenate((x, encoding), dim=2)
        self.percepts = self.forwardEnd(x,t)
        #self.stats(x)
        return self.percepts
    # Manifesting
    def reverse(self, y, t=0):
        assert False, "Perceptual layer is not currently invertible"
        self.percepts = self.reverseBegin(y, t)
        if self.processSymbols:
            self.percepts = self.dereference(self.percepts)
        # preserve the codebook's positional encoding
        encoding = self.percepts[:, :, -TheObjectEncoding.objectSize:]
        if self.useVQ:
            y = y[:,:,0:-TheObjectEncoding.objectSize]
        if self.hasAttention:
            y = self.attention.reverse(y, t)
        x = self.reversePi(y, t)
        #self.concepts = torch.concatenate((self.concepts, encoding), dim=2)
        x = self.reverseEnd(x, t)
        return x
    @staticmethod
    def test():
        pass
class ConceptualSpace(Space):
    name = "Concepts"
    hasAttention = False

    def __init__(self, inputShape, outputShape, nVectors, nDim, useVQ=True, reversePass=False, nPrototypes=0, processSymbols=False, invertible=False, hasNorm=False, ergodic=False):
        super(ConceptualSpace, self).__init__(inputShape, outputShape, nVectors, nDim, useVQ=useVQ, nPrototypes=nPrototypes, reversePass=reversePass, processSymbols=processSymbols)
        self.ergodic = ergodic
        # When objectSize==0 and vector counts differ between input and output,
        # the sigma must operate on *flattened* vectors (nVectors*nDim) so it
        # can change both count and dimension (matching DerivedConceptualSpace).
        self._ergodic_flat = (TheObjectEncoding.objectSize == 0
                              and inputShape[0] != outputShape[0])
        if self._ergodic_flat:
            input  = self.inputShape[0] * self.inputShape[1]
            output = self.outputShape[0] * self.outputShape[1]
        else:
            input, output = self.getEmbeddedIO()
        self.hasNorm = hasNorm
        if self._ergodic_flat:
            self.attention = AttentionLayer(output, output)
        else:
            self.attention = AttentionLayer(output, output)
        if hasNorm:
            self.norm = NormLayer(input, input + 2)
            input += 2
        sigmaOut = output if self._ergodic_flat else self.nDim
        if invertible:
            self.sigma = ReversibleSigmaLayer(input, sigmaOut, permuteInput=False)
            self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
            self.params = self.sigma.getParameters()
            self.layers = [self.sigma]
        elif reversePass:
            self.sigma1 = SigmaLayer(input, sigmaOut, permuteInput=False, deterministic=not ergodic)
            self.sigma2 = SigmaLayer(sigmaOut, input, permuteInput=False, deterministic=not ergodic)
            self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.forward
            self.params = self.sigma1.getParameters() + self.sigma2.getParameters()
            self.layers = [self.sigma1, self.sigma2]
        else:
            self.sigma = SigmaLayer(input, sigmaOut, permuteInput=False, deterministic=not ergodic)
            self.forwardSigma = self.sigma.forward
            self.params = self.sigma.getParameters()
            self.layers = [self.sigma]
        self.createVectorSet()
    def distance(self, x, y):
        # This is a dot-product distance that assumes the X are normalized.
        # However, if the X are not normalized, the magnitudes may be taken as a degree of certainty or knowing.
        # In which case, how do they grow from ignorance to certainty?
        # They would do so naturally if the input vectors are normalized.
        # It would also be possible to use a tunable transfer function.
        return x.T @ y
    def certainty(self, x):
        return x.T @ x
    # Knowing
    def forward(self, x, t=0):
        x = self.forwardBegin(x, t, reshape=self._ergodic_flat)
        if self.hasNorm:
            x = self.norm.forward(x)
        y = self.forwardSigma(x) # Pass through SigmaLayer
        if self.hasAttention:
            if self._ergodic_flat:
                y = self.attention.forward(y)
            else:
                y = self.attention.forward(y, t)
        # Get the concept vectors from the codebook
        # replace some of the Dynamic Percepts with Static Percepts if their distance is low
        if self.useVQ:
            y = self.vectors().forward(y, t) # This must be 4x8x24
        if self.processSymbols:
            # Turn the concept dimensionality into the symbol dimensionality, but preserve the codebook's positional encoding
            encoding = y[:,:,-TheObjectEncoding.objectSize:]
            y = torch.sum(y[:,:,0:-TheObjectEncoding.objectSize], dim=2) / (2*self.nVectors)
            y = y.unsqueeze(-1)
            y = torch.concatenate((y, encoding), dim=2)
        # Reshape the output tensor
        self.concepts = self.forwardEnd(y, t, reshape=self._ergodic_flat)
        #self.stats(x)
        return self.concepts
    # Visualizing
    def reverse(self, y, t=0):
        self.concepts = self.reverseBegin(y, t, reshape=self._ergodic_flat)
        # we are receiving symbols, and we turn them into concepts.
        if self.processSymbols:
            self.concepts = self.dereference(self.concepts)
        if self.hasAttention:
            if self._ergodic_flat:
                self.concepts = self.attention.reverse(self.concepts)
            else:
                self.concepts = self.attention.reverse(self.concepts, t)
        if self._ergodic_flat:
            # Flattened path — no object encoding to strip
            self.concepts = self.reverseSigma(self.concepts)
        else:
            # preserve the codebook's positional encoding
            encoding = self.concepts[:, :, -TheObjectEncoding.objectSize:]
            if TheObjectEncoding.objectSize > 0:
                self.concepts = self.reverseSigma(self.concepts[:,:,0:-TheObjectEncoding.objectSize])
            else:
                self.concepts = self.reverseSigma(self.concepts)
        if self.hasNorm:
            self.concepts = self.norm.reverse(self.concepts)
        #self.concepts = torch.concatenate((self.concepts, encoding), dim=2)
        self.concepts = self.reverseEnd(self.concepts, t, reshape=self._ergodic_flat)
        return self.concepts
    @staticmethod
    def test():
        pass
class SymbolicSpace(Space):
    name = "Symbols"
    threshold        = 0
    serialActivation = False
    symbols          = None

    # The current implementation merely symbolizes all concepts.
    # Therefore, no symbol learning is necessary.
    def __init__(self, inputShape, outputShape, nVectors, nDim, reversePass=False, conceptualSpace=None, processSymbols=False):
        super(SymbolicSpace, self).__init__( inputShape, outputShape, nVectors, nDim, customVQ=True, reversePass=reversePass, processSymbols=processSymbols)
        assert(inputShape[0] == nVectors) # 1:1 mapping
        self.conceptualSpace = conceptualSpace
        #self.mapping     = SoftMap(inputShape[1], nDim, soft=False)
        #self.createVectorSet()
    def distance(self, x, y):
        return x == y
    def certainty(self, x):
        return x.T @ x
    def discretize(self, symbols):
        batch = symbols.shape[0]
        if self.serialActivation:
            for b in range(0,batch):
                top, indices = torch.topk(symbols[b,:], k=1)
                symbols[b,:] = 0
                symbols[b,indices] = top[indices]
        elif self.threshold:
            symbols[symbols > self.threshold] =  1
            symbols[symbols < self.threshold] = -1
        return symbols
    def computeActivation(self, x):
        # we get [ batch x nConcepts x conceptEmbedding ],
        # and must compute [ batch x nConcepts x symbolEmbedding ]
        activations = torch.norm( x[:,:,0:self.outputShape[1]] , dim=2)
        activations = activations.unsqueeze(2)
        activations = torch.concatenate((activations, x[:,:,self.inputShape[1]:]), dim=2)
        return activations

    # Naming
    def forward(self, x, t=0):
        self.symbols = self.forwardBegin(x, t)
        if self.processSymbols: # reduce the embedding vector to the symbolic encoding
            self.symbols = self.computeActivation(self.symbols)
        self.symbols = self.discretize(self.symbols)
        self.symbols = self.forwardEnd(self.symbols, t)
        if self.useVQ:
            self.symbols  = self.vectors().forward(self.symbols , t)
        return self.symbols
    # Interpretation
    def reverse(self, y, t=0):
        self.symbols = self.reverseBegin(y,t)
        if self.processSymbols: # map the symbolic encoding to the embedding vector
            self.symbols = self.conceptualSpace.dereference(self.symbols)
        self.symbols = self.reverseEnd(self.symbols, t)
        return self.symbols

    @staticmethod
    def test():
        pass
class SyntacticSpace(Space):
    name  = "Syntactic"
    words = None

    # The current implementation merely symbolizes all concepts.
    # Therefore, no symbol learning is necessary.
    def __init__(self, inputShape, outputShape, nVectors, nDim, reversePass=False, conceptualSpace=None):
        super(SyntacticSpace, self).__init__( inputShape, outputShape, nVectors, nDim, customVQ=False, reversePass=reversePass, processSymbols=True)
        assert(inputShape[0] == nVectors) # 1:1 mapping
        self.conceptualSpace = conceptualSpace
        #self.mapping     = SoftMap(inputShape[1], nDim, soft=False)
        #self.createVectorSet()
    def distance(self, x, y):
        return x == y
    def certainty(self, x):
        return x.T @ x
    def computeActivation(self, x):
        # we get [ batch x nConcepts x conceptEmbedding ],
        # and must compute [ batch x nConcepts x symbolEmbedding ]
        if x.size(-1) != TheObjectEncoding.symbolDim:
            activations = torch.norm( x[:,:,0:self.outputShape[1]] , dim=2)
            activations = activations.unsqueeze(2)
            activations = torch.concatenate((activations, x[:,:,self.inputShape[1]:]), dim=2)
        else:
            activations = x
        return activations
    # Naming
    def forward(self, x, t=0):
        self.symbols = self.forwardBegin(x, t)
        #self.symbols = self.computeActivation(self.symbols)
        self.symbols = self.forwardEnd(self.symbols, t)
        return self.symbols
    # Interpretation
    def reverse(self, y, t=0):
        self.symbols = self.reverseBegin(y,t)
        self.symbols = self.reverseEnd(self.symbols, t)
        return self.symbols

    @staticmethod
    def test():
        pass
class OutputSpace(Space):
    name = "Outputs"
    def __init__(self, inputShape, outputShape, nVectors, nDim, reversePass=False):
        super(OutputSpace, self).__init__(inputShape, outputShape, nVectors, nDim)
        input, output = self.getEmbeddedIO()
        # the output is reshaped, so we can't use the above formula
        input  = self.inputShape[0]  * input
        output = self.outputShape[0] * self.outputShape[1]

        # output is 0 of reshaped and not embedded
        if reversePass:
            self.linear1 = LinearLayer(input, output)
            self.linear2 = LinearLayer(output, input)
            self.forwardLinear, self.reverseLinear = self.linear1.forward, self.linear2.forward
            #self.linear = ReversibleLinearLayer(input, output)
            #self.forwardLinear, self.reverseLinear = self.linear.forward, self.linear.reverse
        else:
            self.forwardLinear = LinearLayer(input, output)
    # Acting
    def forward(self, x, t=0):
        y = super().forwardBegin(x, t, reshape=True)
        # input is batch x nConcepts
        output = self.forwardLinear(y)
        output = self.forwardEnd(output, t, reshape=True)
        if self.useVQ:
            self.output  = self.vectors().output(self.percepts , t)
        return output
    # Being acted upon
    def reverse(self, y, t=0):
        y = self.reverseBegin(y, t, reshape=True)
        #assert list(y.shape) == [self.batch, self.outputShape[0], self.outputShape[1]]
        y = self.reverseLinear(y)
        output = self.reverseEnd(y, t, reshape=True)
        return output


class BaseModel(nn.Module):
    """Shared training, plotting, and persistence infrastructure for all models."""
    name           = "BaseModel"
    spaces         = []
    reversePass    = False
    plot           = False

    @staticmethod
    def load_config(config_path=None):
        """Load model settings from an XML config file.

        Returns a dict with architecture, training, weights, and server
        settings.  Missing fields use defaults from the create() signature.
        """
        import xml.etree.ElementTree as ET
        if config_path is None:
            config_path = os.path.join(PROJECT_DIR, "model.xml")
        if not os.path.exists(config_path):
            return {}
        tree = ET.parse(config_path)
        root = tree.getroot()
        cfg = {}
        for section in root:
            sec = {}
            for child in section:
                text = child.text.strip() if child.text else ""
                if text.lower() in ("true", "false"):
                    sec[child.tag] = text.lower() == "true"
                else:
                    try:
                        sec[child.tag] = int(text)
                    except ValueError:
                        try:
                            sec[child.tag] = float(text)
                        except ValueError:
                            sec[child.tag] = text
            cfg[section.tag] = sec
        return cfg

    @staticmethod
    def from_config(config_path=None, LM=None):
        """Factory: create the right model type from XML config."""
        cfg = BaseModel.load_config(config_path)
        arch = cfg.get("architecture", {})
        model_type = arch.get("type", "basic")
        if model_type == "simple":
            model = SimpleModel()
        else:
            model = BasicModel()
        model.create_from_config(config_path, LM)
        return model, cfg

    def create(self, **kwargs):
        """Override in subclasses to build model architecture."""
        pass

    def getOptimizer(self, lr=0.01):
        """Build an Adam optimizer over all space parameters.

        Uses getParameters() from each Space (the universal training contract),
        which excludes temperature params managed by alpha_update.
        Falls back to standard PyTorch parameters() when not in ergodic mode.
        """
        if getattr(self, 'ergodic', True):
            params = []
            for s in self.spaces:
                params.extend(s.getParameters())
        else:
            params = list(self.parameters())
        return optim.Adam(params, lr=lr)

    def runTrials(self, numTrials=1, numEpochs=1, batchSize=10, lr=0.001):
        acc = np.zeros([numTrials, numEpochs])
        print(f"\n\n==== {self.name} ====")
        for trial in range(numTrials):
            print(f"\nTrial [{trial + 1}/{numTrials}]")
            self.create(nInput=self.nInput, nConcepts=self.nConcepts, nOutput=self.nOutput)
            acc[trial, :] = self.run(numEpochs=numEpochs, batchSize=batchSize, lr=lr)
        np.savetxt(output_path(f"{self.name}.csv"), np.array(acc), delimiter=",")
        return acc

    def paramUpdate(self):
        for s in self.spaces:
            s.paramUpdate()

    def save_weights(self, path=None):
        """Persist model weights and ergodic state to disk."""
        if path is None:
            path = os.path.join(OUTPUT_DIR, "weights.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[{self.name}] Weights saved to {path}")

    def load_weights(self, path=None, strict=False):
        """Load model weights from disk."""
        if path is None:
            path = os.path.join(OUTPUT_DIR, "weights.pt")
        if not os.path.exists(path):
            print(f"[{self.name}] No weights found at {path}")
            return False
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=strict)
        print(f"[{self.name}] Weights loaded from {path}")
        return True

    def mnistReport(model):
        _, _, y_pred, last_x_pred = model.runEpoch(TheData.test_input, TheData.test_output, lr=0)
        _, predicted = torch.max(y_pred, 1)
        _, actual = torch.max(TheData.test_output, 1)

        rCorrect = torch.zeros((10))
        for i in range(0,10):
            total    = (actual == i).sum().item()
            correct  = (actual==i) & (predicted==actual)
            nCorrect = correct.sum().item()
            rCorrect[i] = nCorrect / total
            print(f"Correctly predicted {i}: {rCorrect[i]}")

        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(0, 10), rCorrect, label="Error (per Input)", marker='o')
        plt.xlabel("Digit")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy per Digit: {model.name}")
        plt.legend()
        plt.grid(True)
        TheReport.save_figure(fig, f"{model.name} Accuracy")
        plt.show(block=False)
        return rCorrect

    def plotLoss(model, trainErr, valErr, testErr):
        """Plots the training, validation, and test losses over time."""
        fig = plt.figure(figsize=(10, 5))

        # Training starts at epoch 2 (epoch 1 is test-only), so offset by +2
        plt.plot(range(2, len(trainErr[0]) + 2), trainErr[0], label="Training Error", marker='o')
        if len(trainErr) > 1 and trainErr[1]:
            plt.plot(range(2, len(trainErr[1]) + 2), trainErr[1], label="Training Error (Input)", marker='o')

        if testErr:
            plt.plot(range(1, len(testErr[0]) + 1), testErr[0], label="Test Error", marker='x')
            if len(testErr) > 1 and testErr[1]:
                plt.plot(range(1, len(testErr[1]) + 1), testErr[1], label="Test Error (Input)", marker='x')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Error per Epoch: {model.name}")
        plt.legend()
        plt.grid(True)

        TheReport.save_figure(fig, f"{model.name} Error")
        plt.show(block=False)

    def plotActivations(model, figure=1, percepts=None, concepts=None, symbols=None):
        fig = plt.figure(figure, figsize=(12, 4))
        fig.clf()

        if percepts is not None:
            p = percepts[-1, :, :].squeeze()
            pAct = torch.norm(p, dim=1).detach().numpy()
            plt.plot(pAct, marker='o', color='r')
            plt.xlabel("Activation")
            plt.ylabel("Percepts")
            plt.title("Perceptual Activation")

        if concepts is not None:
            c = concepts[-1, :, :].squeeze()
            cAct = torch.norm(c, dim=-1).detach().numpy()
            plt.plot(cAct, marker='o', color='b')
            plt.xlabel("Epoch")
            plt.ylabel("Concepts")
            plt.title("Conceptual Activation")

        if symbols is not None:
            s = symbols[-1, :, 0].squeeze()
            sAct = s.detach().numpy()
            plt.plot(sAct, marker='o', color='g')
            plt.xlabel("Epoch")
            plt.ylabel("Symbols")
            plt.title("Symbolic Activations")

        plt.tight_layout()
        plt.show(block=False)

    def plotSpace(model):
        """Visualizes learned weight parameters via PCA projections."""
        perc_weights = model.prototypes.data.cpu().numpy()
        pca = PCA(n_components=2)
        perc_2d = pca.fit_transform(perc_weights)

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(perc_2d[:, 0], perc_2d[:, 1], c='blue', label="Perceptual Prototypes")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Perceptual Space Prototypes")
        plt.legend()

        perc_mean = perc_2d.mean(axis=0)

        conc_weights = model.conceptual.fc_p.weight.data.cpu().numpy()
        conc_proj = pca.transform(conc_weights)

        plt.subplot(1, 3, 2)
        plt.scatter(perc_2d[:, 0], perc_2d[:, 1], c='blue', label="Perceptual Prototypes")
        x_vals = np.linspace(np.min(perc_2d[:, 0]) - 1, np.max(perc_2d[:, 0]) + 1, 100)
        for i in range(conc_proj.shape[0]):
            n = conc_proj[i]
            if np.abs(n[1]) > 1e-3:
                y_vals = perc_mean[1] - (n[0] / n[1]) * (x_vals - perc_mean[0])
                plt.plot(x_vals, y_vals, '--', label=f"Hyperplane {i + 1}" if i == 0 else None, alpha=0.7)
            else:
                plt.axvline(x=perc_mean[0], linestyle='--', alpha=0.7, label=f"Hyperplane {i + 1}" if i == 0 else None)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Conceptual Hyperplanes")
        plt.legend()

        symb_weights = model.fc_symbolic.weight.data.cpu().numpy()
        symb_norms = np.linalg.norm(symb_weights, axis=1)
        x_symb = np.arange(len(symb_norms))

        plt.subplot(1, 3, 3)
        plt.vlines(x_symb, 0, symb_norms, color='k', alpha=0.7)
        plt.scatter(x_symb, symb_norms, color='red', s=100, zorder=3)
        plt.xlabel("Symbolic Feature Index")
        plt.ylabel("L2 Norm")
        plt.title("Symbolic Weights (Lollipop Plot)")

        plt.tight_layout()
        plt.show(block=False)

    def plotNetwork(model):
        """Uses Torchviz to visualize the computation graph."""
        model.eval()
        output, input, _, _ = model.runTest(TheData.test_input, TheData.test_output)
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = "png"
        graph_path = dot.render(output_stem(f"graph_{model.name}"))
        print(f"Saved network graph as {graph_path}")

    def plotErrorbars(model, acc):
        x = list(range(1, len(acc[0]) + 1))
        y = np.array(np.mean(acc, axis=0))
        y_err = np.std(acc, axis=0)
        plt.errorbar(x, y, yerr=y_err, fmt='-o', label=model.name, capsize=4)
class SimpleModel(BaseModel):
    """Unified parameterized model: InputSpace → ConceptualSpace → OutputSpace.

    Three independent levers:
      ergodic   – SigmaLayer (with temperature/alpha adaptation) vs LinearLayer+Tanh
      certainty – CertaintyWeightedCrossEntropy vs nn.CrossEntropyLoss
      quantized – VectorSet codebook snapping vs pass-through
    """
    name       = "SimpleModel"
    vSet       = None
    invertible = False
    hasNorm    = False
    ergodic    = True
    certainty  = True
    quantized  = False

    def create_from_config(self, config_path=None, LM=None):
        """Create the model using settings from an XML config file."""
        cfg = self.load_config(config_path)
        arch = cfg.get("architecture", {})
        self.ergodic = arch.get("ergodic", self.ergodic)
        self.certainty = arch.get("certainty", self.certainty)
        self.quantized = arch.get("quantized", self.quantized)
        self.create(
            nInput=arch.get("nInput", 32),
            nConcepts=arch.get("nConcepts", 256),
            nOutput=arch.get("nOutput", 32),
        )
        return cfg

    def create(self, nInput=32, nConcepts=256, nOutput=32):
        self.nInput    = nInput
        self.nConcepts = nConcepts
        self.nOutput   = nOutput

        inputDim   = 1
        conceptDim = 1
        outputDim  = 1

        self.inputSpace = InputSpace([self.nInput, inputDim],
                                     [self.nInput, inputDim],
                                     self.nInput, nDim=inputDim,
                                     useVQ=self.quantized)
        self.conceptualSpace = ConceptualSpace([self.nInput, inputDim],
                                               [self.nConcepts, conceptDim],
                                               self.nConcepts, conceptDim,
                                               reversePass=self.reversePass,
                                               invertible=self.invertible,
                                               hasNorm=self.hasNorm,
                                               ergodic=self.ergodic,
                                               useVQ=self.quantized)
        self.outputSpace = OutputSpace([self.nConcepts, conceptDim],
                                       [self.nOutput, outputDim],
                                       self.nOutput, outputDim,
                                       reversePass=False)

        self.spaces = [self.inputSpace, self.conceptualSpace, self.outputSpace]
    def forward(self, data):
        percepts = self.inputSpace(data)
        concepts = self.conceptualSpace(percepts)
        symbols  = self.outputSpace(concepts)
        return symbols, concepts
    def reverse(self, concepts):
        percepts = self.conceptualSpace.reverse(concepts)
        data     = self.inputSpace.reverse(percepts)
        return data, percepts
    def runEpoch(self, input, output, lr=0.01, batchSize=10):
        if lr:
            optimizer = self.getOptimizer(lr=lr)

        if self.certainty:
            criterionOutput = CertaintyWeightedCrossEntropy()
        else:
            criterionOutput = nn.CrossEntropyLoss()
        criterionInput  = nn.MSELoss()

        allOutput = []
        allInput  = []
        outErr    = 0
        inErr     = 0
        self.train(lr != 0)
        for i in range(0, len(input), batchSize):
            inputBatch  = input[i:i + batchSize]
            outputBatch = output[i:i + batchSize]
            batchSize   = len(inputBatch)

            inputTensor  = inputBatch
            inputTensor  = inputTensor.unsqueeze(2)
            outputTensor = outputBatch
            outputTensor = outputTensor.unsqueeze(2)

            if lr:
                optimizer.zero_grad()
            outputPred, concepts = self.forward(inputTensor)
            lossOut = criterionOutput(outputPred.squeeze(), outputTensor.squeeze())
            if lr:
                lossOut.backward()
                if self.ergodic:
                    self.paramUpdate()
                optimizer.step()
            outErr = lossOut.item()
            outputPred = outputPred.clone().detach().squeeze()
            if i == 0:
                allOutput = outputPred
            else:
                allOutput = torch.concat((allOutput, outputPred), dim=0)

            # Next run reverse
            if self.reversePass:
                if lr:
                    optimizer.zero_grad()
                inputPred, percepts = self.reverse(concepts.detach())
                lossIn = criterionInput(inputPred.squeeze(), inputTensor.squeeze())
                if lr:
                    lossIn.backward()
                    if self.ergodic:
                        self.paramUpdate()
                    optimizer.step()
                inErr   = lossIn.item()
                inputPred = inputPred.clone().detach().squeeze()
                allInput = inputPred
        return outErr, inErr, allOutput, allInput

    def mnistReport(model):
        _, _, y_pred, last_x_pred = model.runEpoch(TheData.test_input, TheData.test_output, lr=0)
        _, predicted = torch.max(y_pred, 1)
        _, actual = torch.max(TheData.test_output, 1)

        norms = torch.linalg.norm(model.outputSpace.linear.W, dim=0)
        rCorrect = torch.zeros_like(norms)
        for i in range(0,10):
            total    = (actual == i).sum().item()
            correct  = (actual==i) & (predicted==actual)
            nCorrect = correct.sum().item()
            rCorrect[i] = nCorrect / total
            print(f"Correctly predicted {i}: {rCorrect[i]}")
            print(f"Weight norm: {norms[i]}")

        input_matrix = torch.stack((rCorrect, norms))
        correlation_matrix = torch.corrcoef(input_matrix)
        correlation_value = correlation_matrix[0, 1]
        print(f"Pearson Correlation: {correlation_value}")

        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(0, 10), rCorrect, label="Error (per Input)", marker='o')
        plt.xlabel("Digit")
        plt.ylabel("Accuracy & Certainty")
        plt.title(f"Accuracy and Certainty: {model.name}")
        plt.legend()
        plt.grid(True)
        TheReport.save_figure(fig, f"{model.name} Accuracy")
        plt.show(block=False)

        if model.reversePass:
            for i in range(0, 10):
                fig = plt.figure(figsize=(10, 5))
                j = TheData.test_output[-i-1]
                _, num = torch.max(j, axis=0)
                plt.title(f"Reconstruction {num}: {model.name}")
                image = last_x_pred[9-i, :]
                image = np.reshape(image, (28, 28))
                plt.imshow(image)
                TheReport.save_figure(fig, f"{model.name} Reconstruction {num}")
                plt.show(block=False)
        return rCorrect

    def run(self, numEpochs=1, batchSize=10, lr=0.001, stoppingCriterion=0.1):
        trainLosses       = [[],[]]
        validationLosses  = [[],[]]
        minValidationLoss = math.inf
        testLosses        = [[],[]]
        self.plot         = True
        accuracy          = []
        for epoch in range(numEpochs):
            print(f"Epoch [{epoch + 1}/{numEpochs}]")
            if epoch != 0:
                outErr,inErr,allOut,lastIn = self.runEpoch(TheData.train_input, TheData.train_output, lr=lr, batchSize=batchSize)
                trainLosses[0].append(outErr)
                trainLosses[1].append(inErr)
                print(f"Train Loss: {outErr:.4f},  {inErr:.4f}")
                # Anneal global temperature: force convergence over training
                if self.ergodic and numEpochs > 1:
                    progress = epoch / (numEpochs - 1)
                    for s in self.spaces:
                        for l in s.layers:
                            if hasattr(l, 'global_temp_anneal'):
                                l.global_temp_anneal(progress)
            outErr,inErr,allOut,lastIn = self.runEpoch(TheData.test_input, TheData.test_output, lr=0, batchSize=batchSize)
            testLosses[0].append(outErr)
            testLosses[1].append(inErr)
            _, predicted = torch.max(allOut, 1)
            _, actual = torch.max(TheData.test_output, 1)
            total   = predicted.size(0)
            correct = (predicted == actual).sum().item()
            accuracy += [correct / total]
            print(f"Test Accuracy: {100 * correct / total:.2f}%")
            TheData.shuffle()
            if outErr > minValidationLoss + stoppingCriterion:
                print(f"Validation increasing")
                minValidationLoss = outErr
            if outErr < minValidationLoss:
                minValidationLoss = outErr
        if self.plot:
            print(f"Final Stats:")
            self.plotLoss(trainLosses, validationLosses, testLosses)
            self.rCorrect = self.mnistReport()
        self.trainLosses = trainLosses
        self.testLosses  = testLosses
        return accuracy
class BasicModel(BaseModel):
    nSubThoughts   = 1
    nThoughts      = 0
    processSymbols = False
    name           = "BasicModel"

    def create_from_config(self, config_path=None, LM=None):
        """Create the model using settings from model.xml.

        Loads architecture parameters from config, creates the model,
        and optionally loads saved weights if autoload is true.
        """
        cfg = self.load_config(config_path)
        arch = cfg.get("architecture", {})
        self.create(
            nInput=arch.get("nInput", 32),
            nPercepts=arch.get("nPercepts", 64),
            nConcepts=arch.get("nConcepts", 256),
            nSymbols=arch.get("nSymbols", 2),
            nWords=arch.get("nWords", 16),
            nOutput=arch.get("nOutput", 32),
            reversePass=arch.get("reversePass", True),
            LM=LM,
        )
        # Auto-load weights if configured
        wcfg = cfg.get("weights", {})
        if wcfg.get("autoload", True):
            wpath = wcfg.get("path", "output/weights.pt")
            if not os.path.isabs(wpath):
                wpath = os.path.join(PROJECT_DIR, wpath)
            self.load_weights(wpath)
        return cfg

    def create(self, nInput=32,  nPercepts = 64, nConcepts=256, nSymbols=2, nWords=16, nOutput=32, reversePass=True, LM=None):
        self.reversePass      = reversePass
        self.nInput           = nInput
        self.nOutput          = nOutput
        self.nPercepts        = nPercepts
        self.nConcepts        = nConcepts
        self.nSymbols         = nSymbols
        self.nWords           = nWords

        symbolicOutputDim     = TheObjectEncoding.symbolDim if self.processSymbols else TheObjectEncoding.conceptDim

        self.inputSpace       = InputSpace([1, TheData.inputLength],
                                           [self.nInput, TheObjectEncoding.inputDim],
                                           self.nInput, nDim=TheObjectEncoding.inputDim, LM=LM)
        self.conceptualSpace  = ConceptualSpace([self.nInput, TheObjectEncoding.inputDim],
                                               [self.nConcepts, TheObjectEncoding.conceptDim],
                                               self.nConcepts, TheObjectEncoding.conceptDim,
                                               reversePass=reversePass,
                                               nPrototypes=2 * self.nConcepts)
        self.symbolicSpace    = SymbolicSpace([self.nConcepts, TheObjectEncoding.conceptDim],
                                              [self.nSymbols, symbolicOutputDim],
                                              self.nSymbols, TheObjectEncoding.symbolDim,
                                              reversePass = reversePass,
                                              conceptualSpace = self.conceptualSpace,
                                              processSymbols = self.processSymbols)
        self.spaces.extend([self.inputSpace, self.conceptualSpace, self.symbolicSpace])
        if self.nSubThoughts == 1:
            self.perceptualSpace2 = PerceptualSpace([self.nConcepts, symbolicOutputDim],
                                                    [self.nPercepts, TheObjectEncoding.perceptDim],
                                                    self.nPercepts, TheObjectEncoding.perceptDim,
                                                    reversePass = reversePass,
                                                    nPrototypes = 2*self.nPercepts)
            self.conceptualSpace2 = ConceptualSpace([self.nPercepts, TheObjectEncoding.perceptDim],
                                                    [self.nConcepts, TheObjectEncoding.conceptDim],
                                                    self.nConcepts, TheObjectEncoding.conceptDim,
                                                    reversePass = reversePass,
                                                    nPrototypes = 2*self.nConcepts)
            self.symbolicSpace2 = SymbolicSpace([self.nConcepts, TheObjectEncoding.conceptDim],
                                                [self.nSymbols, symbolicOutputDim],
                                                self.nSymbols, TheObjectEncoding.symbolDim,
                                                reversePass = reversePass,
                                                conceptualSpace = self.conceptualSpace2,
                                                processSymbols = self.processSymbols)
            self.spaces.extend([self.perceptualSpace2, self.conceptualSpace2, self.symbolicSpace2])
        if self.nThoughts == 1:
            self.syntacticSpace3    = SyntacticSpace([self.nSymbols, symbolicOutputDim],
                                               [self.nWords, symbolicOutputDim],
                                                self.nWords, TheObjectEncoding.symbolDim,
                                                reversePass = reversePass)
            self.symbolicSpace3 = SymbolicSpace([self.nWords, symbolicOutputDim],
                                                [self.nWords, symbolicOutputDim],
                                                self.nWords, TheObjectEncoding.symbolDim,
                                                reversePass = reversePass)
            self.spaces.extend([self.syntacticSpace3, self.symbolicSpace3])
        # The input dimensionality of the output layer must be equal to the sum of the output dimensionalities of the symbolic layers.
        self.outputSpace     = OutputSpace([ (self.nSubThoughts+self.nThoughts+1) * self.nSymbols, symbolicOutputDim],
                                           [self.nOutput, TheObjectEncoding.outputDim],
                                           self.nOutput, TheObjectEncoding.outputDim,
                                           reversePass = reversePass)
        self.spaces.extend([self.outputSpace])

        # The output dimensionality of the input layer must be equal to the output dimensionality of the perceptual layer, since the conceptual layer operates on both.
        #assert self.inputSpace.outputShape[1] == self.perceptualSpace2.outputShape[1] # inputDim == perceptDim
        # The input dimensionality of the symbolic layer must be equal to the input dimensionality of the perceptual layer, since they both operate on the output of the conceptual layer.
        #assert self.symbolicSpace.inputShape[1] == self.perceptualSpace2.inputShape[1] == self.conceptualSpace.outputShape[1]#  conceptDim = conceptDim
        # The output shape of the symbolic space is equal to the input shape of the output space
        #assert self.symbolicSpace.outputShape[1] == self.outputSpace.inputShape[1] # these are in conceptual space, or symbolic space if symbols emit objectSize symbols (processSymbols == True)

    def Start(self, data, t=0.0):
        input = self.inputSpace(data, t)
        concepts = self.conceptualSpace(input, t)
        symbols = self.symbolicSpace(concepts, t)
        if self.plot:
            self.plotActivations(figure=1, concepts=concepts)
        return concepts, input, symbols
    def StartReverse(self, concepts, input, symbols, t=0.0):
        concepts = self.symbolicSpace.reverse(symbols, t)
        input = self.conceptualSpace.reverse(concepts, t)
        data  = self.inputSpace.reverse(input, t)
        return data, input
    def SubsymbolicThought(self, data, t=0.0):
        percepts = self.perceptualSpace2(data, t)
        concepts = self.conceptualSpace2(percepts, t)
        symbols  = self.symbolicSpace2(concepts, t)
        if self.plot:
            self.plotActivations(figure=1, percepts=percepts, concepts=concepts)
        return concepts, symbols
    def SubsymbolicThoughtReverse(self, concepts, symbols, t=0.0):
        concepts = self.symbolicSpace2.reverse(symbols, t)
        percepts = self.conceptualSpace2.reverse(concepts, t)
        #data = self.perceptualSpace2.reverse(percepts, t)
        data = None
        return data
    def SymbolicThought(self, data, t=0.0):
        words   = self.syntacticSpace3(data, t)
        symbols = self.symbolicSpace3(words, t)
        if self.plot:
            self.plotActivations(figure=1, symbols=symbols)
        return symbols, words
    def SymbolicThoughtReverse(self, symbols, words, t=0.0):
        symbols = self.syntacticSpace3.reverse(words, t)
        data    = self.symbolicSpace3.reverse(symbols, t)
        return data
    def Finish(self, symbols, t=0.0):
            self.words = symbols
            data = self.outputSpace(symbols, t)
            if self.plot:
                self.plotActivations(figure=1, symbols=symbols)
            return data
    def FinishReverse(self, data, t=0.0):
            # cache this non-invertible step, since we watn to reverse our behavior based on our understanding,
            # not based on the action that we emitted.
            symbols = self.words.detach()
            #symbols = self.outputSpace.reverse(data, t)
            return symbols
    def forward(self, data, t=0.0):
        # We may at some point wish to combine concepts over symbolic and subsymbolic pathways
        # concepts = (alpha) * concepts1 + (1 - alpha) * concepts2
        symbols = None
        data, input, symbols = self.Start(data, t)
        for n in range(self.nSubThoughts):
            data, symbols1 = self.SubsymbolicThought(data, t)
            symbols = torch.cat((symbols, symbols1), dim=1)
        for n in range(self.nThoughts):
            data, symbols2 = self.SymbolicThought(data, t)
            symbols = torch.cat((symbols, symbols2), dim=1)
        data = self.Finish(symbols, t)
        batch = input.shape[0]
        TheObjectEncoding.when.increment(batch)
        return data, input
    def reverse(self, data, t=0.0):
        symbols = self.FinishReverse(data, t)
        nSym = round(self.nSymbols)
        symbolIndex = 0
        for n in range(self.nThoughts):
            symbols1 = symbols[:, symbolIndex*nSym:(symbolIndex+1)*nSym]
            symbolIndex += 1
            data = self.SymbolicThoughtReverse(data, symbols1, t)
        for n in range(self.nSubThoughts):
            symbols1 = symbols[:, symbolIndex*nSym:(symbolIndex+1)*nSym]
            symbolIndex += 1
            data = self.SubsymbolicThoughtReverse(data, symbols1, t)
        symbols1 = symbols[:, symbolIndex * nSym:(symbolIndex + 1) * nSym]
        data, input = self.StartReverse(data, None, symbols1, t)
        return data, input

    def run(self, numEpochs=1, batchSize=10, temperature=0.0001, lr=0.01, stoppingCriterion=0.1):
        """
        Runs the Transformer model.
        """

        trainLosses       = [[],[]]
        validationLosses  = [[],[]]
        minValidationLoss = math.inf
        testLosses        = [[],[]]
        self.plot         = False

        for epoch in range(numEpochs):
            if epoch % 10 == 0:
                print(f"Epoch [{epoch + 1}/{numEpochs}]")

            outErr,inErr,_,_ = self.runTrain(TheData.train_input, TheData.train_output, temperature=temperature, lr=lr, batchSize=batchSize)
            trainLosses[0].append(outErr)
            trainLosses[1].append(inErr)
            print(f"Train Loss: {outErr:.4f},  {inErr:.4f}")

            outErr,inErr,_,_ = self.runTest(TheData.validation_input, TheData.validation_output)
            validationLosses[0].append(outErr)
            validationLosses[1].append(inErr)
            #print(f"Validation Loss: {outErr:.4f},  {inErr:.4f}")

            outErr,inErr,_,_ = self.runTest(TheData.test_input, TheData.test_output)
            testLosses[0].append(outErr)
            testLosses[1].append(inErr)
            #print(f"Test Loss: {outErr:.4f},  {inErr:.4f}")

            if outErr > minValidationLoss + stoppingCriterion:
                temperature = temperature/2
                print(f"Validation increasing, dropping temperature to {temperature}")
                minValidationLoss = outErr
            if outErr < minValidationLoss:
                minValidationLoss = outErr
            if epoch < (numEpochs / 2):
                temperature = 0 # This will stop the temperature from being used.
                # It might better be eliminated by the optimizer

        # Plot the loss over time
        self.plot = True
        self.runTest(TheData.test_input, TheData.test_output)
        self.plotLoss(trainLosses, validationLosses, testLosses)
        #self.plotNetwork()
    def runTrain(self, input, output, temperature=0.00001, lr=0.01, batchSize=10):
        """
        Trains the Transformer model using labeled training data.
        """
        optimizer1 = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(self.parameters(), lr=lr)

        # Define LRscheduler
        #scheduler = ReduceLROnPlateau(optimizer1, 'min', patience=5, factor=0.1, verbose=True)
        #scheduler.step(val_loss)

        lossFnOutput = nn.MSELoss()
        lossFnInput = nn.MSELoss()

        allOutput = []
        allInput  = []
        outErr    = 0
        inErr     = 0
        self.train()
        for i in range(0, len(input), batchSize):
            inputBatch  = input[i:i + batchSize]
            outputBatch = output[i:i + batchSize]
            batchSize   = len(inputBatch)

            # train only one layer at a time
            #layer = i % 5
            #self.inputSpace.freeze(layer==0)
            #self.perceptualSpace.freeze(layer==1)
            #self.conceptualSpace.freeze(layer==2)
            #self.symbolicSpace.freeze(layer==3)
            #self.outputSpace.freeze(layer==4)

            # Combine the padded tensors
            inputTensor  = torch.stack(inputBatch, dim=0)
            inputTensor  = inputTensor.unsqueeze(1)
            outputTensor = torch.stack(outputBatch, dim=0)
            outputTensor = outputTensor.unsqueeze(1)

            # First run forward
            optimizer1.zero_grad()
            eOutput, aInput  = self.forward(inputTensor, temperature)
            lossOut  = lossFnOutput(eOutput.squeeze(), outputTensor.squeeze())
            outErr   = lossOut.item()
            lossOut.backward()
            #list(map(lambda obj: obj.zeroGradient(), self.spaces))
            optimizer1.step()
            out = eOutput.clone().detach()
            for i in range(0, batchSize):
                allOutput.append(out[i,:,:].numpy())

            # Next run reverse
            if self.reversePass:
                optimizer2.zero_grad()
                eInput, aOutput  = self.reverse(eOutput.detach(), temperature)
                #lossIn = lossFnInput(eInput.squeeze().float(), inputTensor.squeeze().float())
                lossIn  = lossFnInput(aOutput, aInput)
                inErr   = lossIn.item()
                lossIn.backward()
                #list(map(lambda obj: obj.zeroGradient(), self.spaces))
                optimizer2.step()
                inp = eInput.clone().detach()
                for i in range(0, batchSize):
                    allInput.append(inp[i,:,:].numpy())
        return outErr, inErr, allOutput, allInput
    def runTest(self, input, output, batchSize=10):
        lossFnOutput = nn.MSELoss()
        lossFnInput = nn.MSELoss()

        allOutput = []
        allInput  = []
        outErr    = 0
        inErr     = 0
        self.eval()
        with torch.no_grad():
            for i in range(0, len(input), batchSize):
                inputBatch   = input[i:i + batchSize]
                outputBatch  = output[i:i + batchSize]
                batchSize    = len(inputBatch)
                # Combine the padded tensors
                inputTensor  = torch.stack(inputBatch, dim=0)
                inputTensor  = inputTensor.unsqueeze(1)
                outputTensor = torch.stack(outputBatch, dim=0)
                outputTensor = outputTensor.unsqueeze(1)
                eOutput,pIn  = self.forward(inputTensor)
                lossOut      = lossFnOutput(eOutput.squeeze(), outputTensor.squeeze())
                outErr       = lossOut.item()
                out = eOutput.clone().detach()
                for i in range(0, batchSize):
                    allOutput.append(out[i, :, :].numpy())

                # Next run reverse
                if self.reversePass:
                    eInput, pOut = self.reverse(eOutput.detach())
                    lossIn    = lossFnInput(pOut, pIn)
                    inErr     = lossIn.item()
                    inp       = eInput.clone().detach()
                    for i in range(0, batchSize):
                        allInput.append(inp[i,:,:].numpy())
        return outErr, inErr, allOutput, allInput

    def classificationReport(self, min=0, max=1):
        _, _, y_pred, x_pred = self.runTest(TheData.test_input, TheData.test_output)
        y_actual = TheData.test_output
        y_pred_sat = np.maximum(min, np.minimum(max, np.round(np.array(y_pred)).squeeze()))
        performance = classification_report(
            y_actual, y_pred_sat,
            target_names=["Negative Review", "Positive Review"]
        )
        print(performance)
TheBasicModel = BasicModel()


def createLMModel(dataset, pretrained=False):
    nInput    = 2 ** 3  # the size of the context window
    nPercepts = 2 ** 3  # the number of percepts cannot be greater than the number of inputs
    nConcepts = 2 ** 3  # the number of concepts may vary freely
    nSymbols  = 2 ** 3  # must be equal to nConcepts
    nWords    = 2 ** 3
    nOutput   = 1       # The output (prediction) size

    TheData.load(dataset)
    # load the langauge model, which will determine the number of inputs
    languageModel = LanguageModel()
    languageModel.create(TheData.combinedTokens, nInput=0, nVectors=nInput, pretrained=pretrained)

    # pre-tokenization for speed
    #TheData.tokenize(languageModel)

    inputDim     = languageModel.nDim
    outputDim    = TheData.getOutputSize()
    perceptDim   = inputDim
    conceptDim   = inputDim
    TheObjectEncoding.setDimensions(inputDim, perceptDim, conceptDim, outputDim)

    TheBasicModel.create(nInput=nInput, nPercepts=nPercepts, nConcepts=nConcepts, nSymbols=nSymbols, nWords=nWords, nOutput=nOutput,
                         LM=languageModel)
def createVQModel():
    nInput    = 2 ** 7  #
    nPercepts = 2 ** 8  #
    nConcepts = 2 ** 5  #
    nSymbols  = 2 ** 5  #
    nOutput   = 1  # The output (prediction) size

    inputDim  = TheData.getInputSize()
    outputDim = TheData.getOutputSize()
    TheObjectEncoding.setInputDim(inputDim)
    TheObjectEncoding.setPerceptDim(inputDim)
    TheObjectEncoding.setConceptDim(inputDim)
    TheObjectEncoding.setSymbolDim(0)
    TheObjectEncoding.setOutputDim(outputDim)

    vectorSet = VectorSet()
    vectorSet.create(nInput=nInput, nOutput=nPercepts, nDim =inputDim, nVectors=nInput)

    TheBasicModel.create(nInput=nInput, nPercepts=nPercepts, nConcepts=nConcepts, nSymbols=nSymbols, nOutput=nOutput,
                         reversePass=False, LM=vectorSet)
def createPassThroughModel(dataset):
    nInput    = 1       #
    nPercepts = 2 ** 3  #
    nConcepts = 2 ** 3  #
    nSymbols  = 2 ** 3  #
    nOutput   = 1       # The output (prediction) size

    TheData.load(dataset)
    # Set the network's dimensionality
    inputDim = TheData.getInputSize()
    outputDim = TheData.getOutputSize()
    TheObjectEncoding.setInputDim(inputDim)
    TheObjectEncoding.setPerceptDim(inputDim)
    TheObjectEncoding.setConceptDim(inputDim)
    TheObjectEncoding.setSymbolDim(0)
    TheObjectEncoding.setOutputDim(outputDim)


    passThrough = PassThrough()
    passThrough.create(nInput=nInput, nOutput=nPercepts, nDim=TheObjectEncoding.inputDim)

    TheBasicModel.create(nInput=nInput, nPercepts=nPercepts, nConcepts=nConcepts, nSymbols=nSymbols, nOutput=nOutput,
                         reversePass=False, LM=passThrough)


def test():
    PositionalEncoding.test()
    TemporalEncoding.test()

    # test XOR
    createLMModel('xor', pretrained=False, reversePass=True)
    TheBasicModel.run(numEpochs=200, stoppingCriterion=1, temperature=0.01, lr=0.01)
    TheBasicModel.classificationReport()

def plotErrorbarsFromFile(fn):
    """Load a CSV of trial accuracies and add an errorbar series to the current plot."""
    acc = np.loadtxt(output_path(f"{fn}.csv"), delimiter=",")
    y = np.mean(acc, axis=0)
    y_err = np.std(acc, axis=0)
    x = list(range(1, len(y) + 1))
    plt.errorbar(x, y, yerr=y_err, fmt='-o', label=fn, capsize=4)

def plotComparison(models):
    """Plot per-digit accuracy comparison across model variants.

    Args:
        models: list of (name, rCorrect_tensor) tuples
    """
    digits = list(range(10))
    n = len(models)
    width = 0.8 / n
    fig = plt.figure(figsize=(12, 6))
    for i, (name, rCorrect) in enumerate(models):
        offsets = [d + (i - n/2 + 0.5) * width for d in digits]
        vals = rCorrect.detach().cpu().numpy() if hasattr(rCorrect, 'detach') else rCorrect
        plt.bar(offsets, vals, width, label=name)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.title("Per-Digit Accuracy Comparison")
    plt.xticks(digits)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    path = TheReport.save_figure(fig, "Digit Comparison")
    plt.show(block=False)
    print(f"Comparison saved to {path}")

def plotCombinedLoss(models):
    """Overlay training and test loss curves from multiple models on shared axes.

    Args:
        models: list of SimpleModel instances with .trainLosses and .testLosses.
    """
    fig = plt.figure(figsize=(12, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, m in enumerate(models):
        color = colors[i % len(colors)]
        if hasattr(m, 'trainLosses') and m.trainLosses[0]:
            train = m.trainLosses[0]
            # Training starts at epoch 2 (epoch 1 is test-only)
            plt.plot(range(2, len(train) + 2), train,
                     label=f"{m.name} - Training",
                     marker='o', color=color, linestyle='-')
        if hasattr(m, 'testLosses') and m.testLosses[0]:
            test = m.testLosses[0]
            plt.plot(range(1, len(test) + 1), test,
                     label=f"{m.name} - Test",
                     marker='x', color=color, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Error per Epoch: Model Comparison")
    plt.legend()
    plt.grid(True)
    TheReport.save_figure(fig, "Combined Error Comparison")
    plt.show(block=False)

def plotCombinedAccuracy(models):
    """Overlay per-digit accuracy curves from multiple models on shared axes.

    Args:
        models: list of (name, rCorrect_tensor) tuples.
    """
    fig = plt.figure(figsize=(12, 6))
    digits = list(range(10))
    for name, rCorrect in models:
        vals = rCorrect.detach().cpu().numpy() if hasattr(rCorrect, 'detach') else rCorrect
        plt.plot(digits, vals, label=name, marker='o')
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Digit: Model Comparison")
    plt.xticks(digits)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    TheReport.save_figure(fig, "Combined Accuracy Comparison")
    plt.show(block=False)

def plotEpochComparison():
    """Plot epoch-level accuracy comparison from saved CSV files."""
    fig = plt.figure(figsize=(10, 5))
    for fn in ["SimpleModel", "ErgodicModel", "Ergodic - Normed", "Ergodic - Reversible"]:
        csv_path = output_path(f"{fn}.csv")
        if os.path.exists(csv_path):
            plotErrorbarsFromFile(fn)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(True)
    TheReport.save_figure(fig, "Model Comparison")
    plt.show(block=False)

def _derived_name(ergodic, certainty, quantized, normed=False, reverse=False, invert=False):
    """Generate a human-readable model name from its flags."""
    if not ergodic and not certainty and not quantized:
        return "SimpleModel"
    parts = []
    if ergodic:
        parts.append("Ergodic")
    if certainty:
        parts.append("Certainty")
    if quantized:
        parts.append("Quantized")
    if normed:
        parts.append("Normed")
    if invert:
        parts.append("Invertible")
    elif reverse:
        parts.append("Reversible")
    return " + ".join(parts) if parts else "SimpleModel"


def BasicModelFactory(config_path):
    """Create, train, and evaluate models from an XML config file.

    Dispatches to the right model class based on <architecture> flags:
      - modelType=lm         → BasicModel (language model path)
      - modelType=passthrough → BasicModel (passthrough path)
      - Otherwise             → SimpleModel parameterized by:
            ergodic, certainty, quantized, normed, reverse, invert
    """
    cfg = BaseModel.load_config(config_path)
    arch = cfg.get("architecture", {})
    train = cfg.get("training", {})
    weights = cfg.get("weights", {})

    dataset = train.get("dataset", "xor")
    model_type = train.get("modelType", "lm")

    if train.get("detectAnomaly", False):
        torch.autograd.set_detect_anomaly(True)

    # --- SimpleModel path ---
    # All flags live in <architecture>
    ergodic   = arch.get("ergodic", False)
    certainty = arch.get("certainty", False)
    quantized = arch.get("quantized", False)
    normed    = arch.get("normed", False)
    reverse   = arch.get("reverse", False)
    invert    = arch.get("invert", False)

    is_derived = ergodic or certainty or quantized or normed or reverse or invert \
                 or "ergodic" in arch or "certainty" in arch or "quantized" in arch

    if is_derived:
        TheData.load(dataset)
        nInput = TheData.getInputSize()
        nConcepts = arch.get("nConcepts", 20)
        nOutput = TheData.getOutputSize()

        # Set ObjectEncoding dimensions for the SimpleModel path.
        # nWhere=0, nWhen=0 so objectSize=0 — unified Space with no
        # positional/temporal encoding (matches old DerivedSpace behaviour).
        dim = 1
        TheObjectEncoding.nWhere = 0
        TheObjectEncoding.nWhen = 0
        TheObjectEncoding.objectSize = 0
        TheObjectEncoding.setInputDim(dim)
        TheObjectEncoding.setPerceptDim(dim)
        TheObjectEncoding.setConceptDim(dim)
        TheObjectEncoding.setSymbolDim(0)
        TheObjectEncoding.setOutputDim(dim)

        numTrials = train.get("numTrials", 1)
        numEpochs = train.get("numEpochs", 3)
        batchSize = train.get("batchSize", 10)
        completed_models = []

        m = SimpleModel()
        m.ergodic   = ergodic
        m.certainty = certainty
        m.quantized = quantized
        m.hasNorm   = normed
        if reverse or invert:
            m.reversePass = True
        if invert:
            m.invertible = True
        m.create(nInput=nInput, nConcepts=nConcepts, nOutput=nOutput)
        m.name = _derived_name(ergodic, certainty, quantized, normed, reverse, invert)
        m.runTrials(numTrials, numEpochs, batchSize)
        if hasattr(m, 'rCorrect'):
            completed_models.append((m.name, m.rCorrect, m))

        if len(completed_models) > 1:
            plotComparison([(name, rc) for name, rc, _ in completed_models])

        return completed_models

    # --- BasicModel path (LM or passthrough) ---
    if model_type == "passthrough":
        createPassThroughModel(dataset)
    else:
        createLMModel(dataset, pretrained=train.get("pretrained", False))

    # Build run() kwargs from training config
    run_kwargs = {}
    if "numEpochs" in train:
        run_kwargs["numEpochs"] = train["numEpochs"]
    if "stoppingCriterion" in train:
        run_kwargs["stoppingCriterion"] = train["stoppingCriterion"]
    if "temperature" in train:
        run_kwargs["temperature"] = train["temperature"]
    if "learningRate" in train:
        run_kwargs["lr"] = train["learningRate"]

    TheBasicModel.run(**run_kwargs)

    # Build classificationReport() kwargs
    report_kwargs = {}
    if "classificationMin" in train:
        report_kwargs["min"] = train["classificationMin"]
    if "classificationMax" in train:
        report_kwargs["max"] = train["classificationMax"]
    TheBasicModel.classificationReport(**report_kwargs)

    # Auto-save weights if configured
    if weights.get("autosave", False):
        wpath = weights.get("path", "output/weights.pt")
        if not os.path.isabs(wpath):
            wpath = os.path.join(PROJECT_DIR, wpath)
        TheBasicModel.save_weights(wpath)

    return []


def _resolve_xml(path):
    """Resolve an XML path relative to PROJECT_DIR if not absolute."""
    if not os.path.isabs(path):
        return os.path.join(PROJECT_DIR, path)
    return path


# Standalone execution entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Compare mode: run two XML configs and plot per-digit accuracy side by side
        xml1 = _resolve_xml(sys.argv[2])
        xml2 = _resolve_xml(sys.argv[3])
        TheReport.add_xml(xml1)
        TheReport.add_xml(xml2)
        results = BasicModelFactory(xml1) + BasicModelFactory(xml2)
        if len(results) >= 2:
            plotComparison([(name, rc) for name, rc, _ in results])
            plotCombinedAccuracy([(name, rc) for name, rc, _ in results])
            plotCombinedLoss([m for _, _, m in results])
    else:
        # Single run mode
        xml = _resolve_xml(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(PROJECT_DIR, "data", "xor.xml")
        TheReport.add_xml(xml)
        BasicModelFactory(xml)

    TheReport.write_html()
