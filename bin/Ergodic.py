import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import setuptools.dist
from torchviz import make_dot
from matplotlib import pyplot as plt
from datasets import load_dataset
from sklearn.decomposition import PCA
import pandas as pd
from vector_quantize_pytorch import VectorQuantize
from Model import message, SigmaLayer, ReversibleSigmaLayer
from Model import NormLayer, LinearLayer, AttentionLayer
from Model import GammaMem, CertaintyWeightedCrossEntropy, epsilon

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def output_path(filename):
    return os.path.join(ensure_output_dir(), filename)

def output_stem(stem):
    return os.path.join(ensure_output_dir(), stem)

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
        #print("MNIST mean: ", mnistMean)
        self.train_input = self.train_input - mnistMean
        mnistSTD = torch.std(self.train_input)
        #print("MNIST std: ", mnistSTD)
        self.train_input = self.train_input / mnistSTD
        self.train_output = torch.zeros((train.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(train[:, 0]):
            self.train_output[i][ndx:ndx+1] = 1.0
        self.test_input  = torch.tensor(test[:, 1:]/255.0, dtype=torch.float)
        self.test_input  = (self.test_input -mnistMean)/ mnistSTD
        self.test_output = torch.zeros((test.shape[0],10), dtype=torch.float)
        for i, ndx in enumerate(test[:, 0]):
            self.test_output[i][ndx:ndx+1] = 1.0
        #self.test_labels = self.test_labels.unsqueeze(0)
        self.validation_input  = torch.tensor(test[:, 1:]/255.0, dtype=torch.float)
        #self.validation_input = self.validation_input - torch.mean(self.validation_input, dim=0)
        #self.validation_input = self.validation_input / torch.var(self.validation_input, dim=0)
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
            self.shuffle()

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

# A set of prototype vectors for each space
class VectorSet(nn.Module):
    nInput           = 0  # the number of inputs passed to the forward method
    nVectors         = 0  # the number of outputs expected from the forward method
    nDim             = 0  # the latent dimensionality
    vectors          = nn.Parameter(torch.randn([nVectors, nDim]))  # this is the actual matrix of vectors
    snapDistance     = 0.1  # vectors at or below this distance snap to their corresponding prototypes
    eta              = 0.9
    codebookAct      = GammaMem(nVectors)
    frozen           = []
    returnOnlyFrozen = False
    freezingTemp     = 0.25
    # linear = nn.Linear(10, 5)
    # tracker = ColumnUsageTracker(linear, freezeThreshold=0.01)

    #def getSize(self):
    #    return self.nVectors

    def create(self, nInput, nVectors, nDim, customVQ=False, signed=False):
        self.nInput    = nInput
        self.nVectors  = nVectors
        self.nDim      = nDim
        self.customVQ  = customVQ
        self.signed    = signed
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
                dim = self.nDim,
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
            vec = torch.randn([nVec, self.nDim])
            self.vectors = vec[:, :]
    def forward(self, input):
        # X should be of size batchSize x nInput x nDim
        # Pad X if necessary
        x     = input
        batchSize = input.shape[0]
        act   = torch.zeros([batchSize, self.codebookSize])
        if self.customVQ:
            #if x.shape[-1] == self.nDim:
            #    x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize])], dim=2)
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
                        act[i, indices_smallest[i, j]] = cosSim # + temp * random.random()
                    else:
                        #message("codebook miss")
                        x[i, j, :] = quantized[i, indices_smallest[i, j], :]
                        #x[i, j, :] = torch.zeros( [1, 1, self.embeddingSize])
        else:
            dists = self.codebookDistance(x)
            #x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize])], dim=2)
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
        x: (batchSize, nDim) input vectors
        target_idx: (batchSize,) indices of the "correct" prototype
        lr: learning rate
        """
        x = F.normalize(x, p=2, dim=-1)
        selected_vectors = self.vectors[target_idx]  # (batchSize, nDim )

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
        #for i in range(0, new_vectors.shape[0]):
        #    new_vectors[i, :] = TheObjectEncoding(new_vectors[i, :])
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
        vec = self.vectors #[:, 0:-TheObjectEncoding.objectSize]
        # dist = self.angle(x.unsqueeze(2), vec.unsqueeze(0).unsqueeze(0))  # (batchSize, nInput, nFeatures)
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
# A PassThroughVSet does no vector quantization (vectors come directly from the input space)
class UnquantizedVSet(VectorSet):
    def create(self, nInput, nVectors, nDim):
        super().create(nInput, nVectors, nDim)
    def forward(self, x):
        y = x
        return y
    def reverse(self, y):
        x = y
        return x

class Space(nn.Module):
    name         = ""
    vectorSet    = []
    activation   = None
    params       = []
    layers       = []

    def __init__(self, inputShape, nVectors, nDim, nOutput, vSet=None, reversePass=False):
        super().__init__()
        self.inputShape   = inputShape
        self.outputShape  = [nVectors, nDim]
        self.nVectors     = nVectors
        self.nDim         = nDim # the latent dimensionality
        self.nOutput      = nOutput
        self.batchSize    = 0
        self.vectorSet    = []
        self.reversePass = reversePass
        if vSet is not None:
            self.vectorSet.append(vSet)
        #self.createVectorSet()

    def lookup(self, x):
        activation = x[0]
        x = x.unsqueeze(0).unsqueeze(0)
        #x = torch.cat([torch.zeros([1,1, TheObjectEncoding.conceptDim]), x[:,:,1:]], dim=2)
        output, index, _ = self.vectors().vq(x)
        #output[:,:,0:TheObjectEncoding.conceptDim] = output[:,:,0:TheObjectEncoding.conceptDim] * activation  # multiply the codebook vector by the activation
        return output

    def stats(self, x):
        #codebookUse = self.vectors().codebookUse
        #message(f"{self.name} Codebook activation: { np.sum(self.vectors().codebookAct.get()) }")
        return
    def vectors(self):
        # this is done to store by reference, since lists are mutable entities (?)
        return self.vectorSet[0]
    def createVectorSet(self, quantized=False):
        if quantized:
            self.vectorSet.append(VectorSet())
            self.vectors().create(self.inputShape[0], self.nVectors, self.nDim) # can be bigger than nVectors, cannot be smaller
            self.vectors().addVectors(nVec=self.nOutput)
        else:
            self.vectorSet.append(UnquantizedVSet()) # pass-through
            self.vectors().create(self.inputShape[0], self.nVectors, self.nDim)

    # self.vectors().create(self.inputShape[0], self.nVectors, self.nDim)
    # self.vectors().addVectors(nVec=LM.getSize(), LM=LM)

    def forwardBegin(self, x, reshape=False):
        self.batchSize = x.shape[0]
        if reshape:
            x = self.flatten(x, True)
            # assert list(x.shape) == [self.batchSize, self.inputShape[0] *self.inputShape[1]]
        else:
            assert list(x.shape) == [self.batchSize, self.inputShape[0], self.inputShape[1]]
        return x
    def forwardEnd(self, x, reshape=False):
        if reshape:
            x = self.reshape(x, True)
        else:
        #    if x.shape[-1] == self.nDim: # zero-pad vectors without a position.
        #        # x = TheObjectEncoding.forward(x)
        #        x = torch.cat([x, torch.zeros([x.shape[0], x.shape[1], TheObjectEncoding.objectSize])], dim=2)
            assert list(x.shape)==[self.batchSize, self.outputShape[0], self.outputShape[1]]
        return x
    def reverseBegin(self, y, reshape=False):
        self.batchSize = y.shape[0]
        if reshape:
            y = self.flatten(y, False)
        else:
            assert list(y.shape) == [self.batchSize, self.outputShape[0], self.outputShape[1]]
        return y
    def reverseEnd(self, y, reshape=False):
        if reshape:
            y = self.reshape(y, False)
        else:
            input, _ = self.getEmbeddedIO()
            assert list(y.shape) == [self.batchSize, self.inputShape[0], input]
        return y
    def flatten(self, x, forward=True):
        if forward:
            x = x.reshape(self.batchSize, self.inputShape[0] * self.inputShape[1])
        else:
            x = x.reshape(self.batchSize, self.outputShape[0] * self.outputShape[1])
        return x
    def reshape(self, y, forward=True):
        if forward:
            y = y.reshape(self.batchSize, self.outputShape[0], self.outputShape[1])
        else:
            y = y.reshape(self.batchSize, self.inputShape[0], self.inputShape[1])
        return y
    def getParameters(self):
        return self.params
    def paramUpdate(self):
        for l in self.layers:
            l.paramUpdate()
class InputSpace(Space):
    name = "Inputs"

    def __init__(self, inputShape, nVectors, nDim, nOutput, vSet=None, tokenizedInput=False):
        super().__init__(inputShape, nVectors, nDim, nOutput, vSet)
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.input          = torch.FloatTensor
        self.tokenizedInput = tokenizedInput
        #fullSize  = outputShape[0]*outputShape[1]
        self.createVectorSet()
    # The world presenting itself
    def forward(self, input):
        self.batchSize = input.shape[0]
        assert list(input.shape) == [self.batchSize, self.inputShape[0], self.inputShape[1]]
        self.input = self.vectors().forward(input)
        #self.input = self.lift.forward(self.input)
        assert list(self.input.shape) == [self.batchSize, self.outputShape[0], self.outputShape[1]]
        return self.input
    def reverse(self, y):
        y = self.reverseBegin(y)
        #y = self.lift.reverse(y)
        #x = x.unsqueeze(0).unsqueeze(0)
        #x = torch.cat([torch.zeros([1, 1, TheObjectEncoding.conceptDim]), x[:, :, 1:]], dim=2)
        #output[:, :, 0:TheObjectEncoding.conceptDim] = output[:, :, 0:TheObjectEncoding.conceptDim] * activation  # multiply the codebook vector by the activation
        self.input = self.vectors().reverse(y)
        #where, when = TheObjectEncoding.reverse(self.percepts)
        #self.input = self.reverseEnd(self.input, t)
        return self.input
class ConceptualSpace(Space):
    name = "Concepts"
    hasAttention = False
    hasNorm      = False

    def __init__(self, inputShape, nVectors, nDim, nOutput, vSet=None, reversePass=False, invertible=False, hasNorm=False):
        super().__init__(inputShape, nVectors, nDim, nOutput, vSet=vSet, reversePass=reversePass)
        input, output   = self.inputShape[0]*self.inputShape[1], self.outputShape[0]*self.outputShape[1]
        self.invertible = invertible
        self.hasNorm    = hasNorm
        if self.hasAttention:
            self.attention  = AttentionLayer(self.outputShape[1], self.outputShape[1])
        if hasNorm:
            self.norm = NormLayer(input, input+2)
            input += 2
        if reversePass:
            if self.invertible:
                self.sigma   = ReversibleSigmaLayer(input, output, permuteInput=False)
                self.forwardSigma, self.reverseSigma = self.sigma.forward, self.sigma.reverse
                self.params  = self.sigma.getParameters()
                self.layers += [self.sigma]
            else:
                self.sigma1  = SigmaLayer(input, output, permuteInput=False)
                self.sigma2  = SigmaLayer(output, input, permuteInput=False)
                self.forwardSigma, self.reverseSigma = self.sigma1.forward, self.sigma2.forward
                self.params  = self.sigma1.getParameters() + self.sigma2.getParameters()
                self.layers += [self.sigma1, self.sigma2]
        else:
            self.sigma   = SigmaLayer(input, output, permuteInput=False)
            self.forwardSigma = self.sigma.forward
            self.params  = self.sigma.getParameters()
            self.layers += [self.sigma]
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
    def forward(self, x):
        x = self.forwardBegin(x, reshape=True)
        if self.hasNorm:
            x = self.norm.forward(x)
        y = self.forwardSigma(x) # Pass through SigmaLayer
        if self.hasAttention:
            y = self.attention.forward(y)
        # Get the concept vectors from the codebook
        # replace some of the Dynamic Percepts with Static Percepts if their distance is low
        y = self.vectors().forward(y) # This must be 4x8x24
        # Reshape the output tensor
        self.concepts = self.forwardEnd(y, reshape=True)
        return self.concepts
    # Visualizing
    def reverse(self, y):
        self.concepts = self.reverseBegin(y, reshape=True)
        if self.hasAttention:
            self.concepts = self.attention.reverse(self.concepts)
        self.concepts = self.reverseSigma(self.concepts)
        if self.hasNorm:
            self.concepts = self.norm.reverse(self.concepts)
        self.concepts = self.reverseEnd(self.concepts, reshape=True)
        return self.concepts
    @staticmethod
    def test():
        pass
class OutputSpace(Space):
    name = "Outputs"
    def __init__(self, inputShape, nVectors, nDim, nOutput, vSet=None, reversePass=False):
        super(OutputSpace, self).__init__(inputShape, nVectors, nDim, nOutput, vSet=vSet)
        #input, output =self.inputShape[1], self.outputShape[1]
        # the output is reshaped, so we can't use the above formula
        input  = self.inputShape[0]  * self.inputShape[1]
        output = self.outputShape[0] * self.outputShape[1]

        # output is 0 of reshaped and not embedded
        if reversePass:
            self.linear1 = LinearLayer(input, output)
            self.linear2 = LinearLayer(output, input)
            self.forwardLinear, self.reverseLinear = self.linear1.forward, self.linear2.forward
            #self.linear = ReversibleLinearLayer(input, output)
            #self.forwardLinear, self.reverseLinear = self.linear.forward, self.linear.reverse
            self.params = self.linear1.getParameters() + self.linear2.getParameters()
            self.layers += [self.linear1, self.linear2]
        else:
            self.linear = LinearLayer(input, output)
            self.forwardLinear = self.linear.forward
            self.params = self.linear.getParameters()
            self.layers += [self.linear]
        self.createVectorSet()
    # Acting
    def forward(self, x):
        y = super().forwardBegin(x, reshape=True)
        # input is batchSize x nConcepts
        output = self.forwardLinear(y)
        output = self.forwardEnd(output, reshape=True)
        #self.output  = self.vectors().output(self.percepts)
        return output
    # Being acted upon
    def reverse(self, y):
        y = self.reverseBegin(y, reshape=True)
        #assert list(y.shape) == [self.batchSize, self.outputShape[0], self.outputShape[1]]
        y = self.reverseLinear(y)
        output = self.reverseEnd(y, reshape=True)
        return output

class BaseModel(nn.Module):
    name = "BaseModel"
    spaces = []
    reversePass = False
    vSet        = None
    invertible  = False
    hasNorm     = False

    def create(self, nInput=32, nConcepts=20, nOutput=1):
        self.nInput    = nInput
        self.nConcepts = nConcepts
        self.nOutput   = nOutput
    def runTrials(self, numTrials=1, numEpochs=1, batchSize=10, lr=0.001):
        acc = np.zeros([numTrials, numEpochs])
        print(f"\n\n==== {self.name} ====")
        for trial in range(numTrials):
            print(f"\nTrial [{trial + 1}/{numTrials}]")
            self.create(nInput=self.nInput, nConcepts=self.nConcepts, nOutput=self.nOutput)
            acc[trial, :] = self.run(numEpochs=numEpochs, batchSize=batchSize, lr=lr)
        np.savetxt(output_path(f"{self.name}.csv"), np.array(acc), delimiter=",")
        return acc
    def run(self, numEpochs=1, batchSize=10, lr=0.001, stoppingCriterion=0.1):
        """
        Runs the Transformer model.
        """
        trainLosses       = [[],[]]
        validationLosses  = [[],[]]
        minValidationLoss = math.inf
        testLosses        = [[],[]]
        self.plot         = True
        accuracy          = []
        for epoch in range(numEpochs):
            print(f"Epoch [{epoch + 1}/{numEpochs}]")
            if epoch != 0: # Don't train on the first epoch
                outErr,inErr,allOut,lastIn = self.runEpoch(TheData.train_input, TheData.train_output, lr=lr, batchSize=batchSize)
                trainLosses[0].append(outErr)
                trainLosses[1].append(inErr)
                print(f"Train Loss: {outErr:.4f},  {inErr:.4f}")
            #outErr,inErr,_,_ = self.runTest(TheData.validation_input, TheData.validation_output)
            #validationLosses[0].append(outErr)
            #validationLosses[1].append(inErr)
            #print(f"Validation Loss: {outErr:.4f},  {inErr:.4f}")

            outErr,inErr,allOut,lastIn = self.runEpoch(TheData.test_input, TheData.test_output, lr=0, batchSize=batchSize)
            testLosses[0].append(outErr)
            testLosses[1].append(inErr)
            #print(f"Test Loss: {outErr:.4f},  {inErr:.4f}")

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

        # Plot the loss over time
        if self.plot:
            print(f"Final Stats:")
            self.plotLoss(trainLosses, validationLosses, testLosses)
            #self.plotNetwork()
            self.mnistReport()

        return accuracy
    def paramUpdate(self):
        for s in self.spaces:
            s.paramUpdate()
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

        plt.figure(figsize=(10, 5))
        plt.plot(range(0, 10), rCorrect, label="Error (per Input)", marker='o')
        plt.xlabel("Digit")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy per Digit: {model.name}")
        plt.legend()
        plt.grid(True)
        # Save the plot as a high-resolution PNG (300 DPI)
        filename = output_path(model.name + "Accuracy.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show(block=False)
    def plotLoss(model, trainErr, valErr, testErr):
        """
        Plots the training, validation, and test losses over time.
        """
        plt.figure(figsize=(10, 5))

        plt.plot(range(1, len(trainErr[0]) + 1), trainErr[0], label="Training Error", marker='o')
        #plt.plot(range(1, len(trainErr[1]) + 1), trainErr[1], label="Training Error (Input)", marker='o')

        if testErr:
            plt.plot(range(1, len(testErr[0]) + 1), testErr[0], label="Test Error", marker='x')
            #plt.plot(range(1, len(testErr[1]) + 1), testErr[1], label="Test Error (Input)", marker='x')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Error per Epoch: {model.name}")
        plt.legend()
        plt.grid(True)

        # Save the plot as a high-resolution PNG (300 DPI)
        filename = output_path(model.name + "Error.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show(block=False)
    def plotActivations(model, figure=1, percepts=None, concepts=None, symbols=None):

        # Plot training loss and some activation norms.
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
        """
        Visualizes only the learned weight parameters.
        - For the perceptual layer, plots the 64 prototype weight vectors (each 64-d)
          projected via PCA to 2D.
        - For the conceptual mapping (fc_p of the ConceptualLayer, shape (16,64)),
          uses the same PCA transform and, for each of the 16 weight vectors,
          draws a hyperplane in 2D. Each hyperplane is taken to be the line orthogonal
          to the projected weight vector and passing through the mean of the projected
          perceptual prototypes.
        - For the symbolic branch, computes the L2 norm of each of its 8 weight vectors (from fc_symbolic)
          and displays a lollipop plot.
        """
        # ----- Perceptual prototypes -----
        # prototypes: (64, 64)
        perc_weights = model.prototypes.data.cpu().numpy()  # shape (64,64)
        pca = PCA(n_components=2)
        perc_2d = pca.fit_transform(perc_weights)  # (64,2)

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(perc_2d[:, 0], perc_2d[:, 1], c='blue', label="Perceptual Prototypes")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Perceptual Space Prototypes (64 points)")
        plt.legend()

        # Compute the mean of the projected prototypes.
        perc_mean = perc_2d.mean(axis=0)

        # ----- Conceptual hyperplanes -----
        # Use the fc_p weights from the ConceptualSpace (shape (16,64))
        conc_weights = model.conceptual.fc_p.weight.data.cpu().numpy()  # (16,64)
        # Project these 16 weight vectors using the same PCA (note: they are in the same space as prototypes).
        conc_proj = pca.transform(conc_weights)  # (16,2)

        plt.subplot(1, 3, 2)
        plt.scatter(perc_2d[:, 0], perc_2d[:, 1], c='blue', label="Perceptual Prototypes")
        # For each conceptual weight vector, draw its corresponding hyperplane.
        x_vals = np.linspace(np.min(perc_2d[:, 0]) - 1, np.max(perc_2d[:, 0]) + 1, 100)
        for i in range(conc_proj.shape[0]):
            n = conc_proj[i]  # this is the projected weight vector (treated as normal)
            # Define hyperplane: n . (x - perc_mean) = 0, i.e.
            # n0*(x - m0) + n1*(y - m1) = 0  => y = m1 - (n0/n1)*(x - m0), provided n1 != 0.
            if np.abs(n[1]) > 1e-3:
                y_vals = perc_mean[1] - (n[0] / n[1]) * (x_vals - perc_mean[0])
                plt.plot(x_vals, y_vals, '--', label=f"Hyperplane {i + 1}" if i == 0 else None, alpha=0.7)
            else:
                # Vertical line at x = perc_mean[0]
                plt.axvline(x=perc_mean[0], linestyle='--', alpha=0.7, label=f"Hyperplane {i + 1}" if i == 0 else None)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Conceptual Hyperplanes (16 lines)")
        plt.legend()

        # ----- Symbolic branch: Lollipop plot -----
        # fc_symbolic: Linear(784,8) has weight of shape (8,784)
        symb_weights = model.fc_symbolic.weight.data.cpu().numpy()  # (8,784)
        symb_norms = np.linalg.norm(symb_weights, axis=1)  # (8,)
        x_symb = np.arange(8)

        plt.subplot(1, 3, 3)
        plt.vlines(x_symb, 0, symb_norms, color='k', alpha=0.7)
        plt.scatter(x_symb, symb_norms, color='red', s=100, zorder=3)
        plt.xlabel("Symbolic Feature Index")
        plt.ylabel("L2 Norm")
        plt.title("Symbolic Weights (Lollipop Plot)")

        plt.tight_layout()
        plt.show(block=False)
    def plotNetwork(model):
        """
        Uses Torchviz to create a visualization of the network's computation graph.
        The visualization is saved as 'BasicModel_graph.png'.
        """
        #dummy_input = torch.randn(1, 28, 28)
        _, _, output, input = model.runEpoch(TheData.test_input, TheData.test_output, lr=0)
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = "png"
        graph_path = dot.render(output_stem(f"graph_{model.name}"))
        print(f"Saved network graph as {graph_path}")
    def plotErrorbars(model, acc):
        x = list(range(1, 10))
        y = np.array(np.mean(acc, axis=1))
        y = np.expand_dims(y, axis=1)
        y_err = np.std(acc - y, axis=1)
        plt.errorbar(x, np.squeeze(y), yerr=y_err, fmt='-o', label=model.name, capsize=4)
class SimpleModel(BaseModel):
    name = "SimpleModel"

    def create(self, nInput=32, nConcepts=20, nOutput=1):
        super().create(nInput, nConcepts, nOutput)
        self.hidden = nn.Linear(self.nInput, self.nConcepts)  # First layer
        self.relu   = nn.Tanh() #nn.ReLU()  # Activation
        self.output = nn.Linear(self.nConcepts, self.nOutput)  # Output layer (linear)
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)  # Linear output (no activation)
        return x
    def runEpoch(self, input, output, lr=0.01, batchSize=10):
        if lr:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() # MSELoss()

        allOutput = []
        allInput  = []
        outErr    = 0
        inErr     = 0
        self.train(lr!=0)
        for i in range(0, len(input), batchSize):
            inputBatch  = input[i:i + batchSize]
            outputBatch = output[i:i + batchSize]
            batchSize   = len(inputBatch)

            # Combine the padded tensors
            inputTensor  = inputBatch
            outputTensor = outputBatch

            # First run forward
            if lr:
                optimizer.zero_grad()
            outputPred  = self.forward(inputTensor)
            lossOut  = criterion(outputPred, outputTensor)
            if lr:
                lossOut.backward()
                optimizer.step()
            outErr   = lossOut.item()
            out = outputPred.clone().detach()
            if i==0:
                allOutput = out
            else:
                allOutput = torch.concat((allOutput, out), dim=0)
        return outErr, inErr, allOutput, allInput
class ErgodicModel(BaseModel):
    name   = "ErgodicModel"

    def create(self, nInput=32, nConcepts=256, nOutput=32):
        super().create(nInput, nConcepts, nOutput)

        inputDim   = 1
        conceptDim = 1
        outputDim  = 1

        self.inputSpace = InputSpace([self.nInput, inputDim],
                                           self.nInput, inputDim,
                                           nOutput=self.nInput, vSet=self.vSet)
        self.conceptualSpace = ConceptualSpace([self.nInput, inputDim],
                                               self.nConcepts, conceptDim,
                                               nOutput=self.nConcepts,
                                               reversePass=self.reversePass,
                                               invertible=self.invertible,
                                               hasNorm=self.hasNorm)
        # The input dimensionality of the output layer must be equal to the sum of the output dimensionalities of the symbolic layers.
        self.outputSpace = OutputSpace([ self.nConcepts, conceptDim],
                                           self.nOutput, outputDim,
                                           nOutput = self.nOutput,
                                           reversePass = False)

        self.spaces += [self.inputSpace, self.conceptualSpace, self.outputSpace]
    def forward(self, data):
        percepts = self.inputSpace(data)
        concepts = self.conceptualSpace(percepts)
        symbols  = self.outputSpace(concepts)
        return symbols, concepts
    def reverse(self, concepts):
        percepts = self.conceptualSpace.reverse(concepts)
        data     = self.inputSpace.reverse(percepts)
        return data, percepts
    def getOptimizer(self, lr=0.01):
        params    = self.inputSpace.getParameters() + self.conceptualSpace.getParameters() + self.outputSpace.getParameters()
        #params    = list(filter([], params))
        optimizer = optim.Adam(params, lr=lr)  # For other params, if any
        return optimizer
    def runEpoch(self, input, output, lr=0.01, batchSize=10):
        if lr:
            optimizer = self.getOptimizer(lr=lr)
            #inputOptimizer = self.getOptimizer(lr=lr)

        criterionOutput = CertaintyWeightedCrossEntropy()
        #criterionOutput = CertaintyWeightedMSELoss()
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

            # Combine the padded tensors
            inputTensor  = inputBatch
            inputTensor  = inputTensor.unsqueeze(2)
            outputTensor = outputBatch
            outputTensor = outputTensor.unsqueeze(2)

            # First run forward
            if lr:
                optimizer.zero_grad()
            outputPred, concepts = self.forward(inputTensor)
            lossOut = criterionOutput(outputPred.squeeze(), outputTensor.squeeze())
            if lr:
                lossOut.backward()
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
                #lossIn = lossFnInput(eInput.squeeze().float(), inputTensor.squeeze().float())
                lossIn = criterionInput(inputPred.squeeze(), inputTensor.squeeze())
                if lr:
                    lossIn.backward()
                    self.paramUpdate()
                    optimizer.step()
                inErr   = lossIn.item()
                inputPred = inputPred.clone().detach().squeeze()
                # Can't save all input reconstructions
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

        # Compute the correlation matrix
        input_matrix = torch.stack((rCorrect, norms))
        correlation_matrix = torch.corrcoef(input_matrix)
        correlation_value = correlation_matrix[0, 1]
        print(f"Pearson Correlation: {correlation_value}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(0, 10), rCorrect, label="Error (per Input)", marker='o')
        plt.xlabel("Digit")
        plt.ylabel("Accuracy & Certainty")
        plt.title(f"Accuracy and Certainty: {model.name}")
        plt.legend()
        plt.grid(True)
        # Save the plot as a high-resolution PNG (300 DPI)
        filename = output_path(model.name + "Accuracy.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show(block=False)

        if model.reversePass:
            for i in range(0, 10):
                plt.figure(figsize=(10, 5))
                j = TheData.test_output[-i-1]
                _, num = torch.max(j, axis=0)
                plt.title(f"Reconstruction {num}: {model.name}")
                image = last_x_pred[9-i, :]
                image = np.reshape(image, (28, 28))
                plt.imshow(image)
                filename = output_path(model.name + f"_{num}_Reconstruction.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show(block=False)

def plotErrorbars(fn):
    x = list(range(1, 10))
    acc = np.loadtxt(output_path(f"{fn}.csv"), delimiter=",")
    #acc = acc.T
    y = np.array(np.mean(acc, axis=1))
    y = np.expand_dims(y, axis=1)
    y_err = np.std(acc - y, axis=1)
    plt.errorbar(x, np.squeeze(y), yerr=y_err, fmt='-o', label=fn, capsize=4)

def plotComparison():
    # Plot accuracies over all trials
    plt.figure(figsize=(10, 5))

    plotErrorbars("SimpleModel")
    plotErrorbars("ErgodicModel")
    plotErrorbars("Ergodic - Normed")
    plotErrorbars("Ergodic - Reversible")

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Model Comparison")
    plt.legend()

    # Display the plot
    plt.grid(True)
    filename = output_path("ModelComparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show(block=False)

def ErgodicModelFactory(config_path):
    """Create, train, and evaluate ergodic models from an XML config file.

    Reads architecture and training parameters from the given XML,
    loads the dataset, and runs each enabled model variant
    (traditional, ergodic, normed, reverse, invert).
    """
    from BasicModel import BasicModel
    cfg = BasicModel.load_config(config_path)
    arch = cfg.get("architecture", {})
    train = cfg.get("training", {})

    dataset = train.get("dataset", "mnist")
    TheData.load(dataset)

    nInput = TheData.getInputSize()
    nConcepts = arch.get("nConcepts", 20)
    nOutput = TheData.getOutputSize()

    numTrials = train.get("numTrials", 1)
    numEpochs = train.get("numEpochs", 3)
    batchSize = train.get("batchSize", 10)

    if train.get("traditional", False):
        m = SimpleModel()
        m.create(nInput=nInput, nConcepts=nConcepts, nOutput=nOutput)
        m.name = "SimpleModel"
        m.runTrials(numTrials, numEpochs, batchSize)

    if train.get("ergodic", False):
        m = ErgodicModel()
        m.create(nInput=nInput, nConcepts=nConcepts, nOutput=nOutput)
        m.name = "ErgodicModel"
        m.runTrials(numTrials, numEpochs, batchSize)

    if train.get("normed", False):
        m = ErgodicModel()
        m.create(nInput=nInput, nConcepts=nConcepts, nOutput=nOutput)
        m.hasNorm = True
        m.name = "Ergodic - Normed"
        m.runTrials(numTrials, numEpochs, batchSize)

    if train.get("reverse", False):
        m = ErgodicModel()
        m.create(nInput=nInput, nConcepts=nConcepts, nOutput=nOutput)
        m.reversePass = True
        m.name = "Ergodic - Reversible"
        m.runTrials(numTrials, numEpochs, batchSize)

    if train.get("invert", False):
        m = ErgodicModel()
        m.create(nInput=nInput, nConcepts=nConcepts, nOutput=nOutput)
        m.reversePass = True
        m.invertible = True
        m.name = "Ergodic - Invertible"
        m.runTrials(numTrials, numEpochs, batchSize)


if __name__ == "__main__":
    import sys
    xml = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_DIR, "data", "ergodic.xml")
    if not os.path.isabs(xml):
        xml = os.path.join(PROJECT_DIR, xml)
    ErgodicModelFactory(xml)
