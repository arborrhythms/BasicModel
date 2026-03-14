"""Standalone NumPy neural-network experiment for simple logic problems.

This file predates the PyTorch-based ``BasicModel`` stack and keeps a compact
implementation around for experimentation with activation functions,
bidirectional updates, and XOR-style toy datasets.
"""

# Final project for Machine Learning
import math
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from visualize import TheReport

class SPNN:
    """Small fully-connected network with optional bidirectional learning."""

#region Initialization
    def __init__(self, transfer="sigmoid", bidirectional=False):

        self.name  = ""
        self.debug = False

        # Network topology
        self.nInput  = 2
        self.nHidden = 2
        self.nOutput = 1

        # Training and Testing data
        self.trainInput  = None
        self.trainOutput = None
        self.testInput   = None
        self.testOutput  = None

        # Epoch, learning rate (eta), and momentum values
        self.nEpoch  = 1000
        self.batch   = 100
        self.lr      = 0.05
        self.mom     = 0.1
        self.regular = 0.0001

        # Weight and partial error matrices
        self.M1      = None
        self.W1      = None
        self.deltaW1 = None
        self.M2      = None
        self.W2      = None
        self.deltaW2 = None

        # The type of activation function can be signum, sigmoid, sin, tanh, or linear
        self.transfer      = transfer
        self.bidirectional = bidirectional

        # MSE
        self.trainInput  = np.empty((0,3))
        self.trainOutput = np.empty((0,2))
        self.testInput   = np.empty((0,3))
        self.testOutput  = np.empty((0,2))
        self.testErr     = np.empty((0))
        self.trainErr    = np.empty((0))

        self.loadXOR()
        self.reinit()
    def reinit(self):
        # Weights include an extra column so the network can carry a learned
        # bias-like term alongside the hidden/output activations.
        self.M1 = np.zeros(self.nInput)
        self.M2 = np.zeros(self.nHidden)
        self.M3 = np.zeros(self.nOutput)
        self.W1 = 0.2 * np.random.randn(self.nInput + 1, self.nHidden+1)
        self.W2 = 0.2 * np.random.randn(self.nHidden + 1, self.nOutput+1)
        self.deltaW1  = np.zeros_like(self.W1)
        self.deltaW2  = np.zeros_like(self.W2)
        self.testErr  = []
        self.trainErr = []
    def loadOR(self):
        """Populate the repeated OR truth table used for training and testing."""
        self.name        = "OR"
        input_data       = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        target           = np.array([[1, 0], [1, 1], [1, 1], [1, 1]])
        for i in range(0,self.batch*4):
            self.trainInput  = np.concatenate( (self.trainInput, [input_data[i % 4]]) )
            self.trainOutput = np.concatenate( (self.trainOutput, [target[i % 4]]) )
        self.testInput   = input_data
        self.testOutput  = target
    def loadXOR(self):
        """Populate the repeated XOR truth table used for training and testing."""
        self.name        = "XOR"
        input_data       = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        target           = np.array([[1, 0], [1, 1], [1, 1], [1, 0]])
        for i in range(0, self.batch*4):
            self.trainInput  = np.concatenate( (self.trainInput,  [input_data[i % 4]]) )
            self.trainOutput = np.concatenate( (self.trainOutput, [target[i % 4]]) )
        self.testInput   = input_data
        self.testOutput  = target
#endregion

#region Activation Functions
    def activation(self, x, inverse=False):
        if self.transfer == 'signum':
            x = SPNN.actSignum(x)
        elif self.transfer == 'sigmoid':
            x = SPNN.actSigmoid(x)
        elif self.transfer == 'asinh':
            x = SPNN.actASinh(x)
        elif self.transfer == 'sinh':
            x = SPNN.actSinh(x)
        elif self.transfer == 'sin':
            x = SPNN.actSin(x)
        elif self.transfer == 'cosh':
            x = SPNN.actCosh(x)
        elif self.transfer == 'tanh':
            x = SPNN.actTanh(x)
        elif self.transfer == 'atanh':
            x = SPNN.actAtanh(x)
        elif self.transfer == 'linear':
            x = SPNN.actLinear(x, inverse)
        # The inverse of a function may be periodic, in which case the following formula will not work.
        # for example, we may wish the inverse of a piecewise linear function to be defined everywhere.
        # we might also need to handle extreme values better
        return x
    def gradient(self, x, inverse=False):
        if self.transfer == 'signum':
            x = SPNN.gradSignum(x)
        elif self.transfer == 'sigmoid':
            x = SPNN.gradSigmoid(x)
        elif self.transfer == 'asinh':
            x = SPNN.gradASinh(x)
        elif self.transfer == 'sinh':
            x = SPNN.gradSinh(x)
        elif self.transfer == 'sin':
            x = SPNN.gradSin(x)
        elif self.transfer == 'cosh':
            x = SPNN.gradCosh(x)
        elif self.transfer == 'tanh':
            x = SPNN.gradTanh(x)
        elif self.transfer == 'atanh':
            x = SPNN.gradAtanh(x)
        elif self.transfer == 'linear':
            x = SPNN.gradLinear(x, inverse)
        else:
            raise Exception(f"badness")
        return x
    @staticmethod
    def actSignum(x):
        y = np.zeros_like(x)
        y[x > 0] = 1
        return y
    @staticmethod
    def gradSignum(x):
        return np.ones_like(x)
    @staticmethod
    def actSigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def gradSigmoid(x):
        return x * (1 - x)
    @staticmethod
    def actTanh(x, inverse=False):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            if not inverse:
                y[i] = math.sinh(x[i]) / math.cosh(x[i])
            else:
                y[i] = math.cosh(x[i]) / math.sinh(x[i])
        return y
    @staticmethod
    def gradTanh(x, inverse=False):
        if not inverse:
            return 1 - x ** 2
        else:
            return 1 / (1 - x ** 2)
    @staticmethod
    def actAtanh(x, inverse=False):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            if not inverse:
                y[i] = math.cosh(x[i]) / math.sinh(x[i])
            else:
                y[i] = math.sinh(x[i]) / math.cosh(x[i])
        return y
    @staticmethod
    def gradAtanh(x, inverse=False):
        if not inverse:
            return 1 / (1 - x ** 2)
        else:
            return 1 - x ** 2
    @staticmethod
    def actSinh(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = x[i] / (2*math.pi)
            y[i] = math.sinh(x[i])
        return y
    @staticmethod
    def gradSinh(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = x[i] / (2*math.pi)
            y[i] = math.cosh(x[i])
        return y
    @staticmethod
    def actSin(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = x[i] / (2*math.pi)
            y[i] = math.sin(x[i])
        return y
    @staticmethod
    def gradSin(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = x[i] / (2*math.pi)
            y[i] = math.cos(x[i])
        return y
    @staticmethod
    def actASinh(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = x[i] / (2*math.pi)
            y[i] = -math.asinh(x[i])
        return y
    @staticmethod
    def gradASinh(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = x[i] / (2*math.pi)
            #y[i] = 1/math.cosh(math.asin(x[i]))
            y[i] = -1 / ( math.sqrt(1 - min(0.99, x[i]**2) ) )
        return y
    @staticmethod
    def actCosh(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = 2*x[i]/math.pi
            y[i] = math.cosh(x[i])
        return y
    @staticmethod
    def gradCosh(x):
        y = np.zeros_like(x)
        for i in range(0, len(x)):
            x[i] = 2*x[i]/math.pi
            y[i] = math.sinh(x[i])
        return y
    @staticmethod
    def actLinear(x, inverse=False):
        y = abs(x)
        if inverse:
            y = -y
        return y
    @staticmethod
    def gradLinear(x, inverse=False):
        y = np.zeros_like(x)
        if not inverse:
            y[x >= 0] = 1
        else:
            y[x >= 0] = 1
        return y
    @staticmethod
    # Find a point on the hyperplane
    def pointOnHyperplane(w):
        # zeroth coordinate is bias
        for i in range(1, len(w)):
            if w[i] != 0:
                pt = np.zeros((len(w) - 1))
                pt[i - 1] = -w[0] / w[i]
                break
        return pt
#endregion

#region Train the Network
    def run(self):
        """Train for ``nEpoch`` epochs while tracking train/test reconstruction error."""
        for epoch in range(0, self.nEpoch + 1):
            trainErr = self.runEpoch(self.trainInput, self.trainOutput)
            testErr  = self.runEpoch(self.testInput, self.testOutput, True)
            if self.debug:
                print(f"Epoch {epoch}\n\tMSE={testErr:.4f}")
            self.trainErr.append(trainErr)
            self.testErr.append(testErr)
    def runEpoch(self, input_data, output_data, learn=False):
        """Evaluate one full pass and optionally apply weight updates per sample."""
        nTrials = len(input_data)
        err = 0
        for i in range(0, nTrials):
            (x, act, y) = self.compute(input_data[i,:], output_data[i,:])
            if learn:
                self.update(input_data[i], x, act, y, output_data[i])
            err += np.linalg.norm(y - output_data[i,:])
        mse = err/nTrials
        return mse
    def compute(self, input, output):
        # Unidirectional mode behaves like a conventional MLP; bidirectional
        # mode also attempts an inverse pass from desired outputs back to inputs.
        if self.bidirectional == False:
            x    = np.zeros_like(self.nInput)
            act  = self.activation( np.dot(input, self.W1) )
            #act[0] = 1
            y    = self.activation( np.dot(act, self.W2) )
        else:
            hidden1 = self.activation(np.dot(input, self.W1), inverse=False)
            y       = self.activation(np.dot(hidden1, self.W2), inverse=False)
            hidden2 = self.activation(np.dot(self.W2, output), inverse=True)
            x       = self.activation(np.dot(self.W1, hidden2), inverse=True)
            act     = [hidden1 , hidden2] # combine
        return (x, act, y)
    def update(self, input, x, act, y, output):
        def wta(mean, input, w, act, target, winnerTakeAll=True):
            def project(w, x):
                # Create a difference vector
                pt = SPNN.pointOnHyperplane(w)
                diffVec = x - pt
                # Create a normalized perpendicular vector
                normal = w[1:] / np.linalg.norm(w[1:])
                # Project the difference onto a vector normal to the plane
                # proj = (np.dot(dv, normal) / np.dot(normal, normal)) * normal
                proj = np.dot(diffVec, normal) * normal
                y = x - proj
                return y

            # For each neuron, remove its prediction from the input vector
            # And let the other neurons predict
            point = input  # - mean[i]
            activations = act.copy()
            activations = np.sort(activations)
            # activations[::-1]
            for i in range(0, len(act)):
                a = activations[len(act) - 1 - i]
                # add bias
                pt = np.insert(point, 0, 1)
                # change the weights
                delta = self.gradLinear(a) * (target - a)
                dw = self.lr * (pt * delta)
                reg = 0.00001
                w[:, i] = w[:, i] + dw - reg * w[:, i]
                # weight normalization
                w[:, i] = w[:, i] / np.linalg.norm(w[:, i])

                if a >= 0:
                    output = project(w[:, i], point)
                else:
                    output = project(-w[:, i], point)
                point = output
                if winnerTakeAll:
                    break
            # return output
        def updateLayer(input, x, w, y, output):
            #dw = self.lr * np.outer(np.array(x-input), np.transpose(np.array(y-output)))
            deltaOut = -self.gradLinear(y) * (y-output)
            dw = self.lr * np.outer(input, deltaOut)
            w += dw
            #deltaIn = -self.gradLinear(x) * (x-input)
            #dw = self.lr * np.outer(deltaIn, output)
            #w += dw

        if self.bidirectional == False:
            errOut      = (y - output)
            deltaOut    = self.gradient(y) * errOut
            deltaHidden = self.gradient(act) * np.dot(self.W2, deltaOut)
            dW1 = self.lr * np.outer(input, -deltaHidden)
            dW2 = self.lr * np.outer(act, -deltaOut)
        else:
            dW1 = -0* self.W1
            dW2 = -0 * self.W2
            # Bidirectional mode blends a forward supervised update with a
            # reverse reconstruction update instead of standard backprop.
            errOut        = (y - output)
            deltaOut      = self.gradient(y, inverse=False)  * errOut
            deltaHidden1  = self.gradient(act[0], inverse=False) * np.dot(self.W2, deltaOut)
            dW1 += 0.5*self.lr * np.outer(input, -deltaHidden1)
            dW2 += 0.5*self.lr * np.outer(act[0], -deltaOut)
            errIn         = (x - input)
            deltaIn       = self.gradient(x, inverse=True) * errIn
            deltaHidden2  = self.gradient(act[1], inverse=True) * np.dot(deltaIn, self.W1)
            dW1 += 0.5*self.lr * np.outer(-deltaIn, act[1])
            dW2 += 0.5*self.lr * np.outer(-deltaHidden2, output)

        self.W1 += dW1 + self.mom * self.deltaW1 - self.regular*self.W1
        self.deltaW1 = dW1
        self.W2 += dW2 + self.mom * self.deltaW2 - self.regular*self.W2
        self.deltaW2 = dW2
#endregion

#region Plots and Statistics
    def show(self, ax=None):
        """Plot train/test error traces and print the learned weights."""
        titleText = 'MSE, '
        if self.bidirectional == False:
            titleText += 'UniDir '
        else:
            titleText += 'BiDir '
        titleText += self.transfer
        if ax is None:
            ax = plt.gca()
        ax.grid(True)
        ax.plot(self.trainErr, label='train')
        ax.plot(self.testErr, label='test')
        ax.set_title(titleText, fontsize=16, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=14, fontweight='bold')
        ax.legend(['train', 'test'])
        print('Weight_1', self.W1)
        print('Weight_2', self.W2)
    def showAct(self):
        """Compare the chosen activation with its inverse form and gradients."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        x = np.array([i/100 for i in range(-99,100)], float)

        ax1.plot(x, SPNN.actLinear(x), label='Activation')
        ax1.plot(x, SPNN.gradLinear(x), label='Gradient')
        ax1.legend()

        ax2.plot(x, SPNN.actLinear(x, inverse=True), label='Inverse Activation')
        ax2.plot(x, SPNN.gradLinear(x, inverse=True), label='Inverse Gradient')
        ax2.legend()

        TheReport.save_figure(fig, "SPNN Activation Functions")
        TheReport.show_figure(fig)
#endregion

#region Example Usage
if __name__ == "__main__":
    showActivations = True
    showPictures    = True

    if showActivations:
        n = SPNN()
        n.showAct()

    if showPictures:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        n = SPNN("linear", False)
        n.run()
        n.show(ax=ax1)

        n = SPNN("linear", True)
        n.run()
        n.show(ax=ax2)

        TheReport.save_figure(fig, "SPNN Training Comparison")
        TheReport.show_figure(fig)

    TheReport.write_html()
#endregion
