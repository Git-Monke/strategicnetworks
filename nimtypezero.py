import numpy as np
import random
import datetime as dt

# np.random.seed(430598)

BRAIN_STRUCTURE = [90, 12, 12, 12, 18]
BRAIN_LENGTH = len(BRAIN_STRUCTURE)


def ReLU(x):
    return x if x > 0 else 0


def sigmoid(x):
    return 1 / 1 + np.exp(-x)


def randomList(len):
    l = []

    for _ in range(len):
        l.append(np.random.random())

    return l

# Mixes the values of any two arrays. Will also mix nested arrays if they're at the same index.
# Ex: [[5, 10], 10] and [[20, 40], 3] might produce [[5, 40], 3] as an output


def mixArr(mother, father):
    child = []

    minLength = min(len(mother), len(father))
    difference = abs(len(mother) - len(father))

    for i in range(minLength):
        a = mother[i]
        b = father[i]

        if (isinstance(a, list) and isinstance(b, list)):
            child.append(mixArr(a, b))
        else:
            child.append(random.choice([a, b]))

    if difference > 0:
        longerParent = mother if len(mother) > len(father) else father

        child.extend(longerParent[minLength:])

    return child


class Network(object):
    def __init__(self, numLayers=2, layerNodeCounts=[]):
        self.nodes = []
        self.weights = []
        self.biases = []

        # Initialize all node layers
        for i in range(numLayers):
            nodeCount = 2

            if (i <= len(layerNodeCounts) - 1):
                nodeCount = layerNodeCounts[i]

            self.nodes.append([0] * nodeCount)

        # Initialize weight and bias connections
        for i in range(numLayers - 1):
            # current and next nodeCount
            cNodeCount = len(self.nodes[i])
            nNodeCount = len(self.nodes[i + 1])

            self.weights.append([[0]*nNodeCount]*cNodeCount)
            self.biases.append([[0]*nNodeCount]*cNodeCount)

    # Setters and getters
    @ property
    def outputs(self):
        return self.nodes[len(self.nodes) - 1]

    @ property
    def inputs(self):
        return self.nodes[0]

    @ inputs.setter
    def inputs(self, value):
        self.nodes[0] = value

    # The function used to update the neural networks values
    def propogate(self):
        for i, currentLayer in enumerate(self.nodes[1:]):
            layerWeights = self.weights[i]
            layerBiases = self.biases[i]
            previousLayer = self.nodes[i]

            for j in range(0, len(currentLayer)):
                sum = 0

                for k, node in enumerate(previousLayer):
                    sum += node * layerWeights[k][j] + layerBiases[k][j]

                currentLayer[j] = ReLU(sum)

    # Extra functions for making random networks
    def randomizeConnections(self, connections):
        for layer in connections:
            layerSize = len(layer[0])

            for i in range(len(layer)):
                layer[i] = randomList(layerSize)

    def randomize(self):
        self.randomizeConnections(self.weights)
        self.randomizeConnections(self.biases)


class Player(object):
    def __init__(self, parents=None, shouldMutate=False):
        self.network = Network(BRAIN_LENGTH, BRAIN_STRUCTURE)

        if (parents):
            self.network.weights = mixArr(
                parents[0].weights, parents[1].weights)
            self.network.biases = mixArr(parents[0].biases, parents[1].biases)
        else:
            self.network.randomize()

    # Setters
    @property
    def weights(self):
        return self.network.weights

    @property
    def biases(self):
        return self.network.biases
