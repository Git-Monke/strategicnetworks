import numpy as np
np.random.seed(seed=1000)


def ReLU(x):
    return x if x > 0 else 0


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
            cNodeCount = len(self.nodes[i])
            nNodeCount = len(self.nodes[i + 1])

            self.weights.append([[0]*nNodeCount]*cNodeCount)
            self.biases.append([[0]*nNodeCount]*cNodeCount)

    @property
    def outputs(self):
        return self.nodes[len(self.nodes) - 1]

    @property
    def inputs(self):
        return self.nodes[0]

    @inputs.setter
    def inputs(self, value):
        self.nodes[0] = value

    def calculate(self):
        for i, currentLayer in enumerate(self.nodes[1:]):
            layerWeights = self.weights[i]
            layerBiases = self.biases[i]
            previousLayer = self.nodes[i]

            for j in range(0, len(currentLayer)):
                sum = 0

                for k, node in enumerate(previousLayer):
                    sum += node * layerWeights[k][j] + layerBiases[k][j]

                currentLayer[j] = ReLU(sum)

    def randomizeConnections(self, connections):
        for layer in connections:
            layerSize = len(layer[0])

            for i in range(len(layer)):
                layer[i] = np.random.random(layerSize)
