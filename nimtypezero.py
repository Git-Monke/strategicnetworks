import numpy as np
import random

BRAIN_STRUCTURE = [2, 3, 2]
BRAIN_LENGTH = len(BRAIN_STRUCTURE)

MUTATION_INTENSITY = 0.01


def ReLU(x):
    return x if x > 0 else 0


def randomList(len):
    l = []

    for _ in range(len):
        l.append(np.random.random())

    return l

# Takes in an array of length 9 and returns if there are any 3 in a rows;


def checkForWin(board):
    for i in range(0, 7, 3):
        if (board[i] == 0):
            continue

        if (board[i] == board[i + 1] and board[i] == board[i + 2]):
            return board[i]

    for i in range(3):
        if (board[i] == 0):
            continue

        if (board[i] == board[i + 3] and board[i] == board[i + 6]):
            return board[i]

    if (board[0] != 0 and board[0] == board[4] and board[0] == board[8]):
        return board[0]

    if (board[2] != 0 and board[2] == board[4] and board[2] == board[6]):
        return board[2]

    return 0

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


def mutateArr(arr, mutationIntensity=1):
    for i, value in enumerate(arr):
        if (isinstance(value, list)):
            mutateArr(value, mutationIntensity)
        elif isinstance(value, float):
            arr[i] = value + (((random.random() * 2) - 1) * mutationIntensity)


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

        if (shouldMutate):
            mutateArr(self.network.weights, MUTATION_INTENSITY)
            mutateArr(self.network.biases, MUTATION_INTENSITY)

    # Setters

    @property
    def weights(self):
        return self.network.weights

    @property
    def biases(self):
        return self.network.biases


class Board(object):
    def __init__(self):
        self.claimedBoards = [0]*9
        self.board = [[0]*9 for _ in range(9)]
        self.currentBoard = 0
        self.playerToMove = 1

    def makeMove(self, subsection=0, board=0):
        boardToPlay = board if self.currentBoard == -1 else self.currentBoard

        if (self.claimedBoards[board] != 0):
            raise Exception("Illegal move. Tried to play on a completed board")

        if (self.board[boardToPlay][subsection] != 0):
            raise Exception("Illegal move. Tried to play on occupied square.")

        self.board[boardToPlay][subsection] = self.playerToMove
        self.playerToMove *= -1

        playerWon = checkForWin(self.board[board])

        if (playerWon != 0):
            self.claimedBoards[board] = playerWon
            self.currentBoard = -1
        else:
            self.currentBoard = subsection if self.claimedBoards[subsection] == 0 else -1
