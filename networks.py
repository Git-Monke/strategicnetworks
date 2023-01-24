import numpy as np
import time
import random
import profile

BRAIN_STRUCTURE = [99, 12, 12, 12, 18]
BRAIN_LENGTH = len(BRAIN_STRUCTURE)

MUTATION_INTENSITY = 0.01


def ReLU(x):
    return x if x > 0 else 0


def sigmoid(x):
    return 1 / 1 + (np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    sum = exp_x.sum()
    softmax_x = np.round(exp_x/sum, 3)
    return softmax_x


def filterProbs(probs, board):
    for i, subsection in enumerate(board):
        if (subsection in [-1, 1]):
            probs[i] = 0


def randomList(len):
    l = []

    for _ in range(len):
        l.append(np.random.random())

    return l


def getMaxNumIndx(arr):
    indx = 0
    best = 0

    for index, value in enumerate(arr):
        if value > best:
            indx = index
            best = value

    return indx

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

    if (0 not in board):
        p1Count = board.count(1)
        p2Count = board.count(-1)

        return 1 if p1Count > p2Count else -1

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


def multArr(arr1, arr2):
    result = []

    for x in range(len(arr1)):
        result.append(arr1[x] * arr2[x])

    return result


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
        return softmax(self.nodes[len(self.nodes) - 1])

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

                currentLayer[j] = sigmoid(sum)

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

    # Getters

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

        # This is what the networks whom are competing will look at for their input values.
        # It stores all the board values, and what the current selected board is.
        # The first 81 inputs are what colors occupy what positions (-1, 0, 1)
        # The next 9 represent which board is currently in play
        # The final 9 represent who won which boards
        self.inputArray = [0]*99

        self.currentBoard = -1
        self.playerToMove = 1

        self._player1 = None
        self._player2 = None

        self.winner = 0

    def makeMove(self, subsection=0, board=0):
        board = board if self.currentBoard == -1 else self.currentBoard

        if (self.claimedBoards[board] != 0):
            print(board, self.claimedBoards)
            raise Exception("Illegal move. Tried to play on a completed board")

        if (self.board[board][subsection] != 0):
            raise Exception("Illegal move. Tried to play on occupied square.")

        self.board[board][subsection] = self.playerToMove
        self.inputArray[(board * 9) + subsection] = self.playerToMove
        self.inputArray[80 + board] = 0
        self.playerToMove *= -1

        playerWon = checkForWin(self.board[board])

        if (playerWon != 0):
            self.claimedBoards[board] = playerWon
            self.inputArray[90 + board] = 1

        self.currentBoard = subsection if self.claimedBoards[subsection] == 0 else -1
        self.inputArray[80 + subsection] = 1

    def step(self):
        if (self.player1 == None or self.player2 == None):
            raise Exception("Cannot step unless there are two players")

        outputs = []

        if (self.playerToMove == 1):
            self.player1.network.propogate()
            outputs = self.player1.network.outputs
        else:
            self.player2.network.propogate()
            outputs = self.player2.network.outputs

        # Board probabilities
        boardProbs = outputs[:9]
        subsectionProbs = outputs[9:]

        # Only choose what the newtork thinks is the best board if the scope is unlimited

        if (self.currentBoard != -1):
            boardIndx = self.currentBoard
        else:
            filterProbs(boardProbs, self.claimedBoards)
            boardIndx = getMaxNumIndx(boardProbs)

        # filter the potential moves by whether or not they are legal, then choose the best of those moves
        filterProbs(subsectionProbs, self.board[boardIndx])
        subsectionIndx = getMaxNumIndx(subsectionProbs)

        self.makeMove(subsectionIndx, boardIndx)

        win = checkForWin(self.claimedBoards)

        if win != 0:
            self.winner = win
            return True

        return False

    @property
    def player1(self):
        return self._player1

    @player1.setter
    def player1(self, player):
        if (isinstance(player, Player) != True):
            raise Exception("Can only set player property to Player class")

        self._player1 = player
        player.network.inputs = self.inputArray

    @property
    def player2(self):
        return self._player2

    @player2.setter
    def player2(self, player):
        if (isinstance(player, Player) != True):
            raise Exception("Can only set player property to Player class")

        self._player2 = player
        player.network.inputs = self.inputArray


# total = 0
# count = 0

# gameCount = 100

# for _ in range(gameCount):
#     startTime = time.time()

#     board = Board()
#     player1 = Player()
#     player2 = Player()

#     board.player1 = player1
#     board.player2 = player2

#     finished = False

#     while not finished:
#         finished = board.step()

#     benchmark = time.time() - startTime
#     total += benchmark
#     count += 1
#     print("Game %d took %1.10fms" % (count, benchmark * 1000))

# print("Program took an average of %1.4fms per game (Average of %d games). Total runtime: %1.2fs" %
#       ((total / count)*1000, gameCount, total))
