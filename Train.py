from __future__ import division
from Network import *
from Agent import *
from QuantumChessGame import *

import gym
import argparse, json
import sys, os, datetime

from enumerate_actions import enumerate_all_moves
move_to_index, index_to_move = enumerate_all_moves()


#
# Parse arguments passed to the program (or set defaults)
#
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Config file to load (overrides other settings)")
parser.add_argument("--numTrainSims", type=int, default=200, help="Number of MCTS simulations during training")
parser.add_argument("--maxSearchDepth", type=int, default=200, help="Max MCTS depth")
parser.add_argument("--numHistory", type=int, default=1, help="Number of steps of history tracked by the Agent")
parser.add_argument("--numIterations", type=int, default=100, help="Number of iterations during training. Each iteration has numEpisodes training examples.")
parser.add_argument("--numContests", type=int, default=10, help="Number of contests between current and best network.")

parser.add_argument("--numEpisodes", type=int, default=25, help="Number of episodes per iteration (number of training examples)")
parser.add_argument("--cpuct", type=float, default=1.4, help="Constant in multi-armed bandit")
parser.add_argument("--dropout", type=float, default=0, help="Dropout for the network")
parser.add_argument("--numEpochs", type=int, default=25, help="Number of epochs for training the network")
parser.add_argument("--batchSize", type=int, default=64, help="Batchsize for training the network")
parser.add_argument("--learnRate", type=float, default=1e-1, help="Learning rate for network")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
parser.add_argument("--regConst", type=float, default=0, help="Regularization constant for network")
parser.add_argument("--tempThreshold", type=int, default=0, help="Number of episodes before temperature is set to 0")
parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
args = parser.parse_args()

#
# Make sure the save-dir exists
#
if not os.path.exists("./models"):
    os.mkdir("./models")
checkpoint = './models/%s/'%(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(checkpoint):
    os.mkdir(checkpoint)


if args.file is not None:
    with open(args.file, "r") as f:
        JsonConfig = json.load(f)
else:
    # Make the Json file for convenience, so that we can save it in the
    # corresponding directory
    JsonConfig = {
        "verbose":              args.verbose,
        "savedir":              checkpoint,
        "numHistory":           args.numHistory,
        "action_size":          len(move_to_index),

        "Training": {
            "numIterations":    args.numIterations,
            "numEpisodes":      args.numEpisodes,
            "tempThreshold":    args.tempThreshold,
            "numContests":      args.numContests,
        },

        "MCTS": {
            "numSims":          args.numTrainSims,
            "cpuct":            args.cpuct,
            "max_search_depth": args.maxSearchDepth,
        },

        "Network": {
            "dropout":          args.dropout,
            "learning_rate":    args.learnRate,
            "reg_const":        args.regConst,
            "numEpochs":        args.numEpochs,
            "batchSize":        args.batchSize,
            "momentum":         args.momentum,

            # Three hidden layers w/ 3 filters and kernel size 3
            "hidden_layers":    [{'filters':75, 'kernel_size':(4,4)} for i in range(6)],
        }
    }

# Write the config to file in the corresponding directory (overwrite)
with open('%s/config.json'%checkpoint,'w') as f:
    json.dump(JsonConfig, f, indent=4)

# Set up the environment
#game = QuantumChessGame()

# Create a new neural network
nnet = Network(JsonConfig)
# Create a new agent, whose brain is the network we just created
c = AlphaQuantumAgent(nnet, JsonConfig)

# Start the main training loop
c.learn()
