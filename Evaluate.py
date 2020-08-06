from __future__ import division
import numpy as np
import copy

import json
import argparse
import sys
from multiprocessing import Pool

from MCTS import MCTS
from Brain import NeuralBrain

import gym
import gym_errorzero

def play(error_rate):
    # Load the config
    with open(args.savedir+"/config.json", "r") as f:
        JsonConfig = json.load(f)

    local_env = gym.make('ToricGame-v0')
    local_env.init(JsonConfig["distance"], error_rate, JsonConfig["numHistory"])

    # Make a brain
    net = NeuralBrain(local_env, JsonConfig)
    net.load_checkpoint(folder=args.savedir, filename='checkpoint-%d.pth.tar'%(args.snapshotNum))

    fraction = 0

    JsonConfig["MCTS"]["numSims"] = args.numTestSims;
    numHistory = JsonConfig["numHistory"]

    # Play N games
    for seed in range(args.numTestRuns):
        # Reset the search tree
        mcts = MCTS(local_env, net, JsonConfig)

        # Reset the board
        np.random.seed(seed)
        local_env.reset()

	    # Build the history
        state = local_env.state

        # Keep playing until the game ends
        if state.done:
            if state.reward > 0:
                fraction += 1/args.numTestRuns
            continue

        while state.reward == 0:
            # Use network and MCTS to find the next move, but don't reveal errors!
            action = np.argmax(mcts.getPolicy(state, temperature=0, reveal=False))
            state = state.act(action)

	    # The game ended, increase the number of wins if we won
        if state.reward > 0:
            fraction += 1/args.numTestRuns

    return fraction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("savedir", help="Directory to load config and network from")
    parser.add_argument("snapshotNum", type=int, help="The snapshot number (of network) to load")
    parser.add_argument("--numTestSims", type=int, default=200, help="Number of MCTS simulations during testing")
    parser.add_argument("--numTestRuns", type=int, default=100, help="Number of boards evaluated")
    parser.add_argument("--numProcesses", type=int, default=1, help="Number of processes launched for evaluation")
    args = parser.parse_args()

    pXs = np.arange(0.01, 0.16, 0.01)
    p = Pool(processes=args.numProcesses)
    result = p.map( play, pXs )
    p.terminate()

    np.savetxt(args.savedir + "/snapshot-%d.txt"%(args.snapshotNum), np.column_stack([pXs, result]))
