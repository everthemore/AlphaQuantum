from __future__ import division
import numpy as np
import copy
from collections import deque
import os

from MCTS import MCTS

import gym
import gym_quantumchess
#from gym_errorzero.envs.toricgame import ToricGameState

class AlphaQuantumAgent():
    """
    This class is a RL agent that is based on AlphaZero.
    """
    def __init__(self, env, nnet, config):
        # Store a link to the environment
        self.env = env

        # Store current board and previous board
        self.board = None

        # Give the agent a brain
        self.nnet = nnet

        # Extract Coach specific config
        self.config                = config
        self.maxLenOfQueue         = 10000
        self.numIterations         = config["Training"]["numIterations"]
        self.numEpisodes           = config["Training"]["numEpisodes"]
        self.checkpoint            = config["savedir"]
        self.numHistory            = config["numHistory"]

        # Monte Carlo Tree Search for thinking ahead
        self.mcts = MCTS(self.env, self.nnet, config)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the agent eventually won the game, else -1.
        """

        trainExamples = []

        # Replacement history
        state = self.env.state

        episodeStep = 0
        while True:
            episodeStep += 1

	        # Get the policy vector given the current history
            pi = self.mcts.getPolicy(state, temperature=0)

            #print("Policy: ", pi)
            #pi = np.array([1 for v in range(self.env.action_space.shape)])

            # Mask out the invalid moves
            valids = state.board.get_legal_action()
            valids = np.array([1 if vl in valids else 0 for vl in range(self.env.action_space.shape)])

            if np.sum(valids) == 0:
                valids[0] = 1
            pi = np.array(pi*valids); pi = pi/np.sum(pi)

            # Use symmetric boards
            #sym = self.game.getSymmetries(self.board_history, pi)
            #for b,p in sym:
            #    trainExamples.append([b, p])

            # Only store the boards and the resulting policy
            trainExamples.append([state.board_history, np.array(pi)])

	        # Make a greedy move
            action = np.argmax(pi)
            #action = np.random.choice(len(pi), p=pi)

            # See if we have to update the history
            state = state.act(action)

            #print("Action %d: %d"%(len(trainExamples), action))
            if state.reward !=0 and state.done:
                print("Finished game with r = %.2f in %d moves"%(state.reward, len(trainExamples))) #int(len(trainExamples)/3)))
                return [(x[0],x[1],1 if state.reward > 0 else -1) for x in trainExamples]

    def setnewboard(self):
        """
        Reset the game, until we find a non-trivial board to play.
        """
        # Find the next non-trivial board
        self.currentseed += 1
        np.random.seed(self.currentseed)
        self.env.reset()

        # Keep resetting until non-trivial - meaning this is not a terminal state
        while self.env.state.done:
            self.currentseed += 1
            np.random.seed(self.currentseed)
            self.env.reset()

    def learn(self):
        """
        Performs numIterations iterations with numEpisodes episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).

        The errorrate is increased every so often.
        """

	    # Create an empty queue for storing all the training examples
        trainExamples = deque([], maxlen=self.maxLenOfQueue)

	    # Set the initial seed
        self.currentseed = -1

	    # Initial error rate
        errorrate = 0.01
        self.env.error_rate = errorrate

	    # Keep track of the number of snapshots so we can label them
        num_snapshots = 0

        for i in range(self.numIterations):
            # Change the error rate every 10 iterations
            if (i % 10 == 0) and (i != 0):

                # Cap the error rate at 0.15
                if errorrate < 0.15:
                    errorrate = errorrate + 0.01

                # Update the error rates
                self.env.error_rate = errorrate

                # Save a snapshot of the network for evaluation later
                self.nnet.save_checkpoint(folder=self.checkpoint, filename='checkpoint-%d.pth.tar'%num_snapshots)
                num_snapshots += 1

            print('------ITER ' + str(i+1) + '------')
            for eps in range(self.numEpisodes):
		        # Get a new board with the current error rate
                self.setnewboard()

                self.mcts = MCTS(self.env, self.nnet, self.config) # reset tree
                trainExamples += self.executeEpisode()

	        # Train the network
            #print("Training the network on %d examples"%(len(trainExamples)))
            trainhistory = self.nnet.train(trainExamples)
            #print("Done")

            # Save a copy of the current network
            self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')
