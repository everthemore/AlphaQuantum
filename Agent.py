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

        # The current state of the game
        state = self.env.state

        while True:
	        # Get the policy vector given the current history
            pi = self.mcts.getPolicy(state, temperature=0)

            # Mask out the invalid moves
            valids = state.get_legal_action()
            valids = np.array([1 if vl in valids else 0 for vl in range(self.env.action_space.shape)])

            # Renormalize
            if np.sum(valids) == 0:
                valids[0] = 1
            pi = np.array(pi*valids); pi = pi/np.sum(pi)

            # Only store the boards and the resulting policy, which is what we
            # want the neural network to learn to map
            trainExamples.append([state.board_history, np.array(pi)])

	        # Make a greedy move
            # action = np.argmax(pi)
            # Pick according to the probability distribution
            action = np.random.choice(len(pi), p=pi)

            # Move to the next state
            state = state.act(action)

            # If we get to a terminal state w/ non-zero reward, or a tie
            if state.reward !=0 or state.done:
                print("Finished game with r = %.2f in %d moves"%(state.reward, len(trainExamples))) #int(len(trainExamples)/3)))
                return [(x[0],x[1],state.reward) for x in trainExamples]

    def learn(self):
        """
        Performs numIterations iterations with numEpisodes episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).

        The errorrate is increased every so often.
        """

	    # Create an empty queue for storing all the training examples
        trainExamples = deque([], maxlen=self.maxLenOfQueue)

	    # Keep track of the number of snapshots so we can label them
        num_snapshots = 0

        for i in range(self.numIterations):
            if (i % 10 == 0) and (i != 0):
                # Save a snapshot of the network for evaluation later
                self.nnet.save_checkpoint(folder=self.checkpoint, filename='checkpoint-%d.pth.tar'%num_snapshots)
                num_snapshots += 1

            print('------ITER ' + str(i+1) + '------')
            for eps in range(self.numEpisodes):
                # Reset the game
		        self.env.reset()

                # Reset the MC tree
                self.mcts = MCTS(self.env, self.nnet, self.config)

                # Run an episode and commit it to the list of training examples
                trainExamples += self.executeEpisode()

	        # Train the network
            train_history = self.nnet.train(trainExamples)

            # Save a copy of the current network
            self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')
