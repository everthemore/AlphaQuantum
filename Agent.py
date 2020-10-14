from __future__ import division
import numpy as np
import copy
from collections import deque
import os
from QuantumChessGame import QuantumChessGame

from MCTS import MCTS

class AlphaQuantumAgent():
    """
    This class is a RL agent that is based on AlphaZero.
    """
    def __init__(self, nnet, config):

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
        self.num_contests          = config["Training"]["numContests"]

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

        # Create a new game instance
        game = QuantumChessGame()
        game.reset()

        # Reset the MC tree
        player1 = MCTS(self.nnet, self.config)
        player2 = MCTS(self.nnet, self.config)

        # Start with player 1 (index 0)
        current_player_index = 0
        while True:

            # Set the current player
            current_player = player1 if current_player_index == 0 else player2
            # Get the policy for the current player
            pi = current_player.get_policy(game, temperature=0)

            # TODO
            # We should add the flipped board here (black <-> white)

            # Only store the boards and the resulting policy, which is what we
            # want the neural network to learn to map
            trainExamples.append([game.serialize(), np.array(pi)])

            # Add dirichlet noise, w/ amplitude proportinal to the avg number of available legal moves
            #num_legal_moves = len(state.get_legal_moves())
            #noise = 0.1 * np.random.dirichlet(0.03 * np.ones(num_legal_moves))

            # Pick according to the probability distribution
            action = np.random.choice(len(pi), p=pi)

            # Move to the next state
            game = game.do_move(action)

            # Reflect the move in the MCTrees too, keeping the tree but setting the
            # new state as the root; We don't have to
            # player1.update(action)
            # player2.update(action)

            # Switch player
            current_player_index = (current_player_index + 1) % 2

            # If we get to a terminal state w/ non-zero reward, or a tie
            done, winner = game.done, game.winner

            if done:
                # Board, Policy, v
                return [(x[0], x[1], winner if (current_player_index == 0) else -1*winner) for x in trainExamples]

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

        # Create a new game instance
        game = QuantumChessGame()

        for i in range(self.numIterations):
            print('------ITER ' + str(i+1) + '------')

            if (i % 10 == 0) and (i != 0):
                # Save a snapshot of the network for evaluation later
                self.nnet.save_checkpoint(folder=self.checkpoint, filename='checkpoint-%d.pth.tar'%num_snapshots)
                num_snapshots += 1

            # TODO: Run many in parallel!
            for eps in range(self.numEpisodes):
                # Reset the game
                game.reset()

                # Run an episode and commit it to the list of training examples
                trainExamples += self.executeEpisode()

            # Train the network
            train_history = self.nnet.train(trainExamples)

            if( i == 0 ):
                # The best network is the current one
                self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

            if( i % self.check_freq == 0 and i != 0 ):
                # Run a bunch of games with the current network and with the best network
                # Keep going with the one that is best
                best_network = Network(self.env, self.config)
                best_network.load_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

                currentwins, bestwins, draws = self.contest( self.nnet, best_network, self.num_contests )

                # If the current is 5% better than the best
                if( currentwins + bestwins > 0 and currentwins/(currentwins+bestwins) > 0.05 ):
                    # Overwrite the best
                    self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

    def contest( currentNetwork, bestNetwork, numContests ):

        currentwins, bestwins, draws = 0, 0, 0
        game = QuantumChessGame()

        # TODO: Parallelize
        for game in range(numContests):
            # Start a new game
            game.reset()

            # Reset the search tree
            currentMCTS = MCTS(currentNetwork, self.config)
            bestMCTS = MCTS(bestNetwork, self.config)

            # Start with player 1 (index 0)
            current_player_index = 0
            while True:
                # Set the current player
                current_player = currentMCTS if current_player_index == 0 else bestMCTS
                # Get the policy for the current player
                pi = current_player.get_policy(game, temperature=0)

                # Greedy action selection
                action = np.argmax(pi)
                game = game.do_move(action)

                done, winner = game.get_game_status()

                if done:
                    if winner == 1:
                        currentwins += 1
                    if winner == -1:
                        bestwins += 1
                    if winner == 0:
                        draws += 1

        return currentwins, bestwins, draws
