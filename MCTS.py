from __future__ import division
import math
import numpy as np
import copy

class MCTS:
    """
    This class handles the MCTS.
    """

    def __init__(self, NN, config):
        # Store reference to the neural network
        self.NN   = NN

        # Store the arguments
        self.config         = config
        self.numSims        = config["MCTS"]["numSims"]
        self.cpuct          = config["MCTS"]["cpuct"]

        self.action_size    = 68912

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.num_valid = {} # Number of valid
        self.N   = {}       # stores #times board s was visited
        self.P   = {}       # stores initial policy (returned by neural net) for s
        self.V   = {}       # Stores the value function for state s
        self.Es = {}        # stores reward for board s

    def getPolicy(self, game, temperature=1):
        """
        Get the policy for the current board, by performing numSims MC simulations.

        Parameters:
            env_history    : (list) environments representing the history
            temp           : (float) the inverse exponent of the visit count
            reveal         : (bool) the environment reveals the initial errors if true

        Returns:
            pi: a policy vector with entries representing the probabilities for taking
                the corresponding actions. That is, the ith entry (= ith action) has a
                probability proportional to Nsa[(s,a)]**(1./temp)
        """

        # Convert the latest state to a string so we can index it
        s = game.serialize()

        # Run numSims games from the current board position
        # to sample the available moves

        # TODO this can be done in parallel using a thread pool!
        # @parallel
        for i in range(self.numSims):
            self.search_move(game, depth=0, reveal=reveal)

        # See how often we performed each move
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.action_size)]

        # Sanity check
        if np.sum(counts) == 0:
            print("No moves have been tried.. something is wrong!")
            exit(0)

        # If temperature is zero, we just pick the move with the highest visitcount
        if temperature <= 0:
            pi = np.zeros( len(counts) ); pi[ np.argmax(counts) ] = 1
        else:
            pi = np.array(counts)**(1/temp); pi /= np.sum(counts)

        # Normalize and return
        return pi / np.sum(pi)

    def search_move(self, game, depth, reveal=True):
        """
        Recursively search for moves until we hit a leaf node (by maximizing
        the upper confidence bound (UCB)), counting as one simulation.
        Once we find a leaf, we expand it and evaluate the probabilities P(a|s)
        by asking the neural network.

        The network also returns a value v, which we then propagate back up the
        search path. If the leaf is also a terminal state, we propagate the win
        value back up instead of the value v.

        Returns:
            v: the expected reward in this state
        """

        if depth > 100:
            # Maximum recursion depth
            print("Too many attempts to find an endgame! Game lost")
            return -1

        # Turn board into a string so that we can use it as a dictionary key (hashable)
        s = game.serialize()

        # Make a copy of the state
        this_game = game.copy()

        # Check if this is a terminal state
        if s not in self.Es: # If we haven't seen this state before
            if this_game.done:   # If it is a final state
                self.Es[s] = this_game.reward
            else:            # Otherwise initialize it with 0
                self.Es[s] = 0

        if self.Es[s] != 0:
            return self.Es[s]

        if self.config["verbose"] >= 2:
            print("Non terminal, has value: ", self.Es[s])

        # This must be a leaf node if we haven't encountered it before - expand it and return
        if s not in self.P:
            if self.config["verbose"] >= 2:
                print("Is Leaf")

            # Use NN to predict P and v
            self.P[s], v = self.NN.predict(this_game.toNetworkInput())

            # Mask the illegal actions
            valid_moves = this_game.get_legal_moves() # ['a2a4', 'b1^a3c3']
            valids = np.zeros(self.action_size)
            for v in valid_moves:
                valids[move_to_index[v]] = 1
            valids = np.array(valids)

            # Track number of valid moves for this state
            self.num_valid[s] = len(valids)

            # And adjust the policy
            self.P[s]  = self.P[s]*valids

            if np.sum(self.P[s]) == 0:
                # This can happen, if the network happens to
                # predict a P[s] of zeros, or if the only outputs in
                # P[s] are invalidated by valids. If there are no valid
                # moves, getGameEnded() would have returned
                # Here, we just predict equal probs for each of the moves
                self.P[s] = np.ones( len(self.P[s]) )*valids

            self.P[s] /= np.sum(self.P[s])
            self.N[s] = 0
            return v

        else:
            # If we reach this point, that means that this state is neither a
            # terminating state nor is it a leaf. So let's choose where to move next to
            # in our tree based on the statistics we have.
            cur_best = -float('inf')
            best_act = -1

            # TODO: This for loop is slow! Can we loop over just the legal ones?

            # Choose the action with the highest upper confidence bound
            for a in range(self.action_size):
                # If the action is a valid action for this state
                if self.P[s][a] != 0:
                    if (s,a) in self.Qsa:
                        u = self.Qsa[(s,a)] + self.cpuct*self.P[s][a]*math.sqrt(self.N[s])/(1+self.Nsa[(s,a)])
                    else:
                        u = self.cpuct*self.P[s][a]*math.sqrt(self.N[s])

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act

            # See if we have to update the history
            next_game = this_game.do_move(a)

            # Recursively find the next best move from this state
            v = self.search_move(next_game, depth=depth+1, reveal=reveal)

            # Back up the new statistics
            if (s,a) in self.Qsa:
                self.Qsa[(s,a)]  = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
                self.Nsa[(s,a)] += 1
            else:
                self.Qsa[(s,a)] = v
                self.Nsa[(s,a)] = 1

            # Increase the visit count for this state
            self.N[s] += 1
            return v
