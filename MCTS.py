from __future__ import division
import math
import numpy as np
import copy

class MCTS():
    """
    This class handles the MCTS.
    """

    def __init__(self, env, NN, config):
        # Store references to the environment and the neural network
        self.env  = env
        self.NN   = NN

        # Store the arguments
        self.config         = config
        self.numTrainSims   = config["MCTS"]["numSims"]
        self.cpuct          = config["MCTS"]["cpuct"]

        self.action_size    = self.env.action_space.shape

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited

        self.N = {}         # stores #times board s was visited
        self.P = {}         # stores initial policy (returned by neural net) for s
        self.V = {}         # Stores the value function for state s

        self.Es = {}        # stores reward for board s
        self.Vs = {}        # stores valid moves for board s

    def getPolicy(self, state, temperature=1, reveal=True):
        """
        Get the policy for the current board, by performing numTrainSims MC simulations.

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
        s = state.board.board_state.tostring()

        #print("Searching from ", env_history[0].board.board_state)

        # Run numTrainSims games from the current board position
        # to sample the available moves; this can be done in parallel!
        #env_history_copy = copy.deepcopy(env_history)
        # @parallel
        for i in range(self.numTrainSims):
            self.search_move(state, depth=0, reveal=reveal)

        # Optionally add dirichlet noise
        #eps = 0.25
        #if s in self.P:
        #    self.P[s] = list( (1-eps)*np.array(self.P[s]) + np.array(np.random.dirichlet(0.03, size=self.action_size))*eps )

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
        return pi

    def search_move(self, state, depth, reveal=True):
        """
        Recursively search for moves until we hit a leaf node (by maximizing
        the upper confidence bound (UCB). Once we find a leaf, we expand it and
        evaluate the probabilities P(a|s) by asking the neural network.

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
        s = state.board.board_state.tostring()

        # Make a copy of the history
        #history = copy.deepcopy(env_history)

        # Check if this is a terminal state
        if s not in self.Es:
            if state.done:
                self.Es[s] = state.reward

                # If we don't want to reveal the initial errors, clearing
                # the board always is a win
                if not reveal:
                    self.Es[s] = 1
            else:
                self.Es[s] = 0

        if self.Es[s]!=0:
            return self.Es[s]

        if self.config["verbose"] >= 2:
            print("Non terminal, has value: ", self.Es[s])

        # Leaf node - expand it and return
        if s not in self.P:

            if self.config["verbose"] >= 2:
                print("Is Leaf")

            # Convert history to boards only so that the NN can predict
            self.P[s], v = self.NN.predict(state.board_history)

            # Mask the illegal actions
            valids = np.array(state.board.get_legal_action())
            valids = np.array([1 if vl in valids else 0 for vl in range(self.env.action_space.shape)])


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

        # If we reach this point, that means that this state is neither a
        # terminating state nor is it a leaf. So let's make our move based
        # on the statistics we have.
        cur_best = -float('inf')
        best_act = -1

        valids = np.array(state.board.get_legal_action())
        valids = np.array([1 if vl in valids else 0 for vl in range(self.env.action_space.shape)])

        # Choose the action with the highest upper confidence bound
        for a in range(self.env.action_space.shape):
            # If the action is a valid action for this state
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.cpuct*self.P[s][a]*math.sqrt(self.N[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.cpuct*self.P[s][a]*math.sqrt(self.N[s])

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act

        # See if we have to update the history
        newstate = state.act(a)

        # Recursively find the next best move from this state
        v = self.search_move(newstate, depth=depth+1, reveal=reveal)

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
