from ctypes import *
from sys import platform
from quantumchess import *

def format_gamedata(gamedata):
    return gamedata

def action_to_move_str(action):
    return "a2a4"

class QuantumChessState:
    def __init__(self, game):
        self.game = game
        return

    # Make a move and return a copy of the game (essential for MCTS)
    def act(self, move):
        self.game.do_move(move)
        return QuantumChessState(self.game)

    def get_legal_actions(self):
        # Get legal moves
        legal_moves = self.game.get_legal_moves()

        # All actions are illegal initially
        legal_actions = np.zeros(1000)
        # find indices of moves in list of all actions
        indices = 0
        legal_actions[indices] = 1

        return legal_actions

    def serialize(self):
        gamedata = self.game.get_game_data()

        gamedata_str = gamedata.pieces + gamedata.probabilities;
        return gamedata_str

    def representation(self):
        """
        Return a representation of the state that can be directly
        fed into a neural network.
        """
        white_piece_board = np.zeros((8,8,5))
        black_piece_board = np.zeros((8,8,5))

        gamedata = self.game.get_game_data()
        pieces = gamedata.pieces

        white_pawn_indices = [(i / 8,i % 8) for i, c in enumerate(pieces) if c == 'p']
        white_piece_board[white_pawn_indices*,0] = 1

        black_pawn_indices = [(i / 8,i % 8) for i, c in enumerate(pieces) if c == 'P']
        black_piece_board[white_pawn_indices*,0] = 1

        # ETC

        # Add probability plane
        probability_plane = np.array(gamedata.probabilities).reshape(8,8)

        # Add castle flags plane
        # Add ply plane

        # Stack all the planes
        return np.stack([white_piece_board, black_piece_board], axis=2)

class QuantumChessEnv(gym.Env):
    '''
    Quantum Chess environment.
    '''

    def __init__(self):
        """
        TODO
        """

        # Create a Quantum Chess Game
        self.game = QuantumChessGame()

        self.num_actions = 4
        # self.observation_space=gym.spaces.Box(low=0,high=1,shape=(2*self.d+1, 2*self.d+1, self.volume_depth+self.n_action_layers),dtype=np.uint8)

        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.seed()

        self.completed_actions = np.zeros(self.num_actions,int)
        self.state = None
        self.done = False
        self.reward = 0

        # Create a new game, only so that self.reset() has something to delete
        self.game.new_game()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

    def reset(self):

        # Delete the current game...
        self.game.delete_game()
        # ...and start a new one
        self.game.new_game()

        # Format the gamedata
        gamedata = self.game.get_game_data()
        return format_gamedata(gamedata)

    def step(self, action):
        '''
        Args:
            action: int
        Return:
            observation: board encoding,
            reward: reward of the game,
            done: boolean,
            info: state dict

        '''

        # Convert the action (number) to a move string
        move = action_to_move_str(action)
        move = move.encode('utf-8')

        # perform the action
        gamedata, move_code = self.game.do_move(move)

        # Construct the observation from the gamedata
        observation = format_gamedata(gamedata)

        # Check the reward and terminal condition
        done = (move_code == 0)

        reward = 0
        if( done ):
            reward = 100 if True else -100

        return observation, reward, done, {}

    def close(self):
        return

    def render(self, mode="human", close=False):
        self.game.print_probability_board()
