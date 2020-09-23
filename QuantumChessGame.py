from ctypes import *
from sys import platform
from quantumchess import *

def format_gamedata(gamedata):
    return gamedata

def action_to_move_str(action):
    return "a2a4"

class QuantumChessGame:
    def __init__(self):
        self.game = QuantumChess()
        self.reset()
    
    def get_legal_moves(self):
        return self.game.get_legal_moves()
    
    def do_move(self, move):
        move_str = int(move)
        gameData, movecode = self.game.do_move(move_str)
        return this
    
    def get_game_stats(self):
        gameData = self.game.get_game_data()
    
        # Extract done and winner
        done = False
        blackWin = True
        winner = -1 if blackWin else 1
    
        return done, winner
    
    def reset(self):
        self.game.new_game()
    
    def get_current_player(self):
        gameData = self.game.get_game_data()
        return -1 if gameData.ply % 2 == 0 else 1
    
    def toString(self):
        gamedata = self.game.get_game_data()
        gamedata_str = gamedata.pieces + gamedata.probabilities;
        return gamedata_str

    def toNetworkInput(self):
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
    
    
    

    
class QuantumChessState:
    def __init__(self, game, playerTurn):
        self.game = game
        self.playerTurn = playerTurn
        return

    def act(self, move):
        # Perform the move
        self.game.do_move(move)
        # Return a new game, but switch player turns
        return QuantumChessState(self.game, -self.playerTurn)

    @property
    def done(self):
        return False

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

        #self.num_actions = 4
        # self.observation_space=gym.spaces.Box(low=0,high=1,shape=(2*self.d+1, 2*self.d+1, self.volume_depth+self.n_action_layers),dtype=np.uint8)
        # self.action_space = gym.spaces.Discrete(self.num_actions)
        #self.seed()

        # Create a new game, only so that self.reset() has something to delete
        self.game = QuantumChessGame()
        self.game.new_game()
        self.reset()

    def reset(self):
        # Delete the current game...
        self.game.delete_game()
        # ...and start a new one
        self.game.new_game()

        # Initial game state
        self.state = QuantumChessState(self.game, 1)
        return self.state.representation()

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
        self.state = self.state.act(move)

        reward = 0
        if( self.state.done ):
            reward = 100 if True else -100

        return self.state.representation(), reward, self.state.done, {}

    def close(self):
        return