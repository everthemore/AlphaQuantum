from ctypes import *
from sys import platform
from PyQuantumChess import *

row_names = ["1","2","3","4","5","6","7","8"]
column_names = ["a", "b", "c", "d", "e", "f", "g", "h"]

# Convert move to number
move_to_index = {}

diagonals_board = np.zeros((8,8), dtype=np.int64)
for i in range(8):
  for j in range(8):
      if i + j < 8:
          diagonals_board[i,j] = i + j + 1
      else:
          diagonals_board[j,i] = (16 - (i + j + 1)) % 8

anti_diagonals_board = np.zeros((8,8), dtype=np.int64)
for i in range(8):
  for j in range(8):
      if i <= j:
          num_anti_diagonal = (7 + (i - j)) % 8 + 1
          anti_diagonals_board[i,j] = num_anti_diagonal
          anti_diagonals_board[j,i] = num_anti_diagonal

def get_classical_queen_moves(row, column):
    source = column_names[column] + row_names[row]

    moves = []

    # Add vertical moves
    for i in range(8):
        # Skip source square
        if i == column:
            continue
        moves.append( source + column_names[column] + row_names[i % 8])

    # Add horizontal moves
    for i in range(8):
        # Skip source square
        if i == row:
            continue
        moves.append( source + column_names[i % 8] + row_names[row])

    # Add diagonal moves
    for i in range(diagonals_board[row, column]):
        target = column_names[(column + i) % 8] + row_names[(row + i) % 8]
        # Skip source square
        if source == target:
            continue
        moves.append( source + target )

    for i in range(anti_diagonals_board[row, column]):
        target = column_names[(column - i) % 8] + row_names[(row - i) % 8]
        # Skip source square
        if source == target:
            continue
        moves.append( source + target )

    return moves



def enumerate_all_moves():
    move_counter = 0
    # Loop over all starting squares
    for i in range(8):
        for j in range(8):

            ### Classical moves
            #### Queen moves
            classical_queen_moves = get_classical_queen_moves(i,j)
            for move in classical_queen_moves:
                move_to_index[move] = move_counter
                move_counter += 1

            #### Knight moves
            classical_knight_moves = get_classical_knight_moves(i,j)
            for move in classical_knight_moves:
                move_to_index[move] = move_counter
                move_counter += 1


def action_to_move_str(action):
    return "a2a4"

class QuantumChessGame:
    def __init__(self):
        self.game = QuantumChessEngine()
        self.reset()

    # Initialize with another game, creates a copy
    def __init__(self, other):
        self.game = QuantumChessEngine()

        # This will create a new copy
        self.reset(other.move_history)

        # Overwrite done and reward
        self.done = other.done
        self.reward = other.reward

    def get_legal_moves(self):
        return self.game.get_legal_moves()

    def do_move(self, move):
        move_str = int(move)
        gameData, movecode = self.game.do_move(move_str)

        # Check if we're in a terminal state
        if( movecode > 1 ):
            self.done = True

            # Non-zero reward if there is a winner
            if( movecode == 2 or movecode == 3 ):
                self.reward = 100

        return this

    def get_game_status(self):
        gameData = self.game.get_game_data()

        # Extract done and winner
        self.done = False
        blackWin = True
        winner = -1 if blackWin else 1

        return self.done, winner

    def reset(self, move_history = []):
        # Copy move history
        self.move_history = move_history
        # Start position
        new_game_fen = "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # Move history
        moves = " ".join(self.move_history)
        # Start a new game with the move history replayed
        self.game.new_game(initial_state="position {0} moves {1}".format(new_game_fen, moves))

        # We assume a new game by default, should be overwritten otherwise
        self.done = False
        self.reward = 0

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

        # Add a plane that indicates whose turn it is (i.e. all 0 or all +1)

        # Stack all the planes
        return np.stack([white_piece_board, black_piece_board, probability_plane], axis=2)

    def copy(self):
        # Return a (deep) copy of this game
        return QuantumChessGame(self)

    def delete(self):
        self.game.delete_game()
