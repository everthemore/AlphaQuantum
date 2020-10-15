import numpy as np
from PyQuantumChess import *

class QuantumChessGame:
    # Initialize with another game, creates a copy
    def __init__(self, other = None):
        self.game = QuantumChessEngine()

        if( other is None ):
            self.reset()
            return

        # This will create a new copy
        self.reset(other.move_history)

        # TODO: Check that this is properly COPIED!

        # Overwrite done and reward
        self.done = other.done
        self.reward = other.reward
        # Overwrite piece history
        self.piece_history = other.piece_history
        self.probability_history = other.probability_history

    def get_legal_moves(self):
        moves = self.game.get_legal_moves()
        moves = format_moves(moves)
        return moves

    def do_move(self, move_str):
        gameData, movecode = self.game.do_move(str(move_str))

        self.move_history = self.game.get_history()

        # Keep track of the piece boards for network input
        gamedata = self.game.get_game_data()
        self.piece_history.append(gamedata.pieces)
        self.probability_history.append(gamedata.probabilities)

        # Check if we're in a terminal state
        if( movecode > 1 ):
            self.done = True

            # Non-zero reward if there is a winner
            if( movecode == 2 or movecode == 3 ):
                self.reward = 100
                self.winner = 100 if movecode == 1 else -100

        return self

    def reset(self, move_history = []):
        # Copy move history
        self.move_history = move_history
        # Start position
        new_game_fen = "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # Move history
        moves = format_moves(self.move_history)
        # Start a new game with the move history replayed
        self.game.new_game(initial_state="position {0} moves {1}".format(new_game_fen, " ".join(moves)))

        # We assume a new game by default, should be overwritten otherwise
        self.done = False
        self.winner = 0
        self.reward = 0
        self.piece_history = []
        self.probability_history = []

    def get_current_player(self) -> int:
        gameData = self.game.get_game_data()
        return -1 if gameData.ply % 2 == 0 else 1

    def serialize(self) -> str:
        gamedata = self.game.get_game_data()
        gamedata_str = gamedata.pieces + gamedata.probabilities;
        return gamedata_str

    def toNetworkInput(self):
        """
        Return a representation of the state that can be directly
        fed into the neural network.
        """
        all_planes = []

        # History goes 8 steps back
        for t in range(8):
            if t < len(self.piece_history):
                white_piece_boards, black_piece_boards = self.format_piece_boards(self.piece_history[-t])
                probability_plane = np.array(self.probability_history[-t]).reshape(8,8,1)
            else: # Just zeros
                white_piece_boards, black_piece_boards = np.zeros((8,8,6)), np.zeros((8,8,6))
                probability_plane = np.zeros((8,8,1))

            # TODO: Repetition planes?

            if( len(all_planes) == 0 ):
                all_planes = np.concatenate([white_piece_boards, black_piece_boards, probability_plane], axis=2)
            else:
                all_planes = np.concatenate([all_planes, white_piece_boards, black_piece_boards, probability_plane], axis=2)

        # Add castle flags plane
        # Add ply plane

        # Add a plane that indicates whose turn it is (i.e. all 0 or all +1)
        player_turn_board = np.ones((8,8,1))*self.get_current_player()

        # Stack all the planes
        all_planes = np.concatenate([all_planes, player_turn_board], axis=2)
        return all_planes

    def format_piece_boards(self,pieces):
        white_piece_board = np.zeros((8,8,6))
        black_piece_board = np.zeros((8,8,6))

        white_pawn_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'p']
        white_piece_board[white_pawn_indices, 0] = 1
        white_rook_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'r']
        white_piece_board[white_rook_indices, 1] = 1
        white_knight_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'n']
        white_piece_board[white_knight_indices, 2] = 1
        white_bishop_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'b']
        white_piece_board[white_bishop_indices, 3] = 1
        white_queen_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'q']
        white_piece_board[white_queen_indices, 4] = 1
        white_king_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'k']
        white_piece_board[white_king_indices, 5] = 1

        black_pawn_indices = [(i / 8,i % 8) for i, c in enumerate(pieces) if c == 'P']
        black_piece_board[black_pawn_indices, 0] = 1
        black_rook_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'R']
        black_piece_board[black_rook_indices, 1] = 1
        black_knight_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'N']
        black_piece_board[black_knight_indices, 2] = 1
        black_bishop_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'B']
        black_piece_board[black_bishop_indices, 3] = 1
        black_queen_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'Q']
        black_piece_board[black_queen_indices, 4] = 1
        black_king_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'K']
        black_piece_board[black_king_indices, 5] = 1

        return white_piece_board, black_piece_board

    def copy(self):
        # Return a (deep) copy of this game
        return QuantumChessGame(self)

    def delete(self):
        self.game.delete_game()
