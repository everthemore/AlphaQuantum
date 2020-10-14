from ctypes import *
from sys import platform
from struct import unpack

# GameData struct
class GameData(Structure):
    _fields_ = [
    ("pieces", c_char * 64),
    ("probabilities", c_float * 64),
    ("ply", c_int),
    ("fifty_count", c_int),
    ("castle_flags", c_int),
    ("ep_square", c_int),
    ("pairwise_bell_measures", c_float * 64 * 64 * 8),
    ]

class Move(Structure):
    _fields_ = [
    ("square1", c_uint8),
    ("square2", c_uint8),
    ("square3", c_uint8),
    ("type", c_uint8),
    ("variant", c_uint8),
    ("does_measurement", c_bool),
    ("measurement_outcome", c_uint8),
    ("promotion_piece", c_uint8)
    ]

row_names = ["1","2","3","4","5","6","7","8"]
column_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
MoveCode = dict({0 : "FAIL", 1 : "SUCCESS", 2 : "WHITE_WIN",
                 3 : "BLACK_WIN", 4 : "MUTUAL_WIN", 5 : "DRAW"})

def square_number_to_str(square_number) -> str:
    column = int(square_number % 8)
    row = int(square_number / 8)
    return column_names[column] + row_names[row]

def format_moves(moves) -> str:
    allmoves = []
    for move in moves:
        # Split move
        if( move["type"] == 4 or move["type"] == 5 ):
            movestr = square_number_to_str(move["square1"]) + "^" \
                    + square_number_to_str(move["square2"]) \
                    + square_number_to_str(move["square3"])
        # Merge move
        elif( move["type"] == 6 or move["type"] == 7 ):
            movestr = square_number_to_str(move["square1"]) \
                    + square_number_to_str(move["square2"]) + "^" \
                    + square_number_to_str(move["square3"])
        # Standard
        else:
            movestr = square_number_to_str(move["square1"]) + square_number_to_str(move["square2"])

        # Was this a measurement?
        if( move["does_measurement"] ):
            movestr += ".m" + str(move["measurement_outcome"])

        allmoves.append(movestr)

    return allmoves

class QuantumChessEngine:
    def __init__(self):
        """
        Load the quantum chess shared library
        """
        shared_lib_path = ""

        # Linux
        if platform.startswith("linux"):
            shared_lib_path = "./QuantumChessAPI.so"
        # Windows
        if platform.startswith('win32'):
            shared_lib_path = "./QuantumChessAPI.dll"
        # Mac OSX
        if platform.startswith('darwin'):
            shared_lib_path = "./QuantumChessAPI.dylib"

        if( shared_lib_path == "" ):
            print("This platform is not supported")
            exit(0)

        try:
            self.QChess_lib = CDLL(shared_lib_path)
        except Exception as e:
            print("Could not load the Quantum Chess library")
            print(e)
            exit(0)

        # Set up all the API signatures

        #QUANTUM_CHESS_API GameVariant* new_game(const char * position, bool force_turn, bool force_win, const char * rest_url);
        self.QChess_lib.new_game.argtypes = [c_char_p, c_bool, c_bool, c_char_p]
        self.QChess_lib.new_game.restype = POINTER(c_int)

        #QUANTUM_CHESS_API long delete_game(GameVariant* game);
        self.QChess_lib.delete_game.argtypes = [POINTER(c_int)]
        self.QChess_lib.delete_game.restype = c_long

        #QUANTUM_CHESS_API long do_move(GameVariant* game, const char * move, GameData* out_buffer, int* move_code);
        self.QChess_lib.do_move.argtypes = [POINTER(c_int), c_char_p, POINTER(GameData), POINTER(c_int)]
        self.QChess_lib.do_move.restype = c_long

        #QUANTUM_CHESS_API long undo_move(GameVariant* game, GameData* out_data);
        self.QChess_lib.undo_move.argtypes = [POINTER(c_int), POINTER(GameData)]
        self.QChess_lib.undo_move.restype = c_long

        #QUANTUM_CHESS_API long get_game_data(GameVariant* game, GameData* out_buffer);
        self.QChess_lib.get_game_data.argtypes = [POINTER(c_int), POINTER(GameData)]
        self.QChess_lib.get_game_data.restype = c_long

        #QUANTUM_CHESS_API long get_history(GameVariant* game, QC::Move* out_buffer, size_t buffer_size, size_t* out_size);
        self.QChess_lib.get_history.argtypes = [POINTER(c_int), POINTER(Move), c_size_t, POINTER(c_size_t)]
        self.QChess_lib.get_history.restype = c_long

        #QUANTUM_CHESS_API long get_legal_moves(GameVariant* game, QC::Move* out_buffer, size_t buffer_size, size_t* out_size);
        self.QChess_lib.get_legal_moves.argtypes = [POINTER(c_int), POINTER(Move), c_size_t, POINTER(c_size_t)]
        self.QChess_lib.get_legal_moves.restype = c_long

        # Create a new gamedata struct
        self.gamedata = GameData()

    def new_game(self, initial_state=""):
        initial_state_fen = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves " if initial_state == "" else initial_state
        initial_state_fen = initial_state_fen.encode('utf-8')
        resturl = "".encode('utf-8')
        self.GamePtr = self.QChess_lib.new_game(initial_state_fen, c_bool(True), c_bool(True), resturl);

    def delete_game(self):
        return self.QChess_lib.delete_game(self.GamePtr);

    def do_move(self, move_str):
        move_code = c_int(0)
        result = self.QChess_lib.do_move(self.GamePtr, move_str.encode('utf-8'), byref(self.gamedata), byref(move_code))
        return self.gamedata, move_code.value

    def undo_move(self):
        move_code = c_int(0)
        result = self.QChess_lib.undo_move(self.GamePtr, byref(self.gamedata))
        return result

    def get_game_data(self):
        result = self.QChess_lib.get_game_data(self.GamePtr, byref(self.gamedata))
        return self.gamedata

    def get_history(self):
        moves = (Move*256)()
        out_size = c_size_t()
        result = self.QChess_lib.get_history(self.GamePtr, moves, sizeof(Move)*256, byref(out_size) )

        allMoves = []
        for move in moves[:out_size.value]:
            thisMove = {}
            for field in move._fields_:
                thisMove[field[0]] = getattr(move, field[0])

            allMoves.append(thisMove)
        return allMoves

    def get_legal_moves(self):
        moves = (Move*2048)()
        out_size = c_size_t()
        result = self.QChess_lib.get_legal_moves(self.GamePtr, moves, sizeof(Move)*2048, byref(out_size) )

        allMoves = []
        for move in moves[:out_size.value]:
            thisMove = {}
            for field in move._fields_:
                thisMove[field[0]] = getattr(move, field[0])

            allMoves.append(thisMove)
        return allMoves

    def print_probability_board(self):
        """Renders a ASCII diagram showing the board probabilities."""
        s = ''
        s += ' +----------------------------------+\n'
        for y in reversed(range(8)):
            s += str(y + 1) + '| '
            for x in range(8):
                bit = y * 8 + x
                prob = str(int(100 * self.gamedata.probabilities[bit]))
                if len(prob) <= 2:
                    s += ' '
                if prob == '0':
                    s += '.'
                else:
                    s += prob
                if len(prob) < 2:
                    s += ' '
                s += ' '
            s += ' |\n'
        s += ' +----------------------------------+\n    '
        for x in range(8):
           s += chr(ord('a') + x) + '   '

        print(s)



if( __name__ == "__main__"):
    QChessEngine = QuantumChessEngine()
    QChessEngine.new_game()

    print("Welcome to Quantum Chess")
    print("Use CTRL+C (or CMD+C) to quit")
    while True:

        valid_moves = QChessEngine.get_legal_moves()
        valid_moves = format_moves(valid_moves)
        print(valid_moves[:10])

        # Get input string
        move = input("Enter your next move: ")

        gamedata, move_code = QChessEngine.do_move(move)
        print("Resulted in movecode: ", MoveCode[move_code])
        QChessEngine.print_probability_board()

        move_history = QChessEngine.get_history()
        move_history = format_moves(move_history)
        print(" ".join(move_history[:3]))

        game_data = QChessEngine.get_game_data()
        print(game_data.ply)
