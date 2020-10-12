import numpy as np

row_names = ["1","2","3","4","5","6","7","8"]
column_names = ["a", "b", "c", "d", "e", "f", "g", "h"]

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

def get_classical_knight_moves(row, column):
    source = column_names[column] + row_names[row]

    moves = []

    if( column + 1 < 8 and row + 2 < 8 ):
        moves.append( source + column_names[column+1] + row_names[row+2])
    if( column + 2 < 8 and row + 1 < 8 ):
        moves.append( source + column_names[column+2] + row_names[row+1])

    if( column + 2 < 8 and row - 1 >= 0 ):
        moves.append( source + column_names[column+2] + row_names[row-1])
    if( column + 1 < 8 and row - 2 >= 0 ):
        moves.append( source + column_names[column+1] + row_names[row-2])

    if( column - 1 >= 0 and row - 2 >= 0 ):
        moves.append( source + column_names[column-1] + row_names[row-2])
    if( column - 2 >= 0 and row - 1 >= 0 ):
        moves.append( source + column_names[column-2] + row_names[row-1])

    if( column - 2 >= 0 and row + 1 < 8 ):
        moves.append( source + column_names[column-2] + row_names[row+1])
    if( column - 1 >= 0 and row + 2 < 8 ):
        moves.append( source + column_names[column-1] + row_names[row+2])

    return moves

def enumerate_all_moves():
    move_counter = 0
    move_to_index = {}
    index_to_move = {}

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

    return move_to_index, index_to_move
