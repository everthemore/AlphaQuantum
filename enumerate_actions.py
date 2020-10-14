import numpy as np

row_names = ["1","2","3","4","5","6","7","8"]
column_names = ["a", "b", "c", "d", "e", "f", "g", "h"]

def get_diagonal_squares(row, column):
    squares = []

    for i in range(1,8):
        if( row + i < 8 and column + i < 8 ):
            squares.append( (row+i, column+i) )

    for i in range(1,8):
        if( row - i >= 0 and column + i < 8 ):
            squares.append( (row-i, column+i) )

    for i in range(1,8):
        if( row + i < 8 and column - i >= 0 ):
            squares.append( (row+i, column-i) )

    for i in range(1,8):
        if( row - i >= 0 and column - i >= 0 ):
            squares.append( (row-i, column-i) )

    return squares

def get_classical_queen_moves(row, column):
    source = column_names[column] + row_names[row]

    moves = []

    # Add vertical moves
    for i in range(1,8):
        moves.append( source + column_names[column] + row_names[(row+i) % 8])

    # Add horizontal moves
    for i in range(1,8):
        moves.append( source + column_names[(column+i) % 8] + row_names[row])

    # Add diagonal moves
    diagonal_squares = get_diagonal_squares(row, column)
    for square in diagonal_squares:
        target = column_names[square[1]] + row_names[square[0]]
        moves.append(source + target)

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

def get_quantum_queen_moves(row, column):
    source = column_names[column] + row_names[row]

    moves = []

    classical_moves = get_classical_queen_moves(row, column)

    for move1 in classical_moves:
        target1 = move1[2:]

        for move2 in classical_moves:
            target2 = move2[2:]

            if( target2 == target1 ):
                continue

            movestr = source + "^" + target1 + target2

            if( movestr not in moves ):
                moves.append(movestr)

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
            # classical_queen_moves = get_classical_queen_moves(i,j)
            # for move in classical_queen_moves:
            #     move_to_index[move] = move_counter
            #     move_counter += 1
            #
            # #### Knight moves
            # classical_knight_moves = get_classical_knight_moves(i,j)
            # for move in classical_knight_moves:
            #     move_to_index[move] = move_counter
            #     move_counter += 1

            quantum_queen_moves = get_quantum_queen_moves(i,j)
            for move in quantum_queen_moves:
                move_to_index[move] = move_counter
                move_counter += 1

    return move_to_index, index_to_move
