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

    split_moves = []
    merge_moves = []

    classical_moves = get_classical_queen_moves(row, column)

    for move1 in classical_moves:
        target1 = move1[2:]

        for move2 in classical_moves:
            target2 = move2[2:]

            if( target2 == target1 ):
                continue

            split = source + "^" + target1 + target2
            if( split not in split_moves ):
                split_moves.append(split)

            merge = target1 + target2 + "^" + source
            if( merge not in merge_moves ):
                merge_moves.append(merge)

    return split_moves, merge_moves

def get_quantum_knight_moves(row, column):
    source = column_names[column] + row_names[row]

    split_moves = []
    merge_moves = []

    classical_moves = get_classical_knight_moves(row, column)

    for move1 in classical_moves:
        target1 = move1[2:]

        for move2 in classical_moves:
            target2 = move2[2:]

            if( target2 == target1 ):
                continue

            split = source + "^" + target1 + target2
            if( split not in split_moves ):
                split_moves.append(split)

            merge = target1 + target2 + "^" + source
            if( merge not in merge_moves ):
                merge_moves.append(merge)

    return split_moves, merge_moves

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
                index_to_move[move_counter] = move
                move_counter += 1

            #### Knight moves
            classical_knight_moves = get_classical_knight_moves(i,j)
            for move in classical_knight_moves:
                move_to_index[move] = move_counter
                index_to_move[move_counter] = move
                move_counter += 1

            queen_split_moves, queen_merge_moves = get_quantum_queen_moves(i,j)
            for move in queen_split_moves:
                move_to_index[move] = move_counter
                index_to_move[move_counter] = move
                move_counter += 1

            for move in queen_merge_moves:
                move_to_index[move] = move_counter
                index_to_move[move_counter] = move
                move_counter += 1

            knight_split_moves, knight_merge_moves = get_quantum_knight_moves(i,j)
            for move in knight_split_moves:
                move_to_index[move] = move_counter
                index_to_move[move_counter] = move
                move_counter += 1

            for move in knight_merge_moves:
                move_to_index[move] = move_counter
                index_to_move[move_counter] = move
                move_counter += 1

    return move_to_index, index_to_move
