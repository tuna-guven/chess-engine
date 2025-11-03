import chess

# --- Piece Values (Sunfish-style, in centipawns) ---
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 280,
    chess.BISHOP: 320,
    chess.ROOK: 479,
    chess.QUEEN: 929,
    chess.KING: 60000
}

# --- Piece-Square Tables (PSTs) ---
# White’s perspective; for Black we mirror vertically
PST = {
    chess.PAWN: (
         0,  0,  0,  0,  0,  0,  0,  0,
        78, 83, 86, 73,102, 82, 85, 90,
         7, 29, 21, 44, 40, 31, 44,  7,
       -17, 16, -2, 15, 14,  0, 15,-13,
       -26,  3, 10,  9,  6,  1,  0,-23,
       -22,  9,  5,-11,-10, -2,  3,-19,
       -31,  8, -7,-37,-36,-14,  3,-31,
         0,  0,  0,  0,  0,  0,  0,  0
    ),
    chess.KNIGHT: (
       -66,-53,-75,-75,-10,-55,-58,-70,
        -3, -6,100,-36,  4, 62, -4,-14,
        10, 67,  1, 74, 73, 27, 62, -2,
        24, 24, 45, 37, 33, 41, 25, 17,
        -1,  5, 31, 21, 22, 35,  2,  0,
       -18, 10, 13, 22, 18, 15, 11,-14,
       -23,-15,  2,  0,  2,  0,-23,-20,
       -74,-23,-26,-24,-19,-35,-22,-69
    ),
    chess.BISHOP: (
       -59,-78,-82,-76,-23,-107,-37,-50,
       -11, 20, 35,-42,-39, 31,  2,-22,
        -9, 39,-32, 41, 52,-10, 28,-14,
        25, 17, 20, 34, 26, 25, 15, 10,
        13, 10, 17, 23, 17, 16,  0,  7,
        14, 25, 24, 15,  8, 25, 20, 15,
        19, 20, 11,  6,  7,  6, 20, 16,
        -7,  2,-15,-12,-14,-15,-10,-10
    ),
    chess.ROOK: (
        35, 29, 33,  4, 37, 33, 56, 50,
        55, 29, 56, 67, 55, 62, 34, 60,
        19, 35, 28, 33, 45, 27, 25, 15,
         0,  5, 16, 13, 18, -4, -9, -6,
       -28,-35,-16,-21,-13,-29,-46,-30,
       -42,-28,-42,-25,-25,-35,-26,-46,
       -53,-38,-31,-26,-29,-43,-44,-53,
       -30,-24,-18,  5, -2,-18,-31,-32
    ),
    chess.QUEEN: (
         6,  1, -8,-104, 69, 24, 88, 26,
        14, 32, 60, -10, 20, 76, 57, 24,
        -2, 43, 32, 60, 72, 63, 43,  2,
         1,  3, 52, 23, 15, 24, -1, -9,
       -39, -3, -9,  6, 16, 12, -3,-36,
       -41,-16,-26,-38,-38,-31,-46,-32,
       -51,-44,-33,-51,-46,-50,-13,-36,
       -24,-38,-22,-17,-29,-42,-30,-25
    ),
    chess.KING: (
       -65, 23, 16,-15,-56,-34,  2, 13,
        29, -1,-20, -7, -8, -4,-38,-29,
        -9, 24,  2,-16,-20,  6, 22,-22,
       -17,-20,-12,-27,-30,-25,-14,-36,
       -49, -1,-27,-39,-46,-44,-33,-51,
       -14,-14,-22,-46,-44,-30,-15,-27,
         1,  7, -8,-64,-43,-16,  9,  8,
       -15, 36, 12,-54,  8,-28, 24, 14
    ),
}

# --- Evaluation Function ---
def evaluate_board(board: chess.Board) -> int:
    """Sunfish-style evaluation (material + PST)."""
    value = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        piece_type = piece.piece_type
        color = 1 if piece.color == chess.WHITE else -1

        # Material
        val = piece_values[piece_type]

        # Piece-Square Table bonus
        pst_index = square if piece.color == chess.WHITE else chess.square_mirror(square)
        pst_bonus = PST[piece_type][pst_index]

        value += color * (val + pst_bonus)
    return value
