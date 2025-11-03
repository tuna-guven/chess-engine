import numpy as np
import chess
from chess import Board
from chess_gui import ChessGUI
from chess.pgn import Game

class LichessElite(ChessGUI):
    def __init__(self, model=None, move_to_int=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.move_to_int = move_to_int

    # --- Override AI move ---
    def ai_move(self, board: Board):
        if board.is_game_over() or self.model is None or self.move_to_int is None:
            return None

        board_matrix = self.board_to_matrix(board).reshape(1, 8, 8, 12)
        predictions = self.model.predict(board_matrix, verbose=0)[0]

        int_to_move = {v: k for k, v in self.move_to_int.items()}

        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]

        for move_index in np.argsort(predictions)[::-1]:
            move_uci = int_to_move.get(move_index)
            if move_uci in legal_moves_uci:
                return chess.Move.from_uci(move_uci)

        return None

    # --- Convert board to NN input ---
    @staticmethod
    def board_to_matrix(board: Board):
        matrix = np.zeros((8, 8, 12))
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1
            piece_color = 0 if piece.color else 6
            matrix[row, col, piece_type + piece_color] = 1
        return matrix

    # --- Create input/output data from games ---
    @staticmethod
    def create_input_for_nn(games):
        X, y = [], []
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                X.append(LichessElite.board_to_matrix(board))
                y.append(move.uci())
                board.push(move)
        return X, y

    # --- Encode moves to integers ---
    @staticmethod
    def encode_moves(moves):
        move_to_int = {move: idx for idx, move in enumerate(sorted(set(moves)))}
        encoded_moves = [move_to_int[move] for move in moves]
        return encoded_moves, move_to_int
