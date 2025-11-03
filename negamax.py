import chess
import math
from chess_gui import ChessGUI

# --- Negamax subclass with depth parameter ---
class Negamax(ChessGUI):
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    def __init__(self, current_position=None, depth=3):
        super().__init__(current_position)
        self.transposition_table = {}
        self.max_depth = depth  # Depth for negamax search

    # --- Basic material evaluation ---
    def evaluate_board(self, board):
        value = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_value = self.piece_values[piece.piece_type]
                value += piece_value if piece.color == chess.WHITE else -piece_value
        return value

    # --- Move ordering heuristic ---
    def order_moves(self, board):
        def move_score(move):
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                if captured_piece and attacker_piece:
                    return 10 * self.piece_values[captured_piece.piece_type] - self.piece_values[attacker_piece.piece_type]
            if move.promotion:
                return 1000
            if board.gives_check(move):
                return 50
            return 0
        return sorted(board.legal_moves, key=move_score, reverse=True)

    # --- Negamax with alpha-beta pruning ---
    def negamax(self, board, depth, alpha, beta, color):
        """
        Negamax implementation: color = 1 for white, -1 for black
        """
        board_key = board.fen()
        if board_key in self.transposition_table:
            stored = self.transposition_table[board_key]
            if stored["depth"] >= depth:
                return stored["value"]

        if depth == 0 or board.is_game_over():
            value = color * self.evaluate_board(board)
            self.transposition_table[board_key] = {"value": value, "depth": depth}
            return value

        max_eval = -math.inf
        for move in self.order_moves(board):
            board.push(move)
            eval_score = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break  # Beta cutoff
        self.transposition_table[board_key] = {"value": max_eval, "depth": depth}
        return max_eval

    # --- Search best move ---
    def search_best_move(self, board, depth):
        best_move = None
        max_eval = -math.inf
        color = 1 if board.turn == chess.WHITE else -1
        for move in self.order_moves(board):
            board.push(move)
            eval_score = -self.negamax(board, depth - 1, -math.inf, math.inf, -color)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return best_move

    # --- AI move with iterative deepening ---
    def ai_move(self, board):
        best_move = None
        for depth in range(1, self.max_depth + 1):
            move = self.search_best_move(board, depth)
            if move:
                best_move = move
            print(f"Depth {depth} completed. Best move so far: {best_move}")
        return best_move