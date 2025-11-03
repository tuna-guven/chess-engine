import chess
import chess.svg
import chess.pgn
import ipywidgets as widgets
from IPython.display import display, SVG, clear_output
import random
import datetime

class ChessGUI:
    def __init__(self, current_position=None):
        # --- Core game state ---
        self.original_position = current_position
        self.move_history = []

        # --- Initialize board safely using try/except with normal if/else ---
        try:
            if current_position:
                self.board = chess.Board(current_position)
            else:
                self.board = chess.Board()
        except Exception:
            # Fallback to standard board if anything goes wrong
            print("⚠️ Invalid FEN or input. Starting from initial position.")
            self.board = chess.Board()

        # --- Automatically determine if the player is black ---
        self.is_black = not self.board.turn

        # --- Widgets ---
        self.move_input = widgets.Text(description="Your move:")
        self.make_move_btn = widgets.Button(description="Make Move")
        self.ai_btn = widgets.Button(description="AI Move")
        self.reset_btn = widgets.Button(description="Reset Game")
        self.output = widgets.Output()

        # --- Button actions ---
        self.make_move_btn.on_click(self.handle_move)
        self.ai_btn.on_click(self.handle_ai_move)
        self.reset_btn.on_click(self.handle_reset)

    # --- Display board as SVG ---
    def show_board(self, last_move=None):
        arrows = []
        if last_move:
            arrows.append(chess.svg.Arrow(last_move.from_square, last_move.to_square, color="#00cc00cc"))

        svg = chess.svg.board(
            self.board,
            size=400,
            arrows=arrows,
            lastmove=last_move,
            check=self.board.king(self.board.turn) if self.board.is_check() else None,
            flipped=self.is_black
        )
        display(SVG(svg))

    # --- Handle human move ---
    def handle_move(self, _):
        move_str = self.move_input.value.strip()
        move = None

        try:
            move_obj = chess.Move.from_uci(move_str)
            if move_obj in self.board.legal_moves:
                move = move_obj
        except Exception:
            move = None

        with self.output:
            clear_output(wait=True)
            if move:
                self.board.push(move)
                self.move_history.append(move)
                print(f"🙍‍♂️ You played: {move}")
                self.show_board(move)
            else:
                if self.board.turn == self.is_black:
                    print("❌ It's not your turn! Let the AI make its move.")
                else:
                    print("❌ Invalid move! Use UCI format.")
                self.show_board()

    # --- Handle AI move ---
    def handle_ai_move(self, _):
        with self.output:
            clear_output(wait=True)

            if self.board.is_game_over():
                self.show_game_result()
                return

            if self.board.turn == self.is_black:
                print("🤖 AI is thinking...")
                self.show_board()

                move = self.ai_move(self.board)

                clear_output(wait=True)
                if move is not None:
                    self.board.push(move)
                    self.move_history.append(move)
                    print(f"🤖 AI played: {move}")
                    self.show_board(move)
            else:
                print("❌ It's your turn! Make your move.")
                self.show_board()

    # --- Handle reset ---
    def handle_reset(self, _):
        if self.original_position is None:
            self.board = chess.Board()
        else:
            self.board = chess.Board(self.original_position)
        self.move_history = []
        # Recalculate player color
        self.is_black = not self.board.turn
        with self.output:
            clear_output(wait=True)
            print("♻️ Game reset!")
            self.show_board()

    # --- Print move history ---
    def print_move_history(self):
        print("📜 Move history (PGN):\n")

        # Create PGN game from the current board
        game = chess.pgn.Game.from_board(self.board)

        # --- Determine player sides automatically ---
        if self.is_black:
            white_player = "🤖 AI"
            black_player = "🙍‍♂️ You"
        else:
            white_player = "🙍‍♂️ You"
            black_player = "🤖 AI"

        # --- Fill meaningful PGN headers ---
        game.headers["Event"] = "Friendly Match"
        game.headers["Site"] = "Local Engine"
        game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = white_player
        game.headers["Black"] = black_player
        game.headers["Result"] = self.board.result()

        if self.original_position and self.original_position != chess.STARTING_FEN:
            game.headers["FEN"] = str(self.original_position)
            game.headers["SetUp"] = "1"

        print(str(game))

    # --- Game result helper ---
    def show_game_result(self):
        if self.board.is_checkmate():
            print("🏁 Checkmate! Game over.")
            if self.board.turn == self.is_black:
                print("🙍‍♂️ You win!")
            else:
                print("🤖 AI wins!")
        elif self.board.is_stalemate():
            print("🤝 Draw by Stalemate!")
        else:
            print("🏁 Game over!")

        self.print_move_history()
        self.show_board()

    # --- AI move logic ---
    def ai_move(self, board):
        """Default AI: random move."""
        if board.is_game_over():
            return None
        return random.choice(list(board.legal_moves))

    # --- Launch the GUI ---
    def play(self):
        with self.output:
            clear_output(wait=True)
            self.show_board()

        display(self.move_input, self.make_move_btn, self.ai_btn, self.reset_btn, self.output)
