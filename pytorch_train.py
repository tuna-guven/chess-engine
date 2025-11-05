import os
import torch
import pickle
from tqdm import trange
import chess.pgn
from pytorch_engine import PyTorchChessEngine, ChessModel, create_input_for_nn, encode_moves

# --- 1. Load PGN games ---
LIMIT_OF_GAMES = 20000
pgn_path = r"C:\Users\User\chess_project\data\lichess_elite\lichess_elite_2025-08.pgn"
games = []

with open(pgn_path, "r") as pgn_file:
    for _ in trange(LIMIT_OF_GAMES):
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        games.append(game)

print(f"Loaded {len(games)} games")

# --- 2. Preprocess training data ---
X, y_uci = create_input_for_nn(games)
y, move_to_int = encode_moves(y_uci)

# --- 3. Create engine and train model ---
engine = PyTorchChessEngine(
    model=ChessModel(num_classes=len(move_to_int)),
    move_to_int=move_to_int
)

engine.train_model(
    X, y,
    epochs=50,
    save_model_path=r".\models\TORCH_50EPOCHS.pth",
    save_mapping_path=r".\models\move_to_int"
)

print("Training finished. Model and move mapping saved.")
