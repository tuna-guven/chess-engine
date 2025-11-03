import os
import numpy as np
from tqdm import trange
import chess.pgn
import tensorflow as tf

from tensorflow_engine import LichessElite

# --- Load PGN games ---
PGN_FILE = r"C:\Users\User\chess_project\data\lichess_elite\lichess_elite_2025-08.pgn"
LIMIT_OF_GAMES = 5000

games = []
with open(PGN_FILE, "r") as pgn:
    for _ in trange(LIMIT_OF_GAMES, desc="Loading games"):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        games.append(game)

print(f"Loaded {len(games)} games.")

# --- Preprocess data ---
X, y_uci = LichessElite.create_input_for_nn(games)
y, move_to_int = LichessElite.encode_moves(y_uci)
y = tf.keras.models.to_categorical(y, num_classes=len(move_to_int))
X = np.array(X)

# --- Define and compile model ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)), # type: ignore
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(move_to_int), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# --- Train model ---
model.fit(X, y, epochs=5, validation_split=0.1, batch_size=64)

# --- Save model ---
os.makedirs(r".\models", exist_ok=True)
model.save(r".\models\TF_5EPOCHS.keras")

# --- Save move mapping ---
import pickle
with open(r".\models\TF_move_to_int.pkl", "wb") as f:
    pickle.dump(move_to_int, f)
