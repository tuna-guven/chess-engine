import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from chess_cli import ChessCLI
from chess_gui import ChessGUI
import pickle
from chess import Board
import random
import time
from tqdm import tqdm

# --- Utility functions ---
def board_to_matrix(board: Board):
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    for move in board.legal_moves:
        row_to, col_to = divmod(move.to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix


def prepare_input(board: Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int

# --- Dataset class ---
class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Model class ---
class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- Full Chess Engine with Training ---
class PyTorchChessEngine(ChessGUI):
    def __init__(self, model=None, move_to_int=None, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = model
        self.move_to_int = move_to_int
        if move_to_int:
            self.int_to_move = {v: k for k, v in move_to_int.items()}

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    # --- Override AI move ---
    def ai_move(self, board: Board):
        if not self.model or not self.move_to_int:
            # Fallback to random move if no model is loaded
            return random.choice(list(board.legal_moves))

        X_tensor = prepare_input(board).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
        logits = logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()

        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = np.argsort(probabilities)[::-1]

        for move_index in sorted_indices:
            move = self.int_to_move[move_index]
            if move in legal_moves_uci:
                return board.parse_uci(move)

        return random.choice(legal_moves)

    def train_model(self, X, y, epochs=5, batch_size=64, lr=1e-4, save_model_path=None, save_mapping_path=None):
        if not self.model or not self.move_to_int:
            raise ValueError("Model and move_to_int mapping must be defined before training.")

        dataset = ChessDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()

                # Update tqdm bar with current average loss
                progress_bar.set_postfix({"avg_loss": f"{running_loss / (progress_bar.n+1):.4f}"})

        # Save model and mapping if requested
        if save_model_path:
            torch.save(self.model.state_dict(), save_model_path)
        if save_mapping_path:
            with open(save_mapping_path, "wb") as f:
                pickle.dump(self.move_to_int, f)

        self.model.eval()

