from pathlib import Path

import chess
import chess.pgn
import numpy as np

from state import State

DATA_DIR = "data"


def get_dataset() -> tuple[list[np.ndarray], list[int]]:
    """
    Generate training dataset in format suitable for neural network.
    """
    X: list = []
    Y: list = []
    game_count: int = 0
    path: Path = Path(DATA_DIR)

    for fn in path.iterdir():
        with fn.open() as pgn:
            while True:
                try:
                    game: chess.pgn.Game = chess.pgn.read_game(pgn)
                except Exception:
                    break

                game_count += 1
                print(f"parsing game {game_count} examples {len(X)}")

                board: chess.Board = game.board()
                result: str = game.headers["Result"]

                # skip games that haven't completed
                if result == "*":
                    continue

                value: int = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}[result]

                # input = board state, label = value/result
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    X.append(State(board).serialize())
                    Y.append(value)
    return X, Y


if __name__ == "__main__":
    X, Y = get_dataset()
    np.savez("processed/dataset.npz", X=X, Y=Y)
