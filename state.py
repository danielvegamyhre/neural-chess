import chess
import numpy as np


class State:
    def __init__(self, board: chess.Board):
        self.board = board

    def serialize(self) -> np.ndarray:
        """
        Convert board into matrix representation for use with numpy.

        state[0:63] are board squares (0 = empty, -int=black piece, +int = white piece)
        state[64] = turn
        state[65] = white kingside castling rights
        state[66] = white queenside castling rights
        state[67] = black kingside castling rights
        state[68] = black queenside castling rights
        """
        assert self.board.is_valid()
        state = np.zeros((8 * 8, 5))

        # https://stackoverflow.com/questions/55876336/is-there-a-way-to-convert-a-python-chess-board-into-a-list-of-integers
        state = np.zeros(64 + 5)
        for sq in chess.scan_reversed(
            self.board.occupied_co[chess.WHITE]
        ):  # Check if white
            state[sq] = self.board.piece_type_at(sq)
        for sq in chess.scan_reversed(
            self.board.occupied_co[chess.BLACK]
        ):  # Check if black
            state[sq] = -self.board.piece_type_at(sq)

        # turn
        state[64] = float(self.board.turn)

        # white kingside castling rights
        state[65] = self.board.has_kingside_castling_rights(chess.WHITE)

        # white queenside castling rights
        state[66] = self.board.has_queenside_castling_rights(chess.WHITE)

        # black kingside castling rights
        state[67] = self.board.has_queenside_castling_rights(chess.BLACK)

        # black queenside castling rights
        state[68] = self.board.has_kingside_castling_rights(chess.BLACK)
        return state

    def edges(self) -> list:
        return list(self.board.legal_moves())
