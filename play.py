import base64
import pickle

import chess
import chess.svg
import torch
from flask import Flask, request
from flask.templating import render_template

from state import State
from train import MLP  # noqa: F401


# value net
class Valuator:
    def __init__(self):
        with open("processed/model.pickle", "rb") as fp:
            self.model = pickle.load(fp)

    def __call__(self, state: State):
        return self.model(torch.tensor(state.serialize()).float()).item()


def find_best_move(s: State, v: Valuator) -> chess.Move:
    moves = []
    for move in s.edges():
        s.board.push(move)
        moves.append((v(s), move))
        s.board.pop()
    sorted_moves = sorted(moves, reverse=s.board.turn, key=lambda x: x[0])
    if not sorted_moves:
        return
    return sorted_moves[0][1]


def computer_move():
    move = find_best_move(s, v)
    s.board.push(move)


# globals
app = Flask(__name__)
s = State()
v = Valuator()


# routes
@app.route("/")
def index():
    return render_template("index.html", start=s.board.fen())


@app.route("/newgame")
def newgame():
    """
    Reset board and return board state in FEN format (e.g. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR)
    """
    s.board.reset()
    return s.board.fen(), 200


@app.route("/selfplay")
def selfplay():
    s.board.reset()
    html = "<html><head>"
    while not s.board.is_game_over():
        computer_move()
        svg = base64.b64encode(
            chess.svg.board(board=s.board, size=350).encode("utf-8")
        ).decode("utf-8")
        html += f'<img width=600 height=600 src="data:image/svg+xml;base64,{svg}"></img><br/>'
    return html


@app.route("/move_coordinates")
def move_coordinates():
    # get move info from request
    move_from = int(request.args.get("from"))
    move_to = int(request.args.get("to"))
    promotion = request.args.get("promotion") == "true"

    # input human move into engine
    move = chess.Move(move_from, move_to, promotion=chess.QUEEN if promotion else None)
    san = s.board.san(move)  # san = standard algebraic notation

    try:
        # human move
        s.board.push_san(san)

        # computer responds with move
        computer_move()
    except ValueError:  # illegal move
        pass

    # check if game over
    if s.board.is_game_over():
        return "game over", 200

    # return board state in FEN format (e.g. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR)
    return s.board.fen(), 200


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)
