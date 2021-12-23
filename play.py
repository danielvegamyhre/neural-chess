import pickle

import torch

from state import State
from train import MLP  # noqa: F401


class Valuator:
    def __init__(self):
        with open("processed/model.pickle", "rb") as fp:
            self.model = pickle.load(fp)

    def __call__(self, state: State):
        return self.model(torch.tensor(state.serialize()).float()).item()


if __name__ == "__main__":
    s = State()
    v = Valuator()
    for e in s.edges():
        s.board.push(e)
        print(e, v(s))
        s.board.pop()
