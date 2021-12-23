import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EPOCHS = 20


class ChessDataset(Dataset):
    def __init__(self):
        self.data = np.load(Path("processed/dataset.npz"))
        self.X = self.data["X"]
        self.Y = self.data["Y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class MLP(nn.Module):
    """
    Feed-forward multi-layer perceptron that predicts probabilities of win/loss/draw.
    """

    def __init__(self):
        super(MLP, self).__init__()

        # input layer
        self.l1 = nn.Linear(69, 100)  # input shape (69,) -> output to layer 100 units
        self.relu1 = nn.ReLU()

        # hidden layer 1
        self.l2 = nn.Linear(100, 100)  # input shape 100 -> output to layer 100 units
        self.relu2 = nn.ReLU()

        # hidden layer 3
        self.l3 = nn.Linear(100, 20)  # input shape 100 -> output to layer 20 units
        self.relu3 = nn.ReLU()

        # output layer
        self.l4 = nn.Linear(
            20, 1
        )  # input shape 20 -> output layer 1 unit (value from -1 to 1)

    def forward(self, X):
        out = self.l1(X)
        out = self.relu1(out)

        out = self.l2(out)
        out = self.relu2(out)

        out = self.l3(out)
        out = self.relu3(out)

        out = self.l4(out)
        return torch.tanh(out)


dataset = ChessDataset()


def train():
    # prepare dataloader
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # train model
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    error = nn.MSELoss()

    print("Training model...")
    all_loss, num_loss = 0, 0
    for epoch in range(EPOCHS):
        for X, y in tqdm(train_dataloader):
            y = y.unsqueeze(-1)
            X = X.float()
            y = y.float()

            # clear gradients
            optimizer.zero_grad()

            # forward pass
            out = model(X)

            # compute loss
            loss = error(out, y)

            # backprop
            loss.backward()

            # update parameters
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1

        # print train loss
        print(f"Epoch {epoch} Loss: {all_loss/num_loss}")

    # store model on disk
    with open("processed/model.pickle", "wb") as fp:
        pickle.dump(model, fp)


if __name__ == "__main__":
    train()
